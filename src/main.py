"""
main.py — Orquestador del pipeline de análisis 3D de células.

Uso:
    python src/main.py --config config/config.yaml
    python src/main.py --czi ruta/a/imagen.czi --channel 0

(ejecutar desde la raíz del proyecto cell_3d_analysis/)

Flujo:
    1. Lee config.yaml y metadata.csv.
    2. Crea las carpetas de salida.
    3. Por cada imagen del metadata:
        - Carga el z-stack y lo normaliza a (Z, Y, X).
        - Segmenta en 3D con Cellpose.
        - Crea output/<nombre_imagen>/.
        - Guarda la máscara 3D etiquetada.
        - Mide cada célula y exporta el CSV.
        - Genera proyecciones y figura QC.
    4. Escribe el log en output/logs/pipeline_log.txt.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import tifffile

# Permite ejecutar tanto `python src/main.py` como `python main.py` desde src/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import io_utils  # noqa: E402
import measure_3d  # noqa: E402
import phase1_cache  # noqa: E402
import visualize_qc  # noqa: E402
from utils import setup_logger, stem  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Pipeline de análisis 3D de células a partir de z-stacks (Cellpose)."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Ruta al archivo config.yaml (por defecto: config/config.yaml).",
    )
    parser.add_argument(
        "--czi",
        default=None,
        help="Ruta a un archivo .czi para procesarlo directamente.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Índice de canal a extraer cuando se usa --czi (default: 0).",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Omite la confirmación interactiva en modo --czi (para scripts/CI).",
    )
    return parser.parse_args(argv)


def _is_missing(value):
    """True si un valor de metadata está vacío/NaN."""
    return value is None or (isinstance(value, float) and np.isnan(value))


def _report_progress(logger, progress_callback, message, level="info"):
    getattr(logger, level)(message)
    if progress_callback is not None:
        progress_callback(message)


def _prepare_phase1_context(row, config, logger, volume=None, progress_callback=None):
    filename = str(row["filename"])
    file_stem = stem(filename)
    image_output_dir = os.path.join(config["output_dir"], file_stem, "1")
    out_paths = io_utils.create_output_folders(image_output_dir, include_logs=False)
    _report_progress(
        logger, progress_callback, f"[{filename}] carpeta de salida: {image_output_dir}"
    )

    # Validación de calibración física (requerida para volumen/área reales).
    if _is_missing(row["px_xy_um"]) or _is_missing(row["px_z_um"]):
        raise ValueError(
            f"Faltan px_xy_um o px_z_um para '{filename}' en metadata.csv."
        )
    px_xy_um = float(row["px_xy_um"])
    px_z_um = float(row["px_z_um"])
    if px_xy_um <= 0 or px_z_um <= 0:
        raise ValueError(
            f"px_xy_um y px_z_um deben ser positivos para '{filename}' "
            f"(px_xy_um={px_xy_um}, px_z_um={px_z_um})."
        )

    if volume is None:
        # --- Carga del z-stack ---
        image_path = os.path.join(config["input_dir"], filename)
        _report_progress(logger, progress_callback, f"[{filename}] cargando z-stack")
        raw = io_utils.load_zstack(image_path)  # FileNotFoundError / ValueError si falla
        _report_progress(
            logger,
            progress_callback,
            f"[{filename}] cargado: forma {raw.shape}, dtype {raw.dtype}",
        )

        # --- Normalización a (Z, Y, X) ---
        cp_cfg = config.get("cellpose", {})
        channel_to_segment = row.get("channel_to_segment", 0)
        if _is_missing(channel_to_segment):
            channel_to_segment = 0
        volume = io_utils.prepare_volume(
            raw,
            z_axis=cp_cfg.get("z_axis", 0),
            channel_axis=cp_cfg.get("channel_axis", None),
            channel_to_segment=int(channel_to_segment),
        )
        _report_progress(
            logger,
            progress_callback,
            f"[{filename}] volumen normalizado a (Z,Y,X): {volume.shape}",
        )
    else:
        volume = np.asarray(volume)
        if volume.ndim != 3:
            raise ValueError(
                f"El volumen precargado para '{filename}' no es 3D "
                f"(forma {volume.shape})."
            )
        _report_progress(
            logger,
            progress_callback,
            f"[{filename}] volumen CZI cargado a (Z,Y,X): {volume.shape}",
        )

    if volume.shape[0] < 2:
        logger.warning(
            f"[{filename}] solo {volume.shape[0]} corte(s) Z: una imagen 2D no "
            f"permite reconstrucción 3D real. Se necesita un z-stack."
        )

    return {
        "filename": filename,
        "file_stem": file_stem,
        "image_output_dir": image_output_dir,
        "out_paths": out_paths,
        "volume": volume,
        "px_xy_um": px_xy_um,
        "px_z_um": px_z_um,
        "channel": int(row.get("channel_to_segment", 0) or 0),
        "source_path": row.get(
            "source_path",
            os.path.join(config.get("input_dir", ""), filename),
        ),
    }


def _phase1_qc_note(config, channel):
    cp_cfg = config.get("cellpose", {})
    keys = (
        ("canal", channel),
        ("diameter", cp_cfg.get("diameter")),
        ("flow_threshold", cp_cfg.get("flow_threshold")),
        ("cellprob_threshold", cp_cfg.get("cellprob_threshold")),
        ("min_size_voxels", cp_cfg.get("min_size_voxels")),
        ("gpu", cp_cfg.get("gpu")),
    )
    return " | ".join(f"{key}={value if value is not None else 'null'}" for key, value in keys)


def generate_phase1_qc(
    row,
    config,
    logger,
    volume=None,
    progress_callback=None,
    use_cache=True,
    force_new=False,
):
    """Genera o reutiliza una variante QC de Fase 1 sin CSV/mallas finales."""
    ctx = _prepare_phase1_context(row, config, logger, volume, progress_callback)
    filename = ctx["filename"]
    file_stem = ctx["file_stem"]
    phase1_dir = ctx["image_output_dir"]
    figures_dir = ctx["out_paths"]["figures_qc"]
    cache = phase1_cache.load_cache(figures_dir)

    current_signature = phase1_cache.build_phase1_signature(
        ctx["source_path"],
        filename,
        file_stem,
        ctx["volume"],
        ctx["px_xy_um"],
        ctx["px_z_um"],
        ctx["channel"],
        config,
    )

    if use_cache and not force_new:
        variant, comparison = phase1_cache.find_compatible_variant(
            cache, current_signature
        )
        if variant is not None:
            paths = phase1_cache.variant_paths(
                phase1_dir, file_stem, variant["variant_id"]
            )
            if paths["mask"].is_file():
                mask = tifffile.imread(str(paths["mask"]))
                n_cells = int(np.unique(mask[mask != 0]).size)
                _report_progress(
                    logger,
                    progress_callback,
                    f"[{filename}] actualizando figuras QC cacheadas",
                )
                visualize_qc.create_qc_figures(
                    ctx["volume"],
                    mask,
                    file_stem,
                    projections_dir=str(paths["max_projection"].parent),
                    figures_dir=str(paths["qc_overlay"].parent),
                    config=config,
                    n_cells=n_cells,
                    note=_phase1_qc_note(config, ctx["channel"]),
                )
                _report_progress(
                    logger,
                    progress_callback,
                    f"[{filename}] reutilizando variante QC {variant['variant_id']} "
                    f"({n_cells} célula(s)); Cellpose no se ejecutó.",
                )
                cache["active_variant_id"] = variant["variant_id"]
                cache["finalized"] = cache.get("finalized_variant_id") == variant["variant_id"]
                phase1_cache.write_cache(figures_dir, cache)
                return {
                    "context": ctx,
                    "mask": mask,
                    "variant": variant,
                    "comparison": comparison,
                    "reused": True,
                    "n_cells": n_cells,
                }

    # --- Segmentación 3D (import diferido de cellpose dentro del módulo) ---
    import segment_cellpose_3d  # noqa: E402

    _report_progress(logger, progress_callback, f"[{filename}] iniciando segmentación 3D")
    t0 = time.time()
    mask, segment_info = segment_cellpose_3d.segment_3d_cellpose(
        ctx["volume"],
        config,
        ctx["px_xy_um"],
        ctx["px_z_um"],
        return_info=True,
    )
    seg_secs = time.time() - t0

    n_cells = int(np.unique(mask[mask != 0]).size)
    _report_progress(
        logger,
        progress_callback,
        f"[{filename}] segmentación 3D completada en {seg_secs:.1f}s: "
        f"{n_cells} célula(s) detectada(s).",
    )
    if n_cells == 0:
        logger.warning(f"[{filename}] 0 células detectadas por Cellpose.")

    signature = phase1_cache.build_phase1_signature(
        ctx["source_path"],
        filename,
        file_stem,
        ctx["volume"],
        ctx["px_xy_um"],
        ctx["px_z_um"],
        ctx["channel"],
        config,
        segment_info=segment_info,
    )
    existing_ids = [v.get("variant_id") for v in cache.get("variants", [])]
    variant_id = phase1_cache.make_variant_id(
        signature, existing_ids=existing_ids, force_suffix=force_new
    )
    paths = phase1_cache.variant_paths(phase1_dir, file_stem, variant_id)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    io_utils.save_tiff(str(paths["mask"]), mask.astype(np.uint16))
    _report_progress(
        logger,
        progress_callback,
        f"[{filename}] máscara QC guardada: {paths['mask']}",
    )

    _report_progress(logger, progress_callback, f"[{filename}] generando figuras QC")
    visualize_qc.create_qc_figures(
        ctx["volume"],
        mask,
        file_stem,
        projections_dir=str(paths["max_projection"].parent),
        figures_dir=str(paths["qc_overlay"].parent),
        config=config,
        n_cells=n_cells,
        note=_phase1_qc_note(config, ctx["channel"]),
    )
    _report_progress(
        logger,
        progress_callback,
        f"[{filename}] variante QC generada: {variant_id}",
    )

    variant = {
        "variant_id": variant_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "signature": signature,
        "assets": {key: str(path) for key, path in paths.items()},
        "n_cells": n_cells,
        "finalized": False,
    }
    cache = phase1_cache.add_or_update_variant(cache, variant)
    phase1_cache.write_cache(figures_dir, cache)
    return {
        "context": ctx,
        "mask": mask,
        "variant": variant,
        "comparison": {"compatible": True, "blocking_reasons": [], "warnings": []},
        "reused": False,
        "n_cells": n_cells,
    }


def finalize_phase1(
    row,
    config,
    logger,
    volume=None,
    progress_callback=None,
    variant_id=None,
):
    """Finaliza una variante QC y escribe las salidas canónicas de Fase 1."""
    ctx = _prepare_phase1_context(row, config, logger, volume, progress_callback)
    filename = ctx["filename"]
    file_stem = ctx["file_stem"]
    out_paths = ctx["out_paths"]
    phase1_dir = ctx["image_output_dir"]
    cache = phase1_cache.load_cache(out_paths["figures_qc"])

    variant_id = variant_id or cache.get("active_variant_id")
    variant = phase1_cache.get_variant(cache, variant_id)
    if variant is None:
        raise ValueError(
            f"[{filename}] no hay variante QC activa. Ejecuta Generar y luego Finalizar Fase 1."
        )

    current_signature = phase1_cache.build_phase1_signature(
        ctx["source_path"],
        filename,
        file_stem,
        ctx["volume"],
        ctx["px_xy_um"],
        ctx["px_z_um"],
        ctx["channel"],
        config,
    )
    comparison = phase1_cache.compare_phase1_signature(
        variant.get("signature", {}), current_signature
    )
    if not comparison["compatible"]:
        reasons = "; ".join(comparison["blocking_reasons"])
        raise ValueError(f"[{filename}] cache incompatible: {reasons}")

    variant_paths = phase1_cache.variant_paths(phase1_dir, file_stem, variant_id)
    if not variant_paths["mask"].is_file():
        raise FileNotFoundError(
            f"[{filename}] falta la máscara de la variante QC: {variant_paths['mask']}"
        )

    mask = tifffile.imread(str(variant_paths["mask"]))
    n_cells = int(np.unique(mask[mask != 0]).size)

    # --- Guardar máscara 3D etiquetada canónica ---
    canonical = phase1_cache.canonical_paths(phase1_dir, file_stem)
    io_utils.save_tiff(str(canonical["mask"]), mask.astype(np.uint16))
    _report_progress(
        logger,
        progress_callback,
        f"[{filename}] máscara guardada: {canonical['mask']}",
    )

    # --- Mediciones por célula ---
    # Limpia mallas previas de esta imagen para que una re-ejecución no deje
    # .obj obsoletos (los cell_id pueden cambiar entre corridas).
    old_mesh_pattern = os.path.join(out_paths["meshes"], f"{file_stem}_cell_*.obj")
    for old_obj in glob.glob(old_mesh_pattern):
        os.remove(old_obj)

    df = measure_3d.measure_cells_3d(
        mask,
        ctx["px_xy_um"],
        ctx["px_z_um"],
        config,
        meshes_dir=out_paths["meshes"],
        filename_stem=file_stem,
    )
    # Inserta la columna 'filename' al inicio (requerida en el CSV).
    df.insert(0, "filename", filename)
    csv_path = str(canonical["measurements"])
    df.to_csv(csv_path, index=False)
    _report_progress(logger, progress_callback, f"[{filename}] mediciones guardadas: {csv_path}")

    # --- QC: proyecciones + overlay ---
    _report_progress(logger, progress_callback, f"[{filename}] generando figuras QC")
    visualize_qc.create_qc_figures(
        ctx["volume"],
        mask,
        file_stem,
        projections_dir=out_paths["projections"],
        figures_dir=out_paths["figures_qc"],
        config=config,
        n_cells=n_cells,
        note=_phase1_qc_note(config, ctx["channel"]),
    )
    _report_progress(logger, progress_callback, f"[{filename}] figuras QC y proyecciones generadas.")

    assets = dict(canonical)
    assets["meshes_dir"] = out_paths["meshes"]
    cache = phase1_cache.mark_finalized(cache, variant_id, assets)
    cache = phase1_cache.prune_variants(cache, phase1_dir, file_stem, keep_last=3)
    phase1_cache.write_cache(out_paths["figures_qc"], cache)
    return n_cells


def process_image(row, config, logger, volume=None, progress_callback=None):
    """Procesa una imagen completa: genera QC y finaliza Fase 1.

    Lanza excepciones que el llamador captura para continuar con la siguiente.
    """
    qc_result = generate_phase1_qc(
        row,
        config,
        logger,
        volume=volume,
        progress_callback=progress_callback,
        use_cache=False,
    )
    return finalize_phase1(
        row,
        config,
        logger,
        volume=qc_result["context"]["volume"],
        progress_callback=progress_callback,
        variant_id=qc_result["variant"]["variant_id"],
    )


def main(argv=None, progress_callback=None):
    args = parse_args(argv)

    # --- Carga de configuración ---
    try:
        config = io_utils.load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR de configuración: {e}", file=sys.stderr)
        return 1

    # --- Carpeta de salida + logger general (se crean antes de tocar metadata) ---
    logs_dir = os.path.join(config["output_dir"], "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "pipeline_log.txt")
    logger = setup_logger(log_path)
    logger.info("=== Inicio del pipeline de análisis 3D de células ===")
    logger.info(f"Config: {args.config}")

    if args.czi:
        try:
            volume, px_xy, px_z = io_utils.load_czi(args.czi, channel=args.channel)
            if px_xy is None or px_z is None:
                logger.warning(
                    f"[{os.path.basename(args.czi)}] no se pudo extraer calibración "
                    "completa del CZI."
                )
            filename = os.path.basename(args.czi)
            file_stem = stem(filename)
            preview_dir = os.path.join(config["output_dir"], file_stem, "1")
            px_xy_text = px_xy if px_xy is not None else "NO detectada"
            px_z_text = px_z if px_z is not None else "NO detectada"
            print("Resumen CZI:")
            print(f"  Entrada: {args.czi}")
            print(f"  Canal: {args.channel}")
            print(f"  Volumen (Z,Y,X): {volume.shape}")
            print(f"  px_xy_um: {px_xy_text}")
            print(f"  px_z_um: {px_z_text}")
            print(f"  Salida: {preview_dir}")
            if not args.yes:
                answer = input("¿Continuar con estos parámetros? [y/N]: ")
                if answer.strip().lower() not in {"y", "yes", "s", "si", "sí"}:
                    logger.info(
                        f"[{filename}] operación cancelada por el usuario antes de procesar."
                    )
                    print("Cancelado.")
                    return 0
            row = {
                "filename": filename,
                "px_xy_um": px_xy,
                "px_z_um": px_z,
                "channel_to_segment": args.channel,
                "source_path": args.czi,
            }
            if progress_callback is None:
                n_cells = process_image(row, config, logger, volume=volume)
            else:
                n_cells = process_image(
                    row, config, logger, volume=volume, progress_callback=progress_callback
                )
            logger.info(f"CZI mode: {filename} procesado - {n_cells} células.")
            return 0
        except FileNotFoundError as e:
            logger.error(f"Archivo CZI no encontrado: {e}")
            return 1
        except (ImportError, ValueError) as e:
            logger.error(f"Error en modo CZI: {e}")
            return 1
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Error inesperado en modo CZI: {e}")
            return 1

    # --- Carga de metadata ---
    try:
        metadata = io_utils.load_metadata(config["metadata_file"])
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error al cargar metadata: {e}")
        return 1

    logger.info(f"{len(metadata)} imagen(es) listadas en metadata.csv")

    # --- Procesamiento por imagen ---
    summary = []
    for _, row in metadata.iterrows():
        filename = str(row["filename"])
        try:
            if progress_callback is None:
                n_cells = process_image(row, config, logger)
            else:
                n_cells = process_image(row, config, logger, progress_callback=progress_callback)
            summary.append((filename, "OK", n_cells))
        except FileNotFoundError as e:
            logger.error(f"[{filename}] archivo no encontrado: {e}")
            summary.append((filename, "ERROR: no encontrado", 0))
        except ValueError as e:
            logger.error(f"[{filename}] dato inválido: {e}")
            summary.append((filename, "ERROR: valor inválido", 0))
        except Exception as e:  # noqa: BLE001 — no abortar todo el lote por una imagen
            logger.exception(f"[{filename}] error inesperado: {e}")
            summary.append((filename, f"ERROR: {type(e).__name__}", 0))

    # --- Resumen final ---
    logger.info("=== Resumen del pipeline ===")
    ok = sum(1 for _, status, _ in summary if status == "OK")
    for fname, status, n_cells in summary:
        logger.info(f"  {fname}: {status} ({n_cells} células)")
    logger.info(f"Completado: {ok}/{len(summary)} imágenes procesadas con éxito.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
