"""
main.py — Orquestador del pipeline de análisis 3D de células.

Uso:
    python src/main.py --config config/config.yaml

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

# Permite ejecutar tanto `python src/main.py` como `python main.py` desde src/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import io_utils  # noqa: E402
import measure_3d  # noqa: E402
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
    return parser.parse_args(argv)


def _is_missing(value):
    """True si un valor de metadata está vacío/NaN."""
    return value is None or (isinstance(value, float) and np.isnan(value))


def process_image(row, config, logger):
    """Procesa una sola imagen del metadata. Devuelve nº de células detectadas.

    Lanza excepciones que el llamador captura para continuar con la siguiente.
    """
    filename = str(row["filename"])
    file_stem = stem(filename)
    image_output_dir = os.path.join(config["output_dir"], file_stem, "1")
    out_paths = io_utils.create_output_folders(image_output_dir, include_logs=False)
    logger.info(f"[{filename}] carpeta de salida: {image_output_dir}")

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

    # --- Carga del z-stack ---
    image_path = os.path.join(config["input_dir"], filename)
    raw = io_utils.load_zstack(image_path)  # FileNotFoundError / ValueError si falla
    logger.info(f"[{filename}] cargado: forma {raw.shape}, dtype {raw.dtype}")

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
    logger.info(f"[{filename}] volumen normalizado a (Z,Y,X): {volume.shape}")

    if volume.shape[0] < 2:
        logger.warning(
            f"[{filename}] solo {volume.shape[0]} corte(s) Z: una imagen 2D no "
            f"permite reconstrucción 3D real. Se necesita un z-stack."
        )

    # --- Segmentación 3D (import diferido de cellpose dentro del módulo) ---
    import segment_cellpose_3d  # noqa: E402

    t0 = time.time()
    mask = segment_cellpose_3d.segment_3d_cellpose(volume, config, px_xy_um, px_z_um)
    seg_secs = time.time() - t0

    n_cells = int(np.unique(mask[mask != 0]).size)
    logger.info(
        f"[{filename}] segmentación 3D completada en {seg_secs:.1f}s: "
        f"{n_cells} célula(s) detectada(s)."
    )
    if n_cells == 0:
        logger.warning(f"[{filename}] 0 células detectadas por Cellpose.")

    # --- Guardar máscara 3D etiquetada ---
    mask_path = os.path.join(out_paths["masks_3d"], f"{file_stem}_masks_3d.tif")
    io_utils.save_tiff(mask_path, mask.astype(np.uint16))

    # --- Mediciones por célula ---
    # Limpia mallas previas de esta imagen para que una re-ejecución no deje
    # .obj obsoletos (los cell_id pueden cambiar entre corridas).
    old_mesh_pattern = os.path.join(out_paths["meshes"], f"{file_stem}_cell_*.obj")
    for old_obj in glob.glob(old_mesh_pattern):
        os.remove(old_obj)

    df = measure_3d.measure_cells_3d(
        mask,
        px_xy_um,
        px_z_um,
        config,
        meshes_dir=out_paths["meshes"],
        filename_stem=file_stem,
    )
    # Inserta la columna 'filename' al inicio (requerida en el CSV).
    df.insert(0, "filename", filename)
    csv_path = os.path.join(
        out_paths["measurements"], f"{file_stem}_measurements_3d.csv"
    )
    df.to_csv(csv_path, index=False)
    logger.info(f"[{filename}] mediciones guardadas: {csv_path}")

    # --- QC: proyecciones + overlay ---
    visualize_qc.create_qc_figures(
        volume,
        mask,
        file_stem,
        projections_dir=out_paths["projections"],
        figures_dir=out_paths["figures_qc"],
        config=config,
    )
    logger.info(f"[{filename}] figuras QC y proyecciones generadas.")

    return n_cells


def main(argv=None):
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
            n_cells = process_image(row, config, logger)
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
