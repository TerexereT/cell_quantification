"""
phase2_intensity.py — Clasificación Dbc1+/Dbc1- por intensidad roja.

Uso desde la raíz de cell_3d_analysis:
    python tools/phase2_intensity.py "../.claude/cre+342_17Experiment-2616.czi" output/
"""

import argparse
import csv
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from skimage.color import label2rgb  # noqa: E402
from skimage.filters import threshold_otsu  # noqa: E402

try:
    import czifile
except ImportError:
    czifile = None


DEFAULT_THRESHOLD_FACTOR = 1.6
NEGATIVE_LABEL = "Dbc1-"
POSITIVE_LABEL = "Dbc1+"
MASK_SUFFIX = "_masks_3d"
POSITIVE_MASK_SUFFIX = "_masks_dbc1_positive"


def _size_from_metadata(meta, tag):
    match = re.search(r"<%s>(\d+)</%s>" % (tag, tag), meta)
    return int(match.group(1)) if match else 1


def load_czi_dual_channel(czi_path, red_channel=0, blue_channel=1):
    """Lee un CZI y devuelve (canal_rojo, canal_azul) como volúmenes (Z, Y, X).

    red_channel / blue_channel: índices de canal dentro del CZI (0-based,
    igual que czifile). En Fiji los canales se muestran como C=1, C=2, etc.
    (1-indexed), así que C=1 en Fiji = índice 0 aquí.
    """
    czi_path = Path(czi_path)
    if not czi_path.is_file():
        raise FileNotFoundError(f"No se encontró el CZI: {czi_path}")
    if czifile is None:
        raise ImportError("Falta la dependencia 'czifile'. Instálala con: pip install czifile")

    with czifile.CziFile(str(czi_path)) as czi:
        arr = np.asarray(czi.asarray())
        meta = czi.metadata()

    size_c = _size_from_metadata(meta, "SizeC")
    size_z = _size_from_metadata(meta, "SizeZ")
    if size_c < 2:
        raise ValueError(f"El CZI debe tener al menos 2 canales; SizeC={size_c}")
    for idx, name in ((red_channel, "red_channel"), (blue_channel, "blue_channel")):
        if idx >= size_c:
            raise ValueError(f"--{name} {idx} fuera de rango: el CZI tiene {size_c} canales (0–{size_c-1}).")

    arr = np.squeeze(arr)
    channel_axes = [i for i, size in enumerate(arr.shape) if size == size_c]
    if not channel_axes:
        raise ValueError(f"No se pudo localizar el eje de canales en forma {arr.shape}")
    channel_axis = channel_axes[0]

    print(f"  CZI shape (squeeze): {arr.shape}  |  eje canales: {channel_axis}  |  rojo=canal{red_channel}  azul=canal{blue_channel}")
    red = _extract_channel_zyx(arr, channel_axis, red_channel, size_z)
    blue = _extract_channel_zyx(arr, channel_axis, blue_channel, size_z)
    return red, blue


def _extract_channel_zyx(arr, channel_axis, channel, size_z):
    vol = np.take(arr, channel, axis=channel_axis)
    if vol.ndim != 3:
        vol = np.squeeze(vol)
    if vol.ndim != 3:
        raise ValueError(
            f"Tras extraer canal {channel}, el volumen no es 3D: forma {vol.shape}"
        )

    z_axes = [i for i, size in enumerate(vol.shape) if size == size_z]
    if z_axes and z_axes[0] != 0:
        vol = np.moveaxis(vol, z_axes[0], 0)
    return np.asarray(vol)


def max_project(volume):
    return np.max(np.asarray(volume), axis=0)


def normalize_for_display(image):
    image = np.asarray(image, dtype=np.float64)
    lo, hi = np.percentile(image, (1, 99)) if image.size else (0, 0)
    if hi <= lo:
        lo, hi = float(np.min(image)), float(np.max(image))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float64)
    return np.clip((image - lo) / (hi - lo), 0, 1)


def discover_mask_files(output_dir):
    """Encuentra máscaras originales en subcarpetas output/*/1/masks_3d/*.tif."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return []

    mask_files = []
    for child in sorted(p for p in output_dir.iterdir() if p.is_dir() and p.name != "logs"):
        phase1_dir = child / "1"
        masks_dir = phase1_dir / "masks_3d"
        if not masks_dir.is_dir():
            print(f"Warning: sin 1/masks_3d/ en {child}; skip")
            continue
        masks = sorted(masks_dir.glob("*.tif")) + sorted(masks_dir.glob("*.tiff"))
        originals = [p for p in masks if POSITIVE_MASK_SUFFIX not in p.stem]
        if not originals:
            print(f"Warning: sin máscaras TIFF originales en {masks_dir}; skip")
        mask_files.extend(originals)
    return mask_files


def image_stem_from_mask(mask_path):
    stem = Path(mask_path).stem
    if stem.endswith(MASK_SUFFIX):
        return stem[: -len(MASK_SUFFIX)]
    return stem


def measure_intensity(red_proj, mask_proj, red_volume, mask_3d,
                      threshold_factor=None, threshold_value=None):
    """Calcula métricas 2D (proyección máx) y 3D (volumen completo) por célula.

    2D: area_px, mean_intensity_red, IntDen — sobre la proyección máxima en Z.
    3D: voxel_count_3d, IntDen_3D, mean_intensity_3D — sobre todos los voxeles.
    Clasificación: Otsu sobre mean_intensity_red (2D) salvo override con
    --factor o --threshold.
    """
    red_proj   = np.asarray(red_proj,   dtype=np.float64)
    mask_proj  = np.asarray(mask_proj)
    red_volume = np.asarray(red_volume, dtype=np.float64)
    mask_3d    = np.asarray(mask_3d)

    labels = np.unique(mask_proj)
    labels = labels[labels != 0]

    # BKG 2D (proyección)
    background_pixels = red_proj[mask_proj == 0]
    bkg_pp = float(np.mean(background_pixels)) if background_pixels.size else 0.0

    # BKG 3D (todos los voxeles fuera de máscara)
    background_voxels = red_volume[mask_3d == 0]
    bkg_pp_3d = float(np.mean(background_voxels)) if background_voxels.size else 0.0

    rows = []
    for label in labels:
        # --- 2D ---
        pixels_2d = mask_proj == label
        area      = int(np.count_nonzero(pixels_2d))
        mean_int  = float(np.mean(red_proj[pixels_2d])) if area else 0.0
        intden    = float(area * mean_int)

        # --- 3D ---
        voxels_3d      = mask_3d == label
        voxel_count_3d = int(np.count_nonzero(voxels_3d))
        if voxel_count_3d:
            intden_3d      = float(red_volume[voxels_3d].sum())
            mean_int_3d    = float(red_volume[voxels_3d].mean())
        else:
            intden_3d   = 0.0
            mean_int_3d = 0.0

        rows.append({
            "cell_id":            int(label),
            # 2D
            "area_px":            area,
            "mean_intensity_red": mean_int,
            "mean_intensity_corr": mean_int - bkg_pp,
            "IntDen":             intden,
            # 3D
            "voxel_count_3d":     voxel_count_3d,
            "IntDen_3D":          intden_3d,
            "IntDen_3D_corr":     intden_3d - bkg_pp_3d * voxel_count_3d,
            "mean_intensity_3D":  mean_int_3d,
        })

    if not rows:
        return rows, {
            "bkg_pp": bkg_pp, "bkg_pp_3d": bkg_pp_3d,
            "PromIntDen_BKG": 0.0, "umbral": 0.0,
            "metodo_umbral": "otsu", "n_positivas": 0, "n_negativas": 0,
        }

    areas   = np.array([r["area_px"] for r in rows], dtype=np.float64)
    intdens = np.array([r["IntDen"]  for r in rows], dtype=np.float64)
    bkg_image = bkg_pp * float(np.median(areas))
    bkg_cell  = float(np.min(intdens))
    prom_bkg  = (bkg_image + bkg_cell) / 2.0

    intden_corrected = intdens - prom_bkg
    for row, ic in zip(rows, intden_corrected):
        row["IntDen_corregida"] = float(ic)

    intensities = np.array([r["mean_intensity_red"] for r in rows], dtype=np.float64)

    if threshold_value is not None:
        threshold = float(threshold_value)
        method = f"fixed({threshold_value})"
    elif threshold_factor is not None:
        threshold = float(np.mean(intensities)) - threshold_factor * float(np.std(intensities))
        method = f"mean-{threshold_factor}sd"
    else:
        threshold = float(threshold_otsu(intensities))
        method = "otsu"

    n_pos = 0
    n_neg = 0
    for row in rows:
        classification = POSITIVE_LABEL if row["mean_intensity_red"] >= threshold else NEGATIVE_LABEL
        row["clasificacion"] = classification
        if classification == POSITIVE_LABEL:
            n_pos += 1
        else:
            n_neg += 1

    metadata = {
        "bkg_pp": bkg_pp, "bkg_pp_3d": bkg_pp_3d,
        "PromIntDen_BKG": float(prom_bkg),
        "umbral": float(threshold),
        "metodo_umbral": method,
        "n_positivas": n_pos,
        "n_negativas": n_neg,
    }
    return rows, metadata


def save_dual_overlay_figure(channel_proj, mask_proj, out_path, title):
    img_disp = normalize_for_display(channel_proj)
    overlay = label2rgb(mask_proj, image=img_disp, bg_label=0, alpha=0.45)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_disp, cmap="gray")
    axes[0].set_title("Proyección raw")
    axes[1].imshow(overlay)
    axes[1].set_title("Overlay máscara")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def classification_rgb(mask_proj, classifications):
    rgb = np.zeros((*mask_proj.shape, 3), dtype=np.float32)
    for label, classification in classifications.items():
        pixels = mask_proj == label
        if classification == POSITIVE_LABEL:
            rgb[pixels] = (0.0, 0.85, 0.25)
        else:
            rgb[pixels] = (1.0, 0.05, 0.05)
    return rgb


def save_classification_figure(red_proj, mask_proj, rows, metadata, out_path, title):
    img_disp = normalize_for_display(red_proj)
    classifications = {row["cell_id"]: row["clasificacion"] for row in rows}
    class_rgb = classification_rgb(mask_proj, classifications)
    overlay = np.dstack([img_disp, img_disp, img_disp])
    mask_pixels = mask_proj != 0
    overlay[mask_pixels] = 0.55 * overlay[mask_pixels] + 0.45 * class_rgb[mask_pixels]

    n_total = len(rows)
    n_pos = metadata["n_positivas"]
    n_neg = metadata["n_negativas"]
    threshold = metadata["umbral"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_disp, cmap="gray")
    axes[0].set_title("Canal rojo")
    axes[1].imshow(class_rgb)
    axes[1].set_title("Dbc1+ verde / Dbc1- rojo")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay clasificación")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"{title} | N={n_total}, Dbc1+={n_pos}, Dbc1-={n_neg}, umbral={threshold:.3f}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_measurements_csv(out_path, rows, metadata):
    fieldnames = [
        "cell_id",
        # 2D (proyección máxima)
        "area_px",
        "mean_intensity_red",
        "mean_intensity_corr",
        "IntDen",
        "IntDen_corregida",
        # 3D (volumen completo)
        "voxel_count_3d",
        "IntDen_3D",
        "IntDen_3D_corr",
        "mean_intensity_3D",
        # clasificación y metadatos
        "clasificacion",
        "bkg_pp",
        "bkg_pp_3d",
        "PromIntDen_BKG",
        "umbral",
        "metodo_umbral",
        "n_positivas",
        "n_negativas",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        writer.writerow(
            {
                "cell_id":        "__metadata__",
                "bkg_pp":         metadata["bkg_pp"],
                "bkg_pp_3d":      metadata["bkg_pp_3d"],
                "PromIntDen_BKG": metadata["PromIntDen_BKG"],
                "umbral":         metadata["umbral"],
                "metodo_umbral":  metadata["metodo_umbral"],
                "n_positivas":    metadata["n_positivas"],
                "n_negativas":    metadata["n_negativas"],
            }
        )


def save_positive_mask(mask_path, rows, out_path):
    mask = tifffile.imread(str(mask_path))
    negative_labels = {
        row["cell_id"] for row in rows if row["clasificacion"] == NEGATIVE_LABEL
    }
    positive_mask = np.where(np.isin(mask, list(negative_labels)), 0, mask)
    tifffile.imwrite(str(out_path), positive_mask.astype(mask.dtype), photometric="minisblack")


def process_mask(mask_path, red_proj, blue_proj, red_volume,
                 threshold_factor=None, threshold_value=None):
    mask_path = Path(mask_path)
    # mask_path: output/<exp>/1/masks_3d/<file>.tif
    # Phase 2 outputs go to output/<exp>/2/
    experiment_dir = mask_path.parent.parent.parent
    run_dir = experiment_dir / "2"
    figures_dir = run_dir / "figures_qc"
    measurements_dir = run_dir / "measurements"
    masks_out_dir = run_dir / "masks_3d"
    figures_dir.mkdir(parents=True, exist_ok=True)
    measurements_dir.mkdir(parents=True, exist_ok=True)
    masks_out_dir.mkdir(parents=True, exist_ok=True)

    stem = image_stem_from_mask(mask_path)
    mask_3d   = tifffile.imread(str(mask_path))
    mask_proj = max_project(mask_3d)

    if red_proj.shape != mask_proj.shape or blue_proj.shape != mask_proj.shape:
        raise ValueError(
            f"Forma incompatible en {mask_path}: red={red_proj.shape}, "
            f"blue={blue_proj.shape}, mask_proj={mask_proj.shape}"
        )

    save_dual_overlay_figure(
        blue_proj,
        mask_proj,
        figures_dir / f"{stem}_qc_blue_overlay.png",
        f"{stem} - canal azul",
    )
    save_dual_overlay_figure(
        red_proj,
        mask_proj,
        figures_dir / f"{stem}_qc_red_overlay.png",
        f"{stem} - canal rojo",
    )

    rows, metadata = measure_intensity(
        red_proj, mask_proj, red_volume, mask_3d, threshold_factor, threshold_value
    )
    write_measurements_csv(measurements_dir / f"{stem}_dbc1_intensity.csv", rows, metadata)
    save_positive_mask(mask_path, rows, masks_out_dir / f"{stem}{POSITIVE_MASK_SUFFIX}.tif")
    save_classification_figure(
        red_proj,
        mask_proj,
        rows,
        metadata,
        figures_dir / f"{stem}_dbc1_classification.png",
        stem,
    )

    return {
        "folder": str(run_dir),
        "n_cells": len(rows),
        "n_positive": metadata["n_positivas"],
        "n_negative": metadata["n_negativas"],
        "threshold": metadata["umbral"],
    }


def process_output(czi_path, output_dir, threshold_factor=None, threshold_value=None,
                   red_channel=0, blue_channel=1):
    red_volume, blue_volume = load_czi_dual_channel(czi_path, red_channel, blue_channel)
    red_proj = max_project(red_volume)
    blue_proj = max_project(blue_volume)

    mask_files = discover_mask_files(output_dir)
    if not mask_files:
        print(f"Warning: no se encontraron máscaras en {output_dir}")
        return []

    summaries = []
    for mask_path in mask_files:
        print(f"Procesando {mask_path}")
        summaries.append(process_mask(
            mask_path, red_proj, blue_proj, red_volume,
            threshold_factor, threshold_value,
        ))
    return summaries


def print_summary(summaries):
    if not summaries:
        return
    print("\nResumen:")
    print("carpeta | N_células | N_Dbc1+ | N_Dbc1- | umbral")
    for row in summaries:
        print(
            f"{row['folder']} | {row['n_cells']} | {row['n_positive']} | "
            f"{row['n_negative']} | {row['threshold']:.3f}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fase 2: clasifica células Dbc1+/Dbc1- por intensidad roja."
    )
    parser.add_argument("czi", help="Ruta al archivo CZI con canal rojo=0 y azul=1.")
    parser.add_argument("output", help="Directorio output/ con subcarpetas y masks_3d.")
    parser.add_argument(
        "--factor",
        type=float,
        default=None,
        help=(
            "Usa media - k×SD de mean_intensity_red como umbral. "
            "Ejemplo: --factor 1.0. Si no se pasa, se usa Otsu (default)."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Umbral fijo de mean_intensity_red. Células con valor menor son Dbc1-. "
            "Ejemplo: --threshold 5000. Tiene prioridad sobre --factor y Otsu."
        ),
    )
    parser.add_argument(
        "--red-channel",
        type=int,
        default=0,
        dest="red_channel",
        help="Índice de canal rojo (AF647/Dbc1) en el CZI. Default: 0. En Fiji C=1 = índice 0.",
    )
    parser.add_argument(
        "--blue-channel",
        type=int,
        default=1,
        dest="blue_channel",
        help="Índice de canal azul (DAPI) en el CZI. Default: 1. En Fiji C=2 = índice 1.",
    )
    args = parser.parse_args(argv)

    summaries = process_output(
        args.czi, args.output,
        args.factor, args.threshold,
        args.red_channel, args.blue_channel,
    )
    print_summary(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
