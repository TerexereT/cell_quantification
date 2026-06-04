"""
inspect_results.py — Visualizaciones adicionales sobre resultados ya procesados.

Genera en output/figures_qc/ tres figuras extra a partir de una imagen que ya
pasó por el pipeline (máscara 3D + CSV de mediciones existentes):

  <nombre>_slices.png   — Todos los cortes Z con overlay máscara + imagen original.
  <nombre>_ortho.png    — Vistas ortogonales (XY / XZ / YZ) por el centro del volumen.
  <nombre>_metrics.png  — Distribución de volumen, área de superficie y cortes Z por célula.

Uso (desde cell_3d_analysis/):
    python tools/inspect_results.py cre+342_17Experiment-2616
    python tools/inspect_results.py cre+342_17Experiment-2616 --config config/config.yaml
    python tools/inspect_results.py cre+342_17Experiment-2616 --out-dir output/mi_carpeta
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from skimage.color import label2rgb

# ---------------------------------------------------------------------------
# Rutas por defecto (relativas a la raíz del proyecto)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT_DIR = os.path.join(PROJECT_ROOT, "input", "raw_zstacks")
DEFAULT_MASKS_DIR = os.path.join(PROJECT_ROOT, "output", "masks_3d")
DEFAULT_MEAS_DIR = os.path.join(PROJECT_ROOT, "output", "measurements")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "output", "figures_qc")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path, label):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} no encontrado: {path}")
    return np.asarray(tifffile.imread(path))


def _normalize(arr):
    """Normaliza a [0, 1] float64 para display."""
    arr = arr.astype(np.float64)
    rng = arr.max() - arr.min()
    return (arr - arr.min()) / rng if rng > 0 else arr


def _overlay_slice(img_slice, mask_slice):
    """Devuelve un array RGB con la máscara sobre la imagen."""
    img_n = _normalize(img_slice)
    if mask_slice.max() == 0:
        return np.stack([img_n, img_n, img_n], axis=-1)
    return label2rgb(mask_slice, image=img_n, bg_label=0, alpha=0.45)


# ---------------------------------------------------------------------------
# Figura 1: panel de slices Z
# ---------------------------------------------------------------------------

def save_slices_figure(image, mask, stem, out_dir):
    """Grilla de todos los cortes Z con overlay."""
    n_z = image.shape[0]
    cols = min(n_z, 6)
    rows = (n_z + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = np.array(axes).reshape(rows, cols)

    for z in range(n_z):
        r, c = divmod(z, cols)
        ax = axes[r, c]
        ax.imshow(_overlay_slice(image[z], mask[z]))
        ax.set_title(f"Z={z}", fontsize=9)
        ax.axis("off")

    # Apaga ejes vacíos
    for idx in range(n_z, rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    fig.suptitle(f"{stem} — cortes Z con overlay", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(out_dir, f"{stem}_slices.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  slices  -> {out_path}")


# ---------------------------------------------------------------------------
# Figura 2: vistas ortogonales XY / XZ / YZ
# ---------------------------------------------------------------------------

def save_ortho_figure(image, mask, stem, out_dir):
    """Sección central en los tres planos ortogonales."""
    cz = image.shape[0] // 2
    cy = image.shape[1] // 2
    cx = image.shape[2] // 2

    panels = [
        ("XY  (Z=%d)" % cz, image[cz], mask[cz]),
        ("XZ  (Y=%d)" % cy, image[:, cy, :], mask[:, cy, :]),
        ("YZ  (X=%d)" % cx, image[:, :, cx], mask[:, :, cx]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, img_sl, msk_sl) in zip(axes, panels):
        ax.imshow(_overlay_slice(img_sl, msk_sl))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(f"{stem} — vistas ortogonales (overlay)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(out_dir, f"{stem}_ortho.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ortho   -> {out_path}")


# ---------------------------------------------------------------------------
# Figura 3: distribución de métricas del CSV
# ---------------------------------------------------------------------------

def save_metrics_figure(csv_path, stem, out_dir):
    """Histogramas de volumen, área de superficie y cortes Z detectados."""
    if not os.path.isfile(csv_path):
        print(f"  metrics -> OMITIDO (no existe {csv_path})")
        return

    df = pd.read_csv(csv_path)
    cols_present = [c for c in ["volume_um3", "surface_area_um2", "z_slices_detected"] if c in df.columns]
    if not cols_present:
        print("  metrics -> OMITIDO (columnas no encontradas en el CSV)")
        return

    n = len(cols_present)
    fig, axes = plt.subplots(1, n, figsize=(n * 5, 4))
    if n == 1:
        axes = [axes]

    labels = {
        "volume_um3": "Volumen (µm³)",
        "surface_area_um2": "Área de superficie (µm²)",
        "z_slices_detected": "Cortes Z por célula",
    }
    for ax, col in zip(axes, cols_present):
        data = df[col].dropna()
        ax.hist(data, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.4)
        ax.set_xlabel(labels.get(col, col), fontsize=10)
        ax.set_ylabel("# células", fontsize=10)
        ax.set_title(
            f"n={len(data)}  med={data.median():.1f}  max={data.max():.1f}",
            fontsize=9,
        )

    fig.suptitle(f"{stem} — distribución de métricas ({len(df)} células)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(out_dir, f"{stem}_metrics.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  metrics -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Genera visualizaciones adicionales de resultados ya procesados."
    )
    parser.add_argument(
        "name",
        help=(
            "Nombre base de la imagen (sin extensión). "
            "Ej: cre+342_17Experiment-2616"
        ),
    )
    parser.add_argument(
        "--input-dir", default=DEFAULT_INPUT_DIR,
        help="Carpeta con los z-stacks originales (default: input/raw_zstacks/).",
    )
    parser.add_argument(
        "--masks-dir", default=DEFAULT_MASKS_DIR,
        help="Carpeta con las máscaras 3D (default: output/masks_3d/).",
    )
    parser.add_argument(
        "--meas-dir", default=DEFAULT_MEAS_DIR,
        help="Carpeta con los CSVs de mediciones (default: output/measurements/).",
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help="Carpeta de salida para las figuras (default: output/figures_qc/).",
    )
    parser.add_argument(
        "--no-slices", action="store_true", help="Omite el panel de slices Z."
    )
    parser.add_argument(
        "--no-ortho", action="store_true", help="Omite las vistas ortogonales."
    )
    parser.add_argument(
        "--no-metrics", action="store_true", help="Omite el plot de métricas."
    )
    args = parser.parse_args()

    stem = args.name
    os.makedirs(args.out_dir, exist_ok=True)

    # Busca el TIF original (puede ser .tif o .tiff)
    img_path = None
    for ext in (".tif", ".tiff"):
        candidate = os.path.join(args.input_dir, stem + ext)
        if os.path.isfile(candidate):
            img_path = candidate
            break
    if img_path is None:
        sys.exit(f"No se encontró imagen original para '{stem}' en {args.input_dir}")

    mask_path = os.path.join(args.masks_dir, f"{stem}_masks_3d.tif")
    csv_path  = os.path.join(args.meas_dir,  f"{stem}_measurements_3d.csv")

    print(f"Cargando imagen: {img_path}")
    image = _load(img_path, "imagen")
    print(f"  forma: {image.shape}  dtype: {image.dtype}")

    print(f"Cargando máscara: {mask_path}")
    mask = _load(mask_path, "máscara")
    print(f"  forma: {mask.shape}  células: {int(np.unique(mask[mask != 0]).size)}")

    # Si la imagen tiene más de 3 ejes (multicanal), toma el canal 0 para display
    if image.ndim == 4:
        print("  imagen multicanal detectada → usando canal 0 para display")
        image = image[:, 0, :, :] if image.shape[1] < image.shape[2] else image[..., 0]

    print(f"\nGenerando figuras en {args.out_dir}/")

    if not args.no_slices:
        save_slices_figure(image, mask, stem, args.out_dir)
    if not args.no_ortho:
        save_ortho_figure(image, mask, stem, args.out_dir)
    if not args.no_metrics:
        save_metrics_figure(csv_path, stem, args.out_dir)

    print("Listo.")


if __name__ == "__main__":
    main()
