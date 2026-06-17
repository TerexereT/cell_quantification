"""
visualize_qc.py — Generación de proyecciones y figuras de control de calidad.

Produce:
  - Proyección máxima de la imagen original  (TIFF)
  - Proyección máxima de la máscara etiquetada (TIFF)
  - Figura overlay imagen + máscara            (PNG) para revisión visual rápida
"""

import os

import matplotlib

# Backend no interactivo: permite generar PNG sin entorno gráfico (servidor/Docker).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (debe ir tras matplotlib.use)
import numpy as np  # noqa: E402
from skimage.color import label2rgb  # noqa: E402

from io_utils import save_tiff  # noqa: E402


def _max_projection(volume, axis=0):
    """Proyección de intensidad máxima a lo largo de un eje (por defecto Z)."""
    return np.max(volume, axis=axis)


def create_qc_figures(
    image,
    mask,
    filename_stem,
    projections_dir,
    figures_dir,
    config,
    n_cells=None,
    note=None,
):
    """Crea proyecciones y la figura overlay de QC.

    Parameters
    ----------
    image : np.ndarray
        Imagen original 3D (Z, Y, X) ya reducida a un solo canal.
    mask : np.ndarray (Z, Y, X)
        Máscara etiquetada.
    filename_stem : str
        Nombre base para los archivos de salida.
    projections_dir : str
        Carpeta de salida para las proyecciones TIFF.
    figures_dir : str
        Carpeta de salida para la figura PNG de QC.
    config : dict
        Configuración (sección 'qc').

    Returns
    -------
    dict con las rutas de los archivos generados.
    """
    qc_cfg = config.get("qc", {})
    outputs = {}

    # --- Proyección máxima de la imagen original ---
    img_proj = _max_projection(image, axis=0)
    img_proj_path = os.path.join(projections_dir, f"{filename_stem}_max_projection.tif")
    save_tiff(img_proj_path, img_proj)
    outputs["max_projection"] = img_proj_path

    # --- Proyección máxima de la máscara ---
    mask_proj = _max_projection(mask, axis=0)
    if qc_cfg.get("save_mask_projection", True):
        mask_proj_path = os.path.join(
            projections_dir, f"{filename_stem}_mask_projection.tif"
        )
        save_tiff(mask_proj_path, mask_proj)
        outputs["mask_projection"] = mask_proj_path

    # --- Figura overlay de QC ---
    if qc_cfg.get("save_overlay_projection", True):
        overlay_path = os.path.join(figures_dir, f"{filename_stem}_qc_overlay.png")
        _save_overlay_figure(
            img_proj,
            mask_proj,
            overlay_path,
            filename_stem,
            n_cells=n_cells,
            note=note,
        )
        outputs["qc_overlay"] = overlay_path

    return outputs


def _resolve_cell_count(mask_proj, n_cells=None):
    if n_cells is not None:
        return int(n_cells)
    return int(np.unique(mask_proj[mask_proj != 0]).size)


def _save_overlay_figure(img_proj, mask_proj, out_path, title_stem, n_cells=None, note=None):
    """Genera la figura de 3 paneles: original | máscara | overlay."""
    # Normaliza la imagen a [0, 1] para mostrarla en grises de forma estable.
    img_disp = img_proj.astype(np.float64)
    rng = img_disp.max() - img_disp.min()
    if rng > 0:
        img_disp = (img_disp - img_disp.min()) / rng

    # label2rgb colorea cada célula; bg_label=0 deja el fondo negro.
    overlay = label2rgb(mask_proj, image=img_disp, bg_label=0, alpha=0.45)

    n_cells = _resolve_cell_count(mask_proj, n_cells)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_disp, cmap="gray")
    axes[0].set_title("Proyección máx. original")
    axes[1].imshow(label2rgb(mask_proj, bg_label=0))
    axes[1].set_title("Proyección máx. máscara")
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay ({n_cells} células)")
    for ax in axes:
        ax.axis("off")

    title = title_stem if not note else f"{title_stem}\n{note}"
    fig.suptitle(title, fontsize=14)
    top_margin = 0.88 if note else 0.93
    fig.tight_layout(rect=[0, 0, 1, top_margin])  # deja margen superior para el suptitle
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
