"""
measure_3d.py — Medición de propiedades morfológicas 3D por célula.

A partir de una máscara 3D etiquetada (0 = fondo, 1..N = células), calcula
para cada célula:
  - voxel_count          : número de voxeles
  - volume_um3           : volumen en micras cúbicas
  - surface_area_um2     : área de superficie 3D (marching cubes)
  - projected_area_xy_um2: área de la sombra de la célula en el plano XY
  - z_slices_detected    : número de cortes Z en los que aparece la célula
  - bounding box 3D      : índices min/max en Z, Y, X

Se asume que el orden de ejes de la máscara es (Z, Y, X).
"""

import os

import numpy as np
import pandas as pd
from skimage import measure

# Orden de columnas de salida (sin 'filename'; main.py lo inserta al frente).
MEASUREMENT_COLUMNS = [
    "cell_id",
    "voxel_count",
    "volume_um3",
    "surface_area_um2",
    "projected_area_xy_um2",
    "z_slices_detected",
    "bbox_z_min",
    "bbox_z_max",
    "bbox_y_min",
    "bbox_y_max",
    "bbox_x_min",
    "bbox_x_max",
]


def _write_obj(path, verts, faces):
    """Escribe una malla en formato Wavefront .obj.

    verts : (N, 3) array de coordenadas (en unidades físicas, µm).
    faces : (M, 3) array de índices de vértice (0-indexados); .obj usa 1-indexado.
    """
    with open(path, "w", encoding="utf-8") as f:
        for v in verts:
            # marching_cubes devuelve coords en orden (z, y, x); las escribimos
            # como x y z para que un visor 3D estándar las interprete bien.
            f.write(f"v {v[2]:.6f} {v[1]:.6f} {v[0]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def _surface_area_for_cell(cell_mask, spacing):
    """Calcula el área de superficie 3D de una célula binaria.

    Recorta al bounding box y agrega 1 voxel de padding con ceros para que la
    superficie quede cerrada (evita superficies abiertas en bordes y el error
    "level must be within volume data range" cuando el recorte queda todo en 1).

    Returns
    -------
    (area_um2, verts, faces)
        area_um2 = np.nan y verts/faces = None si marching_cubes no puede
        generar una superficie.
    """
    coords = np.argwhere(cell_mask)
    if coords.size == 0:
        return np.nan, None, None

    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0)

    # Recorte ajustado al bounding box.
    sub = cell_mask[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]

    # Padding de 1 voxel en cada lado -> superficie cerrada y rango de datos [0,1].
    padded = np.pad(sub, pad_width=1, mode="constant", constant_values=0)

    try:
        verts, faces, _normals, _values = measure.marching_cubes(
            padded.astype(np.float32),
            level=0.5,
            spacing=spacing,
        )
        area = float(measure.mesh_surface_area(verts, faces))
        return area, verts, faces
    except (ValueError, RuntimeError):
        # Célula degenerada (p.ej. dimensión demasiado pequeña para marchar cubos).
        return np.nan, None, None


def measure_cells_3d(mask, px_xy_um, px_z_um, config, meshes_dir=None, filename_stem=None):
    """Mide todas las células de una máscara 3D etiquetada.

    Parameters
    ----------
    mask : np.ndarray (Z, Y, X), enteros
        0 = fondo; 1..N = células.
    px_xy_um : float
        Tamaño de pixel lateral en µm.
    px_z_um : float
        Espaciado entre cortes Z en µm.
    config : dict
        Configuración del pipeline (sección 'measurements').
    meshes_dir : str | None
        Carpeta donde guardar mallas .obj individuales (si está habilitado).
    filename_stem : str | None
        Nombre base para nombrar los .obj.

    Returns
    -------
    pandas.DataFrame con columnas MEASUREMENT_COLUMNS.
    """
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError(
            f"La máscara debe ser 3D (Z, Y, X); tiene {mask.ndim} dimensiones "
            f"con forma {mask.shape}."
        )

    meas_cfg = config.get("measurements", {})
    calc_surface = meas_cfg.get("calculate_surface_area", True)
    calc_proj = meas_cfg.get("calculate_projected_area_xy", True)
    save_meshes = meas_cfg.get("save_individual_cell_meshes", False)

    # Espaciado físico para marching_cubes en orden (Z, Y, X).
    spacing = (px_z_um, px_xy_um, px_xy_um)
    voxel_volume_um3 = px_xy_um * px_xy_um * px_z_um
    pixel_area_um2 = px_xy_um * px_xy_um

    # Etiquetas presentes excluyendo el fondo (0).
    labels = np.unique(mask)
    labels = labels[labels != 0]

    rows = []
    for cell_id in labels:
        cell_mask = mask == cell_id

        # a. Conteo de voxeles.
        voxel_count = int(cell_mask.sum())

        # b. Volumen.
        volume_um3 = voxel_count * voxel_volume_um3

        # c. Área proyectada XY (colapsa el eje Z).
        if calc_proj:
            projected_pixels = int(np.any(cell_mask, axis=0).sum())
            projected_area_xy_um2 = projected_pixels * pixel_area_um2
        else:
            projected_area_xy_um2 = np.nan

        # d. Número de cortes Z donde aparece la célula.
        z_slices_detected = int(np.any(cell_mask, axis=(1, 2)).sum())

        # e. Bounding box 3D (índices inclusivos).
        coords = np.argwhere(cell_mask)
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)

        # f. Área de superficie 3D.
        if calc_surface:
            surface_area_um2, verts, faces = _surface_area_for_cell(cell_mask, spacing)
            if save_meshes and meshes_dir and verts is not None:
                obj_path = os.path.join(
                    meshes_dir, f"{filename_stem}_cell_{cell_id}.obj"
                )
                _write_obj(obj_path, verts, faces)
        else:
            surface_area_um2 = np.nan

        rows.append({
            "cell_id": int(cell_id),
            "voxel_count": voxel_count,
            "volume_um3": volume_um3,
            "surface_area_um2": surface_area_um2,
            "projected_area_xy_um2": projected_area_xy_um2,
            "z_slices_detected": z_slices_detected,
            "bbox_z_min": int(z_min),
            "bbox_z_max": int(z_max),
            "bbox_y_min": int(y_min),
            "bbox_y_max": int(y_max),
            "bbox_x_min": int(x_min),
            "bbox_x_max": int(x_max),
        })

    # DataFrame con columnas fijas incluso si no hay células (CSV con cabecera).
    df = pd.DataFrame(rows, columns=MEASUREMENT_COLUMNS)
    return df
