"""
io_utils.py — Lectura y escritura de archivos del pipeline.

Responsabilidades:
- Cargar la configuración YAML.
- Cargar y validar el metadata.csv.
- Leer z-stacks TIFF.
- Crear la estructura de carpetas de salida.
- Guardar arrays como TIFF.
"""

import os

import numpy as np
import pandas as pd
import tifffile
import yaml

# Columnas que metadata.csv debe tener como mínimo.
REQUIRED_METADATA_COLUMNS = [
    "filename",
    "px_xy_um",
    "px_z_um",
    "channel_to_segment",
]

# Subcarpetas que se crean dentro de output_dir.
OUTPUT_SUBFOLDERS = [
    "masks_3d",
    "projections",
    "meshes",
    "measurements",
    "figures_qc",
    "logs",
]

# Extensiones de imagen aceptadas.
VALID_TIFF_EXTENSIONS = (".tif", ".tiff")


def load_config(path):
    """Carga el archivo config.yaml y lo devuelve como dict.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    ValueError
        Si el YAML está vacío o no es un mapeo.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            f"El archivo de configuración {path} está vacío o mal formado "
            f"(se esperaba un mapeo clave: valor)."
        )

    return config


def load_metadata(path):
    """Carga metadata.csv y valida que tenga las columnas requeridas.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    ValueError
        Si falta alguna columna requerida.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontró el archivo de metadata: {path}")

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"A metadata.csv le faltan columnas requeridas: {missing}. "
            f"Columnas presentes: {list(df.columns)}"
        )

    return df


def load_zstack(path):
    """Lee un z-stack TIFF y lo devuelve como ndarray de numpy.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    ValueError
        Si la extensión no es .tif/.tiff.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontró la imagen z-stack: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in VALID_TIFF_EXTENSIONS:
        raise ValueError(
            f"Extensión de imagen no válida: '{ext}'. "
            f"Se aceptan únicamente {VALID_TIFF_EXTENSIONS}. Archivo: {path}"
        )

    array = tifffile.imread(path)
    return np.asarray(array)


def create_output_folders(output_dir, include_logs=True):
    """Crea output_dir y sus subcarpetas estándar. Idempotente.

    Parameters
    ----------
    output_dir : str
        Carpeta raíz donde crear las subcarpetas.
    include_logs : bool
        Si es False, omite la subcarpeta logs.

    Returns
    -------
    dict
        Mapeo nombre_subcarpeta -> ruta absoluta, para uso del pipeline.
    """
    paths = {}
    subfolders = OUTPUT_SUBFOLDERS
    if not include_logs:
        subfolders = [sub for sub in OUTPUT_SUBFOLDERS if sub != "logs"]

    for sub in subfolders:
        full = os.path.join(output_dir, sub)
        os.makedirs(full, exist_ok=True)
        paths[sub] = full
    return paths


def save_tiff(path, array):
    """Guarda un ndarray como archivo TIFF.

    Crea la carpeta contenedora si no existe.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    array = np.asarray(array)
    # Para volúmenes 2D/3D en escala de grises forzamos 'minisblack' para que
    # tifffile no interprete un eje pequeño (p.ej. 3 cortes Z) como canales RGB.
    if array.ndim in (2, 3):
        tifffile.imwrite(path, array, photometric="minisblack")
    else:
        tifffile.imwrite(path, array)


def prepare_volume(image, z_axis=0, channel_axis=None, channel_to_segment=0):
    """Normaliza una imagen cargada a un volumen (Z, Y, X) de un solo canal.

    - Si channel_axis no es None, extrae el canal indicado.
    - Reordena los ejes para que Z quede primero.

    Esto garantiza que measure_3d y visualize_qc reciban siempre (Z, Y, X),
    sin importar el layout original (Z×Y×X, Z×C×Y×X, Z×Y×X×C, etc.).

    Raises
    ------
    ValueError
        Si tras la normalización el volumen no es 3D.
    """
    image = np.asarray(image)

    if channel_axis is not None:
        channel_axis = int(channel_axis)
        image = np.take(image, int(channel_to_segment), axis=channel_axis)
        # Al eliminar el eje de canal, los ejes posteriores se desplazan -1.
        if channel_axis < z_axis:
            z_axis = z_axis - 1

    image = np.moveaxis(image, int(z_axis), 0)

    if image.ndim != 3:
        raise ValueError(
            f"Tras normalizar, el volumen no es 3D (forma {image.shape}). "
            f"Revisa z_axis/channel_axis en config.yaml y las dimensiones del TIFF."
        )

    return image
