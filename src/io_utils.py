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
import re

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


def _size(meta, tag):
    """Lee <Tag>N</Tag> del XML de metadatos del CZI; 1 si no aparece."""
    m = re.search(r"<%s>(\d+)</%s>" % (tag, tag), meta)
    return int(m.group(1)) if m else 1


def _calibration_um(meta):
    """Devuelve (px_xy_um, px_z_um) desde los <Distance Id="X|Y|Z"> en micras."""
    vals = {}
    for axis, value in re.findall(
        r'<Distance Id="([XYZ])">\s*<Value>([0-9eE.\-+]+)', meta
    ):
        vals[axis] = float(value) * 1e6
    px_xy = vals.get("X") or vals.get("Y")
    px_z = vals.get("Z")
    return px_xy, px_z


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


def load_czi(path, channel=0):
    """Lee un CZI, extrae un canal y devuelve (volumen ZYX, px_xy_um, px_z_um).

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    ImportError
        Si falta la dependencia czifile.
    ValueError
        Si el canal no existe o el array reducido no es 3D.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontró el archivo CZI: {path}")

    try:
        import czifile
    except ImportError as exc:
        raise ImportError(
            "Falta la dependencia 'czifile'. Instálala con: pip install czifile"
        ) from exc

    with czifile.CziFile(path) as czi:
        arr = np.asarray(czi.asarray())
        meta = czi.metadata()

    size_c = _size(meta, "SizeC")
    size_z = _size(meta, "SizeZ")
    px_xy, px_z = _calibration_um(meta)

    arr = np.squeeze(arr)

    if channel < 0 or channel >= size_c:
        raise ValueError(
            f"--channel {channel} fuera de rango: el CZI tiene {size_c} canal(es)."
        )

    if size_c > 1:
        c_axes = [i for i, s in enumerate(arr.shape) if s == size_c]
        if not c_axes:
            raise ValueError(
                f"No se pudo identificar el eje de canal del CZI "
                f"(forma {arr.shape}, SizeC={size_c})."
            )
        arr = np.take(arr, int(channel), axis=c_axes[0])

    if arr.ndim == 2 and size_z == 1:
        arr = arr[np.newaxis, :, :]

    if arr.ndim != 3:
        raise ValueError(
            f"Tras reducir canales el array no es 3D (forma {arr.shape}). "
            f"SizeC={size_c}, SizeZ={size_z}. Revisa el archivo."
        )

    z_axes = [i for i, s in enumerate(arr.shape) if s == size_z]
    if z_axes and z_axes[0] != 0:
        arr = np.moveaxis(arr, z_axes[0], 0)

    return arr, px_xy, px_z


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
