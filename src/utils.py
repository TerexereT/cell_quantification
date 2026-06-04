"""
utils.py — Utilidades transversales del pipeline.

Contiene el logger compartido y pequeños helpers reutilizables
(extracción de nombre base de archivo, etc.).
"""

import logging
import os


def stem(filename):
    """Devuelve el nombre de archivo sin la ruta ni la extensión.

    Ejemplo: "input/raw_zstacks/ejemplo_zstack.tif" -> "ejemplo_zstack"
    """
    base = os.path.basename(str(filename))
    return os.path.splitext(base)[0]


def setup_logger(log_path, name="cell_3d_analysis"):
    """Configura un logger que escribe simultáneamente a archivo y a consola.

    Parameters
    ----------
    log_path : str
        Ruta del archivo de log (p.ej. output/logs/pipeline_log.txt).
    name : str
        Nombre del logger.

    Returns
    -------
    logging.Logger
    """
    # Asegura que la carpeta del log exista antes de abrir el FileHandler.
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Evita duplicar handlers si setup_logger se llama más de una vez
    # (p.ej. en tests) — limpia los previos.
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler de archivo (modo 'w' = un log limpio por ejecución)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler de consola
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
