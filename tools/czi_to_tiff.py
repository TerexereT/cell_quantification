"""
czi_to_tiff.py — Convierte un .czi (Zeiss) a un z-stack TIFF (Z, Y, X) listo
para el pipeline, y reporta la calibración física (µm/píxel) del encabezado.

El pipeline principal solo lee .tif/.tiff. Este conversor es el puente para
probar imágenes .czi: extrae el canal indicado, lo reduce a (Z, Y, X), lo guarda
en input/raw_zstacks/ y muestra px_xy_um / px_z_um para que los copies a
metadata.csv (o usa --append-metadata para que lo haga por ti).

Uso:
    python tools/czi_to_tiff.py ruta/a/imagen.czi
    python tools/czi_to_tiff.py ruta/a/imagen.czi --channel 1 --append-metadata

Requiere: pip install czifile
"""

import argparse
import csv
import os
import re

import numpy as np
import tifffile

try:
    import czifile
except ImportError:
    raise SystemExit(
        "Falta la dependencia 'czifile'. Instálala con:  pip install czifile"
    )

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "input", "raw_zstacks")
METADATA_PATH = os.path.join(PROJECT_ROOT, "input", "metadata", "metadata.csv")


def _size(meta, tag):
    """Lee <Tag>N</Tag> del XML de metadatos del CZI; 1 si no aparece."""
    m = re.search(r"<%s>(\d+)</%s>" % (tag, tag), meta)
    return int(m.group(1)) if m else 1


def _calibration_um(meta):
    """Devuelve (px_xy_um, px_z_um) desde los <Distance Id="X|Y|Z"> (en metros)."""
    vals = {}
    for axis, value in re.findall(
        r'<Distance Id="([XYZ])">\s*<Value>([0-9eE.\-+]+)', meta
    ):
        vals[axis] = float(value) * 1e6  # metros → micras
    px_xy = vals.get("X") or vals.get("Y")
    px_z = vals.get("Z")
    return px_xy, px_z


def convert(czi_path, channel=0):
    """Lee el CZI, extrae el canal y devuelve (volumen ZYX, px_xy_um, px_z_um)."""
    with czifile.CziFile(czi_path) as czi:
        arr = np.asarray(czi.asarray())
        meta = czi.metadata()

    size_c = _size(meta, "SizeC")
    size_z = _size(meta, "SizeZ")
    px_xy, px_z = _calibration_um(meta)

    # czifile entrega el array con ejes nombrados; colapsamos los singleton y
    # localizamos C y Z por su tamaño para reducir a (Z, Y, X).
    arr = np.squeeze(arr)

    # Si hay canales, extrae el pedido. Buscamos el eje cuyo tamaño == size_c.
    if size_c > 1:
        c_axes = [i for i, s in enumerate(arr.shape) if s == size_c]
        if c_axes:
            if channel >= size_c:
                raise SystemExit(
                    f"--channel {channel} fuera de rango: el CZI tiene {size_c} canal(es)."
                )
            arr = np.take(arr, channel, axis=c_axes[0])

    # Ahora arr debería ser 3D (Z, Y, X). Si no, error claro.
    if arr.ndim != 3:
        raise SystemExit(
            f"Tras reducir canales el array no es 3D (forma {arr.shape}). "
            f"SizeC={size_c}, SizeZ={size_z}. Revisa el archivo."
        )

    # Coloca Z primero: el eje Z es el que mide size_z (si es ambiguo, asume 0).
    z_axes = [i for i, s in enumerate(arr.shape) if s == size_z]
    if z_axes and z_axes[0] != 0:
        arr = np.moveaxis(arr, z_axes[0], 0)

    return arr, px_xy, px_z


def _append_metadata(filename, px_xy, px_z):
    """Agrega/actualiza la fila de este archivo en metadata.csv."""
    header = ["filename", "px_xy_um", "px_z_um", "channel_to_segment", "notes"]
    rows = []
    if os.path.isfile(METADATA_PATH):
        with open(METADATA_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("filename") != filename]
    rows.append(
        {
            "filename": filename,
            "px_xy_um": px_xy if px_xy is not None else "",
            "px_z_um": px_z if px_z is not None else "",
            "channel_to_segment": 0,
            "notes": "convertido de .czi",
        }
    )
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Convierte un .czi a z-stack TIFF (Z,Y,X).")
    parser.add_argument("czi", help="Ruta al archivo .czi de entrada.")
    parser.add_argument(
        "--channel", type=int, default=0, help="Índice de canal a extraer (default 0)."
    )
    parser.add_argument(
        "--append-metadata",
        action="store_true",
        help="Agrega/actualiza la fila en input/metadata/metadata.csv con la calibración.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.czi):
        raise SystemExit(f"No se encontró el archivo: {args.czi}")

    vol, px_xy, px_z = convert(args.czi, channel=args.channel)

    os.makedirs(RAW_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.czi))[0] + ".tif"
    out_path = os.path.join(RAW_DIR, base)
    tifffile.imwrite(out_path, vol, photometric="minisblack")

    print(f"TIFF guardado: {out_path}")
    print(f"  forma (Z,Y,X): {vol.shape}  dtype: {vol.dtype}")
    print(f"  px_xy_um: {px_xy}")
    print(f"  px_z_um:  {px_z}")
    if vol.shape[0] < 2:
        print("  ⚠️  Solo 1 corte Z: no hay reconstrucción 3D real (se necesita z-stack).")

    if args.append_metadata:
        if px_xy is None or px_z is None:
            print("  ⚠️  No se pudo extraer calibración del CZI; edita metadata.csv a mano.")
        _append_metadata(base, px_xy, px_z)
        print(f"  metadata.csv actualizado con la fila de {base}")


if __name__ == "__main__":
    main()
