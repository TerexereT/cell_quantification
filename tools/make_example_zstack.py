"""
make_example_zstack.py — Genera un z-stack sintético de ejemplo.

Crea esferas brillantes sobre fondo oscuro dentro de un volumen (Z, Y, X) y lo
guarda como input/raw_zstacks/ejemplo_zstack.tif. Sirve para probar el pipeline
sin imágenes reales.

Uso:
    python tools/make_example_zstack.py
"""

import os

import numpy as np
import tifffile


def make_sphere(volume, center, radius, intensity):
    """Dibuja una esfera de intensidad dada en el volumen (in place)."""
    zz, yy, xx = np.ogrid[: volume.shape[0], : volume.shape[1], : volume.shape[2]]
    dist2 = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    volume[dist2 <= radius ** 2] = intensity


def main():
    rng = np.random.default_rng(42)
    shape = (24, 128, 128)  # Z, Y, X
    vol = rng.integers(0, 25, size=shape, dtype=np.uint16)  # fondo con ruido leve

    # Varias "células" esféricas de distintos tamaños y posiciones.
    spheres = [
        ((8, 40, 40), 9, 220),
        ((12, 80, 50), 11, 200),
        ((10, 50, 95), 8, 210),
        ((15, 95, 95), 10, 230),
        ((6, 30, 95), 6, 190),
    ]
    for center, radius, intensity in spheres:
        make_sphere(vol, center, radius, intensity)

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "input",
        "raw_zstacks",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ejemplo_zstack.tif")
    tifffile.imwrite(out_path, vol, photometric="minisblack")
    print(f"Z-stack de ejemplo guardado en: {out_path}  (forma {vol.shape})")


if __name__ == "__main__":
    main()
