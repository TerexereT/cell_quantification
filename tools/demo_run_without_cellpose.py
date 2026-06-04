"""
demo_run_without_cellpose.py — Smoke test / demo del pipeline SIN Cellpose.

Sustituye temporalmente Cellpose por un segmentador simple (umbral + etiquetado
de componentes conexas) para poder ejecutar TODO el pipeline (carga, medición,
QC, exportación) sin instalar torch/cellpose. Útil para verificar el "plumbing"
del proyecto y generar salidas de ejemplo.

NO usa el modelo de deep learning: la segmentación real requiere Cellpose.
Para resultados reales, ejecuta:  python src/main.py --config config/config.yaml

Uso:
    python tools/demo_run_without_cellpose.py
"""

import os
import sys
import types

import numpy as np
from skimage import measure as skmeasure
from skimage.filters import threshold_otsu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)


class _DemoModel:
    """Segmentador sustituto con la misma interfaz que CellposeModel."""

    def __init__(self, **kwargs):
        pass

    def eval(self, image, **kwargs):
        img = np.asarray(image, dtype=np.float64)
        try:
            thr = threshold_otsu(img)
        except ValueError:
            thr = img.mean()
        binary = img > thr
        labels = skmeasure.label(binary, connectivity=1).astype(np.uint16)
        return labels, None, None


def _install_fake_cellpose():
    fake = types.ModuleType("cellpose")
    models = types.ModuleType("cellpose.models")
    models.CellposeModel = _DemoModel
    fake.models = models
    sys.modules["cellpose"] = fake
    sys.modules["cellpose.models"] = models


def main():
    _install_fake_cellpose()
    import main as pipeline_main

    # Ejecuta desde la raíz del proyecto para resolver las rutas del config.
    os.chdir(ROOT)
    print(">>> DEMO: pipeline con segmentador sustituto (NO Cellpose real)\n")
    rc = pipeline_main.main(["--config", "config/config.yaml"])
    print(f"\n>>> DEMO finalizada con código de salida {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
