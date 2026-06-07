"""
conftest.py — Configuración compartida de pytest.

- Agrega src/ al sys.path para importar los módulos del pipeline.
- Provee un fake de Cellpose para testear la segmentación sin instalar torch.
"""

import os
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Agrega src/ al path de importación.
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


class _FakeCellposeModel:
    """Imita cellpose.models.CellposeModel.

    Registra los kwargs con los que se llamó eval() y devuelve una máscara
    predeterminada (controlada por la variable de clase return_mask).
    """

    last_init_kwargs = None
    last_eval_kwargs = None
    return_mask = None

    def __init__(self, **kwargs):
        type(self).last_init_kwargs = kwargs

    def eval(self, image, **kwargs):
        type(self).last_eval_kwargs = kwargs
        if type(self).return_mask is not None:
            mask = type(self).return_mask
        else:
            # Por defecto: una etiqueta en el centro del volumen.
            mask = np.zeros(image.shape[:3], dtype=np.uint16)
        # CellposeModel.eval devuelve (masks, flows, styles).
        return mask, None, None


@pytest.fixture
def fake_cellpose(monkeypatch):
    """Inyecta un módulo 'cellpose' falso en sys.modules.

    Devuelve la clase _FakeCellposeModel para que el test fije return_mask
    e inspeccione last_eval_kwargs / last_init_kwargs.
    """
    _FakeCellposeModel.last_init_kwargs = None
    _FakeCellposeModel.last_eval_kwargs = None
    _FakeCellposeModel.return_mask = None

    fake_cellpose_mod = types.ModuleType("cellpose")
    fake_models_mod = types.ModuleType("cellpose.models")
    fake_models_mod.CellposeModel = _FakeCellposeModel
    fake_cellpose_mod.models = fake_models_mod

    monkeypatch.setitem(sys.modules, "cellpose", fake_cellpose_mod)
    monkeypatch.setitem(sys.modules, "cellpose.models", fake_models_mod)

    return _FakeCellposeModel


def make_cube_mask(shape=(10, 20, 20), z=(2, 7), y=(5, 12), x=(5, 12), label=1):
    """Crea una máscara con un cubo etiquetado. Útil en varios tests."""
    mask = np.zeros(shape, dtype=np.uint16)
    mask[z[0]:z[1], y[0]:y[1], x[0]:x[1]] = label
    return mask
