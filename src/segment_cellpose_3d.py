"""
segment_cellpose_3d.py — Segmentación 3D de células con Cellpose.

Ejecuta Cellpose en modo volumétrico (do_3D=True) sobre un z-stack y devuelve
una máscara 3D etiquetada (0 = fondo, 1..N = células).

El código es robusto a la versión de Cellpose:
  - Cellpose v3 acepta `model_type` (p.ej. "cyto3") y `channels` en eval().
  - Cellpose v4 (Cellpose-SAM) eliminó `channels` y deprecó `model_type`.
Se inspeccionan las firmas para pasar solo los argumentos soportados.
"""

import inspect

import numpy as np


def filter_small_objects(mask, min_size_voxels):
    """Pone a 0 toda etiqueta cuyo número de voxeles sea < min_size_voxels.

    No relabela: las etiquetas conservadas mantienen su id original para poder
    cruzarlas con la máscara TIFF guardada.

    Parameters
    ----------
    mask : np.ndarray de enteros
    min_size_voxels : int

    Returns
    -------
    np.ndarray (misma forma y dtype) con los objetos pequeños eliminados.
    """
    mask = np.asarray(mask)
    if min_size_voxels is None or min_size_voxels <= 1:
        return mask

    # Conteo de voxeles por etiqueta (incluye el fondo en el índice 0).
    max_label = int(mask.max())
    if max_label == 0:
        return mask  # No hay objetos.

    counts = np.bincount(mask.ravel(), minlength=max_label + 1)

    # Etiquetas a eliminar (excluye el fondo, índice 0).
    small_labels = np.where(counts < min_size_voxels)[0]
    small_labels = small_labels[small_labels != 0]

    if small_labels.size == 0:
        return mask

    out = mask.copy()
    out[np.isin(mask, small_labels)] = 0
    return out


def _accepts(sig, name):
    """True si la firma acepta el argumento `name`.

    Considera tanto los parámetros explícitos como la presencia de **kwargs
    (VAR_KEYWORD), que acepta cualquier nombre. Sin esto, una versión de
    Cellpose con firma `eval(self, x, **kwargs)` haría que se descartaran
    silenciosamente argumentos clave como anisotropy o do_3D.
    """
    params = sig.parameters
    if name in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _resolve_gpu(gpu_setting):
    """Resuelve la opción 'gpu' del config a un bool.

    Acepta:
      - True / False  -> se respeta tal cual.
      - "auto" (str)  -> usa GPU solo si torch detecta CUDA disponible.
    Si se pide GPU explícita pero no hay CUDA, cae a CPU con un aviso.
    """
    is_auto = isinstance(gpu_setting, str) and gpu_setting.strip().lower() == "auto"
    want_gpu = None if is_auto else bool(gpu_setting)

    if want_gpu is False:
        return False

    try:
        import torch

        available = bool(torch.cuda.is_available())
    except Exception:
        available = False

    if is_auto:
        print(
            "[cellpose] GPU auto: "
            + ("CUDA disponible -> usando GPU." if available else "sin CUDA -> usando CPU.")
        )
        return available

    if not available:
        print("[cellpose] gpu: true pero CUDA no está disponible -> usando CPU.")
    return available


def _build_model(cellpose_models, model_type, gpu):
    """Instancia CellposeModel pasando model_type solo si la firma lo acepta."""
    ModelClass = cellpose_models.CellposeModel
    sig = inspect.signature(ModelClass.__init__)

    kwargs = {"gpu": gpu}
    if model_type:
        if "model_type" in sig.parameters or _has_var_keyword(sig):
            kwargs["model_type"] = model_type
        elif "pretrained_model" in sig.parameters:
            # Algunas versiones usan pretrained_model en lugar de model_type.
            kwargs["pretrained_model"] = model_type

    return ModelClass(**kwargs)


def _has_var_keyword(sig):
    """True si la firma tiene un parámetro **kwargs."""
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


def _build_eval_kwargs(model, cfg, anisotropy):
    """Construye los kwargs de model.eval() según los soportados por la firma."""
    sig = inspect.signature(model.eval)

    desired = {
        "do_3D": cfg.get("do_3D", True),
        "anisotropy": anisotropy,
        "z_axis": cfg.get("z_axis", 0),
        "channel_axis": cfg.get("channel_axis", None),
        "flow_threshold": cfg.get("flow_threshold", 0.4),
        "cellprob_threshold": cfg.get("cellprob_threshold", 0.0),
        "diameter": cfg.get("diameter", None),
    }

    eval_kwargs = {k: v for k, v in desired.items() if _accepts(sig, k)}

    # 'channels' solo existe en Cellpose v3; en grayscale es [0, 0].
    if "channels" in sig.parameters:
        eval_kwargs["channels"] = [0, 0]

    return eval_kwargs


def segment_3d_cellpose(image, config, px_xy_um, px_z_um):
    """Segmenta un z-stack en 3D con Cellpose.

    Parameters
    ----------
    image : np.ndarray
        Volumen (Z, Y, X) de un solo canal.
    config : dict
        Configuración completa del pipeline (usa la sección 'cellpose').
    px_xy_um : float
        Tamaño de pixel lateral en µm.
    px_z_um : float
        Espaciado entre cortes Z en µm.

    Returns
    -------
    np.ndarray (Z, Y, X) de enteros: máscara etiquetada filtrada por tamaño.
    """
    # Import diferido: permite testear el resto del paquete sin cellpose instalado.
    from cellpose import models as cellpose_models

    cfg = config.get("cellpose", {})

    # Anisotropía = relación entre el espaciado Z y el lateral. Le dice a Cellpose
    # cuánto más "separados" están los cortes Z respecto a los pixeles XY.
    anisotropy = float(px_z_um) / float(px_xy_um)

    model = _build_model(
        cellpose_models,
        model_type=cfg.get("model_type"),
        gpu=_resolve_gpu(cfg.get("gpu", False)),
    )

    eval_kwargs = _build_eval_kwargs(model, cfg, anisotropy)

    # model.eval devuelve (masks, flows, styles) en CellposeModel
    # o (masks, flows, styles, diams) en Cellpose; tomamos siempre el primero.
    result = model.eval(image, **eval_kwargs)
    masks = result[0] if isinstance(result, (tuple, list)) else result

    masks = np.asarray(masks)

    # Descarta objetos demasiado pequeños.
    masks = filter_small_objects(masks, cfg.get("min_size_voxels", 0))

    return masks
