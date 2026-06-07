"""Deteccion de GPU/CUDA para la GUI y mensajes al usuario."""

import json
import subprocess


DRIVER_RECOMMENDATION = (
    "Instala/actualiza los drivers de tu GPU para obtener respuestas mas rapidas."
)


def detect_cuda():
    """Devuelve disponibilidad CUDA sin propagar errores de torch."""
    try:
        import torch

        available = bool(torch.cuda.is_available())
        device_name = torch.cuda.get_device_name(0) if available else None
        return {"available": available, "device_name": device_name}
    except Exception:
        return {"available": False, "device_name": None}


def _vendor_from_name(name):
    lowered = name.lower()
    if "nvidia" in lowered or "geforce" in lowered or "quadro" in lowered:
        return "NVIDIA"
    if "amd" in lowered or "radeon" in lowered:
        return "AMD"
    if "intel" in lowered:
        return "Intel"
    return "Otro"


def detect_video_controllers():
    """Enumera controladoras graficas via PowerShell/Get-CimInstance."""
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "Get-CimInstance Win32_VideoController | "
            "Select-Object -ExpandProperty Name | ConvertTo-Json"
        ),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return []

    output = result.stdout.strip()
    if not output:
        return []

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        parsed = output

    names = parsed if isinstance(parsed, list) else [parsed]
    controllers = []
    for name in names:
        if not name:
            continue
        name = str(name)
        controllers.append({"name": name, "vendor": _vendor_from_name(name)})
    return controllers


def build_gpu_summary():
    """Combina CUDA y GPUs detectadas en un resumen para la GUI."""
    cuda = detect_cuda()
    gpus = detect_video_controllers()
    has_amd = any(gpu["vendor"] == "AMD" for gpu in gpus)
    message = DRIVER_RECOMMENDATION
    if has_amd and not cuda["available"]:
        message += (
            " GPU AMD detectada: en Windows PyTorch no soporta ROCm, "
            "asi que no hay aceleracion GPU disponible para AMD y el pipeline usara CPU."
        )
    elif not cuda["available"]:
        message += " CUDA no esta disponible; el pipeline usara CPU."
    else:
        message += " CUDA esta disponible para acelerar Cellpose."

    return {
        "cuda_available": cuda["available"],
        "cuda_device": cuda["device_name"],
        "gpus_detected": gpus,
        "recommendation_message": message,
    }
