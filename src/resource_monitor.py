"""Lectura de uso de CPU/GPU y formateos para la barra de métricas de la GUI.

Funciones puras (parseo/format) testeables sin Tk; las lecturas de sistema
(`read_gpu_usage`, `read_cpu_percent`) aíslan el subprocess/psutil y nunca
propagan excepciones.
"""

import subprocess

NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=utilization.gpu,memory.used,memory.total",
    "--format=csv,noheader,nounits",
]


def parse_nvidia_smi(text):
    """Parsea la salida de NVIDIA_SMI_QUERY (primera GPU). None si no es válida.

    Espera líneas tipo "3, 1795, 8188" (gracias a nounits). Tolera vacío,
    "[N/A]" y basura devolviendo None.
    """
    if not text:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            return None
        try:
            return {
                "util_pct": float(parts[0]),
                "mem_used_mb": float(parts[1]),
                "mem_total_mb": float(parts[2]),
            }
        except ValueError:
            return None
    return None


def read_gpu_usage(timeout=2.0):
    """Uso de GPU vía nvidia-smi. None si no hay NVIDIA / falla / timeout."""
    try:
        result = subprocess.run(
            NVIDIA_SMI_QUERY,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return parse_nvidia_smi(result.stdout)


def read_cpu_percent():
    """Uso de CPU del sistema vía psutil. None si psutil no está disponible."""
    try:
        import psutil

        return psutil.cpu_percent(interval=None)
    except Exception:
        return None


def format_elapsed(secs):
    """Segundos -> 'mm:ss' (o 'h:mm:ss' si >= 1h)."""
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def phase2_eta_seconds(elapsed, done, total):
    """ETA en segundos para Fase 2. None si no hay base suficiente."""
    if done <= 0 or total <= 0 or done >= total:
        return None
    return elapsed / done * (total - done)


def format_metrics(elapsed_secs, cpu_pct, gpu, *, done=None, total=None):
    """Arma la línea de métricas para mostrar encima del log."""
    segments = [f"Tiempo {format_elapsed(elapsed_secs)}"]

    if total:
        done = done or 0
        pct = min(100, int(done / total * 100)) if total else 0
        eta = phase2_eta_seconds(elapsed_secs, done, total)
        piece = f"{pct}% ({done}/{total})"
        if eta is not None:
            piece += f" ETA ~{format_elapsed(eta)}"
        segments.append(piece)

    segments.append("CPU n/d" if cpu_pct is None else f"CPU {cpu_pct:.0f}%")

    if gpu is None:
        segments.append("GPU n/d")
    else:
        used_gb = gpu["mem_used_mb"] / 1024
        total_gb = gpu["mem_total_mb"] / 1024
        segments.append(
            f"GPU {gpu['util_pct']:.0f}% ({used_gb:.1f}/{total_gb:.1f} GB)"
        )

    return " · ".join(segments)


def is_mask_progress(message, prefix):
    """True si `message` es un evento de progreso por máscara (Fase 2)."""
    return isinstance(message, str) and message.startswith(prefix)
