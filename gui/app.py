"""Tkinter desktop app for the cell_3d_analysis pipeline."""

import copy
import os
import queue
import sys

# En builds windowed de PyInstaller (console=False) sys.stdout/stderr son None;
# tqdm/cellpose escriben a stderr al descargar el modelo y crashean. Damos un
# stream seguro que descarta la salida.
class _NullStream:
    def write(self, *a, **k):
        return 0

    def flush(self):
        pass


for _name in ("stdout", "stderr", "stdin"):
    if getattr(sys, _name, None) is None:
        setattr(sys, _name, _NullStream())

import threading
import time
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk


def resource_root():
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parents[1]


def app_root():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return resource_root()


ROOT = resource_root()
APP_ROOT = app_root()
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
GUI_DIR = Path(__file__).resolve().parent
for path in (SRC, TOOLS, GUI_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import gpu_info  # noqa: E402
import io_utils  # noqa: E402
import main as pipeline_main  # noqa: E402
import phase1_cache  # noqa: E402
import phase2_intensity  # noqa: E402
import resource_monitor  # noqa: E402
from utils import setup_logger, stem  # noqa: E402

CONFIG_PATH = ROOT / "config" / "config.yaml"
FORMULA_DOC_PATH = ROOT / "docs" / "calculos_justificacion.md"

PHASE1_FIELDS = (
    ("diameter", "diameter"),
    ("flow_threshold", "flow_threshold"),
    ("cellprob_threshold", "cellprob_threshold"),
    ("min_size_voxels", "min_size_voxels"),
    ("gpu", "gpu"),
)
PHASE2_FIELDS = (
    ("modo umbral", "threshold_mode"),
    ("factor", "factor"),
    ("threshold", "threshold"),
    ("red_channel", "red_channel"),
    ("blue_channel", "blue_channel"),
)

FIELD_HELP = {
    "diameter": (
        "Diametro esperado del nucleo en pixeles. Usa null para autodeteccion; "
        "si las mascaras salen sub/sobredimensionadas, mide varios nucleos en Fiji "
        "y fija el promedio en pixeles."
    ),
    "flow_threshold": (
        "Controla la separacion de objetos pegados. Bajarlo ayuda a separar nucleos "
        "unidos; subirlo ayuda cuando un nucleo aparece partido en varios fragmentos."
    ),
    "cellprob_threshold": (
        "Controla cuan tenue puede ser una celula para ser detectada. Bajarlo detecta "
        "nucleos tenues; subirlo reduce detecciones de fondo."
    ),
    "min_size_voxels": (
        "Descarta objetos menores que este volumen en voxeles. Subelo si aparecen "
        "muchas manchitas espurias en el overlay de control de calidad."
    ),
    "gpu": (
        "Define si Cellpose usa GPU. 'auto' intenta usar CUDA si esta disponible, "
        "'true' fuerza GPU y 'false' ejecuta en CPU."
    ),
    "threshold_mode": (
        "Modo de umbral para clasificar Dbc1. Otsu usa el valle del histograma, "
        "factor usa mean - k x SD y fixed usa un valor raw fijo."
    ),
    "factor": (
        "Factor k para el modo factor. Valores bajos como 0.5 suben el umbral y "
        "generan mas negativas; valores altos como 1.6 o 2.0 bajan el umbral."
    ),
    "threshold": (
        "Valor fijo de intensidad raw usado solo con modo fixed. Debe estar en la "
        "escala del detector, por ejemplo 0-65535 para imagenes de 16 bits."
    ),
    "red_channel": (
        "Indice del canal rojo/AF647 donde se mide la senal Dbc1. Debe coincidir "
        "con el canal de intensidad del archivo CZI."
    ),
    "blue_channel": (
        "Indice del canal azul/DAPI usado para overlays de control. Debe coincidir "
        "con el canal nuclear del archivo CZI."
    ),
}


def load_gui_defaults(config_path=CONFIG_PATH):
    return io_utils.load_config(str(config_path))


def _load_formula_sections(path=FORMULA_DOC_PATH):
    text = Path(path).read_text(encoding="utf-8")
    marker1 = "## Fase 1"
    marker2 = "## Fase 2"
    start1 = text.find(marker1)
    start2 = text.find(marker2)
    missing = []
    if start1 == -1:
        missing.append(marker1)
    if start2 == -1:
        missing.append(marker2)
    if missing:
        raise ValueError(
            "Falta la seccion de justificacion: " + ", ".join(missing)
        )
    if start2 < start1:
        raise ValueError("La seccion ## Fase 2 debe aparecer despues de ## Fase 1.")

    phase1 = text[start1 + len(marker1):start2].strip()
    next_heading = text.find("\n## ", start2 + len(marker2))
    end2 = next_heading if next_heading != -1 else len(text)
    phase2 = text[start2 + len(marker2):end2].strip()
    return {"phase1": phase1, "phase2": phase2}


def _show_help_popup(parent, title, text):
    popup = tk.Toplevel(parent)
    popup.title(title)
    popup.transient(parent)
    popup.resizable(False, False)
    ttk.Label(popup, text=title, font=("TkDefaultFont", 10, "bold")).grid(
        row=0, column=0, sticky="w", padx=12, pady=(12, 4)
    )
    ttk.Label(popup, text=text, wraplength=420, justify="left").grid(
        row=1, column=0, sticky="ew", padx=12
    )
    ttk.Button(popup, text="Cerrar", command=popup.destroy).grid(
        row=2, column=0, sticky="e", padx=12, pady=12
    )
    popup.grab_set()


def _parse_optional_float(value, field_name):
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} debe ser numerico o null.") from exc


def _parse_int(value, field_name):
    try:
        return int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{field_name} debe ser entero.") from exc


def _parse_float(value, field_name):
    try:
        return float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{field_name} debe ser numerico.") from exc


def resolve_output_preview(czi_path, output_dir):
    name = stem(os.path.basename(czi_path)) if czi_path else "<archivo_czi>"
    root = Path(output_dir)
    return {
        "phase1": str(root / name / "1"),
        "phase2": str(root / name / "2"),
    }


def phase1_output_status(czi_path, output_dir):
    """Indica si existe Fase 1 finalizada para el CZI dado.

    Devuelve dict con:
      - ready: True si hay mascara canonica y cache finalizado.
      - qc_ready: True si hay una variante QC pendiente de finalizar.
      - mask_count: cantidad de mascaras canonicas encontradas.
      - message: texto listo para mostrar en la UI.
    """
    if not czi_path:
        return {
            "ready": False,
            "qc_ready": False,
            "status": "missing",
            "mask_count": 0,
            "message": "Selecciona un archivo CZI para verificar la Fase 1.",
        }

    name = stem(os.path.basename(czi_path))
    phase1_dir = Path(output_dir) / name / "1"
    masks_dir = phase1_dir / "masks_3d"
    status = phase1_cache.cache_status(phase1_dir, name)
    canonical_mask = status["canonical_mask"]

    if status["cache"].get("parse_error"):
        return {
            "ready": False,
            "qc_ready": False,
            "status": "cache_incompatible",
            "mask_count": 0,
            "message": status["message"],
        }

    if status["finalized"]:
        return {
            "ready": True,
            "qc_ready": True,
            "status": "finalized",
            "mask_count": 1,
            "message": f"Fase 1 finalizada: {canonical_mask} lista para Fase 2.",
        }

    if status["has_variant"]:
        return {
            "ready": False,
            "qc_ready": True,
            "status": "qc_pending",
            "mask_count": 0,
            "message": status["message"],
        }

    if not masks_dir.is_dir():
        return {
            "ready": False,
            "qc_ready": False,
            "status": "missing",
            "mask_count": 0,
            "message": f"Falta ejecutar la Fase 1: usa Generar y luego Finalizar (no existe {masks_dir}).",
        }

    masks = [canonical_mask] if canonical_mask.is_file() else []
    originals = [
        p for p in masks if phase2_intensity.POSITIVE_MASK_SUFFIX not in p.stem
    ]
    if not originals:
        return {
            "ready": False,
            "qc_ready": False,
            "status": "missing",
            "mask_count": 0,
            "message": (
                "La Fase 2 requiere la máscara final de Fase 1. "
                "Ejecuta Generar y luego Finalizar Fase 1 para este CZI."
            ),
        }

    return {
        "ready": False,
        "qc_ready": False,
        "status": "legacy_unfinalized",
        "mask_count": len(originals),
        "message": (
            "Se encontró una máscara antigua, pero falta cache finalizado. "
            "Ejecuta Generar y luego Finalizar Fase 1 para habilitar Fase 2."
        ),
    }


def discover_phase1_experiments(output_dir):
    """Lista los experimentos con Fase 1 hecha bajo output_dir.

    Escanea subcarpetas y devuelve, ordenados, los nombres que contienen
    1/masks_3d/*_masks_3d.tif (máscaras originales; excluye las
    *_masks_dbc1_positive). Silencioso y tolerante: si output_dir está vacío o
    no existe, devuelve []. Ignora la subcarpeta 'logs'.
    """
    if not output_dir:
        return []
    root = Path(output_dir)
    if not root.is_dir():
        return []

    found = []
    for child in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "logs"):
        phase1_dir = child / "1"
        masks_dir = phase1_dir / "masks_3d"
        if not masks_dir.is_dir():
            continue
        status = phase1_cache.cache_status(phase1_dir, child.name)
        if not status["finalized"]:
            continue
        masks = [status["canonical_mask"]]
        originals = [
            m for m in masks if phase2_intensity.POSITIVE_MASK_SUFFIX not in m.stem
        ]
        if originals:
            found.append(child.name)
    return found


def build_phase1_config(base_config, form_values):
    config = copy.deepcopy(base_config)
    config["output_dir"] = str(form_values["output_dir"])
    cellpose = config.setdefault("cellpose", {})
    cellpose["diameter"] = _parse_optional_float(form_values["diameter"], "diameter")
    cellpose["flow_threshold"] = _parse_float(
        form_values["flow_threshold"], "flow_threshold"
    )
    cellpose["cellprob_threshold"] = _parse_float(
        form_values["cellprob_threshold"], "cellprob_threshold"
    )
    cellpose["min_size_voxels"] = _parse_int(
        form_values["min_size_voxels"], "min_size_voxels"
    )
    gpu_value = str(form_values["gpu"]).strip().lower()
    if gpu_value == "auto":
        cellpose["gpu"] = "auto"
    elif gpu_value == "true":
        cellpose["gpu"] = True
    elif gpu_value == "false":
        cellpose["gpu"] = False
    else:
        raise ValueError("gpu debe ser auto, true o false.")
    return config


def build_phase2_settings(form_values):
    mode = str(form_values["threshold_mode"]).strip().lower()
    if mode not in {"otsu", "factor", "fixed"}:
        raise ValueError("threshold_mode debe ser otsu, factor o fixed.")
    factor = _parse_float(form_values["factor"], "factor")
    threshold = _parse_optional_float(form_values["threshold"], "threshold")
    return {
        "threshold_factor": factor if mode == "factor" else None,
        "threshold_value": threshold if mode == "fixed" else None,
        "red_channel": _parse_int(form_values["red_channel"], "red_channel"),
        "blue_channel": _parse_int(form_values["blue_channel"], "blue_channel"),
        "output_dir": str(form_values["output_dir"]),
    }


def _cached_value_text(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def apply_phase1_values_to_vars(vars_by_key, values):
    applied = False
    mapping = {
        "diameter": "diameter",
        "flow_threshold": "flow_threshold",
        "cellprob_threshold": "cellprob_threshold",
        "min_size_voxels": "min_size_voxels",
        "gpu": "gpu",
        "channel_to_segment": "channel",
    }
    for var_key, value_key in mapping.items():
        if value_key not in values or var_key not in vars_by_key:
            continue
        vars_by_key[var_key].set(_cached_value_text(values[value_key]))
        applied = True
    return applied


class Cell3DApp:
    def __init__(self, root):
        self.root = root
        self.root.title("cell_3d_analysis")
        self.base_config = load_gui_defaults()
        self.messages = queue.Queue()
        self.worker = None
        self.running = False
        self.metrics_var = tk.StringVar(value="")
        self._t0 = None
        self._metrics = {"cpu": None, "gpu": None}
        self._metrics_thread = None
        self._progress_total = None
        self._progress_done = 0

        self.czi_var = tk.StringVar()
        self.output_var = tk.StringVar(
            value=str(APP_ROOT / self.base_config.get("output_dir", "output"))
        )
        self.channel_var = tk.StringVar(value="0")

        cellpose = self.base_config.get("cellpose", {})
        phase2 = self.base_config.get("phase2", phase2_intensity.DEFAULT_PHASE2_CONFIG)
        self.diameter_var = tk.StringVar(value=str(cellpose.get("diameter")))
        self.flow_var = tk.StringVar(value=str(cellpose.get("flow_threshold")))
        self.cellprob_var = tk.StringVar(value=str(cellpose.get("cellprob_threshold")))
        self.min_size_var = tk.StringVar(value=str(cellpose.get("min_size_voxels")))
        self.gpu_var = tk.StringVar(value=str(cellpose.get("gpu", "auto")).lower())
        self.threshold_mode_var = tk.StringVar(value=str(phase2.get("threshold_mode", "otsu")))
        self.factor_var = tk.StringVar(value=str(phase2.get("factor", 1.6)))
        self.threshold_var = tk.StringVar(value=str(phase2.get("threshold")))
        self.red_var = tk.StringVar(value=str(phase2.get("red_channel", 0)))
        self.blue_var = tk.StringVar(value=str(phase2.get("blue_channel", 1)))
        self.preview_var = tk.StringVar()
        self.phase1_status_var = tk.StringVar()
        self.phase1_ready = False
        self.phase1_qc_ready = False
        self._loaded_phase1_cache_for = None

        self._build_ui()
        self._refresh_gpu()
        self._update_preview()
        self.root.after(150, self._poll_messages)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        common = ttk.Frame(self.root, padding=(10, 10, 10, 0))
        common.grid(row=0, column=0, sticky="ew")
        common.columnconfigure(1, weight=1)

        self._file_row(common, 0, "CZI", self.czi_var, self._pick_czi)
        ttk.Label(common, text="Canal segmentacion").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(common, from_=0, to=20, textvariable=self.channel_var, width=8).grid(row=1, column=1, sticky="w")
        self._file_row(common, 2, "Salida", self.output_var, self._pick_output)
        ttk.Label(common, textvariable=self.preview_var).grid(row=3, column=0, columnspan=3, sticky="w")

        self._paned = ttk.PanedWindow(self.root, orient="vertical")
        self._paned.grid(row=1, column=0, sticky="nsew")
        notebook = ttk.Notebook(self._paned)
        self._paned.add(notebook, weight=3)

        frame = ttk.Frame(notebook, padding=10)
        notebook.add(frame, text="Fase 1")
        frame.columnconfigure(1, weight=1)

        frame2 = ttk.Frame(notebook, padding=10)
        notebook.add(frame2, text="Fase 2")
        frame2.columnconfigure(1, weight=1)

        import plot_panel  # noqa: E402
        plot_tab = plot_panel.build_plot_panel(notebook, self.output_var.get)
        notebook.add(plot_tab, text="Graficar")

        self._notebook = notebook
        notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        ttk.Label(
            frame,
            text=(
                "① Selecciona tu archivo CZI y la carpeta de Salida arriba  "
                "② Ajusta los parametros de segmentacion  ③ Pulsa "
                "\"Paso 1: Ejecutar Fase 1\""
            ),
            font=("TkDefaultFont", 10, "bold"),
            wraplength=700,
            justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        phase1 = ttk.LabelFrame(
            frame, text="Paso 1 / Fase 1 — Segmentación 3D de núcleos celulares", padding=8
        )
        phase1.grid(row=1, column=0, columnspan=3, sticky="ew", pady=6)
        phase1.columnconfigure(1, weight=1)
        ttk.Label(
            phase1,
            text=(
                "Detecta y segmenta cada nucleo en 3D (Cellpose) y guarda sus mascaras "
                "QC como variantes. Usa Finalizar para crear los archivos que necesita Fase 2."
            ),
            wraplength=700,
            justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
        phase1_vars = {
            "diameter": self.diameter_var,
            "flow_threshold": self.flow_var,
            "cellprob_threshold": self.cellprob_var,
            "min_size_voxels": self.min_size_var,
        }
        for idx, (label, key) in enumerate(PHASE1_FIELDS[:-1]):
            row = idx + 1
            self._field_row(
                phase1, row, label,
                lambda c, k=key: ttk.Entry(c, textvariable=phase1_vars[k], width=16),
                key,
            )
        self._field_row(
            phase1, 5, "gpu",
            lambda c: ttk.Combobox(c, textvariable=self.gpu_var, values=("auto", "true", "false"), width=13, state="readonly"),
            "gpu",
        )
        ttk.Button(
            phase1,
            text="Ver justificación de cálculos",
            command=lambda: self._show_formula_section(
                "phase1", "Justificación de cálculos - Fase 1"
            ),
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(8, 0))

        self.gpu_label = ttk.Label(frame, wraplength=700)
        self.gpu_label.grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)

        ttk.Label(
            frame,
            text=(
                "Generar crea variantes QC para ajustar Cellpose. Finalizar confirma "
                "la variante activa y crea la mascara, mediciones y mallas que Fase 2 necesita."
            ),
            wraplength=700,
            justify="left",
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))

        phase1_actions = ttk.Frame(frame)
        phase1_actions.grid(row=4, column=0, columnspan=3, sticky="w", pady=4)
        self.run1_btn = ttk.Button(phase1_actions, text="Generar", command=self._run_phase1)
        self.run1_btn.pack(side="left")
        self.finalize1_btn = ttk.Button(
            phase1_actions, text="Finalizar", command=self._run_phase1_finalize
        )
        self.finalize1_btn.pack(side="left", padx=(6, 0))

        # --- Fase 2 tab ---
        ttk.Label(
            frame2,
            text=(
                "① Verifica que la Fase 1 haya generado mascaras (estado abajo)  "
                "② Ajusta los parametros de Dbc1  ③ Pulsa "
                "\"Paso 2: Ejecutar Fase 2\""
            ),
            font=("TkDefaultFont", 10, "bold"),
            wraplength=700,
            justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        phase2 = ttk.LabelFrame(
            frame2, text="Paso 2 / Fase 2 — Medición de intensidad y clasificación Dbc1+/−", padding=8
        )
        phase2.grid(row=1, column=0, columnspan=3, sticky="ew", pady=6)
        phase2.columnconfigure(1, weight=1)
        ttk.Label(
            phase2,
            text=(
                "Mide la intensidad del canal rojo (Dbc1) en las mascaras generadas "
                "por la Fase 1 y clasifica cada nucleo como Dbc1+ o Dbc1-."
            ),
            wraplength=700,
            justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
        ttk.Label(
            phase2,
            text=(
                "Requiere haber ejecutado la Fase 1 (usa sus mascaras de "
                "output/<archivo>/1/masks_3d)."
            ),
            wraplength=700,
            justify="left",
            foreground="#8a4b00",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 6))
        ttk.Button(
            phase2, text="Detectar Fase 1", command=self._detect_phase1
        ).grid(row=2, column=0, sticky="w", pady=(0, 4))
        self.phase1_status_label = ttk.Label(phase2, textvariable=self.phase1_status_var, wraplength=700)
        self.phase1_status_label.grid(row=3, column=0, columnspan=3, sticky="w", pady=(0, 6))
        phase2_vars = {
            "threshold_mode": self.threshold_mode_var,
            "factor": self.factor_var,
            "threshold": self.threshold_var,
            "red_channel": self.red_var,
            "blue_channel": self.blue_var,
        }
        phase2_values = {"threshold_mode": ("otsu", "factor", "fixed")}
        for idx, (label, key) in enumerate(PHASE2_FIELDS):
            row = idx + 4
            var = phase2_vars[key]
            values = phase2_values.get(key)
            if values:
                factory = lambda c, v=var, vals=values: ttk.Combobox(c, textvariable=v, values=vals, width=13, state="readonly")
            else:
                factory = lambda c, v=var: ttk.Entry(c, textvariable=v, width=16)
            self._field_row(phase2, row, label, factory, key)
        ttk.Button(
            phase2,
            text="Ver justificación de cálculos",
            command=lambda: self._show_formula_section(
                "phase2", "Justificación de cálculos - Fase 2"
            ),
        ).grid(row=9, column=0, columnspan=3, sticky="w", pady=(8, 0))

        self.run2_btn = ttk.Button(frame2, text="Paso 2: Ejecutar Fase 2", command=self._run_phase2)
        self.run2_btn.grid(row=2, column=0, sticky="w", pady=4)

        # --- Panel inferior compartido: métricas + barra + log ---
        # (segundo panel del PanedWindow; visible solo en Fase 1 / Fase 2).
        # El sash da altura editable; el padding superior reducido lo acerca
        # al botón "Ejecutar Fase X".
        self.bottom_panel = ttk.Frame(self.root, padding=(10, 2, 10, 10))
        self.bottom_panel.columnconfigure(0, weight=1)
        self.bottom_panel.rowconfigure(2, weight=1)

        metrics_frame = ttk.Frame(self.bottom_panel)
        metrics_frame.grid(row=0, column=0, sticky="ew")
        metrics_frame.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(metrics_frame, mode="indeterminate")
        self.progress.grid(row=0, column=0, sticky="ew")
        ttk.Label(metrics_frame, textvariable=self.metrics_var).grid(
            row=1, column=0, sticky="w", pady=(2, 0)
        )

        self.log_text = scrolledtext.ScrolledText(
            self.bottom_panel, height=16, width=100, state="disabled"
        )
        self.log_text.grid(row=2, column=0, sticky="nsew", pady=(4, 0))

        self._paned.add(self.bottom_panel, weight=2)
        self.root.after_idle(self._init_sash, notebook)
        self._on_tab_changed()

    def _file_row(self, frame, row, label, var, command):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
        entry = ttk.Entry(frame, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", padx=4)
        entry.bind("<KeyRelease>", lambda _event: self._update_preview())
        ttk.Button(frame, text="...", width=3, command=command).grid(row=row, column=2)

    def _field_row(self, parent, row, label_text, widget_factory, key):
        """Coloca etiqueta + campo + boton de ayuda '?' juntos en la misma fila.

        El campo y el '?' van dentro de un mismo contenedor para que el boton
        quede pegado al valor (no al borde derecho de la columna estirada).
        """
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w")
        cell = ttk.Frame(parent)
        cell.grid(row=row, column=1, sticky="w")
        widget = widget_factory(cell)
        widget.pack(side="left")
        ttk.Button(
            cell,
            text="?",
            width=2,
            command=lambda: _show_help_popup(self.root, label_text, FIELD_HELP[key]),
        ).pack(side="left", padx=(4, 0))
        return widget

    def _show_formula_section(self, section_key, title):
        try:
            sections = _load_formula_sections(FORMULA_DOC_PATH)
        except (OSError, ValueError) as exc:
            messagebox.showwarning(
                "Justificación no disponible",
                f"No se pudo cargar la justificación de cálculos:\n{exc}",
            )
            return

        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.transient(self.root)
        popup.geometry("760x420")
        text = scrolledtext.ScrolledText(popup, wrap="word", width=90, height=22)
        text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(0, weight=1)
        text.insert("1.0", sections[section_key])
        text.configure(state="disabled")
        ttk.Button(popup, text="Cerrar", command=popup.destroy).grid(
            row=1, column=0, sticky="e", padx=10, pady=(0, 10)
        )

    def _pick_czi(self):
        path = filedialog.askopenfilename(filetypes=[("CZI", "*.czi"), ("Todos", "*.*")])
        if path:
            self.czi_var.set(path)
            self._update_preview()
            self._maybe_load_phase1_values()

    def _pick_output(self):
        path = filedialog.askdirectory(initialdir=self.output_var.get())
        if path:
            self.output_var.set(path)
            self._update_preview()
            self._maybe_load_phase1_values()

    def _maybe_load_phase1_values(self):
        czi = self.czi_var.get().strip()
        output = self.output_var.get().strip()
        if not czi or not output:
            return False
        key = (czi, output)
        if self._loaded_phase1_cache_for == key:
            return False

        figures_dir = Path(output) / stem(os.path.basename(czi)) / "1" / "figures_qc"
        cache = phase1_cache.load_cache(figures_dir)
        values = phase1_cache.latest_phase1_values(cache)
        self._loaded_phase1_cache_for = key
        if not values:
            return False

        return apply_phase1_values_to_vars(
            {
                "diameter": self.diameter_var,
                "flow_threshold": self.flow_var,
                "cellprob_threshold": self.cellprob_var,
                "min_size_voxels": self.min_size_var,
                "gpu": self.gpu_var,
                "channel_to_segment": self.channel_var,
            },
            values,
        )

    def _update_preview(self):
        preview = resolve_output_preview(self.czi_var.get(), self.output_var.get())
        self.preview_var.set(f"Fase 1: {preview['phase1']} | Fase 2: {preview['phase2']}")
        self._refresh_phase1_status()

    def _refresh_phase1_status(self):
        status = phase1_output_status(self.czi_var.get(), self.output_var.get())
        self.phase1_ready = status["ready"]
        self.phase1_qc_ready = status.get("qc_ready", False)
        color = "#1a7f37" if status["ready"] else ("#8a4b00" if self.phase1_qc_ready else "#b00020")
        self.phase1_status_label.configure(foreground=color)
        message = status["message"]
        if not status["ready"]:
            experiments = discover_phase1_experiments(self.output_var.get())
            if experiments:
                message += (
                    "  Fase 1 disponible para: "
                    + ", ".join(experiments)
                    + ". Selecciona el CZI correspondiente."
                )
        self.phase1_status_var.set(message)
        if not self.running:
            self.run2_btn.configure(state="normal" if status["ready"] else "disabled")
            self.finalize1_btn.configure(
                state="normal" if self.phase1_qc_ready and not status["ready"] else "disabled"
            )

    def _detect_phase1(self):
        output = self.output_var.get().strip()
        if not output:
            messagebox.showwarning(
                "Detectar Fase 1", "Selecciona primero una carpeta de salida."
            )
            return

        experiments = discover_phase1_experiments(output)
        self._refresh_phase1_status()

        if not experiments:
            self_as_experiment = (Path(output) / "1" / "masks_3d")
            if self_as_experiment.is_dir() and (
                list(self_as_experiment.glob("*.tif"))
                + list(self_as_experiment.glob("*.tiff"))
            ):
                messagebox.showinfo(
                    "Detectar Fase 1",
                    "Parece que seleccionaste la carpeta del experimento.\n"
                    "Elige la carpeta raíz que la contiene (la que tiene dentro "
                    "una subcarpeta por cada imagen).",
                )
            else:
                messagebox.showinfo(
                    "Detectar Fase 1",
                    f"No se encontraron máscaras de Fase 1 en:\n{output}",
                )
            return

        czi_stem = stem(os.path.basename(self.czi_var.get())) if self.czi_var.get() else ""
        if czi_stem in experiments:
            messagebox.showinfo(
                "Detectar Fase 1",
                f"Fase 1 lista para '{czi_stem}'. Puedes ejecutar la Fase 2.",
            )
        else:
            listado = "\n".join(f"  • {name}" for name in experiments)
            messagebox.showinfo(
                "Detectar Fase 1",
                "Se encontró Fase 1 para:\n"
                + listado
                + "\n\nSelecciona el CZI correspondiente (mismo nombre) para "
                "ejecutar la Fase 2.",
            )

    def _refresh_gpu(self):
        summary = gpu_info.build_gpu_summary()
        device = summary["cuda_device"] or "CPU"
        state = "CUDA disponible: si" if summary["cuda_available"] else "CUDA disponible: no"
        gpus = ", ".join(gpu["name"] for gpu in summary["gpus_detected"]) or "no detectadas"
        self.gpu_label.configure(text=f"GPU: {state} ({device}). Detectadas: {gpus}. {summary['recommendation_message']}")

    def _set_running(self, running, total=None):
        self.running = running
        state = "disabled" if running else "normal"
        self.run1_btn.configure(state=state)
        self.finalize1_btn.configure(state="disabled" if running else ("normal" if self.phase1_qc_ready and not self.phase1_ready else "disabled"))
        if running:
            self.run2_btn.configure(state="disabled")
            self._progress_total = total
            self._progress_done = 0
            self._t0 = time.monotonic()
            if total:
                self.progress.configure(mode="determinate", maximum=100, value=0)
            else:
                self.progress.configure(mode="indeterminate")
                self.progress.start(12)
            self._metrics = {"cpu": None, "gpu": None}
            self._metrics_thread = threading.Thread(
                target=self._metrics_loop, daemon=True
            )
            self._metrics_thread.start()
        else:
            self.progress.stop()
            if self._progress_total:
                self.progress.configure(value=100)
            self.run2_btn.configure(state="normal" if self.phase1_ready else "disabled")
            self.finalize1_btn.configure(
                state="normal" if self.phase1_qc_ready and not self.phase1_ready else "disabled"
            )

    def _metrics_loop(self):
        while self.running:
            self._metrics["cpu"] = resource_monitor.read_cpu_percent()
            self._metrics["gpu"] = resource_monitor.read_gpu_usage()
            time.sleep(1.0)

    def _init_sash(self, notebook):
        """Coloca el sash dejando el notebook compacto y el log alto."""
        try:
            self._paned.sashpos(0, notebook.winfo_reqheight())
        except tk.TclError:
            pass

    def _bottom_panel_visible(self):
        return str(self.bottom_panel) in self._paned.panes()

    def _on_tab_changed(self, event=None):
        try:
            current = self._notebook.tab(self._notebook.select(), "text")
        except tk.TclError:
            return
        try:
            if current in ("Fase 1", "Fase 2"):
                if not self._bottom_panel_visible():
                    self._paned.add(self.bottom_panel, weight=2)
            elif self._bottom_panel_visible():
                self._paned.forget(self.bottom_panel)
        except tk.TclError:
            pass

    def _append_log(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", str(message) + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _validate_common(self):
        czi = self.czi_var.get().strip()
        if not czi:
            raise ValueError("Selecciona un archivo CZI.")
        if not Path(czi).is_file():
            raise FileNotFoundError(f"No se encontro el CZI: {czi}")
        output = self.output_var.get().strip()
        if not output:
            raise ValueError("Selecciona una carpeta de salida.")
        return czi, output

    def _run_phase1(self):
        try:
            czi, output = self._validate_common()
            config = build_phase1_config(self.base_config, {
                "output_dir": output,
                "diameter": self.diameter_var.get(),
                "flow_threshold": self.flow_var.get(),
                "cellprob_threshold": self.cellprob_var.get(),
                "min_size_voxels": self.min_size_var.get(),
                "gpu": self.gpu_var.get(),
            })
            channel = _parse_int(self.channel_var.get(), "canal segmentacion")
            volume, px_xy, px_z = io_utils.load_czi(czi, channel=channel)
            row = self._phase1_row(czi, channel, px_xy, px_z)
            cache = phase1_cache.load_cache(
                Path(output) / stem(os.path.basename(czi)) / "1" / "figures_qc"
            )
            signature = phase1_cache.build_phase1_signature(
                czi,
                row["filename"],
                stem(row["filename"]),
                volume,
                px_xy,
                px_z,
                channel,
                config,
            )
            variant, comparison = phase1_cache.find_compatible_variant(cache, signature)
            use_cache = False
            force_new = False
            if variant is not None:
                paths = phase1_cache.variant_paths(
                    Path(output) / stem(os.path.basename(czi)) / "1",
                    stem(os.path.basename(czi)),
                    variant["variant_id"],
                )
                if paths["mask"].is_file():
                    warnings = "\n".join(comparison.get("warnings") or [])
                    msg = (
                        "Se encontró una variante QC compatible.\n\n"
                        f"Variante: {variant['variant_id']}\n"
                        "Reutilizarla evita ejecutar Cellpose otra vez y conserva "
                        "la misma segmentación cacheada para estos parámetros.\n\n"
                        "¿Reutilizar esta variante?"
                    )
                    if warnings:
                        msg += "\n\nAdvertencias:\n" + warnings
                    use_cache = messagebox.askyesno("Reutilizar cache de Fase 1", msg)
                    force_new = not use_cache
            elif cache.get("variants"):
                latest = cache["variants"][-1]
                latest_comparison = phase1_cache.compare_phase1_signature(
                    latest.get("signature", {}), signature
                )
                reasons = "\n".join(latest_comparison.get("blocking_reasons") or [])
                if reasons:
                    messagebox.showwarning(
                        "Cache incompatible",
                        "No se puede reutilizar la variante QC más reciente:\n\n"
                        + reasons
                        + "\n\nSe generará una variante nueva.",
                    )
                force_new = True
            elif cache.get("parse_error"):
                messagebox.showwarning("Cache incompatible", cache["parse_error"])
                force_new = True
        except Exception as exc:
            messagebox.showerror("Error de parametros", str(exc))
            return
        self._start_worker(
            lambda: self._phase1_generate_worker(
                czi, channel, config, volume, px_xy, px_z, use_cache, force_new
            )
        )

    def _phase1_row(self, czi, channel, px_xy, px_z):
        return {
            "filename": os.path.basename(czi),
            "px_xy_um": px_xy,
            "px_z_um": px_z,
            "channel_to_segment": channel,
            "source_path": czi,
        }

    def _phase1_generate_worker(self, czi, channel, config, volume, px_xy, px_z, use_cache, force_new):
        logs_dir = Path(config["output_dir"]) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logger(str(logs_dir / "pipeline_log.txt"))
        row = self._phase1_row(czi, channel, px_xy, px_z)
        result = pipeline_main.generate_phase1_qc(
            row,
            config,
            logger,
            volume=volume,
            progress_callback=self.messages.put,
            use_cache=use_cache,
            force_new=force_new,
        )
        action = "reutilizada" if result["reused"] else "generada"
        self.messages.put(
            f"QC Fase 1 {action}: {result['n_cells']} celula(s). Pulsa Finalizar para habilitar Fase 2."
        )

    def _run_phase1_finalize(self):
        try:
            czi, output = self._validate_common()
            config = build_phase1_config(self.base_config, {
                "output_dir": output,
                "diameter": self.diameter_var.get(),
                "flow_threshold": self.flow_var.get(),
                "cellprob_threshold": self.cellprob_var.get(),
                "min_size_voxels": self.min_size_var.get(),
                "gpu": self.gpu_var.get(),
            })
            channel = _parse_int(self.channel_var.get(), "canal segmentacion")
            status = phase1_output_status(czi, output)
            if not status.get("qc_ready"):
                messagebox.showerror(
                    "Falta generar QC",
                    "Ejecuta Generar y luego Finalizar Fase 1 para este CZI.",
                )
                return
        except Exception as exc:
            messagebox.showerror("Error de parametros", str(exc))
            return
        self._start_worker(lambda: self._phase1_finalize_worker(czi, channel, config))

    def _phase1_finalize_worker(self, czi, channel, config):
        logs_dir = Path(config["output_dir"]) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logger(str(logs_dir / "pipeline_log.txt"))
        volume, px_xy, px_z = io_utils.load_czi(czi, channel=channel)
        row = self._phase1_row(czi, channel, px_xy, px_z)
        n_cells = pipeline_main.finalize_phase1(
            row, config, logger, volume=volume, progress_callback=self.messages.put
        )
        self.messages.put(f"Fase 1 finalizada: {n_cells} celula(s). Fase 2 habilitada.")

    def _run_phase2(self):
        try:
            czi, output = self._validate_common()
            status = phase1_output_status(czi, output)
            if not status["ready"]:
                messagebox.showerror(
                    "Falta finalizar la Fase 1",
                    "La Fase 2 requiere la máscara final de Fase 1.\n\n"
                    + status["message"]
                    + "\n\nEjecuta Generar y luego Finalizar Fase 1 para este CZI.",
                )
                return
            settings = build_phase2_settings({
                "threshold_mode": self.threshold_mode_var.get(),
                "factor": self.factor_var.get(),
                "threshold": self.threshold_var.get(),
                "red_channel": self.red_var.get(),
                "blue_channel": self.blue_var.get(),
                "output_dir": output,
            })
        except Exception as exc:
            messagebox.showerror("Error de parametros", str(exc))
            return
        total = len(
            phase2_intensity.discover_mask_files(
                output, experiment=stem(os.path.basename(czi))
            )
        )
        self._start_worker(
            lambda: self._phase2_worker(czi, settings), total=total or None
        )

    def _phase2_worker(self, czi, settings):
        summaries = phase2_intensity.process_output(
            czi,
            settings["output_dir"],
            settings["threshold_factor"],
            settings["threshold_value"],
            settings["red_channel"],
            settings["blue_channel"],
            progress_callback=self.messages.put,
        )
        self.messages.put(f"Fase 2 completada: {len(summaries)} carpeta(s).")

    def _start_worker(self, target, total=None):
        self._set_running(True, total=total)
        self.worker = threading.Thread(target=self._worker_wrapper, args=(target,), daemon=True)
        self.worker.start()

    def _worker_wrapper(self, target):
        try:
            target()
        except Exception as exc:
            self.messages.put(("error", str(exc), traceback.format_exc()))
        finally:
            self.messages.put(("done",))

    def _poll_messages(self):
        while True:
            try:
                item = self.messages.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple) and item[0] == "error":
                self._append_log(item[2])
                messagebox.showerror("Error de ejecucion", item[1])
            elif isinstance(item, tuple) and item[0] == "done":
                self._refresh_phase1_status()
                self._set_running(False)
                self._update_metrics_label()
            else:
                if (
                    self._progress_total
                    and resource_monitor.is_mask_progress(
                        item, phase2_intensity.MASK_PROGRESS_PREFIX
                    )
                ):
                    self._progress_done += 1
                    self.progress.configure(
                        value=min(100, self._progress_done / self._progress_total * 100)
                    )
                self._append_log(item)
        if self.running:
            self._update_metrics_label()
        self.root.after(150, self._poll_messages)

    def _update_metrics_label(self):
        if self._t0 is None:
            return
        elapsed = time.monotonic() - self._t0
        self.metrics_var.set(
            resource_monitor.format_metrics(
                elapsed,
                self._metrics["cpu"],
                self._metrics["gpu"],
                done=self._progress_done,
                total=self._progress_total,
            )
        )

    def _on_close(self):
        if self.running and not messagebox.askyesno(
            "Proceso en curso",
            "Hay una ejecucion en curso. Cerrar ahora puede dejar salidas incompletas. Cerrar?",
        ):
            return
        self.root.destroy()


def main():
    try:
        root = tk.Tk()
        Cell3DApp(root)
        root.mainloop()
    except Exception as exc:
        log_path = app_root() / "cell3d_gui_error.log"
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        try:
            messagebox.showerror(
                "Error al iniciar",
                f"{exc}\n\nDetalle guardado en:\n{log_path}",
            )
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
