"""Tkinter desktop app for the cell_3d_analysis pipeline."""

import copy
import os
import queue
import sys
import threading
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
for path in (SRC, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import gpu_info  # noqa: E402
import io_utils  # noqa: E402
import main as pipeline_main  # noqa: E402
import phase2_intensity  # noqa: E402
from utils import setup_logger, stem  # noqa: E402

CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_gui_defaults(config_path=CONFIG_PATH):
    return io_utils.load_config(str(config_path))


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


class Cell3DApp:
    def __init__(self, root):
        self.root = root
        self.root.title("cell_3d_analysis")
        self.base_config = load_gui_defaults()
        self.messages = queue.Queue()
        self.worker = None
        self.running = False

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

        self._build_ui()
        self._refresh_gpu()
        self._update_preview()
        self.root.after(150, self._poll_messages)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        self._file_row(frame, 0, "CZI", self.czi_var, self._pick_czi)
        ttk.Label(frame, text="Canal segmentacion").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(frame, from_=0, to=20, textvariable=self.channel_var, width=8).grid(row=1, column=1, sticky="w")

        phase1 = ttk.LabelFrame(frame, text="Fase 1 - Segmentacion", padding=8)
        phase1.grid(row=2, column=0, columnspan=3, sticky="ew", pady=6)
        for idx, (label, var) in enumerate([
            ("diameter", self.diameter_var),
            ("flow_threshold", self.flow_var),
            ("cellprob_threshold", self.cellprob_var),
            ("min_size_voxels", self.min_size_var),
        ]):
            ttk.Label(phase1, text=label).grid(row=idx, column=0, sticky="w")
            ttk.Entry(phase1, textvariable=var, width=16).grid(row=idx, column=1, sticky="w")
        ttk.Label(phase1, text="gpu").grid(row=4, column=0, sticky="w")
        ttk.Combobox(phase1, textvariable=self.gpu_var, values=("auto", "true", "false"), width=13, state="readonly").grid(row=4, column=1, sticky="w")

        phase2 = ttk.LabelFrame(frame, text="Fase 2 - Clasificacion Dbc1", padding=8)
        phase2.grid(row=3, column=0, columnspan=3, sticky="ew", pady=6)
        for idx, (label, var, values) in enumerate([
            ("modo umbral", self.threshold_mode_var, ("otsu", "factor", "fixed")),
            ("factor", self.factor_var, None),
            ("threshold", self.threshold_var, None),
            ("red_channel", self.red_var, None),
            ("blue_channel", self.blue_var, None),
        ]):
            ttk.Label(phase2, text=label).grid(row=idx, column=0, sticky="w")
            if values:
                ttk.Combobox(phase2, textvariable=var, values=values, width=13, state="readonly").grid(row=idx, column=1, sticky="w")
            else:
                ttk.Entry(phase2, textvariable=var, width=16).grid(row=idx, column=1, sticky="w")

        self._file_row(frame, 4, "Salida", self.output_var, self._pick_output)
        ttk.Label(frame, textvariable=self.preview_var).grid(row=5, column=0, columnspan=3, sticky="w")

        self.gpu_label = ttk.Label(frame, wraplength=700)
        self.gpu_label.grid(row=6, column=0, columnspan=3, sticky="ew", pady=4)

        buttons = ttk.Frame(frame)
        buttons.grid(row=7, column=0, columnspan=3, sticky="ew", pady=4)
        self.run1_btn = ttk.Button(buttons, text="Ejecutar Fase 1", command=self._run_phase1)
        self.run1_btn.pack(side="left")
        self.run2_btn = ttk.Button(buttons, text="Ejecutar Fase 2", command=self._run_phase2)
        self.run2_btn.pack(side="left", padx=6)

        self.log_text = scrolledtext.ScrolledText(frame, height=16, width=100, state="disabled")
        self.log_text.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=6)
        frame.rowconfigure(8, weight=1)

    def _file_row(self, frame, row, label, var, command):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
        entry = ttk.Entry(frame, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", padx=4)
        entry.bind("<KeyRelease>", lambda _event: self._update_preview())
        ttk.Button(frame, text="...", width=3, command=command).grid(row=row, column=2)

    def _pick_czi(self):
        path = filedialog.askopenfilename(filetypes=[("CZI", "*.czi"), ("Todos", "*.*")])
        if path:
            self.czi_var.set(path)
            self._update_preview()

    def _pick_output(self):
        path = filedialog.askdirectory(initialdir=self.output_var.get())
        if path:
            self.output_var.set(path)
            self._update_preview()

    def _update_preview(self):
        preview = resolve_output_preview(self.czi_var.get(), self.output_var.get())
        self.preview_var.set(f"Fase 1: {preview['phase1']} | Fase 2: {preview['phase2']}")

    def _refresh_gpu(self):
        summary = gpu_info.build_gpu_summary()
        device = summary["cuda_device"] or "CPU"
        state = "CUDA disponible: si" if summary["cuda_available"] else "CUDA disponible: no"
        gpus = ", ".join(gpu["name"] for gpu in summary["gpus_detected"]) or "no detectadas"
        self.gpu_label.configure(text=f"GPU: {state} ({device}). Detectadas: {gpus}. {summary['recommendation_message']}")

    def _set_running(self, running):
        self.running = running
        state = "disabled" if running else "normal"
        self.run1_btn.configure(state=state)
        self.run2_btn.configure(state=state)

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
        except Exception as exc:
            messagebox.showerror("Error de parametros", str(exc))
            return
        self._start_worker(lambda: self._phase1_worker(czi, channel, config))

    def _phase1_worker(self, czi, channel, config):
        logs_dir = Path(config["output_dir"]) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logger(str(logs_dir / "pipeline_log.txt"))
        volume, px_xy, px_z = io_utils.load_czi(czi, channel=channel)
        row = {
            "filename": os.path.basename(czi),
            "px_xy_um": px_xy,
            "px_z_um": px_z,
            "channel_to_segment": channel,
        }
        n_cells = pipeline_main.process_image(
            row, config, logger, volume=volume, progress_callback=self.messages.put
        )
        self.messages.put(f"Fase 1 completada: {n_cells} celula(s).")

    def _run_phase2(self):
        try:
            czi, output = self._validate_common()
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
        self._start_worker(lambda: self._phase2_worker(czi, settings))

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

    def _start_worker(self, target):
        self._set_running(True)
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
                self._set_running(False)
            else:
                self._append_log(item)
        self.root.after(150, self._poll_messages)

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
