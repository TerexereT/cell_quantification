"""Fase 3 — panel de graficas para los CSV de mediciones.

Separa la logica de datos (testeable sin Tk) de la UI embebida con matplotlib.
"""

import os
from pathlib import Path

import pandas as pd

METADATA_CELL_ID = "__metadata__"

# Etiqueta visible -> kind de matplotlib.
CHART_TYPES = {
    "Dispersion (puntos)": "scatter",
    "Linea": "line",
}
DEFAULT_CHART_LABEL = "Dispersion (puntos)"

MEASUREMENT_GLOBS = ("*_measurements_3d.csv", "*_dbc1_intensity.csv")


def load_measurements(path):
    """Lee un CSV de mediciones y devuelve un DataFrame limpio.

    - Elimina la fila de metadatos (`cell_id == "__metadata__"`) de la Fase 2.
    - Convierte a numerico lo que se pueda (columnas de texto quedan como object).
    """
    df = pd.read_csv(path)
    if "cell_id" in df.columns:
        df = df[df["cell_id"].astype(str) != METADATA_CELL_ID].copy()
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            df[col] = converted
    return df.reset_index(drop=True)


def numeric_columns(df):
    """Columnas aptas para un eje numerico (X o Y)."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def discover_measurement_csvs(output_dir):
    """Encuentra todos los CSV de mediciones bajo output_dir (Fase 1 y Fase 2)."""
    root = Path(output_dir)
    if not root.is_dir():
        return []
    found = []
    for pattern in MEASUREMENT_GLOBS:
        found.extend(root.rglob(pattern))
    return sorted({p for p in found}, key=lambda p: str(p).lower())


def draw_plot(ax, df, x_col, y_col, chart_kind):
    """Dibuja en `ax` la grafica pedida. Devuelve el ax para encadenar.

    Lanza ValueError si las columnas no existen o no hay datos validos.
    """
    if x_col not in df.columns:
        raise ValueError(f"La columna X '{x_col}' no existe en el CSV.")
    if y_col not in df.columns:
        raise ValueError(f"La columna Y '{y_col}' no existe en el CSV.")

    # Construir cada eje como Series independiente: si x_col == y_col (estado
    # transitorio al intercambiar X/Y), df[[x_col, y_col]] daría columnas
    # duplicadas y arrays 2D que cuelgan la UI en modo línea.
    xs = pd.to_numeric(df[x_col], errors="coerce")
    ys = pd.to_numeric(df[y_col], errors="coerce")
    data = pd.DataFrame({"x": xs, "y": ys}).dropna()
    if data.empty:
        raise ValueError("No hay filas con valores numericos para esas columnas.")

    ax.clear()
    x = data["x"].to_numpy()
    y = data["y"].to_numpy()
    if chart_kind == "line":
        order = x.argsort()
        ax.plot(x[order], y[order], marker="o", markersize=3, linewidth=1)
    else:
        ax.scatter(x, y, s=12, alpha=0.7)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}")
    ax.grid(True, alpha=0.3)
    return ax


def default_axes(columns):
    """Sugiere (x, y) iniciales: primeras dos columnas distintas disponibles."""
    if not columns:
        return None, None
    x = columns[0]
    y = columns[1] if len(columns) > 1 else columns[0]
    return x, y


def suggested_png_path(csv_path, x_col, y_col):
    """Nombre por defecto para exportar el PNG, junto al CSV de origen."""
    stem = Path(csv_path).stem
    safe = lambda s: "".join(c if c.isalnum() else "_" for c in str(s))
    return str(Path(csv_path).with_name(f"{stem}_{safe(y_col)}_vs_{safe(x_col)}.png"))


# --------------------------------------------------------------------------- UI

def build_plot_panel(parent, get_output_dir):
    """Construye y devuelve el frame de la pestana Fase 3.

    `get_output_dir` es un callable que devuelve la carpeta de salida actual,
    para autodescubrir CSV de mediciones.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    panel = PlotPanel(parent, get_output_dir, tk, filedialog, messagebox, ttk,
                      FigureCanvasTkAgg, Figure)
    return panel.frame


class PlotPanel:
    def __init__(self, parent, get_output_dir, tk, filedialog, messagebox, ttk,
                 FigureCanvasTkAgg, Figure):
        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.get_output_dir = get_output_dir
        self.df = None
        self.csv_path = None

        frame = ttk.Frame(parent, padding=10)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(6, weight=1)
        self.frame = frame

        ttk.Label(
            frame,
            text=(
                "① Elige el archivo CSV de mediciones  ② Selecciona las columnas X e Y "
                "y el tipo de grafica (se actualiza al instante)  ③ Exporta el resultado "
                "como PNG."
            ),
            wraplength=720,
            justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Label(frame, text="CSV mediciones").grid(row=1, column=0, sticky="w")
        self.csv_var = tk.StringVar()
        self.csv_combo = ttk.Combobox(frame, textvariable=self.csv_var, state="readonly")
        self.csv_combo.grid(row=1, column=1, sticky="ew", padx=4)
        self.csv_combo.bind("<<ComboboxSelected>>", lambda _e: self._load_selected())
        ttk.Button(frame, text="...", width=3, command=self._browse_csv).grid(row=1, column=2)

        ttk.Button(frame, text="Buscar CSV en carpeta de salida",
                   command=self._refresh_csv_list).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(2, 8))

        ctrl = ttk.Frame(frame)
        ctrl.grid(row=3, column=0, columnspan=3, sticky="ew")
        ttk.Label(ctrl, text="X").pack(side="left")
        self.x_var = tk.StringVar()
        self.x_combo = ttk.Combobox(ctrl, textvariable=self.x_var, state="readonly", width=22)
        self.x_combo.pack(side="left", padx=(2, 12))
        self.x_combo.bind("<<ComboboxSelected>>", lambda _e: self._redraw())

        ttk.Label(ctrl, text="Y").pack(side="left")
        self.y_var = tk.StringVar()
        self.y_combo = ttk.Combobox(ctrl, textvariable=self.y_var, state="readonly", width=22)
        self.y_combo.pack(side="left", padx=(2, 12))
        self.y_combo.bind("<<ComboboxSelected>>", lambda _e: self._redraw())

        ttk.Label(ctrl, text="Tipo").pack(side="left")
        self.kind_var = tk.StringVar(value=DEFAULT_CHART_LABEL)
        self.kind_combo = ttk.Combobox(ctrl, textvariable=self.kind_var, state="readonly",
                                       values=list(CHART_TYPES.keys()), width=20)
        self.kind_combo.pack(side="left", padx=2)
        self.kind_combo.bind("<<ComboboxSelected>>", lambda _e: self._redraw())

        self.status_var = tk.StringVar(value="Selecciona un CSV de mediciones para empezar.")
        ttk.Label(frame, textvariable=self.status_var, wraplength=720,
                  foreground="#555").grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 4))

        self.figure = Figure(figsize=(6.5, 4.2), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=frame)
        self.canvas.get_tk_widget().grid(row=6, column=0, columnspan=3, sticky="nsew", pady=6)

        self.export_btn = ttk.Button(frame, text="Exportar PNG",
                                     command=self._export_png, state="disabled")
        self.export_btn.grid(row=7, column=0, sticky="w", pady=4)

        self._refresh_csv_list()

    # ---- acciones

    def _refresh_csv_list(self):
        output_dir = self.get_output_dir()
        paths = discover_measurement_csvs(output_dir) if output_dir else []
        values = [str(p) for p in paths]
        self.csv_combo.configure(values=values)
        if values and not self.csv_var.get():
            self.csv_var.set(values[0])
            self._load_selected()
        elif not values:
            self.status_var.set(
                f"No se encontraron CSV de mediciones en {output_dir or '(carpeta no definida)'}. "
                "Usa '...' para elegir uno manualmente."
            )

    def _browse_csv(self):
        path = self.filedialog.askopenfilename(
            filetypes=[("CSV mediciones", "*.csv"), ("Todos", "*.*")])
        if path:
            self.csv_var.set(path)
            self._load_selected()

    def _load_selected(self):
        path = self.csv_var.get().strip()
        if not path or not os.path.isfile(path):
            self.status_var.set(f"No se encontro el archivo: {path}")
            return
        try:
            self.df = load_measurements(path)
        except Exception as exc:
            self.df = None
            self.status_var.set(f"No se pudo leer el CSV: {exc}")
            return
        self.csv_path = path
        cols = numeric_columns(self.df)
        if not cols:
            self.status_var.set("El CSV no tiene columnas numericas para graficar.")
            self.export_btn.configure(state="disabled")
            return
        self.x_combo.configure(values=cols)
        self.y_combo.configure(values=cols)
        x, y = default_axes(cols)
        self.x_var.set(x)
        self.y_var.set(y)
        self.status_var.set(f"{len(self.df)} fila(s) cargadas de {os.path.basename(path)}.")
        self._redraw()

    def _redraw(self):
        if self.df is None:
            return
        x_col, y_col = self.x_var.get(), self.y_var.get()
        kind = CHART_TYPES.get(self.kind_var.get(), "scatter")
        if not x_col or not y_col:
            return
        try:
            draw_plot(self.ax, self.df, x_col, y_col, kind)
            self.figure.tight_layout()
            self.canvas.draw()
            self.export_btn.configure(state="normal")
            self.status_var.set(f"Grafica: {y_col} vs {x_col} ({self.kind_var.get()}).")
        except ValueError as exc:
            self.export_btn.configure(state="disabled")
            self.status_var.set(str(exc))

    def _export_png(self):
        if self.df is None or self.csv_path is None:
            return
        initial = suggested_png_path(self.csv_path, self.x_var.get(), self.y_var.get())
        path = self.filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=os.path.basename(initial),
            initialdir=os.path.dirname(initial),
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=200, bbox_inches="tight")
            self.status_var.set(f"PNG exportado: {path}")
        except Exception as exc:
            self.messagebox.showerror("Error al exportar", str(exc))
