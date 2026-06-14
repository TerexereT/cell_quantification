from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import pytest

from gui import plot_panel


def _write_csv(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_load_measurements_drops_metadata_row(tmp_path):
    csv = _write_csv(
        tmp_path / "x_dbc1_intensity.csv",
        "cell_id,area_px,clasificacion\n"
        "2,100,Dbc1+\n"
        "5,200,Dbc1-\n"
        "__metadata__,,\n",
    )
    df = plot_panel.load_measurements(csv)
    assert len(df) == 2
    assert "__metadata__" not in df["cell_id"].astype(str).tolist()


def test_numeric_columns_excludes_text(tmp_path):
    csv = _write_csv(
        tmp_path / "m.csv",
        "cell_id,area_px,clasificacion\n1,100,Dbc1+\n2,200,Dbc1-\n",
    )
    df = plot_panel.load_measurements(csv)
    cols = plot_panel.numeric_columns(df)
    assert "cell_id" in cols and "area_px" in cols
    assert "clasificacion" not in cols


def test_discover_measurement_csvs(tmp_path):
    p1 = tmp_path / "img" / "1" / "measurements"
    p1.mkdir(parents=True)
    (p1 / "img_measurements_3d.csv").write_text("cell_id\n1\n", encoding="utf-8")
    p2 = tmp_path / "img" / "2" / "measurements"
    p2.mkdir(parents=True)
    (p2 / "img_dbc1_intensity.csv").write_text("cell_id\n1\n", encoding="utf-8")
    found = plot_panel.discover_measurement_csvs(tmp_path)
    assert len(found) == 2


def test_draw_plot_scatter_and_line(tmp_path):
    csv = _write_csv(
        tmp_path / "m.csv",
        "cell_id,area_px,volume\n1,10,5\n2,20,9\n3,15,7\n",
    )
    df = plot_panel.load_measurements(csv)
    fig = Figure()
    ax = fig.add_subplot(111)
    plot_panel.draw_plot(ax, df, "area_px", "volume", "scatter")
    assert ax.get_xlabel() == "area_px"
    plot_panel.draw_plot(ax, df, "area_px", "volume", "line")
    assert len(ax.lines) == 1


def test_draw_plot_same_column_does_not_explode(tmp_path):
    # Regresión: X == Y (estado al intercambiar) no debe colgar ni dar arrays 2D.
    csv = _write_csv(
        tmp_path / "m.csv",
        "cell_id,area_px,volume\n1,10,5\n2,20,9\n3,15,7\n",
    )
    df = plot_panel.load_measurements(csv)
    fig = Figure()
    ax = fig.add_subplot(111)
    plot_panel.draw_plot(ax, df, "area_px", "area_px", "scatter")
    offsets = ax.collections[0].get_offsets()
    assert offsets.shape == (3, 2)  # 3 puntos (x,y), no explosión
    plot_panel.draw_plot(ax, df, "area_px", "area_px", "line")
    assert len(ax.lines) == 1
    assert ax.lines[0].get_xdata().ndim == 1


def test_draw_plot_rejects_missing_column(tmp_path):
    csv = _write_csv(tmp_path / "m.csv", "cell_id,area_px\n1,10\n2,20\n")
    df = plot_panel.load_measurements(csv)
    fig = Figure()
    ax = fig.add_subplot(111)
    with pytest.raises(ValueError):
        plot_panel.draw_plot(ax, df, "area_px", "no_existe", "scatter")


def test_suggested_png_path(tmp_path):
    out = plot_panel.suggested_png_path(str(tmp_path / "img_dbc1_intensity.csv"),
                                        "area_px", "volume_um3")
    assert out.endswith("img_dbc1_intensity_volume_um3_vs_area_px.png")


def test_default_axes():
    assert plot_panel.default_axes(["a", "b", "c"]) == ("a", "b")
    assert plot_panel.default_axes(["a"]) == ("a", "a")
    assert plot_panel.default_axes([]) == (None, None)
