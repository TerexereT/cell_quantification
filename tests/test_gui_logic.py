from pathlib import Path

import pytest

from gui import app


def _base_config():
    return {
        "output_dir": "output",
        "cellpose": {
            "gpu": "auto",
            "diameter": None,
            "flow_threshold": 0.2,
            "cellprob_threshold": -3.0,
            "min_size_voxels": 2000,
        },
        "phase2": {
            "threshold_mode": "otsu",
            "factor": 1.6,
            "threshold": None,
            "red_channel": 0,
            "blue_channel": 1,
        },
    }


def test_build_phase1_config_uses_form_values_without_mutating_base():
    base = _base_config()

    config = app.build_phase1_config(
        base,
        {
            "output_dir": r"D:\resultados",
            "diameter": "95",
            "flow_threshold": "0.1",
            "cellprob_threshold": "-2.5",
            "min_size_voxels": "1500",
            "gpu": "false",
        },
    )

    assert config["output_dir"] == r"D:\resultados"
    assert config["cellpose"]["diameter"] == 95
    assert config["cellpose"]["flow_threshold"] == 0.1
    assert config["cellpose"]["gpu"] is False
    assert base["cellpose"]["flow_threshold"] == 0.2


def test_resolve_output_preview_with_spaces_and_accents():
    preview = app.resolve_output_preview(
        r"C:\datos\imagen.czi",
        r"C:\Users\ajarc\Mis Resultados\Cultivo Dia 3",
    )

    assert preview["phase1"].endswith(str(Path("imagen") / "1"))
    assert preview["phase2"].endswith(str(Path("imagen") / "2"))


def test_build_phase2_settings_factor_mode():
    settings = app.build_phase2_settings(
        {
            "threshold_mode": "factor",
            "factor": "1.2",
            "threshold": "",
            "red_channel": "0",
            "blue_channel": "1",
            "output_dir": "out",
        }
    )

    assert settings["threshold_factor"] == 1.2
    assert settings["threshold_value"] is None
    assert settings["output_dir"] == "out"


def test_build_phase1_config_rejects_bad_numeric_value():
    with pytest.raises(ValueError):
        app.build_phase1_config(
            _base_config(),
            {
                "output_dir": "out",
                "diameter": "95",
                "flow_threshold": "bad",
                "cellprob_threshold": "-3",
                "min_size_voxels": "2000",
                "gpu": "auto",
            },
        )


def test_load_formula_sections_splits_phase_content(tmp_path):
    doc = tmp_path / "calculos_justificacion.md"
    doc.write_text(
        "## Fase 1\nformula volumen\n\n## Fase 2\nformula intensidad\n",
        encoding="utf-8",
    )

    sections = app._load_formula_sections(doc)

    assert sections["phase1"] == "formula volumen"
    assert sections["phase2"] == "formula intensidad"


def test_load_formula_sections_reports_missing_heading(tmp_path):
    doc = tmp_path / "calculos_justificacion.md"
    doc.write_text("## Fase 1\nsolo una fase\n", encoding="utf-8")

    with pytest.raises(ValueError, match="## Fase 2"):
        app._load_formula_sections(doc)


def test_load_formula_sections_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        app._load_formula_sections(tmp_path / "no_existe.md")


def test_field_help_covers_declared_ui_fields():
    keys = [key for _label, key in app.PHASE1_FIELDS + app.PHASE2_FIELDS]

    assert keys
    for key in keys:
        assert key in app.FIELD_HELP
        assert isinstance(app.FIELD_HELP[key], str)
        assert app.FIELD_HELP[key].strip()
