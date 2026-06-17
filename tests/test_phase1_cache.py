import numpy as np

import phase1_cache


def _signature(tmp_path, diameter=60):
    czi = tmp_path / "sample.czi"
    czi.write_bytes(b"fake")
    config = {
        "cellpose": {
            "diameter": diameter,
            "flow_threshold": 0.2,
            "cellprob_threshold": -3.0,
            "min_size_voxels": 9000,
            "do_3D": True,
            "model_type": "cyto2",
            "z_axis": 0,
            "channel_axis": None,
            "gpu": "auto",
        }
    }
    return phase1_cache.build_phase1_signature(
        czi,
        "sample.czi",
        "sample",
        np.zeros((2, 4, 5), dtype=np.uint16),
        0.1,
        0.3,
        1,
        config,
    )


def test_load_legacy_key_value_cache(tmp_path):
    figures = tmp_path / "figures_qc"
    figures.mkdir()
    (figures / "medidas.md").write_text(
        "diameter=60\nflow_threshold=0.2\ncellprob_threshold=-3.0\n",
        encoding="utf-8",
    )

    cache = phase1_cache.load_cache(figures)

    assert cache["legacy"]["diameter"] == "60"
    assert cache["variants"] == []
    assert cache["parse_error"] is None


def test_write_and_read_structured_cache(tmp_path):
    sig = _signature(tmp_path)
    cache = phase1_cache.empty_cache()
    variant = {
        "variant_id": "v_test",
        "signature": sig,
        "assets": {"mask": "mask.tif"},
        "n_cells": 3,
        "finalized": False,
    }
    cache = phase1_cache.add_or_update_variant(cache, variant)

    phase1_cache.write_cache(tmp_path / "figures_qc", cache)
    loaded = phase1_cache.load_cache(tmp_path / "figures_qc")

    measures_text = (tmp_path / "figures_qc" / "medidas.md").read_text(encoding="utf-8")
    assert "phase1_cache_json" not in measures_text
    assert measures_text == (
        "diameter=60\n"
        "flow_threshold=0.2\n"
        "cellprob_threshold=-3.0\n"
        "min_size_voxels=9000\n"
    )
    assert (tmp_path / "figures_qc" / "phase1_cache.json").is_file()
    assert loaded["active_variant_id"] == "v_test"
    assert loaded["variants"][0]["n_cells"] == 3
    assert loaded["legacy"]["diameter"] == "60"


def test_load_cache_reads_legacy_embedded_json(tmp_path):
    figures = tmp_path / "figures_qc"
    figures.mkdir()
    (figures / "medidas.md").write_text(
        "diameter=60\n"
        "<!-- phase1_cache_json:start -->\n"
        "```json\n"
        '{"active_variant_id": "v_old", "variants": [{"variant_id": "v_old"}]}\n'
        "```\n"
        "<!-- phase1_cache_json:end -->\n",
        encoding="utf-8",
    )

    cache = phase1_cache.load_cache(figures)

    assert cache["active_variant_id"] == "v_old"
    assert cache["legacy"]["diameter"] == "60"


def test_compare_phase1_signature_reports_blocking_change(tmp_path):
    cached = _signature(tmp_path, diameter=60)
    current = _signature(tmp_path, diameter=70)

    comparison = phase1_cache.compare_phase1_signature(cached, current)

    assert comparison["compatible"] is False
    assert any("cellpose.diameter" in reason for reason in comparison["blocking_reasons"])


def test_make_variant_id_adds_suffix_when_forced(tmp_path):
    sig = _signature(tmp_path)
    first = phase1_cache.make_variant_id(sig)
    second = phase1_cache.make_variant_id(sig, existing_ids=[first], force_suffix=True)

    assert second.startswith(first + "_")
