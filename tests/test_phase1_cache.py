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


def test_latest_phase1_values_prefers_active_variant(tmp_path):
    active_sig = _signature(tmp_path, diameter=80)
    old_sig = _signature(tmp_path, diameter=60)
    cache = phase1_cache.empty_cache()
    cache["active_variant_id"] = "v_active"
    cache["variants"] = [
        {"variant_id": "v_old", "signature": old_sig},
        {"variant_id": "v_active", "signature": active_sig},
    ]

    values = phase1_cache.latest_phase1_values(cache)

    assert values["diameter"] == 80
    assert values["flow_threshold"] == 0.2
    assert values["cellprob_threshold"] == -3.0
    assert values["min_size_voxels"] == 9000
    assert values["gpu"] == "auto"
    assert values["channel"] == 1


def test_latest_phase1_values_falls_back_to_legacy_null(tmp_path):
    figures = tmp_path / "figures_qc"
    figures.mkdir()
    (figures / "medidas.md").write_text(
        "diameter=null\n"
        "flow_threshold=0.1\n"
        "cellprob_threshold=-5\n"
        "min_size_voxels=1200\n",
        encoding="utf-8",
    )
    cache = phase1_cache.load_cache(figures)

    values = phase1_cache.latest_phase1_values(cache)

    assert values == {
        "diameter": "null",
        "flow_threshold": "0.1",
        "cellprob_threshold": "-5",
        "min_size_voxels": "1200",
    }


def test_prune_variants_keeps_last_three_and_removes_dirs(tmp_path):
    phase1_dir = tmp_path / "sample" / "1"
    stem = "sample"
    cache = phase1_cache.empty_cache()
    variants = []
    for idx in range(4):
        variant_id = f"v_{idx}"
        variants.append({"variant_id": variant_id, "signature": {}, "finalized": idx == 3})
        for path in phase1_cache.variant_paths(phase1_dir, stem, variant_id).values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x", encoding="utf-8")
    cache.update({
        "active_variant_id": "v_3",
        "finalized_variant_id": "v_3",
        "finalized": True,
        "variants": variants,
    })

    pruned = phase1_cache.prune_variants(cache, phase1_dir, stem, keep_last=3)

    assert [v["variant_id"] for v in pruned["variants"]] == ["v_1", "v_2", "v_3"]
    assert pruned["active_variant_id"] == "v_3"
    assert pruned["finalized_variant_id"] == "v_3"
    assert pruned["finalized"] is True
    assert not (phase1_dir / "masks_3d" / "variants" / "v_0").exists()
    assert not (phase1_dir / "projections" / "variants" / "v_0").exists()
    assert not (phase1_dir / "figures_qc" / "variants" / "v_0").exists()
    assert (phase1_dir / "masks_3d" / "variants" / "v_1").is_dir()
