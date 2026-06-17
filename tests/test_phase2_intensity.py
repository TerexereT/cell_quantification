import csv
import os
import sys

import numpy as np
import pytest
import tifffile

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import phase2_intensity  # noqa: E402


def _write_mask_run(output_dir, run_name, stem, shape=(2, 4, 5), finalized=True):
    run_dir = output_dir / run_name / "1"
    masks_dir = run_dir / "masks_3d"
    masks_dir.mkdir(parents=True)
    mask = np.zeros(shape, dtype=np.uint16)
    label = 1
    for y in range(2):
        for x in range(5):
            mask[:, y, x] = label
            label += 1
    mask_path = masks_dir / f"{stem}_masks_3d.tif"
    tifffile.imwrite(str(mask_path), mask, photometric="minisblack")
    if finalized:
        _write_finalized_cache(output_dir, run_name, stem, mask_path)
    return mask_path


def _write_finalized_cache(output_dir, run_name, stem, mask_path):
    cache = phase2_intensity.phase1_cache.empty_cache()
    variant_id = "v_test"
    cache.update({
        "active_variant_id": variant_id,
        "finalized_variant_id": variant_id,
        "finalized": True,
        "variants": [
            {
                "variant_id": variant_id,
                "signature": {},
                "assets": {"mask": str(mask_path)},
                "finalized": True,
            }
        ],
    })
    phase2_intensity.phase1_cache.write_cache(
        output_dir / run_name / "1" / "figures_qc", cache
    )


def _read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_process_output_only_processes_selected_czi(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    # Carpeta del CZI seleccionado + una carpeta ajena que debe ignorarse.
    _write_mask_run(output_dir, "sample", "sample")
    _write_mask_run(output_dir, "otro", "otro")

    red = np.zeros((2, 4, 5), dtype=np.float32)
    red[:, 0, 0] = 0
    red[:, 0:2, 1:5] = 100
    red[:, 1, 0] = 100
    blue = np.ones((2, 4, 5), dtype=np.float32)
    monkeypatch.setattr(
        phase2_intensity,
        "load_czi_dual_channel",
        lambda _p, red_channel=0, blue_channel=1: (red, blue),
    )

    summaries = phase2_intensity.process_output("sample.czi", output_dir)

    assert len(summaries) == 1
    run_dir = output_dir / "sample" / "2"
    assert (run_dir / "figures_qc" / "sample_qc_blue_overlay.png").is_file()
    assert (run_dir / "figures_qc" / "sample_qc_red_overlay.png").is_file()
    assert (run_dir / "figures_qc" / "sample_dbc1_classification.png").is_file()
    assert (run_dir / "measurements" / "sample_dbc1_intensity.csv").is_file()
    assert (run_dir / "masks_3d" / "sample_masks_dbc1_positive.tif").is_file()
    # La carpeta ajena no se procesa.
    assert not (output_dir / "otro" / "2").exists()


def test_classification_marks_visibly_low_cell_as_negative(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    mask_path = _write_mask_run(output_dir, "sample", "sample")

    red = np.zeros((2, 4, 5), dtype=np.float32)
    red[:, 0, 0] = 0
    red[:, 0:2, 1:5] = 100
    red[:, 1, 0] = 100
    blue = np.ones((2, 4, 5), dtype=np.float32)
    monkeypatch.setattr(
        phase2_intensity,
        "load_czi_dual_channel",
        lambda _p, red_channel=0, blue_channel=1: (red, blue),
    )

    summaries = phase2_intensity.process_output("sample.czi", output_dir)

    assert summaries[0]["n_negative"] == 1
    rows = _read_csv_rows(output_dir / "sample" / "2" / "measurements" / "sample_dbc1_intensity.csv")
    cell_rows = [row for row in rows if row["cell_id"] != "__metadata__"]
    assert cell_rows[0]["clasificacion"] == phase2_intensity.NEGATIVE_LABEL
    assert all(row["clasificacion"] == phase2_intensity.POSITIVE_LABEL for row in cell_rows[1:])

    original = tifffile.imread(str(mask_path))
    filtered = tifffile.imread(
        str(output_dir / "sample" / "2" / "masks_3d" / "sample_masks_dbc1_positive.tif")
    )
    assert np.unique(filtered[filtered != 0]).size < np.unique(original[original != 0]).size
    assert 1 not in np.unique(filtered)


def test_missing_czi_raises_file_not_found():
    with pytest.raises(FileNotFoundError) as exc:
        phase2_intensity.load_czi_dual_channel("no_existe.czi")
    assert "no_existe.czi" in str(exc.value)


def test_no_masks_requires_finalized_phase1(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    (output_dir / "voxels1500").mkdir(parents=True)
    monkeypatch.setattr(
        phase2_intensity,
        "load_czi_dual_channel",
        lambda _p, red_channel=0, blue_channel=1: (
            np.zeros((1, 2, 2)),
            np.zeros((1, 2, 2)),
        ),
    )

    with pytest.raises(ValueError) as exc:
        phase2_intensity.process_output("fake.czi", output_dir)

    assert "Finalizar Fase 1" in str(exc.value)


def test_uniform_cells_have_sd_zero_and_all_positive():
    red_proj = np.full((3, 3), 10, dtype=np.float32)
    mask_proj = np.zeros((3, 3), dtype=np.uint16)
    mask_proj[0, 0] = 1
    mask_proj[1, 1] = 2

    red_volume = np.stack([red_proj])
    mask_3d = np.stack([mask_proj])

    rows, metadata = phase2_intensity.measure_intensity(
        red_proj, mask_proj, red_volume, mask_3d, threshold_factor=1.0
    )

    assert metadata["umbral"] == pytest.approx(10)
    assert [row["clasificacion"] for row in rows] == [
        phase2_intensity.POSITIVE_LABEL,
        phase2_intensity.POSITIVE_LABEL,
    ]


def test_process_output_progress_callback_is_called(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    _write_mask_run(output_dir, "sample", "sample")
    red = np.ones((2, 4, 5), dtype=np.float32)
    blue = np.ones((2, 4, 5), dtype=np.float32)
    monkeypatch.setattr(
        phase2_intensity,
        "load_czi_dual_channel",
        lambda _p, red_channel=0, blue_channel=1: (red, blue),
    )
    messages = []

    summaries = phase2_intensity.process_output(
        "sample.czi", output_dir, progress_callback=messages.append
    )

    assert len(summaries) == 1
    joined = "\n".join(messages)
    assert "Cargando CZI" in joined
    assert "calculando intensidades" in joined
    assert "generando figura" in joined


def test_process_output_raises_on_z_mismatch(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    # Máscara con Z=3 pero el volumen del CZI tiene Z=2.
    _write_mask_run(output_dir, "sample", "sample", shape=(3, 4, 5))
    red = np.ones((2, 4, 5), dtype=np.float32)
    blue = np.ones((2, 4, 5), dtype=np.float32)
    monkeypatch.setattr(
        phase2_intensity,
        "load_czi_dual_channel",
        lambda _p, red_channel=0, blue_channel=1: (red, blue),
    )

    with pytest.raises(ValueError) as exc:
        phase2_intensity.process_output("sample.czi", output_dir)
    assert "incompatible" in str(exc.value)


def test_process_output_rejects_qc_only_phase1(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    phase1_dir = output_dir / "sample" / "1"
    variant_dir = phase1_dir / "masks_3d" / "variants" / "v_test"
    variant_dir.mkdir(parents=True)
    tifffile.imwrite(
        str(variant_dir / "sample_masks_3d.tif"),
        np.zeros((2, 4, 5), dtype=np.uint16),
        photometric="minisblack",
    )
    cache = phase2_intensity.phase1_cache.empty_cache()
    cache.update({
        "active_variant_id": "v_test",
        "finalized": False,
        "variants": [{"variant_id": "v_test", "signature": {}, "finalized": False}],
    })
    phase2_intensity.phase1_cache.write_cache(phase1_dir / "figures_qc", cache)
    monkeypatch.setattr(
        phase2_intensity,
        "load_czi_dual_channel",
        lambda _p, red_channel=0, blue_channel=1: (
            np.zeros((2, 4, 5)),
            np.zeros((2, 4, 5)),
        ),
    )

    with pytest.raises(ValueError) as exc:
        phase2_intensity.process_output("sample.czi", output_dir)

    assert "pendiente de finalizar" in str(exc.value)


def test_load_phase2_config_and_cli_threshold_priority(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "phase2:\n"
        "  threshold_mode: factor\n"
        "  factor: 1.4\n"
        "  threshold: null\n"
        "  red_channel: 2\n"
        "  blue_channel: 3\n",
        encoding="utf-8",
    )

    defaults = phase2_intensity.load_phase2_config(str(config))
    factor, threshold = phase2_intensity.resolve_threshold_settings(
        defaults, cli_threshold=500
    )

    assert defaults["red_channel"] == 2
    assert factor is None
    assert threshold == 500
