"""
Pruebas unitarias del pipeline de análisis 3D de células.

Cubren io_utils, measure_3d, segment_cellpose_3d (con cellpose mockeado),
visualize_qc y el orquestador main, incluyendo casos de éxito, error y borde.
"""

import os

import numpy as np
import pandas as pd
import pytest
import tifffile

import io_utils
import measure_3d
import segment_cellpose_3d
import visualize_qc
from conftest import make_cube_mask
from utils import stem


# ======================================================================
# io_utils
# ======================================================================

def test_load_config_ok(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("output_dir: out\ncellpose:\n  gpu: false\n", encoding="utf-8")
    out = io_utils.load_config(str(cfg))
    assert out["output_dir"] == "out"
    assert out["cellpose"]["gpu"] is False


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        io_utils.load_config("no_existe_12345.yaml")


def test_load_metadata_ok(tmp_path):
    p = tmp_path / "metadata.csv"
    p.write_text(
        "filename,px_xy_um,px_z_um,channel_to_segment,notes\n"
        "a.tif,0.1,0.3,0,nota\n",
        encoding="utf-8",
    )
    df = io_utils.load_metadata(str(p))
    assert len(df) == 1
    assert df.iloc[0]["filename"] == "a.tif"


def test_load_metadata_missing_column(tmp_path):
    p = tmp_path / "metadata.csv"
    # Falta px_z_um.
    p.write_text("filename,px_xy_um,channel_to_segment\na.tif,0.1,0\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        io_utils.load_metadata(str(p))
    assert "px_z_um" in str(exc.value)


def test_load_zstack_ok(tmp_path):
    p = tmp_path / "img.tif"
    arr = np.random.randint(0, 255, size=(4, 8, 8), dtype=np.uint16)
    tifffile.imwrite(str(p), arr)
    loaded = io_utils.load_zstack(str(p))
    assert loaded.shape == (4, 8, 8)


def test_load_zstack_missing():
    with pytest.raises(FileNotFoundError):
        io_utils.load_zstack("no_existe.tif")


def test_load_zstack_bad_extension(tmp_path):
    p = tmp_path / "img.png"
    p.write_bytes(b"not a tiff")
    with pytest.raises(ValueError) as exc:
        io_utils.load_zstack(str(p))
    assert ".png" in str(exc.value)


def test_create_output_folders(tmp_path):
    paths = io_utils.create_output_folders(str(tmp_path / "output"))
    for key in ["masks_3d", "projections", "meshes", "measurements", "figures_qc", "logs"]:
        assert os.path.isdir(paths[key])


def test_save_tiff_roundtrip(tmp_path):
    arr = np.arange(2 * 3 * 3, dtype=np.uint16).reshape(2, 3, 3)
    p = tmp_path / "sub" / "out.tif"
    io_utils.save_tiff(str(p), arr)
    assert os.path.isfile(str(p))
    np.testing.assert_array_equal(tifffile.imread(str(p)), arr)


def test_prepare_volume_grayscale():
    vol = np.random.rand(5, 10, 10)
    out = io_utils.prepare_volume(vol, z_axis=0, channel_axis=None)
    assert out.shape == (5, 10, 10)


def test_prepare_volume_zcyx():
    # Z×C×Y×X, segmentar canal 1.
    raw = np.random.rand(5, 3, 10, 10)
    out = io_utils.prepare_volume(raw, z_axis=0, channel_axis=1, channel_to_segment=1)
    assert out.shape == (5, 10, 10)
    np.testing.assert_array_equal(out, raw[:, 1, :, :])


def test_prepare_volume_zyxc():
    # Z×Y×X×C, segmentar canal 2.
    raw = np.random.rand(5, 10, 10, 3)
    out = io_utils.prepare_volume(raw, z_axis=0, channel_axis=3, channel_to_segment=2)
    assert out.shape == (5, 10, 10)
    np.testing.assert_array_equal(out, raw[:, :, :, 2])


def test_prepare_volume_z_axis_nonzero():
    # Z en el eje 1.
    raw = np.random.rand(8, 5, 12)  # (Y, Z, X) con z_axis=1
    out = io_utils.prepare_volume(raw, z_axis=1, channel_axis=None)
    assert out.shape == (5, 8, 12)


def test_prepare_volume_not_3d_raises():
    raw = np.random.rand(5, 10)  # 2D
    with pytest.raises(ValueError):
        io_utils.prepare_volume(raw, z_axis=0, channel_axis=None)


# ======================================================================
# measure_3d
# ======================================================================

def _base_config():
    return {
        "measurements": {
            "save_individual_cell_meshes": False,
            "calculate_surface_area": True,
            "calculate_projected_area_xy": True,
        }
    }


def test_measure_cube():
    mask = make_cube_mask(shape=(10, 20, 20), z=(2, 7), y=(5, 12), x=(5, 12))
    px_xy, px_z = 0.1, 0.3
    df = measure_3d.measure_cells_3d(mask, px_xy, px_z, _base_config())

    assert len(df) == 1
    row = df.iloc[0]
    # Cubo de 5×7×7 voxeles.
    assert row["voxel_count"] == 5 * 7 * 7
    assert row["volume_um3"] == pytest.approx(5 * 7 * 7 * px_xy * px_xy * px_z)
    assert row["projected_area_xy_um2"] == pytest.approx(7 * 7 * px_xy * px_xy)
    assert row["z_slices_detected"] == 5
    assert row["bbox_z_min"] == 2 and row["bbox_z_max"] == 6
    assert row["bbox_y_min"] == 5 and row["bbox_y_max"] == 11
    assert row["bbox_x_min"] == 5 and row["bbox_x_max"] == 11
    assert np.isfinite(row["surface_area_um2"]) and row["surface_area_um2"] > 0


def test_measure_empty_mask_has_columns():
    mask = np.zeros((5, 10, 10), dtype=np.uint16)
    df = measure_3d.measure_cells_3d(mask, 0.1, 0.3, _base_config())
    assert len(df) == 0
    for col in measure_3d.MEASUREMENT_COLUMNS:
        assert col in df.columns


def test_measure_two_cells():
    mask = np.zeros((10, 20, 20), dtype=np.uint16)
    mask[1:4, 2:5, 2:5] = 1
    mask[6:9, 12:16, 12:16] = 2
    df = measure_3d.measure_cells_3d(mask, 0.1, 0.3, _base_config())
    assert sorted(df["cell_id"].tolist()) == [1, 2]


def test_measure_single_z_slice_surface_finite():
    # Caso borde: célula de un solo corte Z. Con padding, marching_cubes
    # produce una malla fina cerrada -> área finita (no NaN).
    mask = np.zeros((5, 10, 10), dtype=np.uint16)
    mask[3:4, 4:7, 4:7] = 1
    df = measure_3d.measure_cells_3d(mask, 0.1, 0.3, _base_config())
    assert df.iloc[0]["z_slices_detected"] == 1
    assert np.isfinite(df.iloc[0]["surface_area_um2"])


def test_measure_surface_disabled():
    cfg = _base_config()
    cfg["measurements"]["calculate_surface_area"] = False
    mask = make_cube_mask()
    df = measure_3d.measure_cells_3d(mask, 0.1, 0.3, cfg)
    assert np.isnan(df.iloc[0]["surface_area_um2"])


def test_measure_mesh_export(tmp_path):
    cfg = _base_config()
    cfg["measurements"]["save_individual_cell_meshes"] = True
    mask = make_cube_mask()
    measure_3d.measure_cells_3d(
        mask, 0.1, 0.3, cfg, meshes_dir=str(tmp_path), filename_stem="test"
    )
    assert os.path.isfile(str(tmp_path / "test_cell_1.obj"))


def test_measure_non_3d_raises():
    with pytest.raises(ValueError):
        measure_3d.measure_cells_3d(np.zeros((5, 5)), 0.1, 0.3, _base_config())


# ======================================================================
# segment_cellpose_3d
# ======================================================================

def test_filter_small_objects():
    mask = np.zeros((4, 10, 10), dtype=np.uint16)
    mask[0:4, 0:6, 0:6] = 1   # grande: 4*6*6 = 144 voxeles
    mask[0, 9, 9] = 2          # 1 voxel
    out = segment_cellpose_3d.filter_small_objects(mask, min_size_voxels=100)
    assert 1 in np.unique(out)
    assert 2 not in np.unique(out)


def test_filter_small_objects_noop_when_threshold_low():
    mask = make_cube_mask()
    out = segment_cellpose_3d.filter_small_objects(mask, min_size_voxels=1)
    np.testing.assert_array_equal(out, mask)


def test_segment_3d_anisotropy_and_filter(fake_cellpose):
    # La máscara que devuelve el fake: una célula grande + una de 1 voxel.
    ret = np.zeros((6, 12, 12), dtype=np.uint16)
    ret[0:4, 0:6, 0:6] = 1
    ret[5, 11, 11] = 2
    fake_cellpose.return_mask = ret

    config = {
        "cellpose": {
            "gpu": False,
            "model_type": "cyto3",
            "do_3D": True,
            "z_axis": 0,
            "channel_axis": None,
            "flow_threshold": 0.4,
            "cellprob_threshold": 0.0,
            "diameter": None,
            "min_size_voxels": 100,
        }
    }
    image = np.random.rand(6, 12, 12)
    mask = segment_cellpose_3d.segment_3d_cellpose(image, config, px_xy_um=0.1, px_z_um=0.3)

    # anisotropy = px_z / px_xy = 3.0
    assert fake_cellpose.last_eval_kwargs["anisotropy"] == pytest.approx(3.0)
    assert fake_cellpose.last_eval_kwargs["do_3D"] is True
    # La célula de 1 voxel se filtró.
    assert 1 in np.unique(mask)
    assert 2 not in np.unique(mask)


def test_segment_passes_model_type(fake_cellpose):
    fake_cellpose.return_mask = np.zeros((4, 8, 8), dtype=np.uint16)
    config = {"cellpose": {"model_type": "cyto3", "gpu": False, "min_size_voxels": 0}}
    segment_cellpose_3d.segment_3d_cellpose(
        np.random.rand(4, 8, 8), config, 0.1, 0.3
    )
    assert fake_cellpose.last_init_kwargs.get("model_type") == "cyto3"


# ======================================================================
# visualize_qc
# ======================================================================

def test_create_qc_figures(tmp_path):
    image = np.random.rand(6, 20, 20)
    mask = make_cube_mask(shape=(6, 20, 20), z=(1, 4), y=(5, 12), x=(5, 12))
    proj_dir = tmp_path / "projections"
    fig_dir = tmp_path / "figures"
    proj_dir.mkdir()
    fig_dir.mkdir()
    config = {"qc": {"save_overlay_projection": True, "save_mask_projection": True}}

    outputs = visualize_qc.create_qc_figures(
        image, mask, "test", str(proj_dir), str(fig_dir), config
    )
    assert os.path.isfile(outputs["max_projection"])
    assert os.path.isfile(outputs["mask_projection"])
    assert os.path.isfile(outputs["qc_overlay"])


# ======================================================================
# main (end-to-end con cellpose mockeado)
# ======================================================================

def _write_project(tmp_path, mask_to_return, filename="ejemplo_zstack.tif"):
    """Crea una estructura de proyecto mínima en tmp_path y devuelve rutas."""
    (tmp_path / "input" / "raw_zstacks").mkdir(parents=True)
    (tmp_path / "input" / "metadata").mkdir(parents=True)
    (tmp_path / "config").mkdir(parents=True)

    # Imagen sintética.
    img = np.random.randint(0, 255, size=mask_to_return.shape, dtype=np.uint16)
    tifffile.imwrite(str(tmp_path / "input" / "raw_zstacks" / filename), img)

    # metadata.csv
    (tmp_path / "input" / "metadata" / "metadata.csv").write_text(
        "filename,px_xy_um,px_z_um,channel_to_segment,notes\n"
        f"{filename},0.108,0.300,0,control\n",
        encoding="utf-8",
    )

    # config.yaml
    cfg_text = (
        f'input_dir: "{(tmp_path / "input" / "raw_zstacks").as_posix()}"\n'
        f'metadata_file: "{(tmp_path / "input" / "metadata" / "metadata.csv").as_posix()}"\n'
        f'output_dir: "{(tmp_path / "output").as_posix()}"\n'
        "cellpose:\n"
        "  gpu: false\n"
        "  model_type: cyto3\n"
        "  diameter: null\n"
        "  do_3D: true\n"
        "  z_axis: 0\n"
        "  channel_axis: null\n"
        "  flow_threshold: 0.4\n"
        "  cellprob_threshold: 0.0\n"
        "  min_size_voxels: 10\n"
        "measurements:\n"
        "  save_individual_cell_meshes: true\n"
        "  calculate_surface_area: true\n"
        "  calculate_projected_area_xy: true\n"
        "qc:\n"
        "  save_overlay_projection: true\n"
        "  save_mask_projection: true\n"
    )
    cfg_path = tmp_path / "config" / "config.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    return cfg_path


def test_main_end_to_end(tmp_path, fake_cellpose):
    import main

    mask = np.zeros((8, 24, 24), dtype=np.uint16)
    mask[1:5, 4:12, 4:12] = 1
    mask[2:6, 14:20, 14:20] = 2
    fake_cellpose.return_mask = mask

    cfg_path = _write_project(tmp_path, mask)
    rc = main.main(["--config", str(cfg_path)])
    assert rc == 0

    out = tmp_path / "output"
    assert (out / "masks_3d" / "ejemplo_zstack_masks_3d.tif").is_file()
    csv = out / "measurements" / "ejemplo_zstack_measurements_3d.csv"
    assert csv.is_file()
    assert (out / "projections" / "ejemplo_zstack_max_projection.tif").is_file()
    assert (out / "projections" / "ejemplo_zstack_mask_projection.tif").is_file()
    assert (out / "figures_qc" / "ejemplo_zstack_qc_overlay.png").is_file()
    assert (out / "logs" / "pipeline_log.txt").is_file()

    df = pd.read_csv(csv)
    assert len(df) == 2
    assert list(df.columns)[0] == "filename"
    assert set(df["cell_id"]) == {1, 2}


def test_main_cleans_stale_meshes(tmp_path, fake_cellpose):
    import main

    mask = np.zeros((8, 24, 24), dtype=np.uint16)
    mask[1:5, 4:12, 4:12] = 1
    fake_cellpose.return_mask = mask
    cfg_path = _write_project(tmp_path, mask)

    # Primera corrida.
    assert main.main(["--config", str(cfg_path)]) == 0
    meshes_dir = tmp_path / "output" / "meshes"
    # Deja un .obj obsoleto de un cell_id que ya no existe.
    stale = meshes_dir / "ejemplo_zstack_cell_99.obj"
    stale.write_text("v 0 0 0\n", encoding="utf-8")

    # Segunda corrida: debe eliminar el .obj obsoleto.
    assert main.main(["--config", str(cfg_path)]) == 0
    assert not stale.exists()
    assert (meshes_dir / "ejemplo_zstack_cell_1.obj").is_file()


def test_main_missing_config():
    import main
    rc = main.main(["--config", "ruta/inexistente.yaml"])
    assert rc == 1


def test_main_missing_image(tmp_path, fake_cellpose):
    import main
    fake_cellpose.return_mask = np.zeros((4, 8, 8), dtype=np.uint16)
    cfg_path = _write_project(tmp_path, np.zeros((4, 8, 8), dtype=np.uint16))
    # Borra la imagen para forzar archivo no encontrado.
    os.remove(str(tmp_path / "input" / "raw_zstacks" / "ejemplo_zstack.tif"))
    rc = main.main(["--config", str(cfg_path)])
    # El pipeline no aborta: termina con rc=0 pero registra el error.
    assert rc == 0
    log = (tmp_path / "output" / "logs" / "pipeline_log.txt").read_text(encoding="utf-8")
    assert "no encontrado" in log.lower() or "no se encontró" in log.lower()


def test_main_zero_cells_warns(tmp_path, fake_cellpose):
    import main
    fake_cellpose.return_mask = np.zeros((6, 16, 16), dtype=np.uint16)
    cfg_path = _write_project(tmp_path, np.zeros((6, 16, 16), dtype=np.uint16))
    rc = main.main(["--config", str(cfg_path)])
    assert rc == 0
    csv = tmp_path / "output" / "measurements" / "ejemplo_zstack_measurements_3d.csv"
    df = pd.read_csv(csv)
    assert len(df) == 0  # CSV con cabecera pero sin filas
    log = (tmp_path / "output" / "logs" / "pipeline_log.txt").read_text(encoding="utf-8")
    assert "0 células" in log
