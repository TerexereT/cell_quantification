"""Cache y variantes de Fase 1 para QC de segmentacion.

``figures_qc/medidas.md`` queda reservado para medidas legibles. El estado
estructurado que lee el programa vive en ``figures_qc/phase1_cache.json``.
"""

import hashlib
import json
import shutil
from copy import deepcopy
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np

MEASURES_FILENAME = "medidas.md"
CACHE_JSON_FILENAME = "phase1_cache.json"
CACHE_VERSION = 1
JSON_START = "<!-- phase1_cache_json:start -->"
JSON_END = "<!-- phase1_cache_json:end -->"

SEGMENTATION_KEYS = (
    "diameter",
    "flow_threshold",
    "cellprob_threshold",
    "min_size_voxels",
    "do_3D",
    "model_type",
    "z_axis",
    "channel_axis",
    "gpu",
)


def measures_path(figures_dir_or_path):
    path = Path(figures_dir_or_path)
    if path.suffix.lower() == ".md":
        return path
    return path / MEASURES_FILENAME


def cache_path(figures_dir_or_path):
    path = Path(figures_dir_or_path)
    if path.suffix.lower() == ".json":
        return path
    if path.suffix.lower() == ".md":
        return path.with_name(CACHE_JSON_FILENAME)
    return path / CACHE_JSON_FILENAME


def empty_cache():
    return {
        "cache_version": CACHE_VERSION,
        "updated_at": None,
        "active_variant_id": None,
        "finalized_variant_id": None,
        "finalized": False,
        "legacy": {},
        "variants": [],
        "parse_error": None,
    }


def load_cache(figures_dir_or_path):
    md_path = measures_path(figures_dir_or_path)
    json_path = cache_path(figures_dir_or_path)
    cache = empty_cache()

    text = ""
    if md_path.is_file():
        text = md_path.read_text(encoding="utf-8")
        cache["legacy"] = _parse_legacy_key_values(text)

    raw_json = None
    json_source = json_path
    if json_path.is_file():
        raw_json = json_path.read_text(encoding="utf-8")
    elif text:
        # Compatibilidad con corridas generadas antes de separar el JSON.
        raw_json = _extract_json_block(text)
        json_source = md_path
    if not raw_json:
        return cache

    try:
        loaded = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        cache["parse_error"] = f"{json_source.name} tiene JSON invalido: {exc}"
        return cache

    if isinstance(loaded, dict):
        merged = empty_cache()
        merged.update(loaded)
        merged["legacy"] = cache["legacy"]
        merged["parse_error"] = None
        merged.setdefault("variants", [])
        return merged

    cache["parse_error"] = f"{json_source.name} no contiene un objeto JSON de cache."
    return cache


def write_cache(figures_dir_or_path, cache):
    md_path = measures_path(figures_dir_or_path)
    json_path = cache_path(figures_dir_or_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    data = deepcopy(cache)
    data["cache_version"] = CACHE_VERSION
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    data["parse_error"] = None

    active = get_variant(data, data.get("active_variant_id"))
    legacy_lines = []
    if active:
        cellpose = active.get("signature", {}).get("cellpose", {})
        for key in ("diameter", "flow_threshold", "cellprob_threshold", "min_size_voxels"):
            legacy_lines.append(f"{key}={_format_legacy_value(cellpose.get(key))}")
    else:
        legacy = data.get("legacy") or {}
        for key in ("diameter", "flow_threshold", "cellprob_threshold", "min_size_voxels"):
            if key in legacy:
                legacy_lines.append(f"{key}={legacy[key]}")

    md_path.write_text("\n".join([*legacy_lines, ""]), encoding="utf-8")
    json_path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return data


def latest_phase1_values(cache):
    """Devuelve los parametros de Fase 1 mas recientes guardados en cache."""
    values = {}
    variant = get_variant(cache, cache.get("active_variant_id"))
    variants = cache.get("variants") or []
    if variant is None and variants:
        variant = variants[-1]

    if variant:
        signature = variant.get("signature", {})
        cellpose = signature.get("cellpose", {})
        for key in ("diameter", "flow_threshold", "cellprob_threshold", "min_size_voxels", "gpu"):
            if key in cellpose:
                values[key] = cellpose[key]
        extraction = signature.get("extraction", {})
        if "channel" in extraction:
            values["channel"] = extraction["channel"]

    legacy = cache.get("legacy") or {}
    for key in ("diameter", "flow_threshold", "cellprob_threshold", "min_size_voxels", "gpu", "channel"):
        if key not in values and key in legacy:
            values[key] = legacy[key]
    return values


def _format_legacy_value(value):
    if value is None:
        return "null"
    return str(value)


def _parse_legacy_key_values(text):
    values = {}
    in_json = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == JSON_START:
            in_json = True
            continue
        if line == JSON_END:
            in_json = False
            continue
        if in_json or not line or line.startswith("#") or line.startswith("```"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _extract_json_block(text):
    start = text.find(JSON_START)
    end = text.find(JSON_END)
    if start == -1 or end == -1 or end <= start:
        return None
    block = text[start + len(JSON_START):end].strip()
    if block.startswith("```"):
        lines = block.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        block = "\n".join(lines).strip()
    return block


def package_version(package_name):
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def build_phase1_signature(
    czi_path,
    filename,
    stem,
    volume,
    px_xy_um,
    px_z_um,
    channel,
    config,
    segment_info=None,
):
    volume = np.asarray(volume)
    cp_cfg = config.get("cellpose", {})
    path = Path(czi_path) if czi_path else None
    stat = path.stat() if path and path.is_file() else None
    anisotropy = None
    if px_xy_um not in (None, 0) and px_z_um is not None:
        anisotropy = float(px_z_um) / float(px_xy_um)

    segment_info = segment_info or {}
    return {
        "sample": {
            "filename": str(filename),
            "stem": str(stem),
            "path": str(path.resolve()) if path and path.exists() else str(czi_path or ""),
            "size_bytes": stat.st_size if stat else None,
            "mtime_ns": stat.st_mtime_ns if stat else None,
        },
        "extraction": {
            "channel": int(channel),
            "volume_shape": [int(x) for x in volume.shape],
            "volume_dtype": str(volume.dtype),
        },
        "calibration": {
            "px_xy_um": None if px_xy_um is None else float(px_xy_um),
            "px_z_um": None if px_z_um is None else float(px_z_um),
            "anisotropy": anisotropy,
        },
        "cellpose": {key: _jsonable(cp_cfg.get(key)) for key in SEGMENTATION_KEYS},
        "runtime": {
            "gpu_resolved": _jsonable(segment_info.get("gpu_resolved")),
            "eval_kwargs": _jsonable(segment_info.get("eval_kwargs", {})),
        },
        "versions": {
            "cellpose": package_version("cellpose"),
            "czifile": package_version("czifile"),
            "tifffile": package_version("tifffile"),
            "numpy": package_version("numpy"),
            "scikit-image": package_version("scikit-image"),
        },
    }


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def make_variant_id(signature, existing_ids=None, force_suffix=False):
    existing_ids = set(existing_ids or [])
    payload = json.dumps(signature, sort_keys=True, ensure_ascii=False, default=str)
    base = "v_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    if base not in existing_ids and not force_suffix:
        return base
    idx = 2
    while f"{base}_{idx}" in existing_ids:
        idx += 1
    return f"{base}_{idx}"


def variant_paths(phase1_dir, stem, variant_id):
    root = Path(phase1_dir)
    return {
        "mask": root / "masks_3d" / "variants" / variant_id / f"{stem}_masks_3d.tif",
        "max_projection": root / "projections" / "variants" / variant_id / f"{stem}_max_projection.tif",
        "mask_projection": root / "projections" / "variants" / variant_id / f"{stem}_mask_projection.tif",
        "qc_overlay": root / "figures_qc" / "variants" / variant_id / f"{stem}_qc_overlay.png",
    }


def canonical_paths(phase1_dir, stem):
    root = Path(phase1_dir)
    return {
        "mask": root / "masks_3d" / f"{stem}_masks_3d.tif",
        "max_projection": root / "projections" / f"{stem}_max_projection.tif",
        "mask_projection": root / "projections" / f"{stem}_mask_projection.tif",
        "qc_overlay": root / "figures_qc" / f"{stem}_qc_overlay.png",
        "measurements": root / "measurements" / f"{stem}_measurements_3d.csv",
    }


def add_or_update_variant(cache, variant):
    cache = deepcopy(cache)
    variants = [v for v in cache.get("variants", []) if v.get("variant_id") != variant.get("variant_id")]
    variants.append(variant)
    cache["variants"] = variants
    cache["active_variant_id"] = variant.get("variant_id")
    cache["finalized"] = bool(cache.get("finalized") and cache.get("finalized_variant_id") == variant.get("variant_id"))
    return cache


def mark_finalized(cache, variant_id, assets):
    cache = deepcopy(cache)
    cache["active_variant_id"] = variant_id
    cache["finalized_variant_id"] = variant_id
    cache["finalized"] = True
    variant = get_variant(cache, variant_id)
    if variant is not None:
        variant.setdefault("assets", {}).update({k: str(v) for k, v in assets.items()})
        variant["finalized"] = True
    return cache


def prune_variants(cache, phase1_dir, stem, keep_last=3):
    """Conserva las ultimas variantes y elimina sus carpetas de cache antiguas."""
    cache = deepcopy(cache)
    variants = list(cache.get("variants") or [])
    if keep_last is None or keep_last < 1 or len(variants) <= keep_last:
        cache["variants"] = variants
        return cache

    keep = variants[-keep_last:]
    remove = variants[:-keep_last]
    phase1_dir = Path(phase1_dir)
    for variant in remove:
        variant_id = variant.get("variant_id")
        if not variant_id:
            continue
        paths = variant_paths(phase1_dir, stem, variant_id)
        for path in paths.values():
            _remove_variant_parent(path, variant_id)

    keep_ids = {variant.get("variant_id") for variant in keep}
    if cache.get("active_variant_id") not in keep_ids:
        cache["active_variant_id"] = keep[-1].get("variant_id") if keep else None
    if cache.get("finalized_variant_id") not in keep_ids:
        cache["finalized_variant_id"] = None
        cache["finalized"] = False
    cache["variants"] = keep
    return cache


def _remove_variant_parent(file_path, variant_id):
    variant_dir = Path(file_path).parent
    if variant_dir.name != variant_id or variant_dir.parent.name != "variants":
        return
    if variant_dir.is_dir():
        shutil.rmtree(variant_dir)


def get_variant(cache, variant_id):
    if not variant_id:
        return None
    for variant in cache.get("variants", []):
        if variant.get("variant_id") == variant_id:
            return variant
    return None


def find_compatible_variant(cache, current_signature):
    for variant in reversed(cache.get("variants", [])):
        comparison = compare_phase1_signature(
            variant.get("signature", {}), current_signature
        )
        if comparison["compatible"]:
            return variant, comparison
    return None, None


def compare_phase1_signature(cached, current):
    reasons = []
    warnings = []
    checks = [
        ("sample.stem", True),
        ("sample.size_bytes", True),
        ("sample.mtime_ns", True),
        ("extraction.channel", True),
        ("extraction.volume_shape", True),
        ("calibration.px_xy_um", True),
        ("calibration.px_z_um", True),
    ]
    for path, required in checks:
        _compare_path(path, cached, current, reasons if required else warnings)

    for key in SEGMENTATION_KEYS:
        _compare_path(f"cellpose.{key}", cached, current, reasons)

    for key in ("cellpose", "czifile", "tifffile", "numpy", "scikit-image"):
        _compare_path(f"versions.{key}", cached, current, warnings)

    return {
        "compatible": not reasons,
        "blocking_reasons": reasons,
        "warnings": warnings,
    }


def _compare_path(path, cached, current, out):
    old = _get_path(cached, path)
    new = _get_path(current, path)
    if old != new:
        out.append(f"{path} cambio: cache={old}, actual={new}")


def _get_path(data, dotted):
    cur = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def cache_status(phase1_dir, stem):
    root = Path(phase1_dir)
    cache = load_cache(root / "figures_qc")
    canonical = canonical_paths(root, stem)
    active = get_variant(cache, cache.get("active_variant_id"))
    finalized = bool(cache.get("finalized")) and canonical["mask"].is_file()
    has_variant = active is not None
    return {
        "cache": cache,
        "active_variant": active,
        "finalized": finalized,
        "has_variant": has_variant,
        "canonical_mask": canonical["mask"],
        "message": _status_message(cache, finalized, has_variant, canonical["mask"]),
    }


def _status_message(cache, finalized, has_variant, canonical_mask):
    if cache.get("parse_error"):
        return f"Cache incompatible: {cache['parse_error']}"
    if finalized:
        return "Fase 1 finalizada: mascara canonica y cache listos para Fase 2."
    if has_variant:
        return "QC generado pendiente de finalizar. Pulsa Finalizar para habilitar Fase 2."
    if canonical_mask.is_file():
        return "Fase 1 antigua detectada sin cache finalizado. Ejecuta Generar y Finalizar Fase 1."
    return "Falta ejecutar Fase 1: usa Generar y luego Finalizar."
