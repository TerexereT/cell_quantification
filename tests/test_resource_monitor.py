import resource_monitor as rm


def test_parse_nvidia_smi_normal():
    out = rm.parse_nvidia_smi("3, 1795, 8188\n")
    assert out == {"util_pct": 3.0, "mem_used_mb": 1795.0, "mem_total_mb": 8188.0}


def test_parse_nvidia_smi_takes_first_gpu():
    out = rm.parse_nvidia_smi("10, 100, 8000\n80, 7000, 8000\n")
    assert out["util_pct"] == 10.0
    assert out["mem_used_mb"] == 100.0


def test_parse_nvidia_smi_na_returns_none():
    assert rm.parse_nvidia_smi("[N/A], [N/A], [N/A]") is None


def test_parse_nvidia_smi_empty_or_garbage():
    assert rm.parse_nvidia_smi("") is None
    assert rm.parse_nvidia_smi("basura") is None


def test_format_elapsed():
    assert rm.format_elapsed(0) == "00:00"
    assert rm.format_elapsed(59) == "00:59"
    assert rm.format_elapsed(60) == "01:00"
    assert rm.format_elapsed(3661) == "1:01:01"


def test_phase2_eta_seconds():
    assert rm.phase2_eta_seconds(10, 2, 5) == 15.0
    assert rm.phase2_eta_seconds(10, 0, 5) is None
    assert rm.phase2_eta_seconds(10, 5, 5) is None


def test_format_metrics_phase1_no_percent():
    line = rm.format_metrics(65, 35.0, None)
    assert "%" not in line.split("CPU")[0]  # no progress percent before CPU
    assert "ETA" not in line
    assert "GPU n/d" in line
    assert "Tiempo 01:05" in line


def test_format_metrics_cpu_none():
    assert "CPU n/d" in rm.format_metrics(5, None, None)


def test_format_metrics_full():
    gpu = {"util_pct": 80.0, "mem_used_mb": 3276.8, "mem_total_mb": 8192.0}
    line = rm.format_metrics(30, 50.0, gpu, done=2, total=5)
    assert "GB" in line
    assert "ETA" in line
    assert "40% (2/5)" in line
    assert "GPU 80%" in line


def test_read_cpu_percent_returns_float_or_none():
    val = rm.read_cpu_percent()
    assert val is None or isinstance(val, float)


def test_is_mask_progress():
    assert rm.is_mask_progress("Procesando x.tif", "Procesando ")
    assert not rm.is_mask_progress("otro", "Procesando ")
    assert not rm.is_mask_progress(("done",), "Procesando ")
