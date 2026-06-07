import importlib
import subprocess
import sys
import types

import gpu_info


def _fake_torch(available, name="NVIDIA RTX 4060"):
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: available,
            get_device_name=lambda _idx: name,
        )
    )


def test_build_gpu_summary_nvidia_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, "NVIDIA RTX 4060"))
    monkeypatch.setattr(
        gpu_info,
        "detect_video_controllers",
        lambda: [{"name": "NVIDIA RTX 4060", "vendor": "NVIDIA"}],
    )

    summary = gpu_info.build_gpu_summary()

    assert summary["cuda_available"] is True
    assert summary["cuda_device"] == "NVIDIA RTX 4060"
    assert "AMD" not in summary["recommendation_message"]


def test_build_gpu_summary_amd_without_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False))
    monkeypatch.setattr(
        gpu_info,
        "detect_video_controllers",
        lambda: [{"name": "AMD Radeon RX", "vendor": "AMD"}],
    )

    summary = gpu_info.build_gpu_summary()

    assert summary["cuda_available"] is False
    assert "ROCm" in summary["recommendation_message"]
    assert "CPU" in summary["recommendation_message"]


def test_detect_video_controllers_intel_only(monkeypatch):
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout='"Intel Iris Xe Graphics"',
        stderr="",
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)

    assert gpu_info.detect_video_controllers() == [
        {"name": "Intel Iris Xe Graphics", "vendor": "Intel"}
    ]


def test_detect_cuda_torch_not_available(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    real_import = importlib.import_module

    def fake_import(name, package=None):
        if name == "torch":
            raise ImportError("missing")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.setitem(sys.modules, "torch", None)

    summary = gpu_info.detect_cuda()

    assert summary == {"available": False, "device_name": None}
