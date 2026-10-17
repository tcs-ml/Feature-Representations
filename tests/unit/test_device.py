"""Device auto-detection and provenance metadata."""

from __future__ import annotations

import torch

from utils.device import detect_device, device_metadata


def test_detect_device_matches_cuda_availability():
    device = detect_device()
    assert device.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_cpu_metadata_fields():
    meta = device_metadata(torch.device("cpu"))
    assert meta == {
        "device": "cpu",
        "device_name": "cpu",
        "cuda_version": None,
        "gpu_total_memory_gb": None,
    }


def test_detected_device_metadata_has_schema_fields():
    meta = device_metadata(detect_device())
    assert {"device", "device_name", "cuda_version", "gpu_total_memory_gb"} <= meta.keys()
