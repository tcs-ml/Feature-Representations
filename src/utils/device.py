"""Compute-device detection and provenance.

The device is auto-detected: cuda when available, cpu otherwise. There are no
device config keys. SAE training runs on the detected device; the greedy
stage is numpy and always CPU. Results themselves are always saved from CPU
tensors, and run identity excludes the device, so a grid computed on one host
resumes on any other.
"""

from __future__ import annotations

import torch


def detect_device() -> torch.device:
    """Return cuda when available, else cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_metadata(device: torch.device) -> dict:
    """Provenance fields recorded in every learned-run summary."""
    meta: dict = {"device": str(device)}
    if device.type == "cuda":
        index = device.index or 0
        props = torch.cuda.get_device_properties(index)
        meta.update(
            device_name=props.name,
            cuda_version=torch.version.cuda,
            gpu_total_memory_gb=round(props.total_memory / 2**30, 2),
        )
    else:
        meta.update(device_name="cpu", cuda_version=None, gpu_total_memory_gb=None)
    return meta
