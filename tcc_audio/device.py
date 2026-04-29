"""Unified accelerator selection helpers."""

from __future__ import annotations

import platform
from typing import Any


def _load_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def resolve_device_type() -> str:
    torch = _load_torch()
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if platform.system() == "Darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_torch_device() -> Any:
    torch = _load_torch()
    if torch is None:
        return "cpu"
    device_type = resolve_device_type()
    return torch.device(device_type) if device_type != "cpu" else "cpu"


def resolve_transformers_pipeline_device() -> object:
    device_type = resolve_device_type()
    if device_type == "cuda":
        return 0
    if device_type == "mps":
        torch = _load_torch()
        if torch is None:
            return -1
        return torch.device("mps")
    return -1


def resolve_speechbrain_run_opts() -> dict[str, str]:
    device_type = resolve_device_type()
    if device_type == "cuda":
        return {"device": "cuda:0"}
    if device_type == "mps":
        return {"device": "mps"}
    return {"device": "cpu"}
