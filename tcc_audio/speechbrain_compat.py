"""Compatibility helpers for SpeechBrain speaker encoders."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_encoder_classifier():
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError as exc:
        raise SystemExit(
            "speechbrain is required. Install with requirements-cpu-macos.txt or requirements-gpu.txt."
        ) from exc
    return EncoderClassifier


def encode_audio_path(classifier, audio_path: str | Path) -> np.ndarray:
    audio_path = str(audio_path)
    if hasattr(classifier, "encode_file"):
        embedding = classifier.encode_file(audio_path)
    else:
        waveform = classifier.load_audio(audio_path)
        embedding = classifier.encode_batch(waveform)
    return embedding.detach().cpu().numpy().reshape(-1)
