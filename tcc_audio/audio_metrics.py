"""Automatic audio metrics used by the TCC pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tcc_audio.runtime import load_samples, refresh_sample_status, save_samples
from tcc_audio.speechbrain_compat import encode_audio_path, load_encoder_classifier


def _load_librosa():
    try:
        import librosa
    except ImportError as exc:
        raise SystemExit("librosa is required for F0 RMSE.") from exc
    return librosa


def compute_speaker_similarity(
    samples_path: str | Path,
    out_path: str | Path,
    model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
) -> pd.DataFrame:
    EncoderClassifier = load_encoder_classifier()
    classifier = EncoderClassifier.from_hparams(source=model_name)
    samples = load_samples(samples_path)
    subset = samples[samples["audio_path"].astype(str).str.strip().ne("")]

    for _, row in subset.iterrows():
        sample_id = row["sample_id"]
        audio_path = Path(row["audio_path"])
        reference_audio = Path(row["reference_audio_path"])
        if not audio_path.exists() or not reference_audio.exists():
            continue
        try:
            generated = encode_audio_path(classifier, audio_path)
            reference = encode_audio_path(classifier, reference_audio)
            score = float(np.dot(generated, reference) / (np.linalg.norm(generated) * np.linalg.norm(reference)))
            samples.loc[samples["sample_id"].eq(sample_id), "speaker_similarity"] = score
        except Exception as exc:  # pragma: no cover - runtime integration path
            samples.loc[samples["sample_id"].eq(sample_id), "failure_reason"] = str(exc)

    samples = refresh_sample_status(samples)
    save_samples(samples, out_path)
    return samples


def _load_nisqa_predictor():
    candidates = [
        ("nisqa.NISQA_model", "nisqaModel"),
        ("NISQA.nisqa.NISQA_model", "nisqaModel"),
    ]
    for module_name, symbol in candidates:
        try:
            module = __import__(module_name, fromlist=[symbol])
            return getattr(module, symbol)
        except Exception:
            continue
    raise SystemExit("NISQA is not installed. Install a local NISQA package before running this metric.")


def compute_nisqa(
    samples_path: str | Path,
    out_path: str | Path,
    batch_size: int = 4,
) -> pd.DataFrame:
    nisqaModel = _load_nisqa_predictor()
    samples = load_samples(samples_path)
    subset = samples[samples["audio_path"].astype(str).str.strip().ne("")]
    audio_paths = [path for path in subset["audio_path"].tolist() if Path(path).exists()]
    predictor = nisqaModel()
    scores = predictor.predict(audio_paths, bs=batch_size)
    score_map = {Path(path).as_posix(): score for path, score in zip(audio_paths, scores)}
    for _, row in subset.iterrows():
        score = score_map.get(Path(row["audio_path"]).as_posix())
        if score is not None:
            samples.loc[samples["sample_id"].eq(row["sample_id"]), "nisqa"] = float(score)
    samples = refresh_sample_status(samples)
    save_samples(samples, out_path)
    return samples


def compute_f0_rmse(samples_path: str | Path, out_path: str | Path, sample_rate: int = 16000) -> pd.DataFrame:
    librosa = _load_librosa()
    samples = load_samples(samples_path)
    subset = samples[samples["audio_path"].astype(str).str.strip().ne("")]

    for _, row in subset.iterrows():
        audio_path = Path(row["audio_path"])
        reference_audio = Path(row["reference_audio_path"])
        if not audio_path.exists() or not reference_audio.exists():
            continue
        try:
            generated, _ = librosa.load(audio_path, sr=sample_rate)
            reference, _ = librosa.load(reference_audio, sr=sample_rate)
            gen_f0 = librosa.yin(generated, fmin=50, fmax=500, sr=sample_rate)
            ref_f0 = librosa.yin(reference, fmin=50, fmax=500, sr=sample_rate)
            limit = min(len(gen_f0), len(ref_f0))
            if limit == 0:
                continue
            rmse = float(np.sqrt(np.mean((gen_f0[:limit] - ref_f0[:limit]) ** 2)))
            samples.loc[samples["sample_id"].eq(row["sample_id"]), "f0_rmse"] = rmse
        except Exception as exc:  # pragma: no cover - runtime integration path
            samples.loc[samples["sample_id"].eq(row["sample_id"]), "failure_reason"] = str(exc)

    samples = refresh_sample_status(samples)
    save_samples(samples, out_path)
    return samples


def build_arg_parser(metric_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Compute {metric_name} for generated samples.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    return parser
