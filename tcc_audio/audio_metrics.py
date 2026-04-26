"""Automatic audio metrics used by the TCC pipeline."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from tcc_audio.config import load_experiment_config, resolve_speaker_similarity_model
from tcc_audio.runtime import load_samples, refresh_sample_status, save_samples, stringify_csv_value
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
    model_name: str | None = None,
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    config = load_experiment_config(config_path)
    resolved_model_name = model_name or resolve_speaker_similarity_model(config)
    EncoderClassifier = load_encoder_classifier()
    classifier = EncoderClassifier.from_hparams(source=resolved_model_name)
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
            samples.loc[samples["sample_id"].eq(sample_id), "speaker_similarity"] = stringify_csv_value(score)
        except Exception as exc:  # pragma: no cover - runtime integration path
            samples.loc[samples["sample_id"].eq(sample_id), "failure_reason"] = str(exc)

    samples = refresh_sample_status(samples)
    save_samples(samples, out_path)
    return samples


def _iter_nisqa_search_roots(explicit_path: str | Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    for raw_path in [
        explicit_path,
        os.environ.get("NISQA_PATH"),
        Path.cwd() / "NISQA",
        Path.cwd() / "third_party" / "NISQA",
        Path.cwd() / "vendor" / "NISQA",
    ]:
        if not raw_path:
            continue
        path = Path(raw_path).expanduser().resolve()
        if path.exists() and path not in candidates:
            candidates.append(path)
    return candidates


def _load_nisqa_predictor(nisqa_path: str | Path | None = None):
    candidates = [
        ("nisqa.NISQA_model", "nisqaModel"),
        ("NISQA.nisqa.NISQA_model", "nisqaModel"),
    ]

    def _try_import():
        for module_name, symbol in candidates:
            try:
                module = importlib.import_module(module_name)
                return getattr(module, symbol)
            except Exception:
                continue
        return None

    predictor = _try_import()
    if predictor is not None:
        return predictor

    for search_root in _iter_nisqa_search_roots(nisqa_path):
        root_str = str(search_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        predictor = _try_import()
        if predictor is not None:
            return predictor

    raise SystemExit(
        "NISQA is not installed. Install it in the active environment or point to a valid local checkout with "
        "`--nisqa-path /path/to/NISQA` or `NISQA_PATH=/path/to/NISQA`. "
        "In this project, the expected checkout path is `./NISQA`."
    )


def _resolve_nisqa_root(nisqa_path: str | Path | None = None) -> Path:
    roots = _iter_nisqa_search_roots(nisqa_path)
    if not roots:
        raise SystemExit(
            "NISQA path not found. Clone the official repository into `./NISQA`, pass "
            "`--nisqa-path /path/to/NISQA`, or export `NISQA_PATH=/path/to/NISQA`."
        )
    return roots[0]


def _resolve_nisqa_pretrained_model(nisqa_root: Path) -> Path:
    model_path = nisqa_root / "weights" / "nisqa_tts.tar"
    if not model_path.exists():
        raise SystemExit(
            f"NISQA checkpoint not found at `{model_path}`. The project expects a valid checkout at `./NISQA` "
            "with the TTS naturalness model `weights/nisqa_tts.tar` present."
        )
    return model_path


def compute_nisqa(
    samples_path: str | Path,
    out_path: str | Path,
    batch_size: int = 4,
    nisqa_path: str | Path | None = None,
) -> pd.DataFrame:
    nisqaModel = _load_nisqa_predictor(nisqa_path=nisqa_path)
    nisqa_root = _resolve_nisqa_root(nisqa_path=nisqa_path)
    pretrained_model = _resolve_nisqa_pretrained_model(nisqa_root)
    samples = load_samples(samples_path)
    subset = samples[samples["audio_path"].astype(str).str.strip().ne("")]
    audio_paths = [path for path in subset["audio_path"].tolist() if Path(path).exists()]
    if not audio_paths:
        samples = refresh_sample_status(samples)
        save_samples(samples, out_path)
        return samples

    with tempfile.TemporaryDirectory(prefix="nisqa_predict_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "nisqa_input.csv"
        result_dir = tmpdir_path / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"audio_path": [str(Path(path).resolve()) for path in audio_paths]}).to_csv(csv_path, index=False)

        predictor = nisqaModel(
            {
                "mode": "predict_csv",
                "pretrained_model": str(pretrained_model),
                "data_dir": "",
                "csv_file": str(csv_path),
                "csv_deg": "audio_path",
                "output_dir": str(result_dir),
                "tr_bs_val": batch_size,
                "tr_num_workers": 0,
                "ms_channel": None,
            }
        )
        results = predictor.predict()

    score_map = {
        str(Path(row["audio_path"]).resolve().as_posix()): row["mos_pred"]
        for _, row in results.iterrows()
        if pd.notna(row.get("mos_pred"))
    }
    for _, row in subset.iterrows():
        score = score_map.get(Path(row["audio_path"]).resolve().as_posix())
        if score is not None:
            samples.loc[samples["sample_id"].eq(row["sample_id"]), "nisqa"] = stringify_csv_value(float(score))
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
            samples.loc[samples["sample_id"].eq(row["sample_id"]), "f0_rmse"] = stringify_csv_value(rmse)
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
