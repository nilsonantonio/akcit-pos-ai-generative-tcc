"""SpeechT5 LoRA training and inference helpers for the TCC pipeline."""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from tcc_audio.cli_defaults import DEFAULT_AUDIO_BASE_DIR
from tcc_audio.device import resolve_device_type, resolve_torch_device
from tcc_audio.io import ensure_parent_dir, read_csv, read_yaml
from tcc_audio.runtime import (
    EmbeddingIndex,
    load_samples,
    now_utc_iso,
    refresh_sample_status,
    save_samples_atomic,
    stringify_csv_value,
)
from tcc_audio.samples import materialize_checkpoint_samples
from tcc_audio.speecht5_text import normalize_text_for_speecht5


def _load_torch_stack():
    try:
        import librosa
        import soundfile as sf
        import torch
        from torch.utils.data import Dataset
        from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
    except ImportError as exc:
        raise SystemExit("Missing SpeechT5 runtime dependencies. Install with requirements-gpu.txt.") from exc
    return {
        "librosa": librosa,
        "sf": sf,
        "torch": torch,
        "Dataset": Dataset,
        "SpeechT5ForTextToSpeech": SpeechT5ForTextToSpeech,
        "SpeechT5HifiGan": SpeechT5HifiGan,
        "SpeechT5Processor": SpeechT5Processor,
    }


def _ensure_lzma_available() -> None:
    try:
        import _lzma  # noqa: F401
        import lzma  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name not in {"_lzma", "lzma"}:
            raise
        if sys.platform == "darwin":
            hint = "No macOS com Homebrew/asdf: instale `xz`, reinstale o Python e recrie `.venv`."
        elif sys.platform.startswith("linux"):
            hint = "No Linux: instale `xz-utils`/`liblzma-dev`, reinstale o Python e recrie o ambiente virtual."
        else:
            hint = "Reinstale o Python com suporte a `xz/liblzma` e recrie o ambiente virtual."
        raise SystemExit(
            "SpeechT5 LoRA training requires Python stdlib `lzma` support (`_lzma`). "
            f"Current interpreter: {sys.executable}. "
            f"{hint} "
            'Validate with `python3 -c "import lzma, _lzma"`.'
        ) from exc


def _load_training_stack():
    stack = _load_torch_stack()
    _ensure_lzma_available()
    try:
        from transformers import EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments
    except ImportError as exc:
        raise SystemExit("Missing SpeechT5 training dependencies. Install with requirements-gpu.txt.") from exc
    stack["EarlyStoppingCallback"] = EarlyStoppingCallback
    stack["Seq2SeqTrainer"] = Seq2SeqTrainer
    stack["Seq2SeqTrainingArguments"] = Seq2SeqTrainingArguments
    return stack


def _load_peft():
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as exc:
        raise SystemExit("PEFT is required for SpeechT5 LoRA. Install with requirements-gpu.txt.") from exc
    return LoraConfig, PeftModel, get_peft_model


@dataclass
class SpeechT5Context:
    processor: object
    model: object
    vocoder: object
    torch: object
    sf: object
    model_name: str
    vocoder_name: str
    device: str
    active_checkpoint_path: str | None = None


@dataclass
class CheckpointRecord:
    condition_id: str
    checkpoint_step: int
    checkpoint_path: Path
    checkpoint_run_ts: str
    training_scope: str
    training_unit: str
    total_train_gpu_hours: float


@dataclass
class TrainingUnitState:
    training_unit: str
    status: str
    checkpoint_path: str
    best_checkpoint_path: str
    train_gpu_hours: float


def _path_safe_utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_condition_run_root(checkpoint_dir: str | Path, condition_id: str, run_ts: str) -> Path:
    return Path(checkpoint_dir) / condition_id / run_ts


def _training_log_path(checkpoint_dir: str | Path, condition_id: str, run_ts: str) -> Path:
    return _build_condition_run_root(checkpoint_dir, condition_id, run_ts) / "training.log"


def _training_metadata_path(checkpoint_dir: str | Path, condition_id: str, run_ts: str) -> Path:
    return _build_condition_run_root(checkpoint_dir, condition_id, run_ts) / "metadata.json"


class _TeeTextIO:
    def __init__(self, *streams: object) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


@contextlib.contextmanager
def _tee_console_output(log_path: str | Path):
    ensure_parent_dir(log_path)
    with Path(log_path).open("a", encoding="utf-8") as handle:
        stdout = _TeeTextIO(sys.stdout, handle)
        stderr = _TeeTextIO(sys.stderr, handle)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            yield


def _write_training_metadata(
    *,
    metadata_path: str | Path,
    condition_id: str,
    run_ts: str,
    training_scope: str,
    started_at: str,
    finished_at: str,
    training_seconds: float,
    training_gpu_hours_total: float,
    training_units_total: int,
    checkpoints_total: int,
    dataset_train_rows_total: int,
    dataset_val_rows_total: int,
    dataset_inference_rows_total: int,
    training_units: Iterable[TrainingUnitState] | None = None,
) -> None:
    payload = {
        "condition_id": condition_id,
        "run_ts": run_ts,
        "training_scope": training_scope,
        "started_at": started_at,
        "finished_at": finished_at,
        "training_seconds": training_seconds,
        "training_gpu_hours_total": training_gpu_hours_total,
        "training_units_total": training_units_total,
        "checkpoints_total": checkpoints_total,
        "dataset_train_rows_total": dataset_train_rows_total,
        "dataset_val_rows_total": dataset_val_rows_total,
        "dataset_inference_rows_total": dataset_inference_rows_total,
        "training_units": [
            {
                "training_unit": state.training_unit,
                "status": state.status,
                "checkpoint_path": state.checkpoint_path,
                "best_checkpoint_path": state.best_checkpoint_path,
                "train_gpu_hours": state.train_gpu_hours,
            }
            for state in (training_units or [])
        ],
    }
    ensure_parent_dir(metadata_path)
    Path(metadata_path).write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _load_training_metadata(metadata_path: str | Path) -> dict[str, Any]:
    path = Path(metadata_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_lora_conditions(
    config: Mapping[str, Any], selected_condition_ids: Iterable[str] | None = None
) -> list[Mapping[str, Any]]:
    selected = {value for value in (selected_condition_ids or []) if str(value).strip()}
    conditions = []
    for condition in config.get("conditions", []):
        if not isinstance(condition, Mapping):
            continue
        train_strategy = str(condition.get("train_strategy", "")).strip().lower()
        if train_strategy != "lora" and "lora" not in condition:
            continue
        condition_id = str(condition["id"])
        if selected and condition_id not in selected:
            continue
        conditions.append(condition)
    if selected:
        found = {str(condition["id"]) for condition in conditions}
        missing = sorted(selected.difference(found))
        if missing:
            raise ValueError(f"Unknown LoRA condition ids: {', '.join(missing)}")
    return conditions


def _training_scope(condition: Mapping[str, Any]) -> str:
    scope = str(condition.get("training", {}).get("scope", "per_speaker")).strip().lower()
    if scope not in {"per_speaker", "unique"}:
        raise ValueError(f"Unsupported training.scope for `{condition['id']}`: {scope}")
    return scope


def _resolve_gpu_hourly_rate(condition: Mapping[str, Any], cli_override: float | None = None) -> float:
    if cli_override is not None:
        return float(cli_override)
    training = condition.get("training", {})
    if isinstance(training, Mapping):
        configured = training.get("gpu_hourly_rate")
        if configured not in (None, ""):
            return float(configured)
    return 0.0


def _load_embedding(path: str | Path) -> np.ndarray:
    try:
        embedding = np.load(path)
    except FileNotFoundError as exc:
        raise ValueError(
            "Speaker embedding file is missing in the SpeechT5 synthesis path: "
            f"{path}. Reextract synthesis embeddings with `scripts/extract_speaker_embeddings.py`."
        ) from exc
    except (OSError, ValueError) as exc:
        raise ValueError(
            "Speaker embedding file is corrupted or unreadable in the SpeechT5 synthesis path: "
            f"{path}. Reextract synthesis embeddings with `scripts/extract_speaker_embeddings.py`."
        ) from exc
    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.size == 0:
        raise ValueError(
            "Speaker embedding file is empty in the SpeechT5 synthesis path: "
            f"{path}. Reextract synthesis embeddings with `scripts/extract_speaker_embeddings.py`."
        )
    return embedding.reshape(1, -1)


def _load_speaker_embedding(path: str | Path, expected_dim: int | None = None) -> np.ndarray:
    embedding = _load_embedding(path)
    if expected_dim is not None and embedding.shape[1] != expected_dim:
        message = (
            "Speaker embedding dimension mismatch in the SpeechT5 synthesis path: "
            f"got {embedding.shape[1]}, expected {expected_dim}. "
            "`speaker_embedding_path` must point to synthesis embeddings, not evaluation embeddings. "
        )
        if expected_dim == 512:
            message += "Regenerate TTS synthesis embeddings with 'speechbrain/spkrec-xvect-voxceleb'."
        raise ValueError(message)
    norm = float(np.linalg.norm(embedding, ord=2))
    if not np.isfinite(norm):
        raise ValueError(
            "Speaker embedding contains non-finite values in the SpeechT5 synthesis path: "
            f"{path}. Reextract synthesis embeddings with `scripts/extract_speaker_embeddings.py`."
        )
    if norm == 0.0:
        raise ValueError(
            "Speaker embedding has zero L2 norm in the SpeechT5 synthesis path: "
            f"{path}. Reextract synthesis embeddings with `scripts/extract_speaker_embeddings.py`."
        )
    return embedding / norm


def _cached_speaker_embedding(
    path: str | Path,
    *,
    expected_dim: int,
    cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    cache_key = str(Path(path))
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    embedding = _load_speaker_embedding(path, expected_dim=expected_dim)
    if cache is not None:
        cache[cache_key] = embedding
    return embedding


def load_speecht5_context(model_name: str, vocoder_name: str, device: str | None = None) -> SpeechT5Context:
    stack = _load_torch_stack()
    torch = stack["torch"]
    processor = stack["SpeechT5Processor"].from_pretrained(model_name)
    model = stack["SpeechT5ForTextToSpeech"].from_pretrained(model_name)
    vocoder = stack["SpeechT5HifiGan"].from_pretrained(vocoder_name)
    resolved_device = device or str(resolve_torch_device())
    selected_device = resolved_device
    model.to(selected_device)
    vocoder.to(selected_device)
    model.eval()
    vocoder.eval()
    return SpeechT5Context(
        processor=processor,
        model=model,
        vocoder=vocoder,
        torch=torch,
        sf=stack["sf"],
        model_name=model_name,
        vocoder_name=vocoder_name,
        device=selected_device,
    )


def _activate_lora_checkpoint(context: SpeechT5Context, checkpoint_path: str | Path) -> None:
    checkpoint = str(Path(checkpoint_path))
    if context.active_checkpoint_path == checkpoint:
        return

    _, PeftModel, _ = _load_peft()
    unload = getattr(context.model, "unload", None)
    if callable(unload):
        context.model = unload()
    elif hasattr(context.model, "base_model") and hasattr(context.model.base_model, "model"):
        context.model = context.model.base_model.model
    elif getattr(context.model, "config", None) is None or context.active_checkpoint_path is not None:
        stack = _load_torch_stack()
        context.model = stack["SpeechT5ForTextToSpeech"].from_pretrained(context.model_name)
        context.model.to(context.device)

    context.model = PeftModel.from_pretrained(context.model, checkpoint)
    context.model.to(context.device)
    context.model.eval()
    context.active_checkpoint_path = checkpoint


def generate_speech_to_file(
    context: SpeechT5Context,
    text: str,
    speaker_embedding_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16000,
    speaker_embedding_cache: dict[str, np.ndarray] | None = None,
) -> tuple[float, float]:
    torch = context.torch
    inputs = context.processor(text=text, return_tensors="pt")
    device = next(context.model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    expected_dim = int(getattr(context.model.config, "speaker_embedding_dim", 512))
    speaker_embeddings = torch.tensor(
        _cached_speaker_embedding(
            speaker_embedding_path,
            expected_dim=expected_dim,
            cache=speaker_embedding_cache,
        )
    )
    speaker_embeddings = speaker_embeddings.to(device)

    started = time.perf_counter()
    with torch.no_grad():
        waveform = context.model.generate_speech(
            input_ids=input_ids,
            speaker_embeddings=speaker_embeddings,
            vocoder=context.vocoder,
        )
    inference_seconds = time.perf_counter() - started
    output = Path(output_path)
    ensure_parent_dir(output)
    context.sf.write(output, waveform.cpu().numpy(), sample_rate)
    audio_duration = len(waveform) / sample_rate
    rtf = inference_seconds / audio_duration if audio_duration else math.nan
    return inference_seconds, rtf


def run_condition_inference(
    config_path: str | Path,
    samples_path: str | Path,
    condition_id: str,
    checkpoint_dir: str | Path | None = None,
    sample_ids: Iterable[str] | None = None,
    speaker_id: str | None = None,
    checkpoint_step: int | None = None,
    prompt_ids: Iterable[str] | None = None,
    only_pending: bool = True,
    force: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    config = read_yaml(config_path)
    samples = load_samples(samples_path)
    context = load_speecht5_context(
        model_name=config["project"]["primary_model"],
        vocoder_name=config["project"]["vocoder_model"],
    )
    subset = samples[samples["condition"].eq(condition_id)].copy()
    if sample_ids is not None:
        subset = subset[subset["sample_id"].isin(list(sample_ids))].copy()
    if speaker_id:
        subset = subset[subset["speaker_id"].eq(speaker_id)].copy()
    if checkpoint_step is not None:
        subset = subset[subset["checkpoint_step"].astype(str).eq(str(checkpoint_step))].copy()
    selected_prompts = {str(value).strip() for value in (prompt_ids or []) if str(value).strip()}
    if selected_prompts:
        subset = subset[subset["prompt_id"].isin(selected_prompts)].copy()
    if checkpoint_dir is not None:
        subset = subset[subset["checkpoint_path"].astype(str).eq(str(Path(checkpoint_dir)))].copy()
    if force:
        only_pending = False
    if only_pending:
        subset = subset[
            subset["audio_path"].astype(str).apply(lambda value: not (str(value).strip() and Path(value).exists()))
            | subset["status"].astype(str).str.lower().isin({"pending", "failed"})
        ].copy()
    if limit:
        subset = subset.head(limit).copy()
    speaker_embedding_cache: dict[str, np.ndarray] = {}
    if subset.empty:
        samples = refresh_sample_status(samples)
        save_samples_atomic(samples, samples_path)
        return samples

    reconciled_mask = subset["audio_path"].astype(str).apply(lambda value: Path(value).exists() if str(value).strip() else False)
    for _, row in subset[reconciled_mask].iterrows():
        sample_mask = samples["sample_id"].eq(row["sample_id"])
        samples.loc[sample_mask, "model_name"] = row["checkpoint_label"] or row["model_name"] or condition_id
        samples.loc[sample_mask, "failure_reason"] = ""
        samples.loc[sample_mask, "status"] = "generated"
        samples.loc[sample_mask, "lora_gate_status"] = "stable_lora"
    samples = refresh_sample_status(samples)
    save_samples_atomic(samples, samples_path)
    subset = subset[~reconciled_mask].copy()

    for checkpoint_path, checkpoint_rows in subset.groupby("checkpoint_path", dropna=False):
        if not str(checkpoint_path).strip():
            continue
        checkpoint = Path(str(checkpoint_path))
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint}")
        _activate_lora_checkpoint(context, checkpoint)
        for _, row in checkpoint_rows.iterrows():
            sample_id = row["sample_id"]
            started_at = now_utc_iso()
            try:
                inference_seconds, rtf = generate_speech_to_file(
                    context=context,
                    text=row["target_text"],
                    speaker_embedding_path=row["speaker_embedding_path"],
                    output_path=row["audio_path"],
                    sample_rate=int(config["data"].get("sample_rate", 16000)),
                    speaker_embedding_cache=speaker_embedding_cache,
                )
                sample_mask = samples["sample_id"].eq(sample_id)
                samples.loc[sample_mask, "model_name"] = row["checkpoint_label"] or row["model_name"] or condition_id
                samples.loc[sample_mask, "run_started_at"] = started_at
                samples.loc[sample_mask, "run_finished_at"] = now_utc_iso()
                samples.loc[sample_mask, "failure_reason"] = ""
                samples.loc[sample_mask, "inference_seconds"] = stringify_csv_value(inference_seconds)
                samples.loc[sample_mask, "rtf"] = stringify_csv_value(rtf)
                samples.loc[sample_mask, "status"] = "generated"
                samples.loc[sample_mask, "lora_gate_status"] = "stable_lora"
            except Exception as exc:  # pragma: no cover - runtime integration path
                sample_mask = samples["sample_id"].eq(sample_id)
                samples.loc[sample_mask, "run_started_at"] = started_at
                samples.loc[sample_mask, "run_finished_at"] = now_utc_iso()
                samples.loc[sample_mask, "failure_reason"] = str(exc)
                samples.loc[sample_mask, "status"] = "failed"
            samples = refresh_sample_status(samples)
            save_samples_atomic(samples, samples_path)

    samples = refresh_sample_status(samples)
    save_samples_atomic(samples, samples_path)
    return samples


def _preload_speaker_embedding_cache(
    speaker_embedding_map: Mapping[str, str | Path],
    *,
    expected_dim: int,
) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    for speaker_id, embedding_path in speaker_embedding_map.items():
        cache[str(speaker_id)] = _load_speaker_embedding(embedding_path, expected_dim=expected_dim)[0]
    return cache


def _build_dataset(
    manifest: pd.DataFrame,
    speaker_embedding_map: Mapping[str, str | Path],
    sample_rate: int,
    expected_dim: int = 512,
):
    stack = _load_torch_stack()
    Dataset = stack["Dataset"]
    sf = stack["sf"]
    librosa = stack["librosa"]
    speaker_embedding_cache = _preload_speaker_embedding_cache(speaker_embedding_map, expected_dim=expected_dim)

    class SpeechT5TTSDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, processor):
            self.frame = frame.reset_index(drop=True).copy()
            self.processor = processor
            self.texts = [
                normalize_text_for_speecht5(str(text).strip() or fallback)
                for text, fallback in zip(
                    self.frame.get("target_text_speecht5", pd.Series([""] * len(self.frame), dtype=str)).astype(str),
                    self.frame["target_text"].astype(str),
                    strict=False,
                )
            ]

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, index: int) -> dict[str, object]:
            row = self.frame.iloc[index]
            audio, native_sample_rate = sf.read(str(row["audio_path"]), dtype="float32", always_2d=False)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if native_sample_rate != sample_rate:
                audio = librosa.resample(audio, orig_sr=native_sample_rate, target_sr=sample_rate)
            processed = self.processor(
                text=self.texts[index],
                audio_target=audio,
                sampling_rate=sample_rate,
                return_attention_mask=False,
            )
            return {
                "input_ids": processed["input_ids"],
                "labels": processed["labels"][0],
                "speaker_embeddings": speaker_embedding_cache[str(row["speaker_id"])],
            }

    return SpeechT5TTSDataset


def _make_collator(processor, model):
    stack = _load_torch_stack()
    torch = stack["torch"]

    def collate(features: list[dict[str, object]]) -> dict[str, object]:
        batch = processor.pad(
            input_ids=[{"input_ids": feature["input_ids"]} for feature in features],
            labels=[{"input_values": feature["labels"]} for feature in features],
            return_tensors="pt",
        )
        reduction_factor = getattr(model.config, "reduction_factor", 1)
        if reduction_factor > 1:
            target_lengths = torch.tensor([feature["labels"].shape[0] for feature in features])
            target_lengths = target_lengths - (target_lengths % reduction_factor)
            batch["labels"] = batch["labels"][:, : int(target_lengths.max()), :]
        batch["speaker_embeddings"] = torch.tensor(
            np.stack([feature["speaker_embeddings"] for feature in features]),
            dtype=torch.float32,
        )
        return batch

    return collate


def _filter_supported_kwargs(
    kwargs: Mapping[str, Any],
    supported_fields: Iterable[str],
) -> dict[str, Any]:
    supported = set(supported_fields)
    return {key: value for key, value in kwargs.items() if key in supported}


def _apply_lora_adapter(model, lora_config: Mapping[str, Any] | None = None):
    LoraConfig, _, get_peft_model = _load_peft()
    signature = inspect.signature(LoraConfig.__init__)
    kwargs = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "bias": "none",
        "init_lora_weights": True,
        "use_rslora": False,
    }
    if lora_config:
        kwargs.update(dict(lora_config))
    if not any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        kwargs = _filter_supported_kwargs(kwargs, signature.parameters)
    peft_config = LoraConfig(**kwargs)
    return get_peft_model(model, peft_config)


def _prepare_speaker_embedding_map(embedding_index_path: str | Path) -> dict[str, str]:
    index = EmbeddingIndex.load(embedding_index_path)
    mapping: dict[str, str] = {}
    for _, row in index.frame.iterrows():
        mapping[row["speaker_id"]] = row["speaker_embedding_path"]
    return mapping


def _iter_training_units(
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    scope: str,
    selected_speaker_ids: Iterable[str] | None = None,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    selected = {str(value).strip() for value in (selected_speaker_ids or []) if str(value).strip()}
    if scope == "unique":
        if selected:
            raise ValueError("Speaker filters are not supported for training.scope=unique.")
        return [("unique", train_rows.copy(), val_rows.copy())]

    units = []
    for speaker_id in sorted(train_rows["speaker_id"].unique()):
        if selected and str(speaker_id) not in selected:
            continue
        speaker_train = train_rows[train_rows["speaker_id"].eq(speaker_id)].copy()
        speaker_val = val_rows[val_rows["speaker_id"].eq(speaker_id)].copy()
        if speaker_train.empty:
            continue
        units.append((str(speaker_id), speaker_train, speaker_val))
    return units


def _dataset_minutes(train_rows: pd.DataFrame) -> float:
    duration_s = pd.to_numeric(train_rows["duration_s"], errors="coerce").fillna(0.0)
    return float(duration_s.sum() / 60.0)


def _speaker_dataset_minutes(train_rows: pd.DataFrame) -> list[tuple[str, float]]:
    speaker_minutes: list[tuple[str, float]] = []
    for speaker_id in sorted(train_rows["speaker_id"].astype(str).unique()):
        speaker_rows = train_rows[train_rows["speaker_id"].eq(speaker_id)].copy()
        if speaker_rows.empty:
            continue
        speaker_minutes.append((str(speaker_id), _dataset_minutes(speaker_rows)))
    return speaker_minutes


def _log_training_dataset_summary(
    *,
    condition_id: str,
    scope: str,
    training_unit: str,
    train_rows: pd.DataFrame,
    phase: str,
) -> None:
    if scope == "per_speaker":
        print(
            f"[LoRA {phase}] condition={condition_id} scope=per_speaker "
            f"speaker_id={training_unit} dataset_minutes={_dataset_minutes(train_rows):.2f}"
        )
        return

    speaker_minutes = _speaker_dataset_minutes(train_rows)
    details = ", ".join(f"{speaker_id}={minutes:.2f}m" for speaker_id, minutes in speaker_minutes)
    print(
        f"[LoRA {phase}] condition={condition_id} scope=unique "
        f"speaker_count={len(speaker_minutes)} dataset_minutes_by_speaker=[{details}]"
    )


def _build_checkpoint_root(checkpoint_dir: str | Path, condition_id: str, run_ts: str, training_unit: str) -> Path:
    return _build_condition_run_root(checkpoint_dir, condition_id, run_ts) / training_unit


def _checkpoint_dirs(output_dir: str | Path) -> list[Path]:
    checkpoints = []
    for path in Path(output_dir).iterdir():
        if not path.is_dir() or not path.name.startswith("checkpoint-"):
            continue
        try:
            int(path.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append(path)
    return sorted(checkpoints, key=lambda path: int(path.name.split("-", 1)[1]))


def _checkpoint_step(checkpoint_path: str | Path) -> int:
    name = Path(checkpoint_path).name
    return int(name.split("-", 1)[1])


def _validate_effective_batch_size(training_config: Mapping[str, Any]) -> None:
    if "effective_batch_size" not in training_config:
        return
    world_size = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
    expected = (
        int(training_config.get("per_device_train_batch_size", 1))
        * int(training_config.get("gradient_accumulation_steps", 1))
        * world_size
    )
    configured = int(training_config["effective_batch_size"])
    if configured != expected:
        raise ValueError(
            "training.effective_batch_size mismatch: "
            f"configured={configured}, derived={expected} for WORLD_SIZE={world_size}."
        )


def _resolve_early_stopping_config(training_config: Mapping[str, Any]) -> dict[str, float | int] | None:
    patience_raw = training_config.get("early_stopping_patience")
    if patience_raw in (None, ""):
        return None
    try:
        patience = int(patience_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("training.early_stopping_patience must be an integer greater than 0.") from exc
    if patience <= 0:
        raise ValueError("training.early_stopping_patience must be greater than 0.")

    threshold_raw = training_config.get("early_stopping_threshold", 0.0)
    try:
        threshold = float(threshold_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("training.early_stopping_threshold must be a float greater than or equal to 0.") from exc
    if threshold < 0:
        raise ValueError("training.early_stopping_threshold must be greater than or equal to 0.")

    if not bool(training_config.get("load_best_model_at_end", False)):
        raise ValueError("training.early_stopping_patience requires training.load_best_model_at_end=true.")

    return {
        "early_stopping_patience": patience,
        "early_stopping_threshold": threshold,
    }


def _build_training_args(
    stack: Mapping[str, Any],
    output_dir: str | Path,
    training_config: Mapping[str, Any],
    has_eval_dataset: bool,
):
    torch = stack["torch"]
    fields = stack["Seq2SeqTrainingArguments"].__dataclass_fields__
    _validate_effective_batch_size(training_config)
    early_stopping_config = _resolve_early_stopping_config(training_config)

    if "warmup_ratio" in training_config:
        raise ValueError(
            "training.warmup_ratio is no longer supported. Replace it with training.warmup_steps."
        )

    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(getattr(mps_backend, "is_available", lambda: False)())
    accelerator_available = cuda_available or mps_available
    fp16_requested = bool(training_config.get("fp16", False))
    bf16_requested = bool(training_config.get("bf16", False))
    if fp16_requested and bf16_requested:
        raise ValueError("training.fp16 and training.bf16 are mutually exclusive; enable only one of them.")
    if fp16_requested and not cuda_available:
        print("Warning: training.fp16=true ignored because CUDA is not available.")
    if bf16_requested and not cuda_available:
        print("Warning: training.bf16=true ignored because CUDA is not available.")
    if bf16_requested and "bf16" not in fields:
        print(
            "Warning: training.bf16=true ignored because this transformers version does not support "
            "Seq2SeqTrainingArguments.bf16."
        )

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(training_config.get("per_device_train_batch_size", 2)),
        "per_device_eval_batch_size": int(training_config.get("per_device_eval_batch_size", 2)),
        "gradient_accumulation_steps": int(training_config.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(training_config.get("learning_rate", 5e-5)),
        "warmup_steps": float(training_config.get("warmup_steps", 0)),
        "lr_scheduler_type": training_config.get("lr_scheduler_type", "linear"),
        "max_steps": int(training_config.get("max_steps", 100)),
        "save_strategy": training_config.get("save_strategy", "steps"),
        "save_steps": int(training_config.get("save_steps", 100)),
        "logging_strategy": training_config.get("logging_strategy", "steps"),
        "logging_steps": int(training_config.get("logging_steps", 10)),
        "load_best_model_at_end": bool(training_config.get("load_best_model_at_end", False)),
        "metric_for_best_model": training_config.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": bool(training_config.get("greater_is_better", False)),
        "weight_decay": float(training_config.get("weight_decay", 0.0)),
        "max_grad_norm": float(training_config.get("max_grad_norm", 1.0)),
        "gradient_checkpointing": bool(training_config.get("gradient_checkpointing", False)),
        "report_to": [],
        "label_names": ["labels"],
        "fp16": bool(cuda_available and fp16_requested),
        "bf16": bool(cuda_available and bf16_requested and "bf16" in fields),
        "dataloader_pin_memory": cuda_available,
        "dataloader_num_workers": int(training_config.get("dataloader_num_workers", 0)),
        "use_cpu": not accelerator_available,
    }
    if kwargs["dataloader_num_workers"] > 0:
        kwargs["dataloader_persistent_workers"] = bool(training_config.get("dataloader_persistent_workers", True))
        if training_config.get("dataloader_prefetch_factor") not in (None, ""):
            kwargs["dataloader_prefetch_factor"] = int(training_config["dataloader_prefetch_factor"])

    if training_config.get("save_total_limit") not in (None, ""):
        print(
            "Warning: training.save_total_limit is ignored because checkpoint-level evaluation "
            "preserves every checkpoint saved during the run."
        )

    if has_eval_dataset:
        eval_strategy_value = training_config.get("eval_strategy", training_config.get("evaluation_strategy", "steps"))
        if early_stopping_config and str(eval_strategy_value).strip().lower() == "no":
            raise ValueError("training.early_stopping_patience requires training.eval_strategy/evaluation_strategy != 'no'.")
        kwargs["eval_steps"] = int(training_config.get("eval_steps", kwargs["save_steps"]))
        if "eval_strategy" in fields:
            kwargs["eval_strategy"] = eval_strategy_value
        else:
            kwargs["evaluation_strategy"] = eval_strategy_value
        if (
            early_stopping_config
            and str(training_config.get("save_strategy", "steps")).strip().lower() == "steps"
            and str(eval_strategy_value).strip().lower() == "steps"
            and int(kwargs["save_steps"]) != int(kwargs["eval_steps"])
        ):
            print(
                "Warning: early stopping may stop only at the next save step because "
                "training.save_steps != training.eval_steps."
            )
    else:
        kwargs["load_best_model_at_end"] = False
        if "eval_strategy" in fields:
            kwargs["eval_strategy"] = "no"
        else:
            kwargs["evaluation_strategy"] = "no"

    if not has_eval_dataset and bool(training_config.get("load_best_model_at_end", False)):
        print("Warning: load_best_model_at_end ignored because the selected training unit has no validation rows.")

    filtered = _filter_supported_kwargs(kwargs, fields)
    return stack["Seq2SeqTrainingArguments"](**filtered)


def _build_trainer_callbacks(
    stack: Mapping[str, Any],
    training_config: Mapping[str, Any] | None,
    has_eval_dataset: bool,
) -> list[object]:
    resolved_training_config = training_config or {}
    early_stopping_config = _resolve_early_stopping_config(resolved_training_config)
    if early_stopping_config is None:
        return []
    if not has_eval_dataset:
        print("Warning: early stopping ignored because the selected training unit has no validation rows.")
        return []
    return [
        stack["EarlyStoppingCallback"](
            early_stopping_patience=int(early_stopping_config["early_stopping_patience"]),
            early_stopping_threshold=float(early_stopping_config["early_stopping_threshold"]),
        )
    ]


def _build_trainer(
    stack: Mapping[str, Any],
    model: object,
    training_args: object,
    processor: object,
    train_dataset: object,
    eval_dataset: object | None,
    training_config: Mapping[str, Any] | None = None,
):
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": _make_collator(processor, model),
    }
    trainer_signature = inspect.signature(stack["Seq2SeqTrainer"].__init__)
    callbacks = _build_trainer_callbacks(
        stack=stack,
        training_config=training_config,
        has_eval_dataset=eval_dataset is not None,
    )
    if callbacks and "callbacks" in trainer_signature.parameters:
        trainer_kwargs["callbacks"] = callbacks
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor
    return stack["Seq2SeqTrainer"](**trainer_kwargs)


def _condition_run_dirs(checkpoint_dir: str | Path, condition_id: str) -> list[Path]:
    root = Path(checkpoint_dir) / condition_id
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.name)


def _resolve_run_ts(
    checkpoint_dir: str | Path,
    condition_id: str,
    *,
    resume_train: bool,
    checkpoint_run_ts: str | None,
) -> str:
    if checkpoint_run_ts:
        return checkpoint_run_ts
    run_dirs = _condition_run_dirs(checkpoint_dir, condition_id)
    if resume_train and run_dirs:
        return run_dirs[-1].name
    return _path_safe_utc_timestamp()


def _latest_checkpoint_dir(output_dir: str | Path) -> Path | None:
    checkpoints = _checkpoint_dirs(output_dir)
    return checkpoints[-1] if checkpoints else None


def _best_checkpoint_dir(output_dir: str | Path) -> Path | None:
    latest = _latest_checkpoint_dir(output_dir)
    if latest is None:
        return None
    trainer_state_path = latest / "trainer_state.json"
    if trainer_state_path.exists():
        try:
            trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            trainer_state = {}
        best_model_checkpoint = trainer_state.get("best_model_checkpoint")
        if isinstance(best_model_checkpoint, str) and best_model_checkpoint.strip():
            candidate = Path(best_model_checkpoint)
            if not candidate.is_absolute():
                candidate = Path(output_dir) / candidate
            if candidate.exists():
                return candidate
    return latest


def _unit_final_model_exists(output_dir: str | Path) -> bool:
    return (Path(output_dir) / "adapter_model.safetensors").exists()


def _build_training_unit_state(
    training_unit: str,
    output_dir: str | Path,
    elapsed_hours: float,
    *,
    status: str,
) -> TrainingUnitState:
    latest_checkpoint = _latest_checkpoint_dir(output_dir)
    best_checkpoint = _best_checkpoint_dir(output_dir)
    return TrainingUnitState(
        training_unit=training_unit,
        status=status,
        checkpoint_path=str(latest_checkpoint or ""),
        best_checkpoint_path=str(best_checkpoint or ""),
        train_gpu_hours=float(elapsed_hours),
    )


def _training_unit_state_map(metadata_path: str | Path) -> dict[str, TrainingUnitState]:
    payload = _load_training_metadata(metadata_path)
    output: dict[str, TrainingUnitState] = {}
    for raw_state in payload.get("training_units", []):
        if not isinstance(raw_state, Mapping):
            continue
        training_unit = str(raw_state.get("training_unit", "")).strip()
        if not training_unit:
            continue
        output[training_unit] = TrainingUnitState(
            training_unit=training_unit,
            status=str(raw_state.get("status", "")),
            checkpoint_path=str(raw_state.get("checkpoint_path", "")),
            best_checkpoint_path=str(raw_state.get("best_checkpoint_path", "")),
            train_gpu_hours=float(raw_state.get("train_gpu_hours", 0.0) or 0.0),
        )
    return output


def materialize_condition_samples(
    *,
    config_path: str | Path,
    samples_path: str | Path,
    run_matrix_path: str | Path,
    checkpoint_dir: str | Path,
    condition: Mapping[str, Any],
    checkpoint_run_ts: str | None,
    best_only: bool = True,
    checkpoint_steps: Iterable[int] | None = None,
    speaker_ids: Iterable[str] | None = None,
    prompt_ids: Iterable[str] | None = None,
    audio_base_dir: str | Path = DEFAULT_AUDIO_BASE_DIR,
    gpu_hourly_rate_override: float | None = None,
) -> list[str]:
    condition_id = str(condition["id"])
    training_scope = _training_scope(condition)
    resolved_run_ts = _resolve_run_ts(
        checkpoint_dir,
        condition_id,
        resume_train=True,
        checkpoint_run_ts=checkpoint_run_ts,
    )
    training_units_state = _training_unit_state_map(_training_metadata_path(checkpoint_dir, condition_id, resolved_run_ts))
    if training_scope == "unique":
        unit_names = ["unique"]
    else:
        run_root = _build_condition_run_root(checkpoint_dir, condition_id, resolved_run_ts)
        fallback_unit_names = [path.name for path in run_root.iterdir() if path.is_dir()] if run_root.exists() else []
        unit_names = sorted(training_units_state) if training_units_state else fallback_unit_names

    selected_steps = {int(step) for step in (checkpoint_steps or [])}
    selected_sample_ids: list[str] = []
    config = read_yaml(config_path)
    for training_unit in unit_names:
        output_dir = _build_checkpoint_root(checkpoint_dir, condition_id, resolved_run_ts, training_unit)
        if not output_dir.exists():
            continue
        state = training_units_state.get(training_unit)
        checkpoint_paths: list[Path] = []
        if best_only and not selected_steps:
            best_checkpoint = _best_checkpoint_dir(output_dir)
            if best_checkpoint is not None:
                checkpoint_paths = [best_checkpoint]
        else:
            checkpoint_paths = _checkpoint_dirs(output_dir)
            if selected_steps:
                checkpoint_paths = [path for path in checkpoint_paths if _checkpoint_step(path) in selected_steps]

        for checkpoint_path in checkpoint_paths:
            sample_ids = materialize_checkpoint_samples(
                samples_path=samples_path,
                run_matrix_path=run_matrix_path,
                condition_id=condition_id,
                checkpoint_step=_checkpoint_step(checkpoint_path),
                checkpoint_path=checkpoint_path,
                checkpoint_run_ts=resolved_run_ts,
                training_scope=training_scope,
                training_unit=training_unit,
                audio_base_dir=audio_base_dir,
                total_train_gpu_hours=(state.train_gpu_hours if state is not None else 0.0),
                gpu_hourly_rate=_resolve_gpu_hourly_rate(condition, cli_override=gpu_hourly_rate_override),
                speaker_selection_path=config["data"].get("speaker_selection_path"),
                speaker_embeddings_path=config["data"].get("speaker_embeddings_index"),
                speaker_ids=speaker_ids,
                prompt_ids=prompt_ids,
            )
            selected_sample_ids.extend(sample_ids)
    return selected_sample_ids


def _train_condition(
    config_path: str | Path,
    manifest_path: str | Path,
    checkpoint_dir: str | Path,
    condition: Mapping[str, Any],
    gpu_hourly_rate_override: float | None,
    checkpoint_run_ts: str | None = None,
    resume_train: bool = False,
    speaker_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    config = read_yaml(config_path)
    manifest = read_csv(manifest_path)
    condition_id = str(condition["id"])
    training_scope = _training_scope(condition)
    training_config = dict(condition.get("training", {}))
    lora_config = dict(condition.get("lora", {}))
    run_ts = _resolve_run_ts(
        checkpoint_dir,
        condition_id,
        resume_train=resume_train,
        checkpoint_run_ts=checkpoint_run_ts,
    )
    condition_run_root = _build_condition_run_root(checkpoint_dir, condition_id, run_ts)
    condition_run_root.mkdir(parents=True, exist_ok=True)
    log_path = _training_log_path(checkpoint_dir, condition_id, run_ts)
    metadata_path = _training_metadata_path(checkpoint_dir, condition_id, run_ts)

    train_rows = manifest[manifest["split"].eq("train")].copy()
    val_rows = manifest[manifest["split"].eq("val")].copy()

    sample_rate = int(config["data"].get("sample_rate", 16000))
    project_config = config.get("project", {})
    expected_speaker_embedding_dim = (
        int(project_config.get("tts_speaker_embedding_dim", 512))
        if isinstance(project_config, Mapping)
        else 512
    )
    stack = _load_training_stack()
    processor = stack["SpeechT5Processor"].from_pretrained(config["project"]["primary_model"])
    speaker_embedding_map = _prepare_speaker_embedding_map(config["data"]["speaker_embeddings_index"])
    dataset_class = _build_dataset(manifest, speaker_embedding_map, sample_rate, expected_dim=expected_speaker_embedding_dim)
    run_started_at = now_utc_iso()
    run_started_perf = time.perf_counter()
    training_units_total = 0
    checkpoints_total = 0
    training_gpu_hours_total = 0.0
    existing_states = _training_unit_state_map(metadata_path)
    training_unit_states: dict[str, TrainingUnitState] = dict(existing_states)
    units = _iter_training_units(train_rows, val_rows, training_scope, selected_speaker_ids=speaker_ids)

    with _tee_console_output(log_path):
        for training_unit, unit_train_rows, unit_val_rows in units:
            training_units_total += 1
            output_dir = _build_checkpoint_root(checkpoint_dir, condition_id, run_ts, training_unit)
            output_dir.mkdir(parents=True, exist_ok=True)
            if resume_train and _unit_final_model_exists(output_dir):
                state = training_unit_states.get(training_unit)
                training_unit_states[training_unit] = state or _build_training_unit_state(
                    training_unit,
                    output_dir,
                    0.0,
                    status="trained",
                )
                continue

            model = stack["SpeechT5ForTextToSpeech"].from_pretrained(config["project"]["primary_model"])
            model = _apply_lora_adapter(model, lora_config)

            training_args = _build_training_args(
                stack=stack,
                output_dir=output_dir,
                training_config=training_config,
                has_eval_dataset=not unit_val_rows.empty,
            )
            trainer = _build_trainer(
                stack=stack,
                model=model,
                training_args=training_args,
                processor=processor,
                train_dataset=dataset_class(unit_train_rows, processor),
                eval_dataset=dataset_class(unit_val_rows, processor) if not unit_val_rows.empty else None,
                training_config=training_config,
            )

            _log_training_dataset_summary(
                condition_id=condition_id,
                scope=training_scope,
                training_unit=training_unit,
                train_rows=unit_train_rows,
                phase="train:start",
            )
            started = time.perf_counter()
            latest_checkpoint = _latest_checkpoint_dir(output_dir) if resume_train else None
            if latest_checkpoint is not None:
                trainer.train(resume_from_checkpoint=str(latest_checkpoint))
            else:
                trainer.train()
            elapsed_hours = (time.perf_counter() - started) / 3600.0
            training_gpu_hours_total += elapsed_hours
            trainer.save_model()
            _log_training_dataset_summary(
                condition_id=condition_id,
                scope=training_scope,
                training_unit=training_unit,
                train_rows=unit_train_rows,
                phase="train:end",
            )

            checkpoints = _checkpoint_dirs(output_dir)
            checkpoints_total += len(checkpoints)
            if not checkpoints:
                print(f"Warning: no checkpoints were saved for `{condition_id}` training unit `{training_unit}`.")
                training_unit_states[training_unit] = _build_training_unit_state(
                    training_unit,
                    output_dir,
                    elapsed_hours,
                    status="training",
                )
                continue
            training_unit_states[training_unit] = _build_training_unit_state(
                training_unit,
                output_dir,
                elapsed_hours,
                status="trained",
            )

    run_finished_at = now_utc_iso()
    training_seconds = time.perf_counter() - run_started_perf
    _write_training_metadata(
        metadata_path=metadata_path,
        condition_id=condition_id,
        run_ts=run_ts,
        training_scope=training_scope,
        started_at=run_started_at,
        finished_at=run_finished_at,
        training_seconds=training_seconds,
        training_gpu_hours_total=training_gpu_hours_total,
        training_units_total=training_units_total,
        checkpoints_total=checkpoints_total,
        dataset_train_rows_total=len(train_rows),
        dataset_val_rows_total=len(val_rows),
        dataset_inference_rows_total=0,
        training_units=training_unit_states.values(),
    )
    return pd.DataFrame(
        [
            {
                "training_unit": state.training_unit,
                "status": state.status,
                "checkpoint_path": state.checkpoint_path,
                "best_checkpoint_path": state.best_checkpoint_path,
                "train_gpu_hours": state.train_gpu_hours,
            }
            for state in training_unit_states.values()
        ]
    )


def run_lora_pipeline(
    config_path: str | Path,
    manifest_path: str | Path,
    checkpoint_dir: str | Path,
    gpu_hourly_rate: float | None = None,
    condition_ids: Iterable[str] | None = None,
    checkpoint_run_ts: str | None = None,
    resume_train: bool = False,
    speaker_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    config = read_yaml(config_path)
    conditions = _resolve_lora_conditions(config, condition_ids)
    if not conditions:
        raise ValueError("No LoRA conditions found in the experiment config.")

    frames: list[pd.DataFrame] = []
    for condition in conditions:
        frames.append(
            _train_condition(
                config_path=config_path,
                manifest_path=manifest_path,
                checkpoint_dir=checkpoint_dir,
                condition=condition,
                gpu_hourly_rate_override=gpu_hourly_rate,
                checkpoint_run_ts=checkpoint_run_ts,
                resume_train=resume_train,
                speaker_ids=speaker_ids,
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_lora_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SpeechT5 LoRA training.")
    parser.add_argument("-c", "--config")
    parser.add_argument("-m", "--manifest")
    parser.add_argument("-k", "--checkpoint-dir")
    parser.add_argument(
        "-C",
        "--condition",
        action="append",
        default=[],
        help="LoRA condition id to run. Repeat the flag to select multiple conditions. "
        "If omitted, all LoRA conditions are executed sequentially.",
    )
    parser.add_argument(
        "--gpu-hourly-rate",
        type=float,
        default=None,
        help="Optional global override for training.gpu_hourly_rate defined in each LoRA condition.",
    )
    parser.add_argument("--resume-train", action="store_true", help="Resume from the latest saved checkpoint for the run.")
    parser.add_argument("--checkpoint-run-ts", help="Existing checkpoint run timestamp to resume.")
    parser.add_argument("--speaker-id", action="append", default=[], help="Optional per-speaker training filter.")
    return parser
