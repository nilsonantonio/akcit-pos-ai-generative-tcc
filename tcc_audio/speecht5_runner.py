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
from tcc_audio.io import ensure_parent_dir, read_csv, read_yaml
from tcc_audio.runtime import (
    EmbeddingIndex,
    load_samples,
    now_utc_iso,
    refresh_sample_status,
    save_samples,
    stringify_csv_value,
)
from tcc_audio.samples import materialize_checkpoint_samples, reset_condition_checkpoint_rows
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
        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
    except ImportError as exc:
        raise SystemExit("Missing SpeechT5 training dependencies. Install with requirements-gpu.txt.") from exc
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


@dataclass
class CheckpointRecord:
    condition_id: str
    checkpoint_step: int
    checkpoint_path: Path
    checkpoint_run_ts: str
    training_scope: str
    training_unit: str
    total_train_gpu_hours: float


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
    }
    ensure_parent_dir(metadata_path)
    Path(metadata_path).write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


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
    embedding = np.load(path)
    return embedding.reshape(1, -1).astype(np.float32)


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
    return embedding


def load_speecht5_context(model_name: str, vocoder_name: str, device: str | None = None) -> SpeechT5Context:
    stack = _load_torch_stack()
    torch = stack["torch"]
    processor = stack["SpeechT5Processor"].from_pretrained(model_name)
    model = stack["SpeechT5ForTextToSpeech"].from_pretrained(model_name)
    vocoder = stack["SpeechT5HifiGan"].from_pretrained(vocoder_name)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
    )


def generate_speech_to_file(
    context: SpeechT5Context,
    text: str,
    speaker_embedding_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16000,
) -> tuple[float, float]:
    torch = context.torch
    inputs = context.processor(text=text, return_tensors="pt")
    device = next(context.model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    expected_dim = int(getattr(context.model.config, "speaker_embedding_dim", 512))
    speaker_embeddings = torch.tensor(_load_speaker_embedding(speaker_embedding_path, expected_dim=expected_dim)).to(
        device
    )

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
    checkpoint_dir: str | Path,
    sample_ids: Iterable[str] | None = None,
    speaker_id: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    config = read_yaml(config_path)
    samples = load_samples(samples_path)
    context = load_speecht5_context(
        model_name=config["project"]["primary_model"],
        vocoder_name=config["project"]["vocoder_model"],
    )

    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
    _, PeftModel, _ = _load_peft()
    context.model = PeftModel.from_pretrained(context.model, str(checkpoint_path))
    context.model.to(next(context.vocoder.parameters()).device)
    context.model.eval()

    subset = samples[samples["condition"].eq(condition_id)].copy()
    if sample_ids is not None:
        subset = subset[subset["sample_id"].isin(list(sample_ids))].copy()
    if speaker_id:
        subset = subset[subset["speaker_id"].eq(speaker_id)].copy()
    if limit:
        subset = subset.head(limit).copy()

    for _, row in subset.iterrows():
        sample_id = row["sample_id"]
        started_at = now_utc_iso()
        try:
            inference_seconds, rtf = generate_speech_to_file(
                context=context,
                text=row["target_text"],
                speaker_embedding_path=row["speaker_embedding_path"],
                output_path=row["audio_path"],
                sample_rate=int(config["data"].get("sample_rate", 16000)),
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
    save_samples(samples, samples_path)
    return samples


def _build_dataset(manifest: pd.DataFrame, speaker_embedding_map: dict[str, str], sample_rate: int):
    stack = _load_torch_stack()
    librosa = stack["librosa"]
    Dataset = stack["Dataset"]

    def _resolve_training_text(row: pd.Series) -> str:
        normalized_text = str(row.get("target_text_speecht5", "")).strip()
        if normalized_text:
            return normalized_text
        return normalize_text_for_speecht5(row["target_text"])

    class SpeechT5TTSDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, processor):
            self.frame = frame.reset_index(drop=True)
            self.processor = processor

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, index: int) -> dict[str, object]:
            row = self.frame.iloc[index]
            audio, _ = librosa.load(row["audio_path"], sr=sample_rate)
            processed = self.processor(
                text=_resolve_training_text(row),
                audio_target=audio,
                sampling_rate=sample_rate,
                return_attention_mask=False,
            )
            return {
                "input_ids": processed["input_ids"],
                "labels": processed["labels"][0],
                "speaker_embeddings": _load_speaker_embedding(
                    speaker_embedding_map[row["speaker_id"]],
                    expected_dim=512,
                )[0],
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
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    if scope == "unique":
        return [("unique", train_rows.copy(), val_rows.copy())]

    units = []
    for speaker_id in sorted(train_rows["speaker_id"].unique()):
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


def _build_training_args(
    stack: Mapping[str, Any],
    output_dir: str | Path,
    training_config: Mapping[str, Any],
    has_eval_dataset: bool,
):
    torch = stack["torch"]
    fields = stack["Seq2SeqTrainingArguments"].__dataclass_fields__
    _validate_effective_batch_size(training_config)

    fp16_requested = bool(training_config.get("fp16", False))
    if fp16_requested and not torch.cuda.is_available():
        print("Warning: training.fp16=true ignored because CUDA is not available.")

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(training_config.get("per_device_train_batch_size", 2)),
        "per_device_eval_batch_size": int(training_config.get("per_device_eval_batch_size", 2)),
        "gradient_accumulation_steps": int(training_config.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(training_config.get("learning_rate", 5e-5)),
        "warmup_ratio": float(training_config.get("warmup_ratio", 0.0)),
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
        "fp16": bool(torch.cuda.is_available() and fp16_requested),
        "dataloader_pin_memory": bool(torch.cuda.is_available()),
    }

    if training_config.get("save_total_limit") not in (None, ""):
        print(
            "Warning: training.save_total_limit is ignored because checkpoint-level evaluation "
            "preserves every checkpoint saved during the run."
        )

    if has_eval_dataset:
        eval_strategy_value = training_config.get("eval_strategy", training_config.get("evaluation_strategy", "steps"))
        kwargs["eval_steps"] = int(training_config.get("eval_steps", kwargs["save_steps"]))
        if "eval_strategy" in fields:
            kwargs["eval_strategy"] = eval_strategy_value
        else:
            kwargs["evaluation_strategy"] = eval_strategy_value
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


def _build_trainer(
    stack: Mapping[str, Any],
    model: object,
    training_args: object,
    processor: object,
    train_dataset: object,
    eval_dataset: object | None,
):
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": _make_collator(processor, model),
    }
    trainer_signature = inspect.signature(stack["Seq2SeqTrainer"].__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor
    return stack["Seq2SeqTrainer"](**trainer_kwargs)


def _train_condition(
    config_path: str | Path,
    manifest_path: str | Path,
    samples_path: str | Path,
    checkpoint_dir: str | Path,
    condition: Mapping[str, Any],
    gpu_hourly_rate_override: float | None,
    audio_base_dir: str | Path,
) -> pd.DataFrame:
    config = read_yaml(config_path)
    manifest = read_csv(manifest_path)
    condition_id = str(condition["id"])
    training_scope = _training_scope(condition)
    training_config = dict(condition.get("training", {}))
    lora_config = dict(condition.get("lora", {}))
    gpu_hourly_rate = _resolve_gpu_hourly_rate(condition, cli_override=gpu_hourly_rate_override)
    run_ts = _path_safe_utc_timestamp()
    condition_run_root = _build_condition_run_root(checkpoint_dir, condition_id, run_ts)
    condition_run_root.mkdir(parents=True, exist_ok=True)
    log_path = _training_log_path(checkpoint_dir, condition_id, run_ts)
    metadata_path = _training_metadata_path(checkpoint_dir, condition_id, run_ts)

    train_rows = manifest[manifest["split"].eq("train")].copy()
    val_rows = manifest[manifest["split"].eq("val")].copy()

    samples = reset_condition_checkpoint_rows(load_samples(samples_path), condition_id)
    save_samples(samples, samples_path)

    sample_rate = int(config["data"].get("sample_rate", 16000))
    stack = _load_training_stack()
    processor = stack["SpeechT5Processor"].from_pretrained(config["project"]["primary_model"])
    speaker_embedding_map = _prepare_speaker_embedding_map(config["data"]["speaker_embeddings_index"])
    dataset_class = _build_dataset(manifest, speaker_embedding_map, sample_rate)
    run_started_at = now_utc_iso()
    run_started_perf = time.perf_counter()
    training_units_total = 0
    checkpoints_total = 0
    dataset_inference_rows_total = 0
    training_gpu_hours_total = 0.0

    with _tee_console_output(log_path):
        for training_unit, unit_train_rows, unit_val_rows in _iter_training_units(train_rows, val_rows, training_scope):
            training_units_total += 1
            model = stack["SpeechT5ForTextToSpeech"].from_pretrained(config["project"]["primary_model"])
            model = _apply_lora_adapter(model, lora_config)

            output_dir = _build_checkpoint_root(checkpoint_dir, condition_id, run_ts, training_unit)
            output_dir.mkdir(parents=True, exist_ok=True)

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
            )

            _log_training_dataset_summary(
                condition_id=condition_id,
                scope=training_scope,
                training_unit=training_unit,
                train_rows=unit_train_rows,
                phase="train:start",
            )
            started = time.perf_counter()
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
                continue

            for checkpoint in checkpoints:
                checkpoint_step = _checkpoint_step(checkpoint)
                sample_ids = materialize_checkpoint_samples(
                    samples_path=samples_path,
                    condition_id=condition_id,
                    checkpoint_step=checkpoint_step,
                    checkpoint_path=checkpoint,
                    checkpoint_run_ts=run_ts,
                    training_scope=training_scope,
                    training_unit=training_unit,
                    audio_base_dir=audio_base_dir,
                    total_train_gpu_hours=elapsed_hours,
                    gpu_hourly_rate=gpu_hourly_rate,
                )
                dataset_inference_rows_total += len(sample_ids)
                if not sample_ids:
                    continue
                run_condition_inference(
                    config_path=config_path,
                    samples_path=samples_path,
                    condition_id=condition_id,
                    checkpoint_dir=checkpoint,
                    sample_ids=sample_ids,
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
        dataset_inference_rows_total=dataset_inference_rows_total,
    )

    return load_samples(samples_path)


def run_lora_pipeline(
    config_path: str | Path,
    manifest_path: str | Path,
    samples_path: str | Path,
    checkpoint_dir: str | Path,
    gpu_hourly_rate: float | None = None,
    condition_ids: Iterable[str] | None = None,
    audio_base_dir: str | Path = "artifacts/audio",
) -> pd.DataFrame:
    config = read_yaml(config_path)
    conditions = _resolve_lora_conditions(config, condition_ids)
    if not conditions:
        raise ValueError("No LoRA conditions found in the experiment config.")

    for condition in conditions:
        _train_condition(
            config_path=config_path,
            manifest_path=manifest_path,
            samples_path=samples_path,
            checkpoint_dir=checkpoint_dir,
            condition=condition,
            gpu_hourly_rate_override=gpu_hourly_rate,
            audio_base_dir=audio_base_dir,
        )
    return load_samples(samples_path)


def build_lora_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SpeechT5 LoRA checkpoint-evaluation pipeline.")
    parser.add_argument("-c", "--config")
    parser.add_argument("-s", "--samples")
    parser.add_argument("-m", "--manifest")
    parser.add_argument("-k", "--checkpoint-dir")
    parser.add_argument("-a", "--audio-base-dir", default=str(DEFAULT_AUDIO_BASE_DIR))
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
    return parser
