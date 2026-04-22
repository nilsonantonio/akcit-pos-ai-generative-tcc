"""SpeechT5 training and inference helpers for the TCC pipeline."""

from __future__ import annotations

import argparse
import inspect
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from tcc_audio.io import ensure_parent_dir, read_csv, read_yaml
from tcc_audio.runtime import EmbeddingIndex, load_samples, now_utc_iso, refresh_sample_status, save_samples, stringify_csv_value
from tcc_audio.speecht5_text import normalize_text_for_speecht5


def _load_torch_stack():
    try:
        import librosa
        import soundfile as sf
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            SpeechT5ForTextToSpeech,
            SpeechT5HifiGan,
            SpeechT5Processor,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing SpeechT5 runtime dependencies. Install with requirements-gpu.txt."
        ) from exc
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
            "SpeechT5 few-shot/LoRA training requires Python stdlib `lzma` support (`_lzma`). "
            f"Current interpreter: {sys.executable}. "
            f"{hint} "
            "Validate with `python3 -c \"import lzma, _lzma\"`."
        ) from exc


def _load_training_stack():
    stack = _load_torch_stack()
    _ensure_lzma_available()
    try:
        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
    except ImportError as exc:
        raise SystemExit(
            "Missing SpeechT5 training dependencies. Install with requirements-gpu.txt."
        ) from exc
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
    speaker_embeddings = torch.tensor(
        _load_speaker_embedding(speaker_embedding_path, expected_dim=expected_dim)
    ).to(device)

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
    speaker_id: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    config = read_yaml(config_path)
    samples = load_samples(samples_path)
    context = load_speecht5_context(
        model_name=config["project"]["primary_model"],
        vocoder_name=config["project"]["vocoder_model"],
    )

    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir)
        if checkpoint_path.exists():
            try:
                _, PeftModel, _ = _load_peft()
                context.model = PeftModel.from_pretrained(context.model, str(checkpoint_path))
                context.model.to(next(context.vocoder.parameters()).device)
            except Exception:
                # Few-shot checkpoints may be full-model checkpoints.
                stack = _load_torch_stack()
                context.model = stack["SpeechT5ForTextToSpeech"].from_pretrained(str(checkpoint_path))
                context.model.to(next(context.vocoder.parameters()).device)
            context.model.eval()

    subset = samples[samples["condition"].eq(condition_id)].copy()
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
            samples.loc[samples["sample_id"].eq(sample_id), "model_name"] = row["model_name"] or condition_id
            samples.loc[samples["sample_id"].eq(sample_id), "run_started_at"] = started_at
            samples.loc[samples["sample_id"].eq(sample_id), "run_finished_at"] = now_utc_iso()
            samples.loc[samples["sample_id"].eq(sample_id), "failure_reason"] = ""
            samples.loc[samples["sample_id"].eq(sample_id), "inference_seconds"] = stringify_csv_value(inference_seconds)
            samples.loc[samples["sample_id"].eq(sample_id), "rtf"] = stringify_csv_value(rtf)
            samples.loc[samples["sample_id"].eq(sample_id), "status"] = "generated"
            if condition_id == "speecht5_lora":
                samples.loc[samples["sample_id"].eq(sample_id), "lora_gate_status"] = "stable_lora"
        except Exception as exc:  # pragma: no cover - runtime integration path
            samples.loc[samples["sample_id"].eq(sample_id), "run_started_at"] = started_at
            samples.loc[samples["sample_id"].eq(sample_id), "run_finished_at"] = now_utc_iso()
            samples.loc[samples["sample_id"].eq(sample_id), "failure_reason"] = str(exc)
            samples.loc[samples["sample_id"].eq(sample_id), "status"] = "failed"

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


def _freeze_for_decoder_finetune(model) -> int:
    trainable = 0
    allowed = ("speech_decoder", "speech_decoder_prenet", "speech_decoder_postnet", "postnet")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(token in name for token in allowed)
        if parameter.requires_grad:
            trainable += parameter.numel()
    return trainable


def _apply_lora_adapter(model):
    LoraConfig, _, get_peft_model = _load_peft()
    peft_config = LoraConfig(
        # SpeechT5ForTextToSpeech does not accept the generic seq2seq `inputs_embeds`
        # argument that PEFT injects when `task_type=SEQ_2_SEQ_LM`.
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        bias="none",
    )
    return get_peft_model(model, peft_config)


def _prepare_speaker_embedding_map(embedding_index_path: str | Path) -> dict[str, str]:
    index = EmbeddingIndex.load(embedding_index_path)
    mapping: dict[str, str] = {}
    for _, row in index.frame.iterrows():
        mapping[row["speaker_id"]] = row["speaker_embedding_path"]
    return mapping


def _fine_tune(
    config_path: str | Path,
    manifest_path: str | Path,
    samples_path: str | Path,
    condition_id: str,
    checkpoint_dir: str | Path,
    lora: bool = False,
    learning_rate: float = 1e-4,
    max_steps: int = 100,
    per_device_batch_size: int = 2,
    gpu_hourly_rate: float = 0.0,
) -> pd.DataFrame:
    config = read_yaml(config_path)
    manifest = read_csv(manifest_path)
    samples = load_samples(samples_path)
    sample_rate = int(config["data"].get("sample_rate", 16000))
    stack = _load_training_stack()
    torch = stack["torch"]
    processor = stack["SpeechT5Processor"].from_pretrained(config["project"]["primary_model"])
    speaker_embedding_map = _prepare_speaker_embedding_map(config["data"]["speaker_embeddings_index"])
    DatasetClass = _build_dataset(manifest, speaker_embedding_map, sample_rate)

    train_rows = manifest[manifest["split"].eq("train")].copy()
    val_rows = manifest[manifest["split"].eq("val")].copy()

    for speaker_id in sorted(train_rows["speaker_id"].unique()):
        speaker_train = train_rows[train_rows["speaker_id"].eq(speaker_id)].copy()
        speaker_val = val_rows[val_rows["speaker_id"].eq(speaker_id)].copy()
        if speaker_train.empty:
            continue

        model = stack["SpeechT5ForTextToSpeech"].from_pretrained(config["project"]["primary_model"])
        if lora:
            model = _apply_lora_adapter(model)
        else:
            _freeze_for_decoder_finetune(model)

        output_dir = Path(checkpoint_dir) / condition_id / speaker_id
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args_kwargs = {
            "output_dir": str(output_dir),
            "per_device_train_batch_size": per_device_batch_size,
            "per_device_eval_batch_size": per_device_batch_size,
            "learning_rate": learning_rate,
            "max_steps": max_steps,
            "save_strategy": "steps",
            "save_steps": max(10, max_steps // 5),
            "eval_steps": max(10, max_steps // 5),
            "logging_steps": max(5, max_steps // 10),
            "load_best_model_at_end": False,
            "report_to": [],
            "fp16": torch.cuda.is_available(),
            "dataloader_pin_memory": torch.cuda.is_available(),
        }
        if "eval_strategy" in stack["Seq2SeqTrainingArguments"].__dataclass_fields__:
            training_args_kwargs["eval_strategy"] = "steps"
        else:
            training_args_kwargs["evaluation_strategy"] = "steps"
        training_args = stack["Seq2SeqTrainingArguments"](**training_args_kwargs)

        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": DatasetClass(speaker_train, processor),
            "eval_dataset": DatasetClass(speaker_val, processor) if not speaker_val.empty else None,
            "data_collator": _make_collator(processor, model),
        }
        trainer_signature = inspect.signature(stack["Seq2SeqTrainer"].__init__)
        if "processing_class" in trainer_signature.parameters:
            trainer_kwargs["processing_class"] = processor
        else:
            trainer_kwargs["tokenizer"] = processor
        trainer = stack["Seq2SeqTrainer"](**trainer_kwargs)

        started = time.perf_counter()
        trainer.train()
        elapsed_hours = (time.perf_counter() - started) / 3600
        trainer.save_model()

        evaluation_rows = samples[
            samples["condition"].eq(condition_id) & samples["speaker_id"].eq(speaker_id)
        ].copy()
        train_cost_share = elapsed_hours / max(len(evaluation_rows), 1)
        usd_share = train_cost_share * gpu_hourly_rate
        samples.loc[evaluation_rows.index, "train_gpu_hours"] = stringify_csv_value(train_cost_share)
        samples.loc[evaluation_rows.index, "cost_usd"] = stringify_csv_value(usd_share)
        samples.loc[evaluation_rows.index, "model_name"] = config["project"]["primary_model"]
        if lora:
            samples.loc[evaluation_rows.index, "lora_gate_status"] = "stable_lora"

        run_condition_inference(
            config_path=config_path,
            samples_path=samples_path,
            condition_id=condition_id,
            checkpoint_dir=output_dir,
            speaker_id=speaker_id,
        )
        samples = load_samples(samples_path)

    return samples


def run_few_shot_pipeline(
    config_path: str | Path,
    manifest_path: str | Path,
    samples_path: str | Path,
    checkpoint_dir: str | Path,
    learning_rate: float = 1e-4,
    max_steps: int = 100,
    per_device_batch_size: int = 2,
    gpu_hourly_rate: float = 0.0,
) -> pd.DataFrame:
    return _fine_tune(
        config_path=config_path,
        manifest_path=manifest_path,
        samples_path=samples_path,
        condition_id="speecht5_few_shot_decoder_ft",
        checkpoint_dir=checkpoint_dir,
        lora=False,
        learning_rate=learning_rate,
        max_steps=max_steps,
        per_device_batch_size=per_device_batch_size,
        gpu_hourly_rate=gpu_hourly_rate,
    )


def run_lora_pipeline(
    config_path: str | Path,
    manifest_path: str | Path,
    samples_path: str | Path,
    checkpoint_dir: str | Path,
    learning_rate: float = 5e-5,
    max_steps: int = 100,
    per_device_batch_size: int = 2,
    gpu_hourly_rate: float = 0.0,
) -> pd.DataFrame:
    return _fine_tune(
        config_path=config_path,
        manifest_path=manifest_path,
        samples_path=samples_path,
        condition_id="speecht5_lora",
        checkpoint_dir=checkpoint_dir,
        lora=True,
        learning_rate=learning_rate,
        max_steps=max_steps,
        per_device_batch_size=per_device_batch_size,
        gpu_hourly_rate=gpu_hourly_rate,
    )


def build_arg_parser(condition_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run the {condition_name} SpeechT5 pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gpu-hourly-rate", type=float, default=0.0)
    return parser
