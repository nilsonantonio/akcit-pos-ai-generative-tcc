"""Smoke checks for the execution scaffold."""

import io
import json
import os
import sys
import tarfile
import types
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcc_audio.common_voice import prepare_common_voice_metadata
from tcc_audio.audio_preprocess import build_arg_parser as build_audio_preprocess_arg_parser
from tcc_audio.common_voice_download import (
    download_common_voice_pt,
    request_dataset_download_session,
    stage_common_voice_archive,
)
from tcc_audio.audio_metrics import (
    _load_nisqa_predictor,
    _resolve_nisqa_pretrained_model,
    _resolve_nisqa_root,
    compute_speaker_similarity,
)
from tcc_audio.cli_defaults import (
    DEFAULT_AUDIO_BASE_DIR,
    DEFAULT_EMBEDDINGS_INDEX_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PROCESSED_METADATA_PATH,
    DEFAULT_PROMPTS_PATH,
    DEFAULT_RAW_CLIPS_DIR,
    DEFAULT_RAW_METADATA_PATH,
    DEFAULT_RAW_TSV,
    DEFAULT_REPORT_ASSETS_DIR,
    DEFAULT_RUN_MATRIX_PATH,
    DEFAULT_SAMPLES_PATH,
    DEFAULT_SPEAKER_SELECTION_PATH,
    load_cli_config,
    resolve_data_path,
    resolve_deliverable_path,
    resolve_sample_rate,
    resolve_samples_path,
)
from tcc_audio.common_voice import build_arg_parser as build_common_voice_arg_parser
from tcc_audio.config import (
    DEFAULT_SPEAKER_SIMILARITY_MODEL,
    DEFAULT_TTS_SPEAKER_EMBEDDING_DIM,
    resolve_speaker_similarity_model,
    resolve_tts_speaker_embedding_dim,
    resolve_tts_speaker_embedding_model,
)
from tcc_audio.dataset_inventory import (
    build_arg_parser as build_dataset_inventory_arg_parser,
    build_dataset_inventory,
    main as dataset_inventory_main,
    render_dataset_inventory,
    resolve_inventory_paths,
)
from tcc_audio.experiments import build_arg_parser as build_run_matrix_arg_parser, generate_run_matrix
from tcc_audio.evaluation import aggregate_metrics
from tcc_audio.manifest import build_arg_parser as build_manifest_arg_parser, validate_data_manifest, validate_prompts
from tcc_audio.report_assets import build_arg_parser as build_report_assets_arg_parser, make_report_assets
from tcc_audio.runtime import load_samples
from tcc_audio.samples import (
    build_arg_parser as build_samples_arg_parser,
    initialize_samples,
    materialize_checkpoint_samples,
    reset_condition_checkpoint_rows,
)
from tcc_audio.speaker_selection import build_arg_parser as build_speaker_selection_arg_parser, select_speakers
from tcc_audio.speaker_embeddings import build_arg_parser as build_speaker_embeddings_arg_parser, extract_speaker_embeddings
from tcc_audio.speecht5_runner import (
    _build_condition_run_root,
    _apply_lora_adapter,
    _log_training_dataset_summary,
    _resolve_gpu_hourly_rate,
    _load_speaker_embedding,
    _load_torch_stack,
    _load_training_stack,
    _tee_console_output,
    _training_log_path,
    _training_metadata_path,
    _write_training_metadata,
    build_lora_arg_parser,
)
from tcc_audio.training_cleanup import (
    build_cleanup_arg_parser,
    build_remove_condition_arg_parser,
    cleanup_training_results,
    remove_condition,
    resolve_cleanup_paths,
)
from tcc_audio.speecht5_text import count_unk_tokens, has_unk_tokens, normalize_text_for_speecht5
from tcc_audio.whisper_batch import build_arg_parser as build_whisper_arg_parser, main as whisper_main
from tcc_audio.wer import build_arg_parser as build_wer_arg_parser, compute_wer_from_asr, main as wer_main, word_error_rate


class MockUrlopenResponse:
    def __init__(self, payload: bytes) -> None:
        self._buffer = io.BytesIO(payload)

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def __enter__(self) -> "MockUrlopenResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def _build_common_voice_archive_bytes() -> bytes:
    payload = io.BytesIO()
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        dataset_root = tmp / "cv-corpus-25.0-2026-03-09" / "pt"
        clips_dir = dataset_root / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        (clips_dir / "sample.mp3").write_bytes(b"fake-mp3")
        (dataset_root / "validated.tsv").write_text(
            "client_id\tpath\ttext\tgender\tlocale\tvariant\n"
            "spk1\tsample.mp3\tTexto um.\tmale\tpt\tpt-BR\n",
            encoding="utf-8",
        )
        (dataset_root / "validated_sentences.tsv").write_text(
            "sentence_id\tsentence\tis_used\n"
            "s1\tTexto um.\t1\n",
            encoding="utf-8",
        )
        (dataset_root / "clip_durations.tsv").write_text(
            "clip\tduration[ms]\n"
            "sample.mp3\t1234\n",
            encoding="utf-8",
        )
        with tarfile.open(fileobj=payload, mode="w:gz") as archive:
            archive.add(dataset_root, arcname=dataset_root.relative_to(tmp))
    return payload.getvalue()


def _write_common_voice_metadata_inputs(
    tmp: Path,
    *,
    validated_rows: list[str],
    validated_sentences_rows: list[str],
    clip_duration_rows: list[str],
) -> tuple[Path, Path, Path]:
    validated_tsv = tmp / "validated.tsv"
    validated_sentences_tsv = tmp / "validated_sentences.tsv"
    clip_durations_tsv = tmp / "clip_durations.tsv"

    validated_tsv.write_text(
        "client_id\tpath\tsentence_id\tsentence\ttext\tgender\tlocale\tvariant\n"
        + "".join(validated_rows),
        encoding="utf-8",
    )
    validated_sentences_tsv.write_text(
        "sentence_id\tsentence\tvariant\n"
        + "".join(validated_sentences_rows),
        encoding="utf-8",
    )
    clip_durations_tsv.write_text(
        "clip\tduration[ms]\n"
        + "".join(clip_duration_rows),
        encoding="utf-8",
    )
    return validated_tsv, validated_sentences_tsv, clip_durations_tsv


def _write_cleanup_config(tmp: Path) -> Path:
    config = tmp / "config.yaml"
    config.write_text(
        "project:\n"
        "  primary_model: microsoft/speecht5_tts\n"
        "  vocoder_model: microsoft/speecht5_hifigan\n"
        "data:\n"
        "  manifest_path: data/manifests/data_manifest.csv\n"
        "conditions:\n"
        "  - id: cond_a\n"
        "    label: Cond A\n"
        "    train_strategy: lora\n"
        "    lora:\n"
        "      r: 8\n"
        "    training:\n"
        "      scope: per_speaker\n"
        "  - id: cond_b\n"
        "    label: Cond B\n"
        "    train_strategy: lora\n"
        "    lora:\n"
        "      r: 8\n"
        "    training:\n"
        "      scope: unique\n"
        "deliverables:\n"
        f"  run_matrix: {tmp / 'artifacts/run_matrix.csv'}\n"
        f"  samples: {tmp / 'artifacts/evaluation/samples.csv'}\n"
        f"  metrics_summary: {tmp / 'artifacts/evaluation/metrics_summary.csv'}\n"
        f"  cost_summary: {tmp / 'artifacts/evaluation/cost_summary.csv'}\n"
        f"  report_assets_dir: {tmp / 'report_assets'}\n",
        encoding="utf-8",
    )
    return config


def _write_inventory_config(tmp: Path) -> Path:
    config = tmp / "config.yaml"
    config.write_text(
        "project:\n"
        "  primary_model: microsoft/speecht5_tts\n"
        "data:\n"
        f"  manifest_path: {tmp / 'data/manifests/data_manifest.csv'}\n"
        f"  speaker_selection_path: {tmp / 'data/manifests/speaker_selection.csv'}\n"
        "conditions:\n"
        "  - id: cond_per_speaker\n"
        "    train_strategy: lora\n"
        "    lora:\n"
        "      r: 8\n"
        "    training:\n"
        "      scope: per_speaker\n"
        "  - id: cond_unique\n"
        "    train_strategy: lora\n"
        "    lora:\n"
        "      r: 8\n"
        "    training:\n"
        "      scope: unique\n",
        encoding="utf-8",
    )
    return config


def _write_wav(path: Path, *, sample_rate: int = 16000, channels: int = 1, frames: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames * channels)


def _prepare_dataset_inventory_fixture(tmp: Path) -> dict[str, Path]:
    config = _write_inventory_config(tmp)
    raw_dir = tmp / "data/raw/common_voice_pt"
    clips_dir = raw_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for name in ["clip_a.mp3", "clip_b.mp3", "clip_c.mp3"]:
        (clips_dir / name).write_bytes(b"fake")
    (raw_dir / "validated.tsv").write_text(
        "client_id\tpath\ttext\tgender\tlocale\tvariant\n"
        "spk1\tclip_a.mp3\tTexto A\tmale\tpt\tpt-BR\n"
        "spk2\tclip_b.mp3\tTexto B\tfemale\tpt\tpt-PT\n",
        encoding="utf-8",
    )

    processed_dir = tmp / "data/processed/common_voice_pt"
    processed_metadata = tmp / "data/manifests/common_voice_processed.csv"
    processed_metadata.parent.mkdir(parents=True, exist_ok=True)
    wav_a = processed_dir / "source_1/a.wav"
    wav_b = processed_dir / "source_1/b.wav"
    wav_c = processed_dir / "source_2/c.wav"
    for wav_path in [wav_a, wav_b, wav_c]:
        _write_wav(wav_path)
    pd.DataFrame(
        [
            {"source_speaker_id": "source_1", "duration_s": 1.5, "audio_path": str(wav_a)},
            {"source_speaker_id": "source_1", "duration_s": 2.5, "audio_path": str(wav_b)},
            {"source_speaker_id": "source_2", "duration_s": 5.0, "audio_path": str(wav_c)},
        ]
    ).to_csv(processed_metadata, index=False)

    manifest_path = tmp / "data/manifests/data_manifest.csv"
    pd.DataFrame(
        [
            {"speaker_id": "speaker_01", "split": "train", "duration_s": 1.5, "audio_path": str(wav_a)},
            {"speaker_id": "speaker_01", "split": "val", "duration_s": 2.5, "audio_path": str(wav_b)},
            {"speaker_id": "speaker_02", "split": "train", "duration_s": 5.0, "audio_path": str(wav_c)},
        ]
    ).to_csv(manifest_path, index=False)

    speaker_selection = tmp / "data/manifests/speaker_selection.csv"
    pd.DataFrame([{"speaker_id": "speaker_01"}, {"speaker_id": "speaker_02"}]).to_csv(speaker_selection, index=False)
    json_out = tmp / "artifacts/dataset_inventory.json"

    return {
        "config": config,
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "processed_metadata": processed_metadata,
        "manifest_path": manifest_path,
        "speaker_selection": speaker_selection,
        "json_out": json_out,
    }


def test_prompt_file_has_expected_contract() -> None:
    report = validate_prompts(ROOT / "data/prompts/ptbr_test_prompts.csv")
    assert report.ok, report.errors
    assert report.row_count == 24
    assert report.summary["commercial_subset_count"] == 8


def test_build_common_voice_arg_parser_uses_canonical_defaults() -> None:
    parser = build_common_voice_arg_parser()
    args = parser.parse_args([])

    assert args.tsv == str(DEFAULT_RAW_TSV)
    assert args.clips_dir == str(DEFAULT_RAW_CLIPS_DIR)
    assert args.out == str(DEFAULT_RAW_METADATA_PATH)
    assert args.locale == "pt"
    assert args.variant == "pt-BR"


def test_build_audio_preprocess_arg_parser_uses_canonical_defaults() -> None:
    parser = build_audio_preprocess_arg_parser()
    args = parser.parse_args([])

    assert args.metadata == str(DEFAULT_RAW_METADATA_PATH)
    assert args.out_dir == "data/processed/common_voice_pt"
    assert args.out_metadata == str(DEFAULT_PROCESSED_METADATA_PATH)
    assert args.sample_rate is None


def test_build_speaker_selection_arg_parser_accepts_aliases() -> None:
    parser = build_speaker_selection_arg_parser()
    args = parser.parse_args(
        [
            "-m",
            "processed.csv",
            "-o",
            "manifest.csv",
            "-s",
            "selection.csv",
            "-n",
            "1000",
        ]
    )

    assert args.metadata == "processed.csv"
    assert args.manifest_out == "manifest.csv"
    assert args.speaker_selection_out == "selection.csv"
    assert args.speaker_target_count == 1000


def test_load_cli_config_and_resolve_paths_from_yaml() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = tmp / "config.yaml"
        config.write_text(
            "data:\n"
            f"  prompts_path: {tmp / 'prompts.csv'}\n"
            f"  manifest_path: {tmp / 'manifest.csv'}\n"
            f"  speaker_selection_path: {tmp / 'speaker_selection.csv'}\n"
            f"  speaker_embeddings_index: {tmp / 'speaker_embeddings.csv'}\n"
            "  sample_rate: 22050\n"
            "deliverables:\n"
            f"  run_matrix: {tmp / 'run_matrix.csv'}\n"
            f"  samples: {tmp / 'samples.csv'}\n"
            f"  report_assets_dir: {tmp / 'report_assets'}\n",
            encoding="utf-8",
        )

        config_path, payload = load_cli_config(config)

    assert config_path == config
    assert resolve_data_path(payload, "manifest_path", DEFAULT_MANIFEST_PATH) == tmp / "manifest.csv"
    assert resolve_data_path(payload, "prompts_path", DEFAULT_PROMPTS_PATH) == tmp / "prompts.csv"
    assert resolve_data_path(payload, "speaker_selection_path", DEFAULT_SPEAKER_SELECTION_PATH) == tmp / "speaker_selection.csv"
    assert resolve_data_path(payload, "speaker_embeddings_index", DEFAULT_EMBEDDINGS_INDEX_PATH) == tmp / "speaker_embeddings.csv"
    assert resolve_deliverable_path(payload, "run_matrix", DEFAULT_RUN_MATRIX_PATH) == tmp / "run_matrix.csv"
    assert resolve_deliverable_path(payload, "samples", DEFAULT_SAMPLES_PATH) == tmp / "samples.csv"
    assert resolve_deliverable_path(payload, "report_assets_dir", DEFAULT_REPORT_ASSETS_DIR) == tmp / "report_assets"
    assert resolve_samples_path(payload) == tmp / "samples.csv"
    assert resolve_sample_rate(payload) == 22050


def test_build_manifest_arg_parser_defaults_to_optional_paths() -> None:
    parser = build_manifest_arg_parser()
    args = parser.parse_args(["--check-files"])

    assert args.manifest is None
    assert args.prompts is None
    assert args.config is None
    assert args.check_files is True


def test_build_speaker_embeddings_arg_parser_defaults_to_optional_paths() -> None:
    parser = build_speaker_embeddings_arg_parser()
    args = parser.parse_args(["-c", "config.yaml"])

    assert args.config == "config.yaml"
    assert args.speaker_selection is None
    assert args.out_index is None
    assert args.out_dir is None


def test_build_run_matrix_arg_parser_defaults_to_optional_paths() -> None:
    parser = build_run_matrix_arg_parser()
    args = parser.parse_args(["-c", "config.yaml"])

    assert args.config == "config.yaml"
    assert args.out is None


def test_build_samples_arg_parser_defaults_to_optional_paths() -> None:
    parser = build_samples_arg_parser()
    args = parser.parse_args(["-c", "config.yaml"])

    assert args.config == "config.yaml"
    assert args.run_matrix is None
    assert args.out is None
    assert args.audio_base_dir == str(DEFAULT_AUDIO_BASE_DIR)


def test_run_matrix_generation() -> None:
    with TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "run_matrix.csv"
        matrix = generate_run_matrix(ROOT / "configs/speecht5_minimal.yaml", out)
        assert out.exists()
        assert set(matrix["condition"]) == {
            "speecht5_lora_conservative",
            "speecht5_lora_unique",
        }
        assert set(matrix["training_scope"]) == {"per_speaker", "unique"}
        assert len(matrix) == 192
        speaker_selection = Path(tmpdir) / "speaker_selection.csv"
        embeddings = Path(tmpdir) / "speaker_embeddings.csv"
        pd.DataFrame(
            [
                {
                    "speaker_id": f"speaker_{index:02d}",
                    "reference_audio": f"audio/speaker_{index:02d}.wav",
                }
                for index in range(1, 5)
            ]
        ).to_csv(speaker_selection, index=False)
        pd.DataFrame(
            [
                {
                    "speaker_id": f"speaker_{index:02d}",
                    "speaker_embedding_path": f"embeddings/speaker_{index:02d}.npy",
                }
                for index in range(1, 5)
            ]
        ).to_csv(embeddings, index=False)
        samples = initialize_samples(
            out,
            Path(tmpdir) / "samples.csv",
            speaker_selection_path=speaker_selection,
            speaker_embeddings_path=embeddings,
        )
        assert len(samples) == len(matrix)
        assert "asr_text" in samples.columns
        assert "reference_audio_path" in samples.columns
        assert "speaker_embedding_path" in samples.columns
        assert "checkpoint_step" in samples.columns
        assert samples["audio_path"].astype(str).str.strip().eq("").all()


def test_lora_run_matrix_supports_raw_and_normalized_modes() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        prompts = tmp / "prompts.csv"
        config = tmp / "config.yaml"
        prompts.write_text(
            "prompt_id,category,raw_text,normalized_text,include_prompt_subexperiment,include_commercial_subset\n"
            "P001,general,\"Texto cru\",\"Texto normalizado\",true,true\n",
            encoding="utf-8",
        )
        config.write_text(
            "project:\n"
            "  seed: 42\n"
            "data:\n"
            "  prompts_path: " + str(prompts) + "\n"
            "  speaker_target_count: 1\n"
            "conditions:\n"
            "  - id: lora_text_modes\n"
            "    label: LoRA Text Modes\n"
            "    train_strategy: lora\n"
            "    normalization_modes: [raw, normalized]\n"
            "    training:\n"
            "      scope: unique\n"
            "    lora:\n"
            "      r: 8\n",
            encoding="utf-8",
        )
        matrix = generate_run_matrix(config)

    assert len(matrix) == 2
    assert set(matrix["text_variant"]) == {"raw", "normalized"}
    assert set(matrix["condition"]) == {"lora_text_modes"}


def test_materialize_checkpoint_samples_expands_rows_per_checkpoint() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        run_matrix_path = tmp / "run_matrix.csv"
        samples_path = tmp / "samples.csv"
        speaker_selection = tmp / "speaker_selection.csv"
        embeddings = tmp / "speaker_embeddings.csv"

        pd.DataFrame(
            [
                {
                    "run_id": "lora_cond__speaker_01__P001__normalized",
                    "condition": "lora_cond",
                    "condition_label": "LoRA Cond",
                    "regime": "light_finetune",
                    "model_name": "microsoft/speecht5_tts",
                    "train_strategy": "lora",
                    "uses_speaker_embeddings": True,
                    "official_hypothesis_arm": True,
                    "speaker_id": "speaker_01",
                    "prompt_id": "P001",
                    "category": "general",
                    "text_variant": "normalized",
                    "target_text": "Texto normalizado.",
                    "training_scope": "per_speaker",
                }
            ]
        ).to_csv(run_matrix_path, index=False)
        pd.DataFrame([{"speaker_id": "speaker_01", "reference_audio": "reference.wav"}]).to_csv(
            speaker_selection, index=False
        )
        pd.DataFrame([{"speaker_id": "speaker_01", "speaker_embedding_path": "embedding.npy"}]).to_csv(
            embeddings, index=False
        )

        initialize_samples(
            run_matrix_path,
            samples_path,
            speaker_selection_path=speaker_selection,
            speaker_embeddings_path=embeddings,
        )
        sample_ids = materialize_checkpoint_samples(
            samples_path=samples_path,
            condition_id="lora_cond",
            checkpoint_step=500,
            checkpoint_path=tmp / "checkpoints/lora_cond/20260426T010203Z/speaker_01/checkpoint-500",
            checkpoint_run_ts="20260426T010203Z",
            training_scope="per_speaker",
            training_unit="speaker_01",
            audio_base_dir=tmp / "audio",
            total_train_gpu_hours=1.5,
            gpu_hourly_rate=2.0,
        )
        materialized = load_samples(samples_path)

    assert sample_ids == ["lora_cond__speaker_01__P001__normalized__20260426T010203Z__step500"]
    assert len(materialized) == 2
    checkpoint_rows = materialized[materialized["checkpoint_step"].astype(str).str.strip().ne("")]
    assert len(checkpoint_rows) == 1
    row = checkpoint_rows.iloc[0]
    assert row["checkpoint_label"] == "lora_cond@step500"
    assert row["training_scope"] == "per_speaker"
    assert row["training_unit"] == "speaker_01"
    assert row["audio_path"].endswith("step_500/P001__normalized.wav")
    assert float(row["train_gpu_hours"]) == 1.5
    assert float(row["cost_usd"]) == 3.0


def test_reset_condition_checkpoint_rows_keeps_base_rows_only() -> None:
    frame = pd.DataFrame(
        [
            {"condition": "lora_cond", "checkpoint_step": "", "sample_id": "base"},
            {"condition": "lora_cond", "checkpoint_step": "500", "sample_id": "materialized"},
            {"condition": "other_cond", "checkpoint_step": "500", "sample_id": "other"},
        ]
    )

    cleaned = reset_condition_checkpoint_rows(frame, "lora_cond")

    assert cleaned["sample_id"].tolist() == ["base", "other"]


def test_build_lora_arg_parser_accepts_repeated_condition_flags() -> None:
    parser = build_lora_arg_parser()
    args = parser.parse_args(
        [
            "--config",
            "config.yaml",
            "--samples",
            "samples.csv",
            "--manifest",
            "manifest.csv",
            "--checkpoint-dir",
            "artifacts/checkpoints",
            "--condition",
            "cond_a",
            "--condition",
            "cond_b",
        ]
    )

    assert args.condition == ["cond_a", "cond_b"]
    assert args.gpu_hourly_rate is None


def test_build_lora_arg_parser_accepts_gpu_hourly_rate_override() -> None:
    parser = build_lora_arg_parser()
    args = parser.parse_args(
        [
            "--config",
            "config.yaml",
            "--samples",
            "samples.csv",
            "--manifest",
            "manifest.csv",
            "--checkpoint-dir",
            "artifacts/checkpoints",
            "--gpu-hourly-rate",
            "1.75",
        ]
    )

    assert args.gpu_hourly_rate == 1.75


def test_build_lora_arg_parser_accepts_short_aliases() -> None:
    parser = build_lora_arg_parser()
    args = parser.parse_args(["-c", "config.yaml", "-s", "samples.csv", "-m", "manifest.csv", "-k", "ckpts", "-C", "cond_a"])

    assert args.config == "config.yaml"
    assert args.samples == "samples.csv"
    assert args.manifest == "manifest.csv"
    assert args.checkpoint_dir == "ckpts"
    assert args.condition == ["cond_a"]


def test_resolve_gpu_hourly_rate_uses_condition_value_without_cli_override() -> None:
    condition = {
        "id": "cond_a",
        "training": {
            "scope": "per_speaker",
            "gpu_hourly_rate": 2.5,
        },
    }

    assert _resolve_gpu_hourly_rate(condition) == 2.5


def test_resolve_gpu_hourly_rate_cli_override_wins() -> None:
    condition = {
        "id": "cond_a",
        "training": {
            "scope": "per_speaker",
            "gpu_hourly_rate": 2.5,
        },
    }

    assert _resolve_gpu_hourly_rate(condition, cli_override=4.0) == 4.0


def test_resolve_gpu_hourly_rate_defaults_to_zero_when_missing() -> None:
    condition = {
        "id": "cond_a",
        "training": {
            "scope": "per_speaker",
        },
    }

    assert _resolve_gpu_hourly_rate(condition) == 0.0


def test_resolve_gpu_hourly_rate_supports_distinct_values_per_condition() -> None:
    conservative = {
        "id": "cond_a",
        "training": {
            "scope": "per_speaker",
            "gpu_hourly_rate": 1.25,
        },
    }
    unique = {
        "id": "cond_b",
        "training": {
            "scope": "unique",
            "gpu_hourly_rate": 2.75,
        },
    }

    assert _resolve_gpu_hourly_rate(conservative) == 1.25
    assert _resolve_gpu_hourly_rate(unique) == 2.75


def test_log_training_dataset_summary_for_per_speaker(capsys) -> None:
    train_rows = pd.DataFrame(
        [
            {"speaker_id": "speaker_01", "duration_s": "120"},
            {"speaker_id": "speaker_01", "duration_s": "180"},
        ]
    )

    _log_training_dataset_summary(
        condition_id="cond_a",
        scope="per_speaker",
        training_unit="speaker_01",
        train_rows=train_rows,
        phase="train:start",
    )
    _log_training_dataset_summary(
        condition_id="cond_a",
        scope="per_speaker",
        training_unit="speaker_01",
        train_rows=train_rows,
        phase="train:end",
    )

    output = capsys.readouterr().out
    assert "scope=per_speaker" in output
    assert "speaker_id=speaker_01" in output
    assert "dataset_minutes=5.00" in output
    assert "[LoRA train:start]" in output
    assert "[LoRA train:end]" in output


def test_log_training_dataset_summary_for_unique(capsys) -> None:
    train_rows = pd.DataFrame(
        [
            {"speaker_id": "speaker_01", "duration_s": "120"},
            {"speaker_id": "speaker_01", "duration_s": "180"},
            {"speaker_id": "speaker_02", "duration_s": "60"},
        ]
    )

    _log_training_dataset_summary(
        condition_id="cond_b",
        scope="unique",
        training_unit="unique",
        train_rows=train_rows,
        phase="train:start",
    )
    _log_training_dataset_summary(
        condition_id="cond_b",
        scope="unique",
        training_unit="unique",
        train_rows=train_rows,
        phase="train:end",
    )

    output = capsys.readouterr().out
    assert "scope=unique" in output
    assert "speaker_count=2" in output
    assert "speaker_01=5.00m" in output
    assert "speaker_02=1.00m" in output
    assert "[LoRA train:start]" in output
    assert "[LoRA train:end]" in output


def test_tee_console_output_writes_to_console_and_log(capsys) -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        log_path = _training_log_path(tmp / "artifacts/checkpoints/lora", "cond_a", "20260427T010203Z")

        with _tee_console_output(log_path):
            print("stdout line")
            print("stderr line", file=sys.stderr)

        output = capsys.readouterr()
        log_text = log_path.read_text(encoding="utf-8")

    assert "stdout line" in output.out
    assert "stderr line" in output.err
    assert "stdout line" in log_text
    assert "stderr line" in log_text


def test_write_training_metadata_creates_expected_payload() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        metadata_path = _training_metadata_path(tmp / "artifacts/checkpoints/lora", "cond_a", "20260427T010203Z")
        _write_training_metadata(
            metadata_path=metadata_path,
            condition_id="cond_a",
            run_ts="20260427T010203Z",
            training_scope="per_speaker",
            started_at="2026-04-27T01:02:03+00:00",
            finished_at="2026-04-27T01:12:03+00:00",
            training_seconds=600.0,
            training_gpu_hours_total=0.5,
            training_units_total=2,
            checkpoints_total=3,
            dataset_train_rows_total=12,
            dataset_val_rows_total=4,
            dataset_inference_rows_total=24,
        )
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert payload == {
        "condition_id": "cond_a",
        "run_ts": "20260427T010203Z",
        "training_scope": "per_speaker",
        "started_at": "2026-04-27T01:02:03+00:00",
        "finished_at": "2026-04-27T01:12:03+00:00",
        "training_seconds": 600.0,
        "training_gpu_hours_total": 0.5,
        "training_units_total": 2,
        "checkpoints_total": 3,
        "dataset_train_rows_total": 12,
        "dataset_val_rows_total": 4,
        "dataset_inference_rows_total": 24,
    }


def test_build_cleanup_arg_parser_and_resolve_defaults() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = _write_cleanup_config(tmp)
        parser = build_cleanup_arg_parser()
        args = parser.parse_args(["--config", str(config), "--condition", "cond_a"])
        paths = resolve_cleanup_paths(config_path=args.config)

    assert args.condition == ["cond_a"]
    assert paths.samples_path == tmp / "artifacts/evaluation/samples.csv"
    assert paths.evaluation_dir == tmp / "artifacts/evaluation"
    assert paths.report_assets_dir == tmp / "report_assets"
    assert paths.run_matrix_path == tmp / "artifacts/run_matrix.csv"


def test_build_cleanup_arg_parser_accepts_aliases_and_all() -> None:
    parser = build_cleanup_arg_parser()
    args = parser.parse_args(["-c", "config.yaml", "-C", "all", "-y"])

    assert args.config == "config.yaml"
    assert args.condition == ["all"]
    assert args.bypass is True


def test_build_remove_condition_arg_parser_accepts_aliases() -> None:
    parser = build_remove_condition_arg_parser()
    args = parser.parse_args(["-c", "config.yaml", "-C", "cond_a", "-y"])

    assert args.config == "config.yaml"
    assert args.condition == ["cond_a"]
    assert args.bypass is True


def test_cleanup_training_results_requires_yes_confirmation() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = _write_cleanup_config(tmp)
        with patch("builtins.input", return_value="no"):
            try:
                cleanup_training_results(config_path=config, condition_ids=["cond_a"])
            except SystemExit as exc:
                message = str(exc)
            else:
                raise AssertionError("expected cleanup confirmation to abort")

    assert message == "Aborted by user."


def test_cleanup_training_results_bypass_skips_confirmation() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = _write_cleanup_config(tmp)
        with patch("builtins.input", side_effect=AssertionError("input should not be called")):
            result = cleanup_training_results(config_path=config, condition_ids=["cond_a"], bypass=True)

    assert result.condition_ids == ["cond_a"]


def test_cleanup_training_results_removes_condition_artifacts() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = _write_cleanup_config(tmp)
        samples_path = tmp / "artifacts/evaluation/samples.csv"
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"condition": "cond_a", "checkpoint_step": "", "sample_id": "base_a"},
                {"condition": "cond_a", "checkpoint_step": "500", "sample_id": "mat_a"},
                {"condition": "cond_b", "checkpoint_step": "750", "sample_id": "mat_b"},
            ]
        ).to_csv(samples_path, index=False)
        checkpoint_a = tmp / "artifacts/checkpoints/lora/cond_a"
        checkpoint_a.mkdir(parents=True, exist_ok=True)
        (checkpoint_a / "marker.txt").write_text("x", encoding="utf-8")
        run_root_a = _build_condition_run_root(tmp / "artifacts/checkpoints/lora", "cond_a", "20260427T010203Z")
        run_root_a.mkdir(parents=True, exist_ok=True)
        (run_root_a / "training.log").write_text("log", encoding="utf-8")
        (run_root_a / "metadata.json").write_text("{}", encoding="utf-8")
        checkpoint_b = tmp / "artifacts/checkpoints/lora/cond_b"
        checkpoint_b.mkdir(parents=True, exist_ok=True)
        (checkpoint_b / "marker.txt").write_text("x", encoding="utf-8")
        audio_a = tmp / "artifacts/audio/cond_a"
        audio_a.mkdir(parents=True, exist_ok=True)
        (audio_a / "sample.wav").write_text("x", encoding="utf-8")
        for filename in ["metrics_summary.csv", "cost_summary.csv", "metrics_by_speaker.csv", "cost_by_speaker.csv"]:
            target = tmp / "artifacts/evaluation" / filename
            target.write_text("x", encoding="utf-8")
        report_assets = tmp / "report_assets"
        report_assets.mkdir(parents=True, exist_ok=True)
        (report_assets / "overview.md").write_text("x", encoding="utf-8")

        result = cleanup_training_results(
            config_path=config,
            condition_ids=["cond_a"],
            checkpoint_dir=tmp / "artifacts/checkpoints/lora",
            audio_base_dir=tmp / "artifacts/audio",
            bypass=True,
        )
        remaining = load_samples(samples_path)
        checkpoint_a_exists = checkpoint_a.exists()
        checkpoint_b_exists = checkpoint_b.exists()
        training_log_exists = (run_root_a / "training.log").exists()
        training_metadata_exists = (run_root_a / "metadata.json").exists()
        audio_a_exists = audio_a.exists()
        report_assets_exists = report_assets.exists()
        evaluation_files_exist = [
            (tmp / "artifacts/evaluation" / filename).exists()
            for filename in ["metrics_summary.csv", "cost_summary.csv", "metrics_by_speaker.csv", "cost_by_speaker.csv"]
        ]

        assert result.condition_ids == ["cond_a"]
        assert result.removed_materialized_rows == 1
        assert checkpoint_a_exists is False
        assert checkpoint_b_exists is True
        assert training_log_exists is False
        assert training_metadata_exists is False
        assert audio_a_exists is False
        assert report_assets_exists is False
        assert remaining["sample_id"].tolist() == ["base_a", "mat_b"]
        assert evaluation_files_exist == [False, False, False, False]


def test_cleanup_training_results_accepts_all() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = _write_cleanup_config(tmp)
        samples_path = tmp / "artifacts/evaluation/samples.csv"
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"condition": "cond_a", "checkpoint_step": "", "sample_id": "base_a"},
                {"condition": "cond_a", "checkpoint_step": "500", "sample_id": "mat_a"},
                {"condition": "cond_b", "checkpoint_step": "750", "sample_id": "mat_b"},
            ]
        ).to_csv(samples_path, index=False)
        for condition_id in ["cond_a", "cond_b"]:
            checkpoint_root = tmp / "artifacts/checkpoints/lora" / condition_id
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            audio_root = tmp / "artifacts/audio" / condition_id
            audio_root.mkdir(parents=True, exist_ok=True)

        result = cleanup_training_results(
            config_path=config,
            condition_ids=["all"],
            checkpoint_dir=tmp / "artifacts/checkpoints/lora",
            audio_base_dir=tmp / "artifacts/audio",
            bypass=True,
        )
        remaining = load_samples(samples_path)
        checkpoint_a_exists = (tmp / "artifacts/checkpoints/lora/cond_a").exists()
        checkpoint_b_exists = (tmp / "artifacts/checkpoints/lora/cond_b").exists()
        audio_a_exists = (tmp / "artifacts/audio/cond_a").exists()
        audio_b_exists = (tmp / "artifacts/audio/cond_b").exists()

        assert result.condition_ids == ["cond_a", "cond_b"]
        assert result.removed_materialized_rows == 2
        assert remaining["sample_id"].tolist() == ["base_a"]
        assert checkpoint_a_exists is False
        assert checkpoint_b_exists is False
        assert audio_a_exists is False
        assert audio_b_exists is False


def test_cleanup_training_results_rejects_all_with_specific_conditions() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = _write_cleanup_config(tmp)
        try:
            cleanup_training_results(config_path=config, condition_ids=["all", "cond_a"], bypass=True)
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected invalid all combination to fail")

    assert "cannot be combined" in message


def test_remove_condition_rejects_all() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = _write_cleanup_config(tmp)
        try:
            remove_condition(config_path=config, condition_id="all", bypass=True)
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected remove_condition to reject all")

    assert "specific LoRA condition id" in message


def test_remove_condition_removes_condition_from_samples_and_config() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = _write_cleanup_config(tmp)
        samples_path = tmp / "artifacts/evaluation/samples.csv"
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"condition": "cond_a", "checkpoint_step": "", "sample_id": "base_a"},
                {"condition": "cond_a", "checkpoint_step": "500", "sample_id": "mat_a"},
                {"condition": "cond_b", "checkpoint_step": "", "sample_id": "base_b"},
            ]
        ).to_csv(samples_path, index=False)
        run_matrix = tmp / "artifacts/run_matrix.csv"
        run_matrix.parent.mkdir(parents=True, exist_ok=True)
        run_matrix.write_text("run_id\nx\n", encoding="utf-8")
        checkpoint_a = tmp / "artifacts/checkpoints/lora/cond_a"
        checkpoint_a.mkdir(parents=True, exist_ok=True)
        audio_a = tmp / "artifacts/audio/cond_a"
        audio_a.mkdir(parents=True, exist_ok=True)

        result = remove_condition(
            config_path=config,
            condition_id="cond_a",
            checkpoint_dir=tmp / "artifacts/checkpoints/lora",
            audio_base_dir=tmp / "artifacts/audio",
            bypass=True,
        )
        remaining = load_samples(samples_path)
        config_text = config.read_text(encoding="utf-8")
        run_matrix_exists = run_matrix.exists()
        checkpoint_a_exists = checkpoint_a.exists()
        audio_a_exists = audio_a.exists()

        assert result.cleanup.removed_materialized_rows == 1
        assert result.removed_base_rows == 1
        assert result.updated_config is True
        assert result.removed_run_matrix is True
        assert remaining["sample_id"].tolist() == ["base_b"]
        assert "id: cond_a" not in config_text
        assert "id: cond_b" in config_text
        assert run_matrix_exists is False
        assert checkpoint_a_exists is False
        assert audio_a_exists is False


def test_build_dataset_inventory_arg_parser_and_resolve_defaults() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = _write_inventory_config(tmp)
        parser = build_dataset_inventory_arg_parser()
        args = parser.parse_args(["-c", str(config), "-j", str(tmp / "inventory.json"), "--sample-size", "10"])
        paths = resolve_inventory_paths(config_path=args.config, json_out=args.json_out)

    assert args.config == str(config)
    assert args.json_out == str(tmp / "inventory.json")
    assert args.sample_size == 10
    assert args.override is False
    assert paths.raw_dir == Path("data/raw/common_voice_pt")
    assert paths.processed_dir == Path("data/processed/common_voice_pt")
    assert paths.processed_metadata_path == Path("data/manifests/common_voice_processed.csv")
    assert paths.manifest_path == tmp / "data/manifests/data_manifest.csv"
    assert paths.speaker_selection_path == tmp / "data/manifests/speaker_selection.csv"


def test_build_dataset_inventory_arg_parser_accepts_override() -> None:
    parser = build_dataset_inventory_arg_parser()
    args = parser.parse_args(["--override"])

    assert args.override is True


def test_build_dataset_inventory_uses_cached_json_when_available(monkeypatch) -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        fixture = _prepare_dataset_inventory_fixture(tmp)
        cached_inventory = {
            "raw": {"overall_totals": {"total_clips": 99}},
            "processed": {"speaker_count": 12},
            "selected_speakers": {"unique": {"speaker_count": 3}},
        }
        fixture["json_out"].parent.mkdir(parents=True, exist_ok=True)
        fixture["json_out"].write_text(json.dumps(cached_inventory) + "\n", encoding="utf-8")
        original_text = fixture["json_out"].read_text(encoding="utf-8")

        monkeypatch.setattr(
            "tcc_audio.dataset_inventory._build_raw_inventory",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("raw inventory should not rebuild when cache is valid")),
        )

        inventory = build_dataset_inventory(
            config_path=fixture["config"],
            raw_dir=fixture["raw_dir"],
            processed_dir=fixture["processed_dir"],
            processed_metadata_path=fixture["processed_metadata"],
            manifest_path=fixture["manifest_path"],
            speaker_selection_path=fixture["speaker_selection"],
            json_out=fixture["json_out"],
            sample_size=50,
        )
        cached_text_after = fixture["json_out"].read_text(encoding="utf-8")

    assert inventory == cached_inventory
    assert cached_text_after == original_text


def test_dataset_inventory_main_reports_cache_hit(monkeypatch, capsys) -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        fixture = _prepare_dataset_inventory_fixture(tmp)
        cached_inventory = {
            "raw": {
                "overall_totals": {
                    "total_clips": 1,
                    "validated_clips": 1,
                    "non_validated_clips": 0,
                    "unique_speakers": 1,
                    "formats": ["mp3"],
                    "sample_rates_khz": ["48.0"],
                    "channel_modes": ["stereo"],
                },
                "grouped_by_locale_variant_gender": [],
            },
            "processed": {
                "speaker_count": 1,
                "clip_count": 1,
                "total_duration_s": 1.0,
                "lowest_clip_speaker": {"speaker_id": "spk", "clip_count": 1, "total_duration_s": 1.0},
                "highest_clip_speaker": {"speaker_id": "spk", "clip_count": 1, "total_duration_s": 1.0},
                "formats": ["wav"],
                "sample_rates_khz": ["16.0"],
                "channel_modes": ["mono"],
            },
            "selected_speakers": {"warnings": []},
            "warnings": [],
        }
        fixture["json_out"].parent.mkdir(parents=True, exist_ok=True)
        fixture["json_out"].write_text(json.dumps(cached_inventory) + "\n", encoding="utf-8")

        monkeypatch.setattr(
            "tcc_audio.dataset_inventory._build_raw_inventory",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("main should load cached inventory")),
        )

        exit_code = dataset_inventory_main(
            [
                "-c",
                str(fixture["config"]),
                "--raw-dir",
                str(fixture["raw_dir"]),
                "--processed-dir",
                str(fixture["processed_dir"]),
                "--processed-metadata",
                str(fixture["processed_metadata"]),
                "--manifest",
                str(fixture["manifest_path"]),
                "--speaker-selection",
                str(fixture["speaker_selection"]),
                "-j",
                str(fixture["json_out"]),
            ]
        )
        output = capsys.readouterr().out

    assert exit_code == 0
    assert "RAW DATASET" in output
    assert f"Loaded JSON inventory from {fixture['json_out']}" in output
    assert f"Wrote JSON inventory to {fixture['json_out']}" not in output


def test_build_dataset_inventory_override_rebuilds_and_overwrites_cache(monkeypatch) -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        fixture = _prepare_dataset_inventory_fixture(tmp)
        fixture["json_out"].parent.mkdir(parents=True, exist_ok=True)
        fixture["json_out"].write_text('{"raw": {"stale": true}, "processed": {}, "selected_speakers": {}}\n', encoding="utf-8")

        def fake_probe(path: Path) -> dict[str, str]:
            if path.suffix.lower() == ".mp3":
                return {"format": "mp3", "sample_rate_khz": "48.0", "channel_mode": "stereo"}
            return {"format": "wav", "sample_rate_khz": "16.0", "channel_mode": "mono"}

        monkeypatch.setattr("tcc_audio.dataset_inventory._probe_audio_file", fake_probe)

        inventory = build_dataset_inventory(
            config_path=fixture["config"],
            raw_dir=fixture["raw_dir"],
            processed_dir=fixture["processed_dir"],
            processed_metadata_path=fixture["processed_metadata"],
            manifest_path=fixture["manifest_path"],
            speaker_selection_path=fixture["speaker_selection"],
            json_out=fixture["json_out"],
            sample_size=50,
            override=True,
        )
        saved_json = json.loads(fixture["json_out"].read_text(encoding="utf-8"))

    assert inventory["raw"]["overall_totals"]["total_clips"] == 3
    assert saved_json["raw"]["overall_totals"]["total_clips"] == 3
    assert "stale" not in saved_json["raw"]


def test_build_dataset_inventory_rebuilds_when_cached_json_is_invalid(monkeypatch) -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        fixture = _prepare_dataset_inventory_fixture(tmp)
        fixture["json_out"].parent.mkdir(parents=True, exist_ok=True)
        fixture["json_out"].write_text("{invalid json\n", encoding="utf-8")

        def fake_probe(path: Path) -> dict[str, str]:
            if path.suffix.lower() == ".mp3":
                return {"format": "mp3", "sample_rate_khz": "48.0", "channel_mode": "stereo"}
            return {"format": "wav", "sample_rate_khz": "16.0", "channel_mode": "mono"}

        monkeypatch.setattr("tcc_audio.dataset_inventory._probe_audio_file", fake_probe)

        inventory = build_dataset_inventory(
            config_path=fixture["config"],
            raw_dir=fixture["raw_dir"],
            processed_dir=fixture["processed_dir"],
            processed_metadata_path=fixture["processed_metadata"],
            manifest_path=fixture["manifest_path"],
            speaker_selection_path=fixture["speaker_selection"],
            json_out=fixture["json_out"],
            sample_size=50,
        )
        saved_json = json.loads(fixture["json_out"].read_text(encoding="utf-8"))

    assert inventory["processed"]["speaker_count"] == 2
    assert saved_json["processed"]["speaker_count"] == 2


def test_build_dataset_inventory_reports_raw_processed_and_selected_speakers(monkeypatch) -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        fixture = _prepare_dataset_inventory_fixture(tmp)

        def fake_probe(path: Path) -> dict[str, str]:
            if path.suffix.lower() == ".mp3":
                return {"format": "mp3", "sample_rate_khz": "48.0", "channel_mode": "stereo"}
            return {"format": "wav", "sample_rate_khz": "16.0", "channel_mode": "mono"}

        monkeypatch.setattr("tcc_audio.dataset_inventory._probe_audio_file", fake_probe)

        inventory = build_dataset_inventory(
            config_path=fixture["config"],
            raw_dir=fixture["raw_dir"],
            processed_dir=fixture["processed_dir"],
            processed_metadata_path=fixture["processed_metadata"],
            manifest_path=fixture["manifest_path"],
            speaker_selection_path=fixture["speaker_selection"],
            json_out=fixture["json_out"],
            sample_size=50,
            override=True,
        )
        report = render_dataset_inventory(inventory)
        saved_json = json.loads(fixture["json_out"].read_text(encoding="utf-8"))

    assert set(inventory.keys()) >= {"raw", "processed", "selected_speakers", "sampling", "paths"}
    assert inventory["raw"]["overall_totals"]["total_clips"] == 3
    assert inventory["raw"]["overall_totals"]["validated_clips"] == 2
    assert inventory["raw"]["overall_totals"]["non_validated_clips"] == 1
    assert inventory["raw"]["overall_totals"]["unique_speakers"] == 2
    assert inventory["raw"]["overall_totals"]["formats"] == ["mp3"]
    assert inventory["raw"]["grouped_by_locale_variant_gender"] == [
        {"locale": "pt", "variant": "pt-BR", "gender": "male", "clip_count": 1, "unique_speakers": 1},
        {"locale": "pt", "variant": "pt-PT", "gender": "female", "clip_count": 1, "unique_speakers": 1},
    ]
    assert inventory["processed"]["speaker_count"] == 2
    assert inventory["processed"]["lowest_clip_speaker"] == {
        "speaker_id": "source_2",
        "clip_count": 1,
        "total_duration_s": 5.0,
    }
    assert inventory["processed"]["highest_clip_speaker"] == {
        "speaker_id": "source_1",
        "clip_count": 2,
        "total_duration_s": 4.0,
    }
    assert inventory["selected_speakers"]["per_speaker"]["speakers"][0]["speaker_id"] == "speaker_01"
    assert inventory["selected_speakers"]["per_speaker"]["speakers"][0]["train_clip_count"] == 1
    assert inventory["selected_speakers"]["per_speaker"]["speakers"][0]["val_clip_count"] == 1
    assert inventory["selected_speakers"]["per_speaker"]["totals"]["train_clip_count"] == 2
    assert inventory["selected_speakers"]["per_speaker"]["totals"]["val_clip_count"] == 1
    assert inventory["selected_speakers"]["unique"]["speaker_count"] == 2
    assert inventory["selected_speakers"]["unique"]["train_clip_count"] == 2
    assert inventory["selected_speakers"]["unique"]["val_clip_count"] == 1
    assert inventory["sampling"] == {
        "sample_size_requested": 50,
        "sample_size_effective_raw": 3,
        "sample_size_effective_processed": 3,
    }
    assert saved_json["raw"]["overall_totals"]["non_validated_clips"] == 1
    assert "RAW DATASET" in report
    assert "PROCESSED DATASET" in report
    assert "SELECTED SPEAKERS" in report
    assert "WARNINGS" not in report


def test_build_dataset_inventory_reports_single_clip_speaker_as_train_only(monkeypatch) -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = _write_inventory_config(tmp)
        raw_dir = tmp / "data/raw/common_voice_pt"
        clips_dir = raw_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        (clips_dir / "clip_a.mp3").write_bytes(b"fake")
        (raw_dir / "validated.tsv").write_text(
            "client_id\tpath\ttext\tgender\tlocale\tvariant\n"
            "spk1\tclip_a.mp3\tTexto A\tmale\tpt\tpt-BR\n",
            encoding="utf-8",
        )

        processed_metadata = tmp / "data/manifests/common_voice_processed.csv"
        processed_metadata.parent.mkdir(parents=True, exist_ok=True)
        wav_a = tmp / "data/processed/common_voice_pt/source_1/a.wav"
        _write_wav(wav_a)
        pd.DataFrame([{"source_speaker_id": "source_1", "duration_s": 1.5, "audio_path": str(wav_a)}]).to_csv(
            processed_metadata, index=False
        )

        manifest_path = tmp / "data/manifests/data_manifest.csv"
        pd.DataFrame([{"speaker_id": "speaker_01", "split": "train", "duration_s": 1.5, "audio_path": str(wav_a)}]).to_csv(
            manifest_path, index=False
        )
        speaker_selection = tmp / "data/manifests/speaker_selection.csv"
        pd.DataFrame([{"speaker_id": "speaker_01"}]).to_csv(speaker_selection, index=False)
        json_out = tmp / "artifacts/dataset_inventory.json"

        monkeypatch.setattr(
            "tcc_audio.dataset_inventory._probe_audio_file",
            lambda path: {"format": "wav", "sample_rate_khz": "16.0", "channel_mode": "mono"},
        )

        inventory = build_dataset_inventory(
            config_path=config,
            raw_dir=raw_dir,
            processed_dir=tmp / "data/processed/common_voice_pt",
            processed_metadata_path=processed_metadata,
            manifest_path=manifest_path,
            speaker_selection_path=speaker_selection,
            json_out=json_out,
            sample_size=50,
        )

    speaker = inventory["selected_speakers"]["per_speaker"]["speakers"][0]
    assert speaker["speaker_id"] == "speaker_01"
    assert speaker["train_clip_count"] == 1
    assert speaker["val_clip_count"] == 0
    assert speaker["train_duration_s"] == 1.5
    assert speaker["val_duration_s"] == 0.0


def test_build_dataset_inventory_degrades_when_ffprobe_is_unavailable(monkeypatch) -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = _write_inventory_config(tmp)
        raw_dir = tmp / "data/raw/common_voice_pt"
        clips_dir = raw_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        (clips_dir / "clip_a.mp3").write_bytes(b"fake")
        (raw_dir / "validated.tsv").write_text(
            "client_id\tpath\ttext\tgender\tlocale\tvariant\n"
            "spk1\tclip_a.mp3\tTexto A\tmale\tpt\tpt-BR\n",
            encoding="utf-8",
        )

        processed_metadata = tmp / "data/manifests/common_voice_processed.csv"
        processed_metadata.parent.mkdir(parents=True, exist_ok=True)
        wav_a = tmp / "data/processed/common_voice_pt/source_1/a.wav"
        _write_wav(wav_a)
        pd.DataFrame([{"source_speaker_id": "source_1", "duration_s": 1.5, "audio_path": str(wav_a)}]).to_csv(
            processed_metadata, index=False
        )
        manifest_path = tmp / "data/manifests/data_manifest.csv"
        pd.DataFrame([{"speaker_id": "speaker_01", "split": "train", "duration_s": 1.5, "audio_path": str(wav_a)}]).to_csv(
            manifest_path, index=False
        )
        speaker_selection = tmp / "data/manifests/speaker_selection.csv"
        pd.DataFrame([{"speaker_id": "speaker_01"}]).to_csv(speaker_selection, index=False)

        monkeypatch.setattr(
            "tcc_audio.dataset_inventory._resolve_ffprobe_command",
            lambda: (_ for _ in ()).throw(FileNotFoundError("ffprobe not found")),
        )

        inventory = build_dataset_inventory(
            config_path=config,
            raw_dir=raw_dir,
            processed_metadata_path=processed_metadata,
            manifest_path=manifest_path,
            speaker_selection_path=speaker_selection,
            json_out=tmp / "artifacts/dataset_inventory.json",
            sample_size=50,
        )
        report = render_dataset_inventory(inventory)

    assert inventory["raw"]["overall_totals"]["formats"] == ["mp3"]
    assert inventory["raw"]["overall_totals"]["sample_rates_khz"] == ["unknown"]
    assert inventory["processed"]["channel_modes"] == ["unknown"]
    assert inventory["warnings"]
    assert "ffprobe not available" in "\n".join(inventory["warnings"])
    assert "WARNINGS" in report


def test_speaker_selection_from_processed_metadata() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        rows = []
        for speaker_index in range(6):
            source_speaker_id = f"speaker_{speaker_index}"
            gender = "unknown" if speaker_index < 4 else "feminine"
            utterance_duration = 150 - (speaker_index * 10)
            for utterance_index in range(12):
                rows.append(
                    {
                        "source_speaker_id": source_speaker_id,
                        "gender": gender,
                        "utterance_id": f"{source_speaker_id}_{utterance_index}",
                        "duration_s": utterance_duration,
                        "audio_path": f"audio/{source_speaker_id}_{utterance_index}.wav",
                        "target_text": 'Ação de teste com “aspas” para seleção de speaker.',
                        "license": "CC0",
                        "source": "common_voice_pt",
                        "locale": "pt",
                    }
                )
        metadata = tmp / "metadata.csv"
        pd.DataFrame(rows).to_csv(metadata, index=False)
        manifest, speaker_selection = select_speakers(
            metadata,
            tmp / "data_manifest.csv",
            tmp / "speaker_selection.csv",
            speaker_target_count=4,
            minutes_per_speaker=20,
        )
        assert speaker_selection["speaker_id"].nunique() == 4
        assert manifest["speaker_id"].nunique() == 4
        assert speaker_selection["source_speaker_id"].tolist() == [
            "speaker_0",
            "speaker_1",
            "speaker_2",
            "speaker_3",
        ]
        assert "audio_path" in manifest.columns
        assert "target_text_speecht5" in manifest.columns
        assert manifest.loc[0, "target_text_speecht5"] == 'Acao de teste com "aspas" para selecao de speaker.'
        assert set(speaker_selection.columns) == {
            "speaker_id",
            "source_speaker_id",
            "gender",
            "total_duration_s",
            "selected_train_duration_s",
            "selected_val_duration_s",
            "reference_audio",
            "reference_duration_s",
            "license",
            "source",
            "notes",
        }


def test_speaker_selection_includes_shorter_speakers_when_ranked_globally() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        rows = []
        speaker_specs = [
            ("speaker_a", "unknown", [400, 350, 300, 250]),
            ("speaker_b", "unknown", [390, 340, 290, 240]),
            ("speaker_c", "masculine", [380, 330]),
            ("speaker_d", "feminine", [370, 320]),
            ("speaker_e", "feminine", [100, 90]),
        ]
        for source_speaker_id, gender, durations in speaker_specs:
            for utterance_index, duration_s in enumerate(durations):
                rows.append(
                    {
                        "source_speaker_id": source_speaker_id,
                        "gender": gender,
                        "utterance_id": f"{source_speaker_id}_{utterance_index}",
                        "duration_s": duration_s,
                        "audio_path": f"audio/{source_speaker_id}_{utterance_index}.wav",
                        "target_text": "Texto de teste.",
                        "license": "CC0",
                        "source": "common_voice_pt",
                    }
                )
        metadata = tmp / "metadata.csv"
        pd.DataFrame(rows).to_csv(metadata, index=False)

        manifest, speaker_selection = select_speakers(
            metadata,
            tmp / "data_manifest.csv",
            tmp / "speaker_selection.csv",
            speaker_target_count=4,
            minutes_per_speaker=20,
        )

        assert speaker_selection["source_speaker_id"].tolist() == [
            "speaker_a",
            "speaker_b",
            "speaker_c",
            "speaker_d",
        ]
        shorter = speaker_selection[speaker_selection["source_speaker_id"].eq("speaker_c")].iloc[0]
        assert float(shorter["total_duration_s"]) < 20 * 60
        assert float(shorter["selected_train_duration_s"]) + float(shorter["selected_val_duration_s"]) == 710.0
        assert manifest["speaker_id"].nunique() == 4


def test_speaker_selection_assigns_single_selected_clip_to_train() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        rows = [
            {
                "source_speaker_id": "speaker_a",
                "gender": "unknown",
                "utterance_id": "speaker_a_0",
                "duration_s": 600,
                "audio_path": "audio/speaker_a_0.wav",
                "target_text": "Texto de teste.",
                "license": "CC0",
                "source": "common_voice_pt",
            },
            {
                "source_speaker_id": "speaker_b",
                "gender": "feminine",
                "utterance_id": "speaker_b_0",
                "duration_s": 590,
                "audio_path": "audio/speaker_b_0.wav",
                "target_text": "Outro texto.",
                "license": "CC0",
                "source": "common_voice_pt",
            },
        ]
        metadata = tmp / "metadata.csv"
        pd.DataFrame(rows).to_csv(metadata, index=False)

        manifest, speaker_selection = select_speakers(
            metadata,
            tmp / "data_manifest.csv",
            tmp / "speaker_selection.csv",
            speaker_target_count=2,
            minutes_per_speaker=20,
        )

    assert manifest["speaker_id"].nunique() == 2
    assert set(manifest["split"]) == {"train"}
    assert not manifest.groupby("speaker_id")["split"].apply(lambda values: (values == "train").any()).eq(False).any()
    assert (speaker_selection["selected_train_duration_s"].astype(float) > 0).all()
    assert (speaker_selection["selected_val_duration_s"].astype(float) == 0.0).all()


def test_speaker_selection_handles_unknown_majority_without_gender_buckets() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        rows = []
        speaker_specs = [
            ("speaker_unknown_1", "unknown", 180),
            ("speaker_unknown_2", "unknown", 170),
            ("speaker_unknown_3", "unknown", 160),
            ("speaker_unknown_4", "unknown", 150),
            ("speaker_unknown_5", "unknown", 140),
            ("speaker_f", "feminine", 100),
        ]
        for source_speaker_id, gender, duration_s in speaker_specs:
            for utterance_index in range(8):
                rows.append(
                    {
                        "source_speaker_id": source_speaker_id,
                        "gender": gender,
                        "utterance_id": f"{source_speaker_id}_{utterance_index}",
                        "duration_s": duration_s,
                        "audio_path": f"audio/{source_speaker_id}_{utterance_index}.wav",
                        "target_text": "Texto de teste.",
                        "license": "CC0",
                        "source": "common_voice_pt",
                    }
                )
        metadata = tmp / "metadata.csv"
        pd.DataFrame(rows).to_csv(metadata, index=False)

        _, speaker_selection = select_speakers(
            metadata,
            tmp / "data_manifest.csv",
            tmp / "speaker_selection.csv",
            speaker_target_count=4,
            minutes_per_speaker=20,
        )

        assert speaker_selection["speaker_id"].nunique() == 4
        assert speaker_selection["source_speaker_id"].tolist() == [
            "speaker_unknown_1",
            "speaker_unknown_2",
            "speaker_unknown_3",
            "speaker_unknown_4",
        ]


def test_speaker_embedding_config_resolution() -> None:
    modern = {
        "project": {
            "tts_speaker_embedding_model": "custom/xvector",
            "tts_speaker_embedding_dim": 256,
        },
        "evaluation": {
            "speaker_similarity_model": "custom/ecapa",
        },
    }
    legacy = {
        "project": {
            "speaker_embedding_model": "legacy/model",
        }
    }

    assert resolve_tts_speaker_embedding_model(modern) == "custom/xvector"
    assert resolve_tts_speaker_embedding_dim(modern) == 256
    assert resolve_speaker_similarity_model(modern) == "custom/ecapa"

    assert resolve_tts_speaker_embedding_model(legacy) == "legacy/model"
    assert resolve_tts_speaker_embedding_dim(legacy) == DEFAULT_TTS_SPEAKER_EMBEDDING_DIM
    assert resolve_speaker_similarity_model(legacy) == DEFAULT_SPEAKER_SIMILARITY_MODEL


def test_extract_speaker_embeddings_uses_configured_tts_model() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config_path = tmp / "config.yaml"
        selection_path = tmp / "speaker_selection.csv"
        out_index = tmp / "speaker_embeddings.csv"
        out_dir = tmp / "embeddings"
        reference_audio = tmp / "reference.wav"
        reference_audio.write_bytes(b"fake")
        config_path.write_text(
            "project:\n"
            "  tts_speaker_embedding_model: speechbrain/spkrec-xvect-voxceleb\n"
            "  tts_speaker_embedding_dim: 512\n",
            encoding="utf-8",
        )
        pd.DataFrame(
            [{"speaker_id": "speaker_01", "reference_audio": str(reference_audio)}]
        ).to_csv(selection_path, index=False)

        calls: dict[str, object] = {}

        class FakeEncoderClassifier:
            @classmethod
            def from_hparams(cls, source: str, run_opts: dict[str, str] | None = None):
                calls["source"] = source
                calls["run_opts"] = run_opts
                return object()

        with patch("tcc_audio.speaker_embeddings.load_encoder_classifier", return_value=FakeEncoderClassifier):
            with patch(
                "tcc_audio.speaker_embeddings.encode_audio_path",
                return_value=np.zeros(512, dtype=np.float32),
            ):
                frame = extract_speaker_embeddings(
                    speaker_selection_path=selection_path,
                    out_index=out_index,
                    out_dir=out_dir,
                    config_path=config_path,
                )

        assert calls["source"] == "speechbrain/spkrec-xvect-voxceleb"
        assert int(frame.iloc[0]["embedding_dim"]) == 512
        assert Path(frame.iloc[0]["speaker_embedding_path"]).exists()


def test_extract_speaker_embeddings_override_wins_over_config() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config_path = tmp / "config.yaml"
        selection_path = tmp / "speaker_selection.csv"
        out_index = tmp / "speaker_embeddings.csv"
        out_dir = tmp / "embeddings"
        reference_audio = tmp / "reference.wav"
        reference_audio.write_bytes(b"fake")
        config_path.write_text(
            "project:\n"
            "  tts_speaker_embedding_model: speechbrain/spkrec-xvect-voxceleb\n"
            "  tts_speaker_embedding_dim: 512\n",
            encoding="utf-8",
        )
        pd.DataFrame(
            [{"speaker_id": "speaker_01", "reference_audio": str(reference_audio)}]
        ).to_csv(selection_path, index=False)

        calls: dict[str, object] = {}

        class FakeEncoderClassifier:
            @classmethod
            def from_hparams(cls, source: str, run_opts: dict[str, str] | None = None):
                calls["source"] = source
                calls["run_opts"] = run_opts
                return object()

        with patch("tcc_audio.speaker_embeddings.load_encoder_classifier", return_value=FakeEncoderClassifier):
            with patch(
                "tcc_audio.speaker_embeddings.encode_audio_path",
                return_value=np.zeros(192, dtype=np.float32),
            ):
                frame = extract_speaker_embeddings(
                    speaker_selection_path=selection_path,
                    out_index=out_index,
                    out_dir=out_dir,
                    config_path=config_path,
                    model_name="override/model",
                    expected_dim=192,
                )

        assert calls["source"] == "override/model"
        assert calls["run_opts"] == {"device": "cpu"}
        assert calls["run_opts"] == {"device": "cpu"}
        assert int(frame.iloc[0]["embedding_dim"]) == 192


def test_compute_speaker_similarity_uses_configured_eval_model() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config_path = tmp / "config.yaml"
        samples_path = tmp / "samples.csv"
        generated_audio = tmp / "generated.wav"
        reference_audio = tmp / "reference.wav"
        generated_audio.write_bytes(b"fake")
        reference_audio.write_bytes(b"fake")
        config_path.write_text(
            "evaluation:\n"
            "  speaker_similarity_model: speechbrain/spkrec-ecapa-voxceleb\n",
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {
                    "sample_id": "s1",
                    "run_id": "r1",
                    "condition": "speecht5_lora_conservative",
                    "speaker_id": "speaker_01",
                    "prompt_id": "P001",
                    "text_variant": "normalized",
                    "target_text": "Texto de teste.",
                    "audio_path": str(generated_audio),
                    "reference_audio_path": str(reference_audio),
                    "speaker_embedding_path": "tts_embedding.npy",
                    "model_name": "microsoft/speecht5_tts",
                    "run_started_at": "",
                    "run_finished_at": "",
                    "failure_reason": "",
                    "wer": "",
                    "speaker_similarity": "",
                    "nisqa": "",
                    "f0_rmse": "",
                    "rtf": "",
                    "train_gpu_hours": "",
                    "inference_seconds": "",
                    "cost_usd": "",
                    "status": "generated",
                    "lora_gate_status": "",
                }
            ]
        ).to_csv(samples_path, index=False)

        calls: dict[str, object] = {}

        class FakeEncoderClassifier:
            @classmethod
            def from_hparams(cls, source: str, run_opts: dict[str, str] | None = None):
                calls["source"] = source
                calls["run_opts"] = run_opts
                return object()

        with patch("tcc_audio.audio_metrics.load_encoder_classifier", return_value=FakeEncoderClassifier):
            with patch(
                "tcc_audio.audio_metrics.encode_audio_path",
                side_effect=[np.array([1.0, 0.0], dtype=np.float32), np.array([1.0, 0.0], dtype=np.float32)],
            ):
                updated = compute_speaker_similarity(samples_path, samples_path, config_path=config_path)

        assert calls["source"] == "speechbrain/spkrec-ecapa-voxceleb"
        assert calls["run_opts"] == {"device": "cpu"}
        assert float(updated.loc[updated["sample_id"].eq("s1"), "speaker_similarity"].iloc[0]) == 1.0


def test_speecht5_runtime_loader_does_not_require_training_stack(monkeypatch) -> None:
    fake_librosa = types.ModuleType("librosa")
    fake_soundfile = types.ModuleType("soundfile")
    fake_torch = types.ModuleType("torch")
    fake_torch_utils = types.ModuleType("torch.utils")
    fake_torch_utils_data = types.ModuleType("torch.utils.data")
    fake_dataset = type("Dataset", (), {})
    fake_torch_utils_data.Dataset = fake_dataset
    fake_torch.utils = fake_torch_utils
    fake_torch_utils.data = fake_torch_utils_data

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.SpeechT5ForTextToSpeech = object()
    fake_transformers.SpeechT5HifiGan = object()
    fake_transformers.SpeechT5Processor = object()

    def _guard_training_import(name: str):
        if name in {"Seq2SeqTrainer", "Seq2SeqTrainingArguments"}:
            raise AssertionError("runtime loader should not import training classes")
        raise AttributeError(name)

    fake_transformers.__getattr__ = _guard_training_import

    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", fake_torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", fake_torch_utils_data)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    stack = _load_torch_stack()

    assert stack["librosa"] is fake_librosa
    assert stack["sf"] is fake_soundfile
    assert stack["torch"] is fake_torch
    assert stack["Dataset"] is fake_dataset
    assert "Seq2SeqTrainer" not in stack
    assert "Seq2SeqTrainingArguments" not in stack


def test_speecht5_training_stack_reports_missing_lzma(monkeypatch) -> None:
    real_import = __import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "_lzma":
            raise ModuleNotFoundError("No module named '_lzma'", name="_lzma")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("tcc_audio.speecht5_runner._load_torch_stack", lambda: {})
    with patch("builtins.__import__", side_effect=_guarded_import):
        try:
            _load_training_stack()
        except SystemExit as exc:
            assert "_lzma" in str(exc)
            assert "python3 -c" in str(exc)
        else:
            raise AssertionError("expected training stack to fail without lzma support")


def test_speecht5_dataset_preserves_full_text_sequence() -> None:
    from tcc_audio.speecht5_runner import _build_dataset

    with patch(
        "tcc_audio.speecht5_runner._load_torch_stack",
        return_value={
            "librosa": types.SimpleNamespace(load=lambda *args, **kwargs: (np.zeros(1600, dtype=np.float32), 16000)),
            "Dataset": type("Dataset", (), {}),
        },
    ), patch(
        "tcc_audio.speecht5_runner._load_speaker_embedding",
        return_value=np.zeros((1, 512), dtype=np.float32),
    ):
        class FakeProcessor:
            def __call__(self, **kwargs):
                assert kwargs["text"] == 'Acao de teste com "aspas".'
                return {
                    "input_ids": [11, 22, 33],
                    "labels": np.zeros((1, 5, 80), dtype=np.float32),
                }

        frame = pd.DataFrame(
            [
                {
                    "audio_path": "dummy.wav",
                    "target_text": 'Ação de teste com “aspas”.',
                    "target_text_speecht5": 'Acao de teste com "aspas".',
                    "speaker_id": "speaker_01",
                }
            ]
        )
        DatasetClass = _build_dataset(frame, {"speaker_01": "speaker.npy"}, sample_rate=16000)
        item = DatasetClass(frame, FakeProcessor())[0]

    assert item["input_ids"] == [11, 22, 33]
    assert item["labels"].shape == (5, 80)
    assert item["speaker_embeddings"].shape == (512,)


def test_speecht5_dataset_normalizes_legacy_manifest_text() -> None:
    from tcc_audio.speecht5_runner import _build_dataset

    with patch(
        "tcc_audio.speecht5_runner._load_torch_stack",
        return_value={
            "librosa": types.SimpleNamespace(load=lambda *args, **kwargs: (np.zeros(1600, dtype=np.float32), 16000)),
            "Dataset": type("Dataset", (), {}),
        },
    ), patch(
        "tcc_audio.speecht5_runner._load_speaker_embedding",
        return_value=np.zeros((1, 512), dtype=np.float32),
    ):
        class FakeProcessor:
            def __call__(self, **kwargs):
                assert kwargs["text"] == "Acao rapida em portugues brasileiro."
                return {
                    "input_ids": [44, 55],
                    "labels": np.zeros((1, 3, 80), dtype=np.float32),
                }

        frame = pd.DataFrame(
            [
                {
                    "audio_path": "dummy.wav",
                    "target_text": "Ação rápida em português brasileiro.",
                    "speaker_id": "speaker_01",
                }
            ]
        )
        DatasetClass = _build_dataset(frame, {"speaker_01": "speaker.npy"}, sample_rate=16000)
        item = DatasetClass(frame, FakeProcessor())[0]

    assert item["input_ids"] == [44, 55]
    assert item["labels"].shape == (3, 80)


def test_apply_lora_adapter_does_not_force_seq2seq_task_type() -> None:
    calls: dict[str, object] = {}

    class FakeConfig:
        def __init__(self, **kwargs) -> None:
            calls["kwargs"] = kwargs

    def fake_get_peft_model(model, config):
        calls["model"] = model
        calls["config"] = config
        return "wrapped-model"

    with patch(
        "tcc_audio.speecht5_runner._load_peft",
        return_value=(FakeConfig, object(), fake_get_peft_model),
    ):
        wrapped = _apply_lora_adapter(object())

    assert wrapped == "wrapped-model"
    assert "task_type" not in calls["kwargs"]
    assert calls["kwargs"]["target_modules"] == ["q_proj", "k_proj", "v_proj", "out_proj"]


def test_load_nisqa_predictor_accepts_explicit_checkout_path(monkeypatch) -> None:
    import importlib

    with TemporaryDirectory() as tmpdir:
        checkout_path = Path(tmpdir).resolve()

        def fake_import_module(module_name: str):
            if module_name == "nisqa.NISQA_model" and str(checkout_path) in sys.path:
                return types.SimpleNamespace(nisqaModel="predictor")
            raise ImportError(module_name)

        monkeypatch.setattr(importlib, "import_module", fake_import_module)

        predictor = _load_nisqa_predictor(nisqa_path=checkout_path)

    assert predictor == "predictor"
    assert str(checkout_path) in sys.path


def test_load_nisqa_predictor_error_mentions_local_checkout(monkeypatch) -> None:
    import importlib

    monkeypatch.delenv("NISQA_PATH", raising=False)
    monkeypatch.setattr(importlib, "import_module", lambda module_name: (_ for _ in ()).throw(ImportError(module_name)))

    try:
        _load_nisqa_predictor()
    except SystemExit as exc:
        assert "--nisqa-path" in str(exc)
        assert "NISQA_PATH" in str(exc)
    else:
        raise AssertionError("expected NISQA loader to report local checkout guidance")


def test_resolve_nisqa_root_error_mentions_project_checkout() -> None:
    with TemporaryDirectory() as tmpdir:
        cwd = Path(tmpdir)
        previous = Path.cwd()
        os.chdir(cwd)
        try:
            try:
                _resolve_nisqa_root()
            except SystemExit as exc:
                assert "./NISQA" in str(exc)
                assert "--nisqa-path" in str(exc)
            else:
                raise AssertionError("expected missing NISQA root to raise SystemExit")
        finally:
            os.chdir(previous)


def test_resolve_nisqa_pretrained_model_requires_tts_checkpoint() -> None:
    with TemporaryDirectory() as tmpdir:
        checkout = Path(tmpdir) / "NISQA"
        (checkout / "weights").mkdir(parents=True, exist_ok=True)
        try:
            _resolve_nisqa_pretrained_model(checkout)
        except SystemExit as exc:
            assert "weights/nisqa_tts.tar" in str(exc)
            assert "./NISQA" in str(exc)
        else:
            raise AssertionError("expected missing NISQA checkpoint to raise SystemExit")


def test_speecht5_text_normalization_and_unk_audit() -> None:
    class FakeTokenizer:
        unk_token_id = 99

        def __call__(self, text: str, return_attention_mask: bool = False) -> dict[str, list[int]]:
            del return_attention_mask
            return {"input_ids": [99 if ord(character) > 127 else 1 for character in text]}

    normalized = normalize_text_for_speecht5('“Ação à noite, café e pinguim ü”')
    guillemet_normalized = normalize_text_for_speecht5("«ação»")

    assert normalized == '"Acao a noite, cafe e pinguim u"'
    assert guillemet_normalized == '"acao"'
    assert count_unk_tokens('“Ação à noite, café e pinguim ü”', FakeTokenizer()) > 0
    assert not has_unk_tokens(normalized, FakeTokenizer())
    assert not has_unk_tokens(guillemet_normalized, FakeTokenizer())


def test_validate_manifest_with_config_audits_speecht5_unknown_tokens() -> None:
    class FakeTokenizer:
        unk_token_id = 99

        def __call__(self, text: str, return_attention_mask: bool = False) -> dict[str, list[int]]:
            del return_attention_mask
            return {"input_ids": [99 if ord(character) > 127 else 1 for character in text]}

    with patch("tcc_audio.manifest._load_speecht5_tokenizer", return_value=FakeTokenizer()):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "data_manifest.csv"
            config_path = tmp / "config.yaml"
            config_path.write_text("project:\n  primary_model: microsoft/speecht5_tts\n", encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "speaker_id": "speaker_01",
                        "utterance_id": "utt_001",
                        "split": "train",
                        "duration_s": "3.2",
                        "source": "common_voice_pt",
                        "license": "CC0-1.0",
                        "audio_path": "audio.wav",
                        "reference_audio": "reference.wav",
                        "target_text": "Ação rápida em português brasileiro.",
                        "target_text_speecht5": "Acao rapida em portugues brasileiro.",
                        "text_variant": "normalized",
                    }
                ]
            ).to_csv(manifest_path, index=False)

            report = validate_data_manifest(manifest_path, config_path=config_path)

            assert report.ok, report.errors
            assert report.summary["speecht5_rows_with_unk_raw"] == 1
            assert report.summary["speecht5_rows_with_unk_normalized"] == 0


def test_validate_manifest_re_normalizes_stale_speecht5_audit_text() -> None:
    class FakeTokenizer:
        unk_token_id = 99

        def __call__(self, text: str, return_attention_mask: bool = False) -> dict[str, list[int]]:
            del return_attention_mask
            return {"input_ids": [99 if ord(character) > 127 else 1 for character in text]}

    with patch("tcc_audio.manifest._load_speecht5_tokenizer", return_value=FakeTokenizer()):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "data_manifest.csv"
            config_path = tmp / "config.yaml"
            config_path.write_text("project:\n  primary_model: microsoft/speecht5_tts\n", encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "speaker_id": "speaker_01",
                        "utterance_id": "utt_001",
                        "split": "train",
                        "duration_s": "3.2",
                        "source": "common_voice_pt",
                        "license": "CC0-1.0",
                        "audio_path": "audio.wav",
                        "reference_audio": "reference.wav",
                        "target_text": "e o estádio marcado por Sanches, quando disse «nem sei se nada sei».",
                        "target_text_speecht5": 'e o estadio marcado por Sanches, quando disse «nem sei se nada sei».',
                        "text_variant": "normalized",
                    }
                ]
            ).to_csv(manifest_path, index=False)

            report = validate_data_manifest(manifest_path, config_path=config_path)

            assert report.ok, report.errors
            assert report.summary["speecht5_rows_with_unk_raw"] == 1
            assert report.summary["speecht5_rows_with_unk_normalized"] == 0


def test_speecht5_embedding_dimension_validation() -> None:
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "speaker.npy"
        np.save(path, np.zeros(192, dtype=np.float32))
        try:
            _load_speaker_embedding(path, expected_dim=512)
        except ValueError as exc:
            assert "synthesis path" in str(exc)
            assert "speaker_embedding_path" in str(exc)
        else:
            raise AssertionError("expected speaker embedding dimension validation to fail")


def test_metric_aggregation_contract() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        samples_path = tmp / "samples.csv"
        rows = []
        for checkpoint_label, speaker_id, wer, similarity, nisqa in [
            ("speecht5_lora_conservative@step500", "speaker_01", 0.31, 0.62, 3.1),
            ("speecht5_lora_unique@step500", "speaker_01", 0.22, 0.71, 3.5),
            ("speecht5_lora_conservative@step500", "speaker_02", 0.28, 0.66, 3.2),
            ("speecht5_lora_unique@step500", "speaker_02", 0.20, 0.73, 3.6),
        ]:
            rows.append(
                {
                    "sample_id": f"{checkpoint_label}_{speaker_id}",
                    "run_id": f"base__{speaker_id}__P001__normalized",
                    "condition": checkpoint_label.split("@", 1)[0],
                    "speaker_id": speaker_id,
                    "prompt_id": "P001",
                    "text_variant": "normalized",
                    "target_text": "Texto de teste.",
                    "audio_path": "audio.wav",
                    "reference_audio_path": "reference.wav",
                    "speaker_embedding_path": "embedding.npy",
                    "model_name": checkpoint_label,
                    "checkpoint_label": checkpoint_label,
                    "checkpoint_step": "500",
                    "checkpoint_path": f"artifacts/checkpoints/{checkpoint_label}",
                    "checkpoint_run_ts": "20260426T010203Z",
                    "training_scope": "per_speaker",
                    "training_unit": speaker_id,
                    "run_started_at": "",
                    "run_finished_at": "",
                    "failure_reason": "",
                    "wer": wer,
                    "speaker_similarity": similarity,
                    "nisqa": nisqa,
                    "f0_rmse": 20.0,
                    "rtf": 0.8,
                    "train_gpu_hours": 0.1,
                    "inference_seconds": 1.2,
                    "cost_usd": 0.05,
                    "status": "ok",
                    "lora_gate_status": "stable_lora",
                }
            )
        pd.DataFrame(rows).to_csv(samples_path, index=False)
        metrics, costs = aggregate_metrics(samples_path, tmp / "evaluation")
        assert not metrics.empty
        assert set(costs["condition"]) == {
            "speecht5_lora_conservative@step500",
            "speecht5_lora_unique@step500",
        }
        assert costs["samples"].sum() == 4
        assert (metrics["condition"] == "speecht5_lora_conservative@step500 vs speecht5_lora_unique@step500").any()
        assert (tmp / "evaluation/metrics_summary.csv").exists()
        assert (tmp / "evaluation/cost_summary.csv").exists()
        assert (tmp / "evaluation/metrics_by_speaker.csv").exists()
        assert (tmp / "evaluation/cost_by_speaker.csv").exists()


def test_wer_computation() -> None:
    assert word_error_rate("a voz ficou clara", "a voz ficou clara") == 0
    assert word_error_rate("a voz ficou clara", "a voz clara") == 0.25
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        samples_path = tmp / "samples.csv"
        pd.DataFrame(
            [
                {
                    "sample_id": "s1",
                    "run_id": "r1",
                    "condition": "speecht5_lora_conservative",
                    "speaker_id": "speaker_01",
                    "prompt_id": "P001",
                    "text_variant": "normalized",
                    "target_text": "a voz ficou clara",
                    "audio_path": "audio.wav",
                    "reference_audio_path": "reference.wav",
                    "speaker_embedding_path": "embedding.npy",
                    "model_name": "speecht5_lora_conservative",
                    "run_started_at": "",
                    "run_finished_at": "",
                    "failure_reason": "",
                    "wer": "",
                    "speaker_similarity": "",
                    "nisqa": "",
                    "f0_rmse": "",
                    "rtf": "",
                    "train_gpu_hours": "",
                    "inference_seconds": "",
                    "cost_usd": "",
                    "status": "pending",
                    "lora_gate_status": "",
                    "asr_text": "a voz ficou clara",
                }
            ]
        ).to_csv(samples_path, index=False)
        computed = compute_wer_from_asr(samples_path, tmp / "samples_with_wer.csv")
        assert float(computed.loc[0, "wer"]) == 0
        assert computed.loc[0, "status"] == "pending"


def test_build_whisper_arg_parser_defaults_to_optional_paths() -> None:
    parser = build_whisper_arg_parser()
    args = parser.parse_args(["--all"])

    assert args.config is None
    assert args.samples is None
    assert args.out is None
    assert args.model_name is None
    assert args.all is True


def test_whisper_main_defaults_out_to_samples() -> None:
    with patch("tcc_audio.whisper_batch.run_whisper_batch") as mocked:
        whisper_main(["-s", "samples.csv"])

    mocked.assert_called_once_with("samples.csv", "samples.csv", "openai/whisper-small", only_missing=True)


def test_build_wer_arg_parser_defaults_to_optional_paths() -> None:
    parser = build_wer_arg_parser()
    args = parser.parse_args([])

    assert args.config is None
    assert args.samples is None
    assert args.out is None
    assert args.asr_column == "asr_text"


def test_wer_main_defaults_out_to_samples() -> None:
    fake = pd.DataFrame([{"wer": "0.0"}])
    with patch("tcc_audio.wer.compute_wer_from_asr", return_value=fake) as mocked:
        wer_main(["-s", "samples.csv"])

    mocked.assert_called_once_with("samples.csv", "samples.csv", "asr_text")


def test_build_report_assets_arg_parser_defaults_to_optional_paths() -> None:
    parser = build_report_assets_arg_parser()
    args = parser.parse_args(["-c", "config.yaml"])

    assert args.config == "config.yaml"
    assert args.samples is None
    assert args.metrics is None
    assert args.costs is None
    assert args.out_dir is None


def test_prepare_common_voice_metadata() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        validated_tsv, _, _ = _write_common_voice_metadata_inputs(
            tmp,
            validated_rows=[
                'spk1\tclip1.mp3\tsid1\tTexto validado 1.\tTexto validado, "com aspas".\tmale_masculine\tpt\t\n',
                "spk2\tclip2.mp3\tsid2\t\t\t\tpt\tPortuguese (Brasil)\n",
                "spk3\tclip3.mp3\tsid3\tTexto pt-pt.\t\tfemale_feminine\tpt\tPortuguese (Portugal)\n",
            ],
            validated_sentences_rows=[
                "sid1\tTexto catalogado 1.\tpt-BR\n",
                'sid2\tTexto catalogado 2, "fallback".\t\n',
                "sid3\tTexto catalogado 3.\tPortuguese (Portugal)\n",
            ],
            clip_duration_rows=[
                "clip1.mp3\t5256\n",
                "clip2.mp3\t4032\n",
                "clip3.mp3\t6000\n",
            ],
        )
        prepared = prepare_common_voice_metadata(
            tsv_path=validated_tsv,
            clips_dir=tmp / "clips",
            out_path=tmp / "metadata.csv",
            locale="pt",
            variant="pt-BR",
        )
        assert list(prepared.columns) == [
            "source_speaker_id",
            "gender",
            "utterance_id",
            "duration_s",
            "audio_path",
            "target_text",
            "license",
            "source",
            "locale",
            "variant",
        ]
        assert len(prepared) == 2
        assert prepared.loc[0, "source_speaker_id"] == "spk1"
        assert prepared.loc[0, "gender"] == "masculine"
        assert prepared.loc[0, "target_text"] == 'Texto validado, "com aspas".'
        assert float(prepared.loc[0, "duration_s"]) == 5.256
        assert prepared.loc[0, "audio_path"] == str(tmp / "clips" / "clip1.mp3")
        assert prepared.loc[0, "variant"] == "pt-BR"
        assert prepared.loc[1, "source_speaker_id"] == "spk2"
        assert prepared.loc[1, "gender"] == "unknown"
        assert prepared.loc[1, "target_text"] == 'Texto catalogado 2, "fallback".'
        assert float(prepared.loc[1, "duration_s"]) == 4.032
        assert prepared.loc[1, "variant"] == "pt-BR"
        assert (tmp / "metadata.csv").exists()
        serialized = (tmp / "metadata.csv").read_text(encoding="utf-8")
        assert '"Texto validado, ""com aspas""."' in serialized
        assert '"Texto catalogado 2, ""fallback""."' in serialized


def test_prepare_common_voice_metadata_keeps_variant_empty_without_tsv_value() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        validated_tsv, _, _ = _write_common_voice_metadata_inputs(
            tmp,
            validated_rows=[
                "spk1\tclip1.mp3\tsid1\tTexto validado.\t\tmale\tpt\t\n",
            ],
            validated_sentences_rows=[
                "sid1\tTexto catalogado.\t\n",
            ],
            clip_duration_rows=[
                "clip1.mp3\t5256\n",
            ],
        )
        prepared = prepare_common_voice_metadata(
            tsv_path=validated_tsv,
            clips_dir=tmp / "clips",
            out_path=tmp / "metadata.csv",
            locale="pt",
        )
        assert len(prepared) == 1
        assert prepared.loc[0, "variant"] == ""


def test_prepare_common_voice_metadata_rejects_missing_validated_sentence_merge() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        validated_tsv, _, _ = _write_common_voice_metadata_inputs(
            tmp,
            validated_rows=[
                "spk1\tclip1.mp3\tsid1\tTexto validado.\t\tmale\tpt\tpt-BR\n",
            ],
            validated_sentences_rows=[],
            clip_duration_rows=[
                "clip1.mp3\t5256\n",
            ],
        )
        try:
            prepare_common_voice_metadata(
                tsv_path=validated_tsv,
                clips_dir=tmp / "clips",
                out_path=tmp / "metadata.csv",
            )
        except ValueError as exc:
            assert "validated_sentences" in str(exc)
        else:
            raise AssertionError("Expected missing validated_sentences merge to raise ValueError.")


def test_prepare_common_voice_metadata_rejects_missing_clip_duration_merge() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        validated_tsv, _, _ = _write_common_voice_metadata_inputs(
            tmp,
            validated_rows=[
                "spk1\tclip1.mp3\tsid1\tTexto validado.\t\tmale\tpt\tpt-BR\n",
            ],
            validated_sentences_rows=[
                "sid1\tTexto catalogado.\tpt-BR\n",
            ],
            clip_duration_rows=[],
        )
        try:
            prepare_common_voice_metadata(
                tsv_path=validated_tsv,
                clips_dir=tmp / "clips",
                out_path=tmp / "metadata.csv",
            )
        except ValueError as exc:
            assert "clip durations" in str(exc)
        else:
            raise AssertionError("Expected missing clip durations merge to raise ValueError.")


def test_request_dataset_download_session_uses_bearer_token() -> None:
    requests_seen: list[object] = []

    def fake_urlopen(req: object) -> MockUrlopenResponse:
        requests_seen.append(req)
        return MockUrlopenResponse(
            (
                '{"downloadUrl":"https://storage.example.com/common-voice-pt.tar.gz",'
                '"filename":"common-voice-pt.tar.gz",'
                '"checksum":"sha256:abc123",'
                '"sizeBytes":"42"}'
            ).encode("utf-8")
        )

    with patch("tcc_audio.common_voice_download.request.urlopen", side_effect=fake_urlopen):
        session = request_dataset_download_session(
            "cmn29f4cb017bmm07pd9yd8mw",
            api_key="token-123",
        )

    request_obj = requests_seen[0]
    assert request_obj.get_header("Authorization") == "Bearer token-123"
    assert session.download_url == "https://storage.example.com/common-voice-pt.tar.gz"
    assert session.filename == "common-voice-pt.tar.gz"
    assert session.checksum == "sha256:abc123"
    assert session.size_bytes == 42


def test_download_common_voice_pt_stages_archive_with_mocked_http() -> None:
    archive_bytes = _build_common_voice_archive_bytes()

    def fake_urlopen(req: object) -> MockUrlopenResponse:
        if hasattr(req, "full_url"):
            return MockUrlopenResponse(
                (
                    '{"downloadUrl":"https://storage.example.com/common-voice-pt.tar.gz",'
                    '"filename":"common-voice-pt.tar.gz",'
                    '"checksum":"",'
                    f'"sizeBytes":"{len(archive_bytes)}"'
                    "}"
                ).encode("utf-8")
            )
        return MockUrlopenResponse(archive_bytes)

    with TemporaryDirectory() as tmpdir, patch.dict(
        os.environ,
        {"MOZILLA_DATA_COLLECTIVE_API_KEY": "token-123"},
        clear=True,
    ), patch("tcc_audio.common_voice_download.request.urlopen", side_effect=fake_urlopen):
        staged = download_common_voice_pt(out_dir=Path(tmpdir) / "data/raw/common_voice_pt")

        assert staged.archive_path.exists()
        assert staged.clips_dir.exists()
        assert staged.validated_tsv.exists()
        assert staged.validated_sentences_path.exists()
        assert staged.clip_durations_path.exists()
        assert (staged.clips_dir / "sample.mp3").read_bytes() == b"fake-mp3"
        assert "Texto um." in staged.validated_tsv.read_text(encoding="utf-8")
        assert "Texto um." in staged.validated_sentences_path.read_text(encoding="utf-8")
        assert "sample.mp3" in staged.clip_durations_path.read_text(encoding="utf-8")


def test_download_common_voice_pt_requires_api_key_env() -> None:
    with TemporaryDirectory() as tmpdir, patch.dict(os.environ, {}, clear=True):
        try:
            download_common_voice_pt(out_dir=Path(tmpdir) / "data/raw/common_voice_pt")
        except RuntimeError as exc:
            assert "MOZILLA_DATA_COLLECTIVE_API_KEY" in str(exc)
        else:
            raise AssertionError("Expected missing API key to raise RuntimeError.")


def test_download_common_voice_pt_reuses_existing_archive_without_api_call() -> None:
    with TemporaryDirectory() as tmpdir, patch.dict(
        os.environ,
        {"MOZILLA_DATA_COLLECTIVE_API_KEY": "token-123"},
        clear=True,
    ), patch("tcc_audio.common_voice_download.request.urlopen") as mocked_urlopen:
        out_dir = Path(tmpdir) / "data/raw/common_voice_pt"
        out_dir.mkdir(parents=True, exist_ok=True)
        archive_path = out_dir / "common-voice-scripted-speech-25-0-portug-0254cce0.tar.gz"
        archive_path.write_bytes(_build_common_voice_archive_bytes())

        staged = download_common_voice_pt(out_dir=out_dir)

        mocked_urlopen.assert_not_called()
        assert staged.archive_path == archive_path
        assert staged.validated_tsv.exists()
        assert staged.validated_sentences_path.exists()
        assert staged.clip_durations_path.exists()


def test_stage_common_voice_archive_respects_overwrite() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        archive_path = tmp / "common-voice-pt.tar.gz"
        archive_path.write_bytes(_build_common_voice_archive_bytes())
        out_dir = tmp / "data/raw/common_voice_pt"

        staged = stage_common_voice_archive(archive_path, out_dir)
        assert (staged.clips_dir / "sample.mp3").exists()
        assert staged.validated_sentences_path.exists()
        assert staged.clip_durations_path.exists()

        try:
            stage_common_voice_archive(archive_path, out_dir)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Expected second extraction without overwrite to fail.")

        replacement_archive = tmp / "common-voice-pt-replacement.tar.gz"
        replacement_payload = io.BytesIO()
        with TemporaryDirectory() as replacement_tmpdir:
            replacement_tmp = Path(replacement_tmpdir)
            dataset_root = replacement_tmp / "release" / "pt"
            clips_dir = dataset_root / "clips"
            clips_dir.mkdir(parents=True, exist_ok=True)
            (clips_dir / "sample.mp3").write_bytes(b"replacement")
            (dataset_root / "validated.tsv").write_text(
                "client_id\tpath\ttext\tgender\tlocale\tvariant\n"
                "spk2\tsample.mp3\tTexto novo.\tfemale\tpt\tpt-BR\n",
                encoding="utf-8",
            )
            (dataset_root / "validated_sentences.tsv").write_text(
                "sentence_id\tsentence\tis_used\n"
                "s2\tTexto novo.\t1\n",
                encoding="utf-8",
            )
            (dataset_root / "clip_durations.tsv").write_text(
                "clip\tduration[ms]\n"
                "sample.mp3\t4321\n",
                encoding="utf-8",
            )
            with tarfile.open(fileobj=replacement_payload, mode="w:gz") as archive:
                archive.add(dataset_root, arcname=dataset_root.relative_to(replacement_tmp))
        replacement_archive.write_bytes(replacement_payload.getvalue())

        restaged = stage_common_voice_archive(replacement_archive, out_dir, overwrite=True)
        assert (restaged.clips_dir / "sample.mp3").read_bytes() == b"replacement"
        assert "Texto novo." in restaged.validated_tsv.read_text(encoding="utf-8")
        assert "Texto novo." in restaged.validated_sentences_path.read_text(encoding="utf-8")
        assert "4321" in restaged.clip_durations_path.read_text(encoding="utf-8")


def test_report_assets_without_human_eval() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        samples_path = tmp / "samples.csv"
        rows = []
        for checkpoint_label, speaker_id in [
            ("speecht5_lora_conservative@step500", "speaker_01"),
            ("speecht5_lora_unique@step500", "speaker_01"),
            ("speecht5_lora_unique@step1000", "speaker_02"),
        ]:
            rows.append(
                {
                    "sample_id": f"{checkpoint_label}_{speaker_id}",
                    "run_id": f"base__{speaker_id}__P001__normalized",
                    "condition": checkpoint_label.split("@", 1)[0],
                    "speaker_id": speaker_id,
                    "prompt_id": "P001",
                    "text_variant": "normalized",
                    "target_text": "Texto de teste.",
                    "audio_path": __file__,
                    "reference_audio_path": __file__,
                    "speaker_embedding_path": "embedding.npy",
                    "model_name": checkpoint_label,
                    "checkpoint_label": checkpoint_label,
                    "checkpoint_step": checkpoint_label.split("step", 1)[1],
                    "checkpoint_path": f"artifacts/checkpoints/{checkpoint_label}",
                    "checkpoint_run_ts": "20260426T010203Z",
                    "training_scope": "per_speaker",
                    "training_unit": speaker_id,
                    "run_started_at": "",
                    "run_finished_at": "",
                    "failure_reason": "",
                    "wer": 0.1,
                    "speaker_similarity": 0.8,
                    "nisqa": 3.5,
                    "f0_rmse": 22.0,
                    "rtf": 0.9,
                    "train_gpu_hours": 0.2,
                    "inference_seconds": 1.0,
                    "cost_usd": 0.05,
                    "status": "ok",
                    "lora_gate_status": "stable_lora",
                    "asr_text": "Texto de teste.",
                }
            )
        pd.DataFrame(rows).to_csv(samples_path, index=False)

        metrics = pd.DataFrame(
            [
                {"condition": "speecht5_lora_conservative@step500", "text_variant": "normalized", "metric": "wer", "n": 1, "mean": 0.3, "median": 0.3, "std": 0, "ci95_low": 0.3, "ci95_high": 0.3, "test": "", "statistic": "", "p_value": ""},
                {"condition": "speecht5_lora_unique@step500", "text_variant": "normalized", "metric": "wer", "n": 1, "mean": 0.1, "median": 0.1, "std": 0, "ci95_low": 0.1, "ci95_high": 0.1, "test": "", "statistic": "", "p_value": ""},
            ]
        )
        costs = pd.DataFrame(
            [
                {"condition": "speecht5_lora_conservative@step500", "samples": 1, "total_cost_usd": 0.05, "total_train_gpu_hours": 0.2, "mean_rtf": 1.0, "mean_inference_seconds": 1.0},
                {"condition": "speecht5_lora_unique@step500", "samples": 1, "total_cost_usd": 0.10, "total_train_gpu_hours": 0.2, "mean_rtf": 0.8, "mean_inference_seconds": 0.8},
            ]
        )
        metrics_by_speaker = pd.DataFrame(
            [
                {"condition": "speecht5_lora_conservative@step500", "speaker_id": "speaker_01", "text_variant": "normalized", "metric": "wer", "n": 1, "mean": 0.3, "median": 0.3, "std": 0, "ci95_low": 0.3, "ci95_high": 0.3},
            ]
        )
        cost_by_speaker = pd.DataFrame(
            [
                {"condition": "speecht5_lora_conservative@step500", "speaker_id": "speaker_01", "samples": 1, "total_cost_usd": 0.05, "total_train_gpu_hours": 0.2, "mean_rtf": 1.0, "mean_inference_seconds": 1.0},
            ]
        )
        metrics_path = tmp / "metrics.csv"
        costs_path = tmp / "costs.csv"
        metrics.to_csv(metrics_path, index=False)
        costs.to_csv(costs_path, index=False)
        metrics_by_speaker.to_csv(tmp / "metrics_by_speaker.csv", index=False)
        cost_by_speaker.to_csv(tmp / "cost_by_speaker.csv", index=False)
        outputs = make_report_assets(samples_path, metrics_path, costs_path, tmp / "report_assets")
        assert (tmp / "report_assets/overview.md").exists()
        assert "metrics_markdown" in outputs
        assert "metrics_by_speaker_markdown" in outputs
        assert "human_markdown" not in outputs


BOOTSTRAP_SMOKE_TESTS = (
    test_prompt_file_has_expected_contract,
    test_run_matrix_generation,
    test_speaker_selection_from_processed_metadata,
    test_speecht5_text_normalization_and_unk_audit,
    test_validate_manifest_with_config_audits_speecht5_unknown_tokens,
    test_speaker_embedding_config_resolution,
    test_extract_speaker_embeddings_uses_configured_tts_model,
    test_extract_speaker_embeddings_override_wins_over_config,
    test_compute_speaker_similarity_uses_configured_eval_model,
    test_speecht5_dataset_preserves_full_text_sequence,
    test_speecht5_dataset_normalizes_legacy_manifest_text,
    test_speecht5_embedding_dimension_validation,
    test_metric_aggregation_contract,
    test_wer_computation,
    test_prepare_common_voice_metadata,
    test_report_assets_without_human_eval,
)

BOOTSTRAP_SMOKE_TEST_PROFILES = {
    "core": BOOTSTRAP_SMOKE_TESTS[:5],
    "extended": BOOTSTRAP_SMOKE_TESTS,
}


def run_bootstrap_smoke_tests(profile: str = "extended") -> None:
    try:
        smoke_tests = BOOTSTRAP_SMOKE_TEST_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown smoke test profile: {profile}") from exc

    for smoke_test in smoke_tests:
        smoke_test()


if __name__ == "__main__":
    selected_profile = sys.argv[1] if len(sys.argv) > 1 else "extended"
    run_bootstrap_smoke_tests(profile=selected_profile)
    print("smoke tests passed")
