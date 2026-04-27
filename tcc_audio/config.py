"""Experiment config helpers with backwards-compatible key resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from tcc_audio.io import read_yaml

DEFAULT_TTS_SPEAKER_EMBEDDING_MODEL = "speechbrain/spkrec-xvect-voxceleb"
DEFAULT_TTS_SPEAKER_EMBEDDING_DIM = 512
DEFAULT_SPEAKER_SIMILARITY_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_ASR_MODEL = "openai/whisper-small"
DEFAULT_ASR_TASK = "transcribe"
DEFAULT_ASR_LANGUAGE = "pt"


def load_experiment_config(config_path: str | Path | None) -> dict[str, Any]:
    if not config_path:
        return {}
    return read_yaml(config_path)


def _config_section(config: Mapping[str, Any] | None, section_name: str) -> Mapping[str, Any]:
    if not config:
        return {}
    section = config.get(section_name, {})
    if isinstance(section, Mapping):
        return section
    return {}


def resolve_tts_speaker_embedding_model(config: Mapping[str, Any] | None = None) -> str:
    project = _config_section(config, "project")
    value = project.get("tts_speaker_embedding_model") or project.get("speaker_embedding_model")
    if not value:
        return DEFAULT_TTS_SPEAKER_EMBEDDING_MODEL
    return str(value)


def resolve_tts_speaker_embedding_dim(config: Mapping[str, Any] | None = None) -> int:
    project = _config_section(config, "project")
    value = project.get("tts_speaker_embedding_dim")
    if value in (None, ""):
        return DEFAULT_TTS_SPEAKER_EMBEDDING_DIM
    return int(value)


def resolve_speaker_similarity_model(config: Mapping[str, Any] | None = None) -> str:
    evaluation = _config_section(config, "evaluation")
    value = evaluation.get("speaker_similarity_model")
    if not value:
        return DEFAULT_SPEAKER_SIMILARITY_MODEL
    return str(value)


def resolve_asr_model(config: Mapping[str, Any] | None = None) -> str:
    evaluation = _config_section(config, "evaluation")
    asr = evaluation.get("asr", {})
    if not isinstance(asr, Mapping):
        return DEFAULT_ASR_MODEL
    value = asr.get("name")
    if not value:
        return DEFAULT_ASR_MODEL
    return str(value)


def resolve_asr_task(config: Mapping[str, Any] | None = None) -> str:
    evaluation = _config_section(config, "evaluation")
    asr = evaluation.get("asr", {})
    if not isinstance(asr, Mapping):
        return DEFAULT_ASR_TASK
    value = asr.get("task")
    if not value:
        return DEFAULT_ASR_TASK
    return str(value)


def resolve_asr_language(config: Mapping[str, Any] | None = None) -> str:
    evaluation = _config_section(config, "evaluation")
    asr = evaluation.get("asr", {})
    if not isinstance(asr, Mapping):
        return DEFAULT_ASR_LANGUAGE
    value = asr.get("language")
    if not value:
        return DEFAULT_ASR_LANGUAGE
    return str(value)
