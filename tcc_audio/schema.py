"""Stable artifact schemas used by the experiment pipeline."""

DATA_MANIFEST_REQUIRED_COLUMNS = [
    "speaker_id",
    "utterance_id",
    "split",
    "duration_s",
    "source",
    "license",
    "audio_path",
    "reference_audio",
    "target_text",
    "text_variant",
]

PROMPTS_REQUIRED_COLUMNS = [
    "prompt_id",
    "category",
    "raw_text",
    "normalized_text",
    "include_prompt_subexperiment",
    "include_commercial_subset",
]

EVAL_SAMPLES_REQUIRED_COLUMNS = [
    "sample_id",
    "run_id",
    "condition",
    "speaker_id",
    "prompt_id",
    "text_variant",
    "target_text",
    "audio_path",
    "reference_audio_path",
    "speaker_embedding_path",
    "model_name",
    "checkpoint_label",
    "checkpoint_step",
    "checkpoint_path",
    "checkpoint_run_ts",
    "training_scope",
    "training_unit",
    "run_started_at",
    "run_finished_at",
    "failure_reason",
    "wer",
    "speaker_similarity",
    "nisqa",
    "f0_rmse",
    "rtf",
    "train_gpu_hours",
    "inference_seconds",
    "cost_usd",
    "status",
    "lora_gate_status",
]

NUMERIC_METRIC_COLUMNS = [
    "wer",
    "speaker_similarity",
    "nisqa",
    "f0_rmse",
    "rtf",
    "train_gpu_hours",
    "inference_seconds",
    "cost_usd",
]

VALID_SPLITS = {"train", "val", "test", "reference", "commercial"}
VALID_TEXT_VARIANTS = {"raw", "normalized"}

SAMPLES_OPTIONAL_COLUMNS = [
    "asr_text",
    "total_train_gpu_hours",
]
