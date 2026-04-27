"""Prepare Common Voice metadata for downstream processing."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from tcc_audio.cli_defaults import (
    DEFAULT_RAW_CLIPS_DIR,
    DEFAULT_RAW_METADATA_PATH,
    DEFAULT_RAW_TSV,
)
from tcc_audio.io import ensure_parent_dir


VALIDATED_SENTENCES_FILENAMES = ("validated_sentences.tsv", "validated_sentences.csv")
CLIP_DURATIONS_FILENAMES = (
    "clip_durations.tsv",
    "clip_duration.tsv",
    "clip_duration.csv",
    "clip_durations.csv",
)
OUTPUT_COLUMNS = [
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
_VARIANT_ALIASES = {
    "pt-br": "pt-BR",
    "portuguese (brasil)": "pt-BR",
    "portuguese (brazil)": "pt-BR",
    "pt-pt": "pt-PT",
    "portuguese (portugal)": "pt-PT",
}


def _normalize_gender(value: object) -> str:
    normalized = str(value).strip().lower()
    mapping = {
        "male": "masculine",
        "m": "masculine",
        "male_masculine": "masculine",
        "masculine": "masculine",
        "female": "feminine",
        "f": "feminine",
        "female_feminine": "feminine",
        "feminine": "feminine",
    }
    return mapping.get(normalized, "unknown")


def _canonicalize_variant(value: object) -> str:
    raw = str(value).strip()
    if not raw:
        return ""
    return _VARIANT_ALIASES.get(raw.lower(), raw)


def _strip_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _coalesce_series(*series_list: pd.Series) -> pd.Series:
    if not series_list:
        return pd.Series(dtype=object)

    result = _strip_series(series_list[0]).copy()
    for candidate in series_list[1:]:
        candidate_clean = _strip_series(candidate)
        result = result.where(result.ne(""), candidate_clean)
    return result


def _read_tsv(path: str | Path, *, label: str) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Missing {label}: {source}")
    return pd.read_csv(source, sep="\t", dtype=str, keep_default_na=False)


def _resolve_sidecar_path(
    validated_tsv_path: str | Path,
    explicit_path: str | Path | None,
    candidates: tuple[str, ...],
    *,
    label: str,
) -> Path:
    if explicit_path is not None:
        resolved = Path(explicit_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Missing {label}: {resolved}")
        return resolved

    base_dir = Path(validated_tsv_path).resolve().parent
    for candidate in candidates:
        resolved = base_dir / candidate
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        f"Missing {label} alongside {validated_tsv_path}. "
        f"Tried: {', '.join(candidates)}"
    )


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for column in columns:
        if column not in normalized.columns:
            normalized[column] = ""
    return normalized


def _ensure_unique(frame: pd.DataFrame, column: str, *, label: str) -> None:
    duplicates = frame[_strip_series(frame[column]).duplicated(keep=False) & _strip_series(frame[column]).ne("")]
    if duplicates.empty:
        return
    samples = ", ".join(_strip_series(duplicates[column]).drop_duplicates().head(5).tolist())
    raise ValueError(f"{label} must have unique {column} values. Sample duplicates: {samples}")


def _validate_join_coverage(merged: pd.DataFrame, indicator_column: str, *, label: str, key_column: str) -> None:
    missing = merged[merged[indicator_column] != "both"].copy()
    if missing.empty:
        return
    samples = ", ".join(_strip_series(missing[key_column]).head(5).tolist())
    raise ValueError(f"Missing {label} rows for {len(missing)} entries. Sample {key_column} values: {samples}")


def prepare_common_voice_metadata(
    tsv_path: str | Path,
    clips_dir: str | Path,
    out_path: str | Path,
    locale: str | None = None,
    variant: str | None = None,
    max_rows: int | None = None,
    validated_sentences_tsv_path: str | Path | None = None,
    clip_durations_tsv_path: str | Path | None = None,
) -> pd.DataFrame:
    validated_path = Path(tsv_path)
    validated_sentences_path = _resolve_sidecar_path(
        validated_path,
        validated_sentences_tsv_path,
        VALIDATED_SENTENCES_FILENAMES,
        label="validated_sentences TSV",
    )
    clip_durations_path = _resolve_sidecar_path(
        validated_path,
        clip_durations_tsv_path,
        CLIP_DURATIONS_FILENAMES,
        label="clip durations TSV",
    )

    validated = _ensure_columns(
        _read_tsv(validated_path, label="validated TSV"),
        ["client_id", "path", "sentence_id", "sentence", "text", "gender", "locale", "variant"],
    )
    validated_sentences = _ensure_columns(
        _read_tsv(validated_sentences_path, label="validated_sentences TSV"),
        ["sentence_id", "sentence", "variant"],
    )
    clip_durations = _ensure_columns(
        _read_tsv(clip_durations_path, label="clip durations TSV"),
        ["clip", "duration[ms]"],
    )

    _ensure_unique(validated_sentences, "sentence_id", label="validated_sentences TSV")
    _ensure_unique(clip_durations, "clip", label="clip durations TSV")

    merged = validated.merge(
        validated_sentences[["sentence_id", "sentence", "variant"]],
        on="sentence_id",
        how="left",
        suffixes=("_validated", "_sentences"),
        indicator="_sentence_merge",
    )
    _validate_join_coverage(
        merged,
        "_sentence_merge",
        label="validated_sentences",
        key_column="sentence_id",
    )

    merged = merged.merge(
        clip_durations[["clip", "duration[ms]"]],
        left_on="path",
        right_on="clip",
        how="left",
        indicator="_duration_merge",
    )
    _validate_join_coverage(
        merged,
        "_duration_merge",
        label="clip durations",
        key_column="path",
    )

    merged["resolved_locale"] = _coalesce_series(merged["locale"], pd.Series([locale or ""] * len(merged)))
    merged["resolved_variant"] = _coalesce_series(
        merged["variant_sentences"].map(_canonicalize_variant),
        merged["variant_validated"].map(_canonicalize_variant),
    )
    merged["target_text"] = _coalesce_series(
        merged["text"],
        merged["sentence_validated"],
        merged["sentence_sentences"],
    )

    if locale:
        merged = merged[merged["resolved_locale"].eq(locale)].copy()

    if variant:
        requested_variant = _canonicalize_variant(variant)
        merged = merged[merged["resolved_variant"].eq(requested_variant)].copy()

    if max_rows:
        merged = merged.head(max_rows).copy()

    merged = merged.reset_index(drop=True)
    duration_ms = pd.to_numeric(merged["duration[ms]"], errors="coerce")
    if duration_ms.isna().any():
        samples = ", ".join(_strip_series(merged.loc[duration_ms.isna(), "path"]).head(5).tolist())
        raise ValueError(f"Invalid duration[ms] values after merge. Sample path values: {samples}")

    source_speaker_ids = _strip_series(merged["client_id"])
    fallback_speaker_ids = pd.Series([f"speaker_{index:06d}" for index in range(len(merged))])
    source_speaker_ids = source_speaker_ids.where(source_speaker_ids.ne(""), fallback_speaker_ids)

    clips_root = Path(clips_dir)
    prepared = pd.DataFrame(
        {
            "source_speaker_id": source_speaker_ids,
            "gender": merged["gender"].map(_normalize_gender),
            "utterance_id": _strip_series(merged["path"]).map(lambda path: Path(path).stem),
            "duration_s": (duration_ms / 1000).round(3),
            "audio_path": _strip_series(merged["path"]).map(lambda path: str(clips_root / path)),
            "target_text": merged["target_text"],
            "license": "CC0-1.0",
            "source": "common_voice_pt",
            "locale": merged["resolved_locale"],
            "variant": merged["resolved_variant"],
        },
        columns=OUTPUT_COLUMNS,
    )

    ensure_parent_dir(out_path)
    prepared.to_csv(
        out_path,
        index=False,
        sep=",",
        quotechar='"',
        quoting=csv.QUOTE_NONNUMERIC,
        float_format="%.3f",
    )
    return prepared


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Common Voice metadata for the TCC pipeline.")
    parser.add_argument("-t", "--tsv", default=str(DEFAULT_RAW_TSV), help="Path to Common Voice validated.tsv.")
    parser.add_argument("--validated-sentences-tsv", help="Path to Common Voice validated_sentences.tsv.")
    parser.add_argument("--clip-durations-tsv", help="Path to Common Voice clip_durations.tsv.")
    parser.add_argument("-d", "--clips-dir", default=str(DEFAULT_RAW_CLIPS_DIR), help="Path to Common Voice clips directory.")
    parser.add_argument("-o", "--out", default=str(DEFAULT_RAW_METADATA_PATH), help="Output CSV path.")
    parser.add_argument("-l", "--locale", default="pt", help="Locale filter when the column exists.")
    parser.add_argument("-v", "--variant", default="pt-BR", help="Variant filter, for example pt-BR.")
    parser.add_argument("--max-rows", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    frame = prepare_common_voice_metadata(
        tsv_path=args.tsv,
        clips_dir=args.clips_dir,
        out_path=args.out,
        locale=args.locale,
        variant=args.variant,
        max_rows=args.max_rows,
        validated_sentences_tsv_path=args.validated_sentences_tsv,
        clip_durations_tsv_path=args.clip_durations_tsv,
    )
    print(f"Wrote {len(frame)} Common Voice metadata rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
