"""Dataset inventory reporting for raw, processed, and selected-speaker assets."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from tcc_audio.io import ensure_parent_dir, read_csv, read_yaml

DEFAULT_CONFIG_PATH = Path("configs/speecht5_minimal.yaml")
DEFAULT_RAW_DIR = Path("data/raw/common_voice_pt")
DEFAULT_PROCESSED_DIR = Path("data/processed/common_voice_pt")
DEFAULT_RAW_METADATA = Path("data/manifests/common_voice_metadata.csv")
DEFAULT_PROCESSED_METADATA = Path("data/manifests/common_voice_curated.csv")
DEFAULT_MANIFEST = Path("data/manifests/data_manifest.csv")
DEFAULT_SPEAKER_SELECTION = Path("data/manifests/speaker_selection.csv")
DEFAULT_JSON_OUT = Path("artifacts/dataset_inventory.json")


@dataclass
class InventoryPaths:
    config_path: Path
    raw_dir: Path
    processed_dir: Path
    raw_metadata_path: Path
    processed_metadata_path: Path
    manifest_path: Path
    speaker_selection_path: Path
    json_out: Path


def _strip(value: object, default: str = "") -> str:
    text = str(value).strip()
    return text if text else default


def _read_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def _resolve_audio_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else Path.cwd() / path


def _resolve_lora_scopes(config: Mapping[str, Any]) -> list[str]:
    scopes: list[str] = []
    for condition in config.get("conditions", []):
        if not isinstance(condition, Mapping):
            continue
        train_strategy = str(condition.get("train_strategy", "")).strip().lower()
        if train_strategy != "lora" and "lora" not in condition:
            continue
        scope = str(condition.get("training", {}).get("scope", "per_speaker")).strip().lower()
        if scope in {"per_speaker", "unique"} and scope not in scopes:
            scopes.append(scope)
    return scopes


def resolve_inventory_paths(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    raw_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
    raw_metadata_path: str | Path | None = None,
    processed_metadata_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    speaker_selection_path: str | Path | None = None,
    json_out: str | Path | None = None,
) -> InventoryPaths:
    config_path = Path(config_path)
    config = read_yaml(config_path)
    data = config.get("data", {})
    data = data if isinstance(data, Mapping) else {}
    return InventoryPaths(
        config_path=config_path,
        raw_dir=Path(raw_dir or DEFAULT_RAW_DIR),
        processed_dir=Path(processed_dir or DEFAULT_PROCESSED_DIR),
        raw_metadata_path=Path(raw_metadata_path or DEFAULT_RAW_METADATA),
        processed_metadata_path=Path(processed_metadata_path or DEFAULT_PROCESSED_METADATA),
        manifest_path=Path(manifest_path or data.get("manifest_path") or DEFAULT_MANIFEST),
        speaker_selection_path=Path(speaker_selection_path or data.get("speaker_selection_path") or DEFAULT_SPEAKER_SELECTION),
        json_out=Path(json_out or DEFAULT_JSON_OUT),
    )


def _distributed_sample(paths: list[Path], sample_size: int) -> list[Path]:
    if sample_size <= 0 or not paths:
        return []
    ordered = sorted(paths)
    if len(ordered) <= sample_size:
        return ordered
    indices = sorted(set(np.linspace(0, len(ordered) - 1, num=sample_size, dtype=int).tolist()))
    return [ordered[index] for index in indices]


def _channel_mode(channels: int | None) -> str:
    if channels is None:
        return "unknown"
    if channels == 1:
        return "mono"
    if channels == 2:
        return "stereo"
    return f"{channels} channels"


def _resolve_ffprobe_command() -> list[str]:
    command = shutil.which("ffprobe")
    if not command:
        raise FileNotFoundError("ffprobe not found")
    return [command]


def _probe_audio_file(path: Path) -> dict[str, str]:
    command = _resolve_ffprobe_command() + [
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "format=format_name",
        "-show_entries",
        "stream=sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout or "{}")
    streams = payload.get("streams") or []
    stream = streams[0] if streams else {}
    format_name = _strip((payload.get("format") or {}).get("format_name", ""), default=path.suffix.lstrip(".").lower())
    format_label = _strip(format_name.split(",", 1)[0], default=path.suffix.lstrip(".").lower())
    sample_rate = _strip(stream.get("sample_rate", ""))
    channels = _strip(stream.get("channels", ""))
    sample_rate_khz = "unknown"
    if sample_rate.isdigit():
        sample_rate_khz = f"{int(sample_rate) / 1000.0:.1f}"
    channel_mode = _channel_mode(int(channels)) if channels.isdigit() else "unknown"
    return {
        "format": format_label or "unknown",
        "sample_rate_khz": sample_rate_khz,
        "channel_mode": channel_mode,
    }


def _fallback_probe(path: Path) -> dict[str, str]:
    return {
        "format": _strip(path.suffix.lstrip(".").lower(), default="unknown"),
        "sample_rate_khz": "unknown",
        "channel_mode": "unknown",
    }


def _sort_metadata_values(values: Iterable[str]) -> list[str]:
    def _key(value: str) -> tuple[int, float | str]:
        if value == "unknown":
            return (1, value)
        try:
            return (0, float(value))
        except ValueError:
            return (0, value)

    return sorted({value for value in values if _strip(value)}, key=_key)


def _probe_audio_collection(paths: Iterable[Path], sample_size: int) -> tuple[dict[str, Any], list[str]]:
    sampled_paths = _distributed_sample(list(paths), sample_size)
    warnings: list[str] = []
    if not sampled_paths:
        return (
            {
                "formats": [],
                "sample_rates_khz": [],
                "channel_modes": [],
                "sampled_files": 0,
            },
            warnings,
        )

    metadata_rows: list[dict[str, str]] = []
    ffprobe_unavailable = False
    for path in sampled_paths:
        try:
            metadata_rows.append(_probe_audio_file(path))
        except FileNotFoundError:
            ffprobe_unavailable = True
            metadata_rows.append(_fallback_probe(path))
        except subprocess.CalledProcessError:
            warnings.append(f"ffprobe failed for {path}")
            metadata_rows.append(_fallback_probe(path))

    if ffprobe_unavailable:
        warnings.append("ffprobe not available; sample rates and channel modes were reported as unknown.")

    return (
        {
            "formats": _sort_metadata_values(row["format"] for row in metadata_rows),
            "sample_rates_khz": _sort_metadata_values(row["sample_rate_khz"] for row in metadata_rows),
            "channel_modes": _sort_metadata_values(row["channel_mode"] for row in metadata_rows),
            "sampled_files": len(sampled_paths),
        },
        warnings,
    )


def _raw_group_key(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].fillna("").astype(str).str.strip().replace("", "unknown")


def _build_raw_inventory(raw_dir: Path, sample_size: int) -> tuple[dict[str, Any], list[str]]:
    validated_tsv = raw_dir / "validated.tsv"
    clips_dir = raw_dir / "clips"
    validated = _read_tsv(validated_tsv)
    clip_paths = sorted(path for path in clips_dir.rglob("*") if path.is_file())
    clip_relative = {path.relative_to(clips_dir).as_posix(): path for path in clip_paths}
    validated["path"] = validated["path"].fillna("").astype(str).str.strip()
    validated_rows = validated[validated["path"].ne("")].copy()
    validated_rows["locale_group"] = _raw_group_key(validated_rows, "locale")
    validated_rows["variant_group"] = _raw_group_key(validated_rows, "variant")
    validated_rows["gender_group"] = _raw_group_key(validated_rows, "gender")

    validated_clip_keys = set(validated_rows["path"].tolist())
    existing_validated = sorted(validated_clip_keys.intersection(clip_relative))
    non_validated = sorted(set(clip_relative).difference(validated_clip_keys))

    grouped_rows: list[dict[str, Any]] = []
    for keys, group in validated_rows.groupby(["locale_group", "variant_group", "gender_group"], dropna=False):
        locale, variant, gender = keys
        grouped_rows.append(
            {
                "locale": locale,
                "variant": variant,
                "gender": gender,
                "clip_count": int(group["path"].nunique()),
                "unique_speakers": int(group["client_id"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().nunique()),
            }
        )

    audio_metadata, warnings = _probe_audio_collection(clip_paths, sample_size)
    inventory = {
        "grouped_by_locale_variant_gender": sorted(
            grouped_rows,
            key=lambda row: (row["locale"], row["variant"], row["gender"]),
        ),
        "overall_totals": {
            "total_clips": len(clip_paths),
            "validated_clips": len(existing_validated),
            "non_validated_clips": len(non_validated),
            "unique_speakers": int(
                validated_rows["client_id"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().nunique()
            ),
            **audio_metadata,
        },
        "warnings": warnings,
    }
    return inventory, warnings


def _speaker_stats(frame: pd.DataFrame, speaker_column: str) -> pd.DataFrame:
    stats = (
        frame.groupby(speaker_column, dropna=False)
        .agg(
            clip_count=("audio_path", "count"),
            total_duration_s=("duration_s", lambda values: float(pd.to_numeric(values, errors="coerce").sum())),
        )
        .reset_index()
    )
    return stats


def _extreme_speaker(stats: pd.DataFrame, speaker_column: str, ascending: bool) -> dict[str, Any]:
    ordered = stats.sort_values(
        ["clip_count", "total_duration_s", speaker_column],
        ascending=[ascending, ascending, True],
    ).reset_index(drop=True)
    row = ordered.iloc[0]
    return {
        "speaker_id": str(row[speaker_column]),
        "clip_count": int(row["clip_count"]),
        "total_duration_s": float(row["total_duration_s"]),
    }


def _build_processed_inventory(processed_metadata_path: Path, sample_size: int) -> tuple[dict[str, Any], list[str]]:
    processed = read_csv(processed_metadata_path)
    if "audio_path" not in processed.columns or "source_speaker_id" not in processed.columns or "duration_s" not in processed.columns:
        raise ValueError("Processed metadata must contain audio_path, source_speaker_id and duration_s.")

    audio_paths = [_resolve_audio_path(value) for value in processed["audio_path"].tolist() if _strip(value)]
    audio_metadata, warnings = _probe_audio_collection(audio_paths, sample_size)
    stats = _speaker_stats(processed, "source_speaker_id")
    inventory = {
        "speaker_count": int(processed["source_speaker_id"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().nunique()),
        "clip_count": int(len(processed)),
        "total_duration_s": float(pd.to_numeric(processed["duration_s"], errors="coerce").sum()),
        "lowest_clip_speaker": _extreme_speaker(stats, "source_speaker_id", ascending=True),
        "highest_clip_speaker": _extreme_speaker(stats, "source_speaker_id", ascending=False),
        **audio_metadata,
        "warnings": warnings,
    }
    return inventory, warnings


def _summarize_manifest_rows(frame: pd.DataFrame, sample_size: int) -> tuple[dict[str, Any], list[str]]:
    train_mask = frame["split"].astype(str).str.lower().eq("train")
    val_mask = frame["split"].astype(str).str.lower().eq("val")
    audio_paths = [_resolve_audio_path(value) for value in frame["audio_path"].tolist() if _strip(value)]
    audio_metadata, warnings = _probe_audio_collection(audio_paths, sample_size)
    return (
        {
            "train_clip_count": int(train_mask.sum()),
            "val_clip_count": int(val_mask.sum()),
            "train_duration_s": float(pd.to_numeric(frame.loc[train_mask, "duration_s"], errors="coerce").sum()),
            "val_duration_s": float(pd.to_numeric(frame.loc[val_mask, "duration_s"], errors="coerce").sum()),
            **audio_metadata,
        },
        warnings,
    )


def _build_selected_speakers_inventory(
    manifest_path: Path,
    config: Mapping[str, Any],
    sample_size: int,
) -> tuple[dict[str, Any], list[str]]:
    manifest = read_csv(manifest_path)
    if "audio_path" not in manifest.columns or "speaker_id" not in manifest.columns or "duration_s" not in manifest.columns:
        raise ValueError("Manifest must contain audio_path, speaker_id and duration_s.")

    warnings: list[str] = []
    inventory: dict[str, Any] = {}
    scopes = _resolve_lora_scopes(config)

    if "per_speaker" in scopes:
        speakers: list[dict[str, Any]] = []
        for speaker_id in sorted(manifest["speaker_id"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().unique()):
            speaker_rows = manifest[manifest["speaker_id"].eq(speaker_id)].copy()
            speaker_summary, speaker_warnings = _summarize_manifest_rows(speaker_rows, sample_size)
            warnings.extend(speaker_warnings)
            speakers.append({"speaker_id": speaker_id, **speaker_summary})
        totals, total_warnings = _summarize_manifest_rows(manifest, sample_size)
        warnings.extend(total_warnings)
        inventory["per_speaker"] = {
            "speakers": speakers,
            "totals": {
                "speaker_count": int(len(speakers)),
                **totals,
            },
        }

    if "unique" in scopes:
        totals, total_warnings = _summarize_manifest_rows(manifest, sample_size)
        warnings.extend(total_warnings)
        inventory["unique"] = {
            "speaker_count": int(manifest["speaker_id"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().nunique()),
            **totals,
        }

    inventory["warnings"] = sorted(set(warnings))
    return inventory, inventory["warnings"]


def build_dataset_inventory(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    raw_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
    raw_metadata_path: str | Path | None = None,
    processed_metadata_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    speaker_selection_path: str | Path | None = None,
    json_out: str | Path | None = None,
    sample_size: int = 50,
) -> dict[str, Any]:
    paths = resolve_inventory_paths(
        config_path=config_path,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        raw_metadata_path=raw_metadata_path,
        processed_metadata_path=processed_metadata_path,
        manifest_path=manifest_path,
        speaker_selection_path=speaker_selection_path,
        json_out=json_out,
    )
    config = read_yaml(paths.config_path)

    raw_inventory, raw_warnings = _build_raw_inventory(paths.raw_dir, sample_size)
    processed_inventory, processed_warnings = _build_processed_inventory(paths.processed_metadata_path, sample_size)
    selected_inventory, selected_warnings = _build_selected_speakers_inventory(paths.manifest_path, config, sample_size)

    inventory = {
        "raw": raw_inventory,
        "processed": processed_inventory,
        "selected_speakers": selected_inventory,
        "sampling": {
            "sample_size_requested": int(sample_size),
            "sample_size_effective_raw": int(raw_inventory["overall_totals"]["sampled_files"]),
            "sample_size_effective_processed": int(processed_inventory["sampled_files"]),
        },
        "paths": {
            "config": str(paths.config_path),
            "raw_dir": str(paths.raw_dir),
            "processed_dir": str(paths.processed_dir),
            "raw_metadata": str(paths.raw_metadata_path),
            "processed_metadata": str(paths.processed_metadata_path),
            "manifest": str(paths.manifest_path),
            "speaker_selection": str(paths.speaker_selection_path),
            "json_out": str(paths.json_out),
        },
        "warnings": sorted(set(raw_warnings + processed_warnings + selected_warnings)),
    }

    ensure_parent_dir(paths.json_out)
    paths.json_out.write_text(json.dumps(inventory, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return inventory


def _list_value(values: Iterable[str]) -> str:
    items = list(values)
    return ", ".join(items) if items else "n/a"


def render_dataset_inventory(inventory: Mapping[str, Any]) -> str:
    raw = inventory["raw"]
    processed = inventory["processed"]
    selected = inventory["selected_speakers"]
    lines = [
        "RAW DATASET",
        f"total_clips={raw['overall_totals']['total_clips']} validated_clips={raw['overall_totals']['validated_clips']} non_validated_clips={raw['overall_totals']['non_validated_clips']} unique_speakers={raw['overall_totals']['unique_speakers']}",
        f"formats={_list_value(raw['overall_totals']['formats'])} sample_rates_khz={_list_value(raw['overall_totals']['sample_rates_khz'])} channel_modes={_list_value(raw['overall_totals']['channel_modes'])}",
        "grouped_by_locale_variant_gender:",
    ]
    for row in raw["grouped_by_locale_variant_gender"]:
        lines.append(
            f"- locale={row['locale']} variant={row['variant']} gender={row['gender']} clip_count={row['clip_count']} unique_speakers={row['unique_speakers']}"
        )

    lines.extend(
        [
            "",
            "PROCESSED DATASET",
            f"speaker_count={processed['speaker_count']} clip_count={processed['clip_count']} total_duration_s={processed['total_duration_s']:.3f}",
            f"lowest_clip_speaker={processed['lowest_clip_speaker']['speaker_id']} clips={processed['lowest_clip_speaker']['clip_count']} duration_s={processed['lowest_clip_speaker']['total_duration_s']:.3f}",
            f"highest_clip_speaker={processed['highest_clip_speaker']['speaker_id']} clips={processed['highest_clip_speaker']['clip_count']} duration_s={processed['highest_clip_speaker']['total_duration_s']:.3f}",
            f"formats={_list_value(processed['formats'])} sample_rates_khz={_list_value(processed['sample_rates_khz'])} channel_modes={_list_value(processed['channel_modes'])}",
            "",
            "SELECTED SPEAKERS",
        ]
    )

    if "per_speaker" in selected:
        lines.append("per_speaker:")
        for row in selected["per_speaker"]["speakers"]:
            lines.append(
                f"- speaker_id={row['speaker_id']} train_clip_count={row['train_clip_count']} val_clip_count={row['val_clip_count']} train_duration_s={row['train_duration_s']:.3f} val_duration_s={row['val_duration_s']:.3f} formats={_list_value(row['formats'])} sample_rates_khz={_list_value(row['sample_rates_khz'])} channel_modes={_list_value(row['channel_modes'])}"
            )
        totals = selected["per_speaker"]["totals"]
        lines.append(
            f"- totals speaker_count={totals['speaker_count']} train_clip_count={totals['train_clip_count']} val_clip_count={totals['val_clip_count']} train_duration_s={totals['train_duration_s']:.3f} val_duration_s={totals['val_duration_s']:.3f} formats={_list_value(totals['formats'])} sample_rates_khz={_list_value(totals['sample_rates_khz'])} channel_modes={_list_value(totals['channel_modes'])}"
        )

    if "unique" in selected:
        unique = selected["unique"]
        lines.append("unique:")
        lines.append(
            f"- speaker_count={unique['speaker_count']} train_clip_count={unique['train_clip_count']} val_clip_count={unique['val_clip_count']} train_duration_s={unique['train_duration_s']:.3f} val_duration_s={unique['val_duration_s']:.3f} formats={_list_value(unique['formats'])} sample_rates_khz={_list_value(unique['sample_rates_khz'])} channel_modes={_list_value(unique['channel_modes'])}"
        )

    warnings = list(inventory.get("warnings", []))
    if warnings:
        lines.append("")
        lines.append("WARNINGS")
        for warning in warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Log a hierarchical inventory of raw, processed, and selected-speaker datasets.")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--raw-metadata", default=str(DEFAULT_RAW_METADATA))
    parser.add_argument("--processed-metadata", default=str(DEFAULT_PROCESSED_METADATA))
    parser.add_argument("--manifest")
    parser.add_argument("--speaker-selection")
    parser.add_argument("-j", "--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--sample-size", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    inventory = build_dataset_inventory(
        config_path=args.config,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        raw_metadata_path=args.raw_metadata,
        processed_metadata_path=args.processed_metadata,
        manifest_path=args.manifest,
        speaker_selection_path=args.speaker_selection,
        json_out=args.json_out,
        sample_size=args.sample_size,
    )
    print(render_dataset_inventory(inventory), end="")
    print(f"Wrote JSON inventory to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
