"""Validation for the data and prompt manifests."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tcc_audio.io import read_csv
from tcc_audio.schema import (
    DATA_MANIFEST_REQUIRED_COLUMNS,
    PROMPTS_REQUIRED_COLUMNS,
    VALID_SPLITS,
    VALID_TEXT_VARIANTS,
)


@dataclass
class ValidationReport:
    path: Path
    row_count: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, str | int | float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _missing_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [column for column in required if column not in df.columns]


def _is_blank(value: object) -> bool:
    return str(value).strip() == ""


def validate_prompts(path: str | Path) -> ValidationReport:
    prompt_path = Path(path)
    prompts = read_csv(prompt_path)
    report = ValidationReport(path=prompt_path, row_count=len(prompts))

    missing = _missing_columns(prompts, PROMPTS_REQUIRED_COLUMNS)
    if missing:
        report.errors.append(f"Missing prompt columns: {', '.join(missing)}")
        return report

    if len(prompts) != 24:
        report.warnings.append(f"Expected 24 prompts, found {len(prompts)}")

    duplicated = prompts["prompt_id"][prompts["prompt_id"].duplicated()].unique()
    if len(duplicated):
        report.errors.append(f"Duplicated prompt_id values: {', '.join(duplicated)}")

    categories = sorted(prompts["category"].unique())
    report.summary["prompt_categories"] = ", ".join(categories)
    report.summary["commercial_subset_count"] = int(
        prompts["include_commercial_subset"].str.lower().eq("true").sum()
    )
    report.summary["prompt_subexperiment_count"] = int(
        prompts["include_prompt_subexperiment"].str.lower().eq("true").sum()
    )
    return report


def validate_data_manifest(
    path: str | Path,
    prompts_path: str | Path | None = None,
    check_files: bool = False,
    project_root: str | Path = ".",
) -> ValidationReport:
    manifest_path = Path(path)
    manifest = read_csv(manifest_path)
    report = ValidationReport(path=manifest_path, row_count=len(manifest))

    missing = _missing_columns(manifest, DATA_MANIFEST_REQUIRED_COLUMNS)
    if missing:
        report.errors.append(f"Missing manifest columns: {', '.join(missing)}")
        return report

    if manifest.empty:
        report.warnings.append("Manifest has only the header; populate it after speaker selection.")
        return report

    for row_number, row in manifest.iterrows():
        line = row_number + 2
        for column in DATA_MANIFEST_REQUIRED_COLUMNS:
            if _is_blank(row[column]):
                report.errors.append(f"Line {line}: blank required field '{column}'")

        split = row["split"].strip()
        if split and split not in VALID_SPLITS:
            report.errors.append(f"Line {line}: invalid split '{split}'")

        variant = row["text_variant"].strip()
        if variant and variant not in VALID_TEXT_VARIANTS:
            report.errors.append(f"Line {line}: invalid text_variant '{variant}'")

        try:
            duration = float(row["duration_s"])
            if duration <= 0:
                report.errors.append(f"Line {line}: duration_s must be > 0")
        except ValueError:
            report.errors.append(f"Line {line}: duration_s must be numeric")

        if "reference_duration_s" in manifest.columns and not _is_blank(row["reference_duration_s"]):
            try:
                ref_duration = float(row["reference_duration_s"])
                if ref_duration < 3 or ref_duration > 10:
                    report.warnings.append(
                        f"Line {line}: reference_duration_s should be between 3 and 10 seconds"
                    )
            except ValueError:
                report.errors.append(f"Line {line}: reference_duration_s must be numeric")

        if check_files:
            for path_field in ["audio_path", "reference_audio"]:
                audio_path = Path(row[path_field])
                if not audio_path.is_absolute():
                    audio_path = Path(project_root) / audio_path
                if not audio_path.exists():
                    report.errors.append(f"Line {line}: {path_field} not found: {audio_path}")

    duplicated_keys = manifest[["speaker_id", "utterance_id", "split"]].duplicated()
    if duplicated_keys.any():
        report.errors.append("Duplicated speaker_id + utterance_id + split combinations found")

    report.summary["speaker_count"] = int(manifest["speaker_id"].nunique())
    report.summary["total_duration_s"] = float(pd.to_numeric(manifest["duration_s"], errors="coerce").sum())

    if prompts_path:
        prompt_report = validate_prompts(prompts_path)
        report.errors.extend([f"Prompts: {error}" for error in prompt_report.errors])
        report.warnings.extend([f"Prompts: {warning}" for warning in prompt_report.warnings])
        report.summary.update({f"prompts_{key}": value for key, value in prompt_report.summary.items()})

    return report


def print_report(report: ValidationReport) -> None:
    status = "OK" if report.ok else "FAILED"
    print(f"{status}: {report.path} ({report.row_count} rows)")
    for key, value in sorted(report.summary.items()):
        print(f"summary.{key}={value}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the TCC data manifest.")
    parser.add_argument("--manifest", required=True, help="Path to data manifest CSV.")
    parser.add_argument("--prompts", help="Path to PT-BR prompts CSV.")
    parser.add_argument("--check-files", action="store_true", help="Check reference_audio paths.")
    parser.add_argument("--project-root", default=".", help="Root for relative audio paths.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = validate_data_manifest(
        args.manifest,
        prompts_path=args.prompts,
        check_files=args.check_files,
        project_root=args.project_root,
    )
    print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
