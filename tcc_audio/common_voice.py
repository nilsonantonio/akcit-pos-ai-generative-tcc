"""Prepare Common Voice metadata for downstream processing."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tcc_audio.io import ensure_parent_dir


def _normalize_gender(value: object) -> str:
    normalized = str(value).strip().lower()
    mapping = {
        "male": "masculine",
        "m": "masculine",
        "masculine": "masculine",
        "female": "feminine",
        "f": "feminine",
        "feminine": "feminine",
    }
    return mapping.get(normalized, "unknown")


def _select_target_text(row: pd.Series) -> str:
    text = str(row.get("text", "")).strip()
    if text:
        return text
    return str(row.get("sentence", "")).strip()


def prepare_common_voice_metadata(
    tsv_path: str | Path,
    clips_dir: str | Path,
    out_path: str | Path,
    locale: str | None = None,
    variant: str | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    validated = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)
    if locale and "locale" in validated.columns:
        validated = validated[validated["locale"].eq(locale)].copy()

    if variant:
        if "variant" not in validated.columns:
            raise ValueError(f"Requested variant filter '{variant}' but TSV has no 'variant' column.")
        validated = validated[validated["variant"].eq(variant)].copy()

    if max_rows:
        validated = validated.head(max_rows).copy()

    rows: list[dict[str, object]] = []
    clips_root = Path(clips_dir)
    for index, row in validated.iterrows():
        filename = row.get("path", "").strip()
        if not filename:
            continue
        rows.append(
            {
                "source_speaker_id": row.get("client_id", f"speaker_{index:06d}"),
                "gender": _normalize_gender(row.get("gender", "")),
                "utterance_id": Path(filename).stem,
                "duration_s": "",
                "audio_path": str(clips_root / filename),
                "target_text": _select_target_text(row),
                "license": "CC0-1.0",
                "source": "common_voice_pt",
                "locale": row.get("locale", locale or ""),
                "variant": row.get("variant", variant or ""),
            }
        )

    prepared = pd.DataFrame(rows)
    ensure_parent_dir(out_path)
    prepared.to_csv(out_path, index=False)
    return prepared


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Common Voice metadata for the TCC pipeline.")
    parser.add_argument("--tsv", required=True, help="Path to Common Voice validated.tsv.")
    parser.add_argument("--clips-dir", required=True, help="Path to Common Voice clips directory.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--locale", default="pt", help="Locale filter when the column exists.")
    parser.add_argument("--variant", help="Variant filter, for example pt-BR.")
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
    )
    print(f"Wrote {len(frame)} Common Voice metadata rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
