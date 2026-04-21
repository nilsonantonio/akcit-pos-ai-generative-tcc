"""Download and stage Common Voice PT raw assets from Mozilla Data Collective."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = "https://mozilladatacollective.com/api"
DEFAULT_DATASET_ID = "cmn29f4cb017bmm07pd9yd8mw"
DEFAULT_DATASET_SLUG = "common-voice-scripted-speech-25-0-portug-0254cce0"
DEFAULT_API_KEY_ENV = "MOZILLA_DATA_COLLECTIVE_API_KEY"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class DownloadSession:
    download_url: str
    filename: str
    checksum: str = ""
    size_bytes: int | None = None


@dataclass(frozen=True)
class StagedCommonVoicePaths:
    archive_path: Path
    clips_dir: Path
    validated_tsv: Path


def _require_api_key(env_var: str) -> str:
    #api_key = os.environ.get(env_var, "").strip()
    api_key = os.getenv(env_var, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Export {env_var} with a Mozilla Data Collective bearer token first."
        )
    return api_key


def _parse_json_response(response: object) -> dict[str, object]:
    payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise RuntimeError("Mozilla Data Collective API returned a non-object JSON response.")
    return data


def _request_json(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    data: bytes | None = None,
) -> dict[str, object]:
    req = request.Request(url, method=method, headers=headers, data=data)
    try:
        with request.urlopen(req) as response:
            return _parse_json_response(response)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        message = body or exc.reason
        raise RuntimeError(f"Mozilla Data Collective API request failed ({exc.code}): {message}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Mozilla Data Collective API request failed: {exc.reason}") from exc


def _parse_size_bytes(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(str(value))


def _checksum_sha256(checksum: str) -> str | None:
    if not checksum:
        return None
    prefix, separator, digest = checksum.partition(":")
    if separator and prefix.lower() == "sha256":
        return digest.strip().lower()
    return None


def request_dataset_download_session(
    dataset_id: str,
    *,
    api_key: str,
    dataset_slug: str = DEFAULT_DATASET_SLUG,
    api_base_url: str = API_BASE_URL,
) -> DownloadSession:
    payload = _request_json(
        f"{api_base_url}/datasets/{dataset_id}/download",
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=b"{}",
    )
    download_url = str(payload.get("downloadUrl", "")).strip()
    if not download_url:
        raise RuntimeError("Mozilla Data Collective API response did not include downloadUrl.")
    filename = str(payload.get("filename", "")).strip() or f"{dataset_slug}.tar.gz"
    checksum = str(payload.get("checksum", "")).strip()
    size_bytes = _parse_size_bytes(payload.get("sizeBytes"))
    return DownloadSession(
        download_url=download_url,
        filename=filename,
        checksum=checksum,
        size_bytes=size_bytes,
    )


def download_archive(
    session: DownloadSession,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    destination_path = Path(destination)
    if destination_path.exists() and not overwrite:
        return destination_path

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination_path.with_name(f"{destination_path.name}.part")
    if temp_path.exists():
        temp_path.unlink()

    expected_sha256 = _checksum_sha256(session.checksum)
    digest = hashlib.sha256()
    bytes_written = 0

    try:
        with request.urlopen(session.download_url) as response, temp_path.open("wb") as output:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                bytes_written += len(chunk)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    if session.size_bytes is not None and bytes_written != session.size_bytes:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded archive size mismatch: expected {session.size_bytes} bytes, got {bytes_written}."
        )

    if expected_sha256 and digest.hexdigest().lower() != expected_sha256:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError("Downloaded archive checksum mismatch for Common Voice PT dataset.")

    if destination_path.exists():
        destination_path.unlink()
    temp_path.replace(destination_path)
    return destination_path


def _select_candidate(paths: list[Path], *, label: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"Could not find {label} in extracted Common Voice archive.")
    return min(paths, key=lambda path: (len(path.parts), str(path)))


def _safe_extract_archive(archive_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            target_path = (destination / member.name).resolve()
            if not target_path.is_relative_to(destination_resolved):
                raise RuntimeError(f"Unsafe archive entry detected: {member.name}")
        archive.extractall(destination)


def _replace_path(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    shutil.move(str(source), str(destination))


def _resolve_existing_archive(out_dir: Path, dataset_slug: str) -> Path | None:
    canonical_archive = out_dir / f"{dataset_slug}.tar.gz"
    if canonical_archive.exists():
        return canonical_archive

    archives = sorted(out_dir.glob("*.tar.gz"))
    if len(archives) == 1:
        return archives[0]
    return None


def stage_common_voice_archive(
    archive_path: str | Path,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
) -> StagedCommonVoicePaths:
    archive = Path(archive_path)
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    clips_dir = out_root / "clips"
    validated_tsv = out_root / "validated.tsv"

    if not overwrite and (clips_dir.exists() or validated_tsv.exists()):
        raise FileExistsError(
            f"Refusing to overwrite existing extraction target in {out_root}. Use --force-extract."
        )

    with tempfile.TemporaryDirectory(prefix="common_voice_extract_", dir=out_root.parent) as tmpdir:
        extract_root = Path(tmpdir)
        _safe_extract_archive(archive, extract_root)

        clips_source = _select_candidate(
            [path for path in extract_root.rglob("clips") if path.is_dir()],
            label="clips directory",
        )
        validated_source = _select_candidate(
            [path for path in extract_root.rglob("validated.tsv") if path.is_file()],
            label="validated.tsv file",
        )

        _replace_path(clips_source, clips_dir)
        _replace_path(validated_source, validated_tsv)

    return StagedCommonVoicePaths(
        archive_path=archive,
        clips_dir=clips_dir,
        validated_tsv=validated_tsv,
    )


def download_common_voice_pt(
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_slug: str = DEFAULT_DATASET_SLUG,
    out_dir: str | Path = "data/raw/common_voice_pt",
    api_key_env: str = DEFAULT_API_KEY_ENV,
    force_download: bool = False,
    force_extract: bool = False,
) -> StagedCommonVoicePaths:
    api_key = _require_api_key(api_key_env)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    archive_path = _resolve_existing_archive(out_root, dataset_slug)
    if force_download or archive_path is None:
        session = request_dataset_download_session(
            dataset_id,
            api_key=api_key,
            dataset_slug=dataset_slug,
        )
        archive_path = out_root / session.filename
        download_archive(session, archive_path, overwrite=force_download)

    return stage_common_voice_archive(archive_path, out_root, overwrite=force_extract)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and stage Common Voice PT raw assets.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG)
    parser.add_argument("--out-dir", default="data/raw/common_voice_pt")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        staged = download_common_voice_pt(
            dataset_id=args.dataset_id,
            dataset_slug=args.dataset_slug,
            out_dir=args.out_dir,
            api_key_env=args.api_key_env,
            force_download=args.force_download,
            force_extract=args.force_extract,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Archive: {staged.archive_path}")
    print(f"Clips: {staged.clips_dir}")
    print(f"Validated TSV: {staged.validated_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
