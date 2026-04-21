"""Smoke checks for the execution scaffold."""

import io
import os
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcc_audio.common_voice import prepare_common_voice_metadata
from tcc_audio.common_voice_download import (
    download_common_voice_pt,
    request_dataset_download_session,
    stage_common_voice_archive,
)
from tcc_audio.experiments import generate_run_matrix
from tcc_audio.evaluation import aggregate_metrics
from tcc_audio.human_eval import build_human_eval_pack, import_human_eval_results
from tcc_audio.manifest import validate_prompts
from tcc_audio.report_assets import make_report_assets
from tcc_audio.samples import initialize_samples
from tcc_audio.speaker_selection import select_speakers
from tcc_audio.wer import word_error_rate, compute_wer_from_asr


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


def test_prompt_file_has_expected_contract() -> None:
    report = validate_prompts(ROOT / "data/prompts/ptbr_test_prompts.csv")
    assert report.ok, report.errors
    assert report.row_count == 24
    assert report.summary["commercial_subset_count"] == 8


def test_run_matrix_generation() -> None:
    with TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "run_matrix.csv"
        matrix = generate_run_matrix(ROOT / "configs/speecht5_minimal.yaml", out)
        assert out.exists()
        assert {"speecht5_zero_shot", "speecht5_few_shot_decoder_ft", "speecht5_lora"}.issubset(
            set(matrix["condition"])
        )
        assert "parler_reference_optional" in set(matrix["condition"])
        assert len(matrix[matrix["condition"].eq("parler_reference_optional")]) == 16
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


def test_speaker_selection_from_curated_metadata() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        rows = []
        for gender in ["masculine", "feminine"]:
            for speaker_index in range(2):
                source_speaker_id = f"{gender}_{speaker_index}"
                for utterance_index in range(12):
                    rows.append(
                        {
                            "source_speaker_id": source_speaker_id,
                            "gender": gender,
                            "utterance_id": f"{source_speaker_id}_{utterance_index}",
                            "duration_s": 120,
                            "audio_path": f"audio/{source_speaker_id}_{utterance_index}.wav",
                            "target_text": "Texto de teste para selecao de speaker.",
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
            speakers_per_gender=2,
            minutes_per_speaker=20,
        )
        assert speaker_selection["speaker_id"].nunique() == 4
        assert manifest["speaker_id"].nunique() == 4
        assert "audio_path" in manifest.columns
        assert "parler_description" in speaker_selection.columns


def test_metric_aggregation_contract() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        samples_path = tmp / "samples.csv"
        rows = []
        for condition, wer, similarity, nisqa in [
            ("speecht5_zero_shot", 0.31, 0.62, 3.1),
            ("speecht5_few_shot_decoder_ft", 0.22, 0.71, 3.5),
            ("speecht5_lora", 0.18, 0.76, 3.7),
        ]:
            rows.append(
                {
                    "sample_id": f"{condition}_sample",
                    "run_id": f"{condition}__speaker_01__P001__normalized",
                    "condition": condition,
                    "speaker_id": "speaker_01",
                    "prompt_id": "P001",
                    "text_variant": "normalized",
                    "target_text": "Texto de teste.",
                    "audio_path": "audio.wav",
                    "reference_audio_path": "reference.wav",
                    "speaker_embedding_path": "embedding.npy",
                    "model_name": condition,
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
                    "lora_gate_status": "stable_lora" if condition == "speecht5_lora" else "",
                }
            )
        pd.DataFrame(rows).to_csv(samples_path, index=False)
        metrics, costs = aggregate_metrics(samples_path, tmp / "evaluation")
        assert not metrics.empty
        assert costs["samples"].sum() == 3
        assert (tmp / "evaluation/metrics_summary.csv").exists()
        assert (tmp / "evaluation/cost_summary.csv").exists()


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
                    "condition": "speecht5_zero_shot",
                    "speaker_id": "speaker_01",
                    "prompt_id": "P001",
                    "text_variant": "normalized",
                    "target_text": "a voz ficou clara",
                    "audio_path": "audio.wav",
                    "reference_audio_path": "reference.wav",
                    "speaker_embedding_path": "embedding.npy",
                    "model_name": "speecht5_zero_shot",
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


def test_human_eval_and_report_assets() -> None:
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        samples_path = tmp / "samples.csv"
        rows = []
        for condition in ["speecht5_zero_shot", "speecht5_few_shot_decoder_ft", "speecht5_lora"]:
            rows.append(
                {
                    "sample_id": f"{condition}_sample",
                    "run_id": f"{condition}__speaker_01__P001__normalized",
                    "condition": condition,
                    "speaker_id": "speaker_01",
                    "prompt_id": "P001",
                    "text_variant": "normalized",
                    "target_text": "Texto de teste.",
                    "audio_path": __file__,
                    "reference_audio_path": __file__,
                    "speaker_embedding_path": "embedding.npy",
                    "model_name": condition,
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
                    "lora_gate_status": "stable_lora" if condition == "speecht5_lora" else "",
                    "asr_text": "Texto de teste.",
                }
            )
        pd.DataFrame(rows).to_csv(samples_path, index=False)

        pack = build_human_eval_pack(samples_path, tmp / "human_eval_pack")
        assert not pack.empty
        results = tmp / "human_eval_results.csv"
        filled = pack.copy()
        filled["mos_left"] = 4
        filled["mos_right"] = 3
        filled["smos_left"] = 4
        filled["smos_right"] = 3
        filled["preferred_condition"] = filled["left_condition"]
        filled.to_csv(results, index=False)
        human_summary = import_human_eval_results(results, tmp / "human_eval_summary.csv")
        assert not human_summary.empty

        metrics = pd.DataFrame(
            [
                {"condition": "speecht5_zero_shot", "text_variant": "normalized", "metric": "wer", "n": 1, "mean": 0.3, "median": 0.3, "std": 0, "ci95_low": 0.3, "ci95_high": 0.3, "test": "", "statistic": "", "p_value": ""},
                {"condition": "speecht5_lora", "text_variant": "normalized", "metric": "wer", "n": 1, "mean": 0.1, "median": 0.1, "std": 0, "ci95_low": 0.1, "ci95_high": 0.1, "test": "", "statistic": "", "p_value": ""},
            ]
        )
        costs = pd.DataFrame(
            [
                {"condition": "speecht5_zero_shot", "samples": 1, "total_cost_usd": 0.05, "total_train_gpu_hours": 0, "mean_rtf": 1.0, "mean_inference_seconds": 1.0},
                {"condition": "speecht5_lora", "samples": 1, "total_cost_usd": 0.10, "total_train_gpu_hours": 0.2, "mean_rtf": 0.8, "mean_inference_seconds": 0.8},
            ]
        )
        metrics_path = tmp / "metrics.csv"
        costs_path = tmp / "costs.csv"
        metrics.to_csv(metrics_path, index=False)
        costs.to_csv(costs_path, index=False)
        outputs = make_report_assets(samples_path, metrics_path, costs_path, tmp / "human_eval_summary.csv", tmp / "report_assets")
        assert (tmp / "report_assets/overview.md").exists()
        assert "metrics_markdown" in outputs


BOOTSTRAP_SMOKE_TESTS = (
    test_prompt_file_has_expected_contract,
    test_run_matrix_generation,
    test_speaker_selection_from_curated_metadata,
    test_metric_aggregation_contract,
    test_wer_computation,
    test_prepare_common_voice_metadata,
    test_human_eval_and_report_assets,
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
