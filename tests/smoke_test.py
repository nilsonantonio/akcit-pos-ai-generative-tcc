"""Smoke checks for the execution scaffold."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from tcc_audio.common_voice import prepare_common_voice_metadata
from tcc_audio.experiments import generate_run_matrix
from tcc_audio.evaluation import aggregate_metrics
from tcc_audio.human_eval import build_human_eval_pack, import_human_eval_results
from tcc_audio.manifest import validate_prompts
from tcc_audio.report_assets import make_report_assets
from tcc_audio.samples import initialize_samples
from tcc_audio.speaker_selection import select_speakers
from tcc_audio.wer import word_error_rate, compute_wer_from_asr


ROOT = Path(__file__).resolve().parents[1]


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
        tsv_path = tmp / "validated.tsv"
        tsv_path.write_text(
            "client_id\tpath\tsentence\tgender\tlocale\n"
            "spk1\tclip1.mp3\tTexto um.\tmale\tpt\n"
            "spk2\tclip2.mp3\tTexto dois.\tfemale\tpt\n",
            encoding="utf-8",
        )
        prepared = prepare_common_voice_metadata(
            tsv_path=tsv_path,
            clips_dir=tmp / "clips",
            out_path=tmp / "metadata.csv",
            locale="pt",
        )
        assert len(prepared) == 2
        assert set(prepared["gender"]) == {"masculine", "feminine"}
        assert (tmp / "metadata.csv").exists()


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
