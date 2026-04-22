# TCC Audio Execution Plan

Base executavel para a parte pratica do TCC em IA generativa de audio.

O projeto implementa o plano operacional atualizado:

- eixo principal em `SpeechT5`: `zero-shot`, `few-shot decoder fine-tune` e `LoRA`;
- `Whisper` sem ajuste como avaliador de WER;
- `SpeechBrain ECAPA` para embeddings e similaridade de speaker;
- `Parler-TTS` como benchmark externo opcional para demo;
- `BRSpeech-DF` como evidencia contextual, nao como braco principal.

## Estrutura

- `configs/speecht5_minimal.yaml`: configuracao oficial do desenho experimental.
- `data/prompts/ptbr_test_prompts.csv`: 24 prompts fixos em PT-BR, com texto cru e normalizado.
- `data/manifests/data_manifest_template.csv`: schema do manifesto unico de dados.
- `data/manifests/speaker_selection_template.csv`: template para selecao dos 4 speakers.
- `tcc_audio/`: pacote com validacao, matriz de runs, inferencia, metricas e relatorios.
- `scripts/`: wrappers de linha de comando.
- `demo/app.py`: demo Gradio minima para comparar amostras A/B/C.
- `MANUAL_EXECUCAO.md`: manual completo por ambiente.
- `tests/smoke_test.py`: checagens locais sem modelos nem datasets.

## Fluxo recomendado

1. Aceite os termos do dataset no site do Mozilla Data Collective e exporte o token:

```bash
export MOZILLA_DATA_COLLECTIVE_API_KEY="seu_token_aqui"
```

2. Baixe e organize o `Common Voice PT` no layout canonico do repo:

```bash
python3 scripts/download_common_voice_pt.py \
  --out-dir data/raw/common_voice_pt
```

3. Prepare os metadados do subconjunto `pt-BR`:

```bash
python3 scripts/prepare_common_voice_metadata.py \
  --tsv data/raw/common_voice_pt/validated.tsv \
  --clips-dir data/raw/common_voice_pt/clips \
  --locale pt \
  --variant pt-BR \
  --out data/manifests/common_voice_metadata.csv
```

4. Converta o audio para WAV mono e calcule duracoes:

```bash
python3 scripts/preprocess_audio_dataset.py \
  --metadata data/manifests/common_voice_metadata.csv \
  --out-dir data/processed/common_voice_pt \
  --out-metadata data/manifests/common_voice_curated.csv
```

5. Gere o manifesto real a partir do subset curado de Common Voice.

```bash
python3 scripts/select_speakers.py \
  --metadata data/manifests/common_voice_curated.csv \
  --manifest-out data/manifests/data_manifest.csv \
  --speaker-selection-out data/manifests/speaker_selection.csv
```

6. Valide o manifesto:

```bash
python3 scripts/validate_manifest.py \
  --manifest data/manifests/data_manifest.csv \
  --prompts data/prompts/ptbr_test_prompts.csv
```

7. Gere embeddings de speaker:

```bash
python3 scripts/extract_speaker_embeddings.py \
  --config configs/speecht5_minimal.yaml \
  --speaker-selection data/manifests/speaker_selection.csv \
  --out-index artifacts/embeddings/speaker_embeddings.csv \
  --out-dir artifacts/embeddings
```

Esse passo prepara os embeddings consumidos pela sintese do SpeechT5. O `speaker_embedding_path` do `samples.csv` aponta para esses embeddings de sintese.

8. Gere a matriz de execucao:

```bash
python3 scripts/generate_run_matrix.py \
  --config configs/speecht5_minimal.yaml \
  --out artifacts/run_matrix.csv
```

9. Inicialize o ledger de amostras:

```bash
python3 scripts/init_samples.py \
  --run-matrix artifacts/run_matrix.csv \
  --speaker-selection data/manifests/speaker_selection.csv \
  --speaker-embeddings artifacts/embeddings/speaker_embeddings.csv \
  --out artifacts/evaluation/samples.csv
```

10. Rode os modelos:

```bash
python3 scripts/run_speecht5_zero_shot.py \
  --config configs/speecht5_minimal.yaml \
  --samples artifacts/evaluation/samples.csv
```

```bash
python3 scripts/run_speecht5_few_shot.py \
  --config configs/speecht5_minimal.yaml \
  --manifest data/manifests/data_manifest.csv \
  --samples artifacts/evaluation/samples.csv \
  --checkpoint-dir artifacts/checkpoints/few_shot \
  --gpu-hourly-rate 0.0
```

```bash
python3 scripts/run_speecht5_lora.py \
  --config configs/speecht5_minimal.yaml \
  --manifest data/manifests/data_manifest.csv \
  --samples artifacts/evaluation/samples.csv \
  --checkpoint-dir artifacts/checkpoints/lora \
  --gpu-hourly-rate 0.0
```

Opcional:

```bash
python3 scripts/run_parler_reference.py \
  --config configs/speecht5_minimal.yaml \
  --samples artifacts/evaluation/samples.csv
```

Se for necessario preencher custos de treino depois da execucao a partir de `run_started_at` e `run_finished_at`, use:

```bash
python3 scripts/backfill_training_costs.py \
  --samples artifacts/evaluation/samples.csv \
  --gpu-hourly-rate 0.0 \
  --summary-out artifacts/evaluation/manual_training_costs_summary.csv
```

11. Rode Whisper e calcule WER:

```bash
python3 scripts/run_whisper_batch.py \
  --samples artifacts/evaluation/samples.csv \
  --out artifacts/evaluation/samples.csv
```

```bash
python3 scripts/compute_wer.py \
  --samples artifacts/evaluation/samples.csv \
  --out artifacts/evaluation/samples.csv
```

12. Calcule as demais metricas:

```bash
python3 scripts/compute_speaker_similarity.py \
  --config configs/speecht5_minimal.yaml \
  --samples artifacts/evaluation/samples.csv \
  --out artifacts/evaluation/samples.csv
```

Esse passo usa o modelo de avaliacao configurado em `evaluation.speaker_similarity_model` e nao consome `speaker_embedding_path`.

```bash
python3 scripts/compute_nisqa.py \
  --samples artifacts/evaluation/samples.csv \
  --out artifacts/evaluation/samples.csv
```

O NISQA nao faz parte deste repositorio. Antes desse passo, baixe o repositorio oficial localmente:

```bash
git clone https://github.com/gabrielmittag/NISQA.git
```

Se o pacote NISQA nao estiver instalado no ambiente ativo, aponte esse checkout local no comando:

```bash
python3 scripts/compute_nisqa.py \
  --samples artifacts/evaluation/samples.csv \
  --out artifacts/evaluation/samples.csv \
  --nisqa-path /caminho/para/NISQA
```

Exemplo, se o clone foi feito na raiz do projeto:

```bash
python3 scripts/compute_nisqa.py \
  --samples artifacts/evaluation/samples.csv \
  --out artifacts/evaluation/samples.csv \
  --nisqa-path "$(pwd)/NISQA"
```

Ou exporte `NISQA_PATH=/caminho/para/NISQA`.

```bash
python3 scripts/compute_f0_rmse.py \
  --samples artifacts/evaluation/samples.csv \
  --out artifacts/evaluation/samples.csv
```

11. Monte o pacote de avaliacao humana e depois importe os resultados:

```bash
python3 scripts/build_human_eval_pack.py \
  --samples artifacts/evaluation/samples.csv \
  --out-dir artifacts/human_eval_pack
```

```bash
python3 scripts/import_human_eval_results.py \
  --results artifacts/human_eval_pack/human_eval_pack.csv \
  --out artifacts/evaluation/human_eval_summary.csv
```

12. Depois de preencher as metricas e marcar linhas concluidas com `status=ok`, agregue resultados:

```bash
python3 scripts/aggregate_metrics.py \
  --samples artifacts/evaluation/samples.csv \
  --out-dir artifacts/evaluation
```

13. Gere assets finais:

```bash
python3 scripts/make_report_assets.py \
  --samples artifacts/evaluation/samples.csv \
  --metrics artifacts/evaluation/metrics_summary.csv \
  --costs artifacts/evaluation/cost_summary.csv \
  --human-eval artifacts/evaluation/human_eval_summary.csv \
  --out-dir report_assets
```

14. Rode a demo, se `gradio` estiver instalado:

```bash
python3 demo/app.py --samples artifacts/evaluation/samples.csv
```

## Ambientes

- `MANUAL_EXECUCAO.md`: runbooks completos para `Mac M2 host-native com MPS`, `Linux local com NVIDIA + Docker`, `Google Colab` e `RunPod com NVIDIA + Docker`.
- `requirements-macos-mps.txt`: dependencia para `macOS Apple Silicon` com `PyTorch MPS`.
- `requirements-linux-gpu.txt`: dependencia para `Linux/RunPod` com `CUDA`.
- `docker/docker-compose.nvidia.yml`: compose oficial para `Linux NVIDIA` e `RunPod`.
- `docker/Dockerfile.cpu`: legado para analise/preprocess, nao e caminho oficial de GPU.
- `notebooks/`: notebooks Colab de smoke test e piloto de treino.

Regra operacional:

- `Mac = GPU no host via MPS, sem Docker`
- `Docker = apenas NVIDIA Linux/RunPod`
- `RunPod + Docker` segue suportado e e o ambiente oficial

## Contrato de artefatos

O manifesto de dados deve conter, no minimo:

```text
speaker_id,utterance_id,split,duration_s,source,license,audio_path,reference_audio,target_text,text_variant
```

O arquivo `samples.csv` de avaliacao deve conter, no minimo:

```text
sample_id,run_id,condition,speaker_id,prompt_id,text_variant,target_text,audio_path,reference_audio_path,speaker_embedding_path,model_name,run_started_at,run_finished_at,failure_reason,wer,speaker_similarity,nisqa,f0_rmse,rtf,train_gpu_hours,inference_seconds,cost_usd,status,lora_gate_status
```

No `samples.csv`, `speaker_embedding_path` representa apenas o embedding de sintese usado pelo TTS.

Campos extras sao permitidos. Use `asr_text` para armazenar a transcricao do Whisper antes de rodar `scripts/compute_wer.py`.

## Gate de LoRA

Se `LoRA/PEFT` em `SpeechT5` nao estiver estavel, marque `lora_gate_status=fallback_decoder_postnet_ft` nos metadados do run e trate `speecht5_few_shot_decoder_ft` como fallback metodologico. LoRA fica documentado como risco metodologico/trabalho futuro.
