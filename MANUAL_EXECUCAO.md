# Manual de Execucao do TCC de Audio

Manual operacional oficial para executar o projeto em:

- `Mac M2 Pro Max host-native com MPS`
- `Linux local com NVIDIA + Docker`
- `Google Colab`
- `RunPod com NVIDIA + Docker`

## Nota técnica

- `macOS + Docker Desktop`: nao e caminho oficial de GPU para este projeto. A documentacao do Docker informa suporte oficial de GPU no Docker Desktop apenas em `Windows + WSL2`, nao em macOS.
- `macOS + Apple Silicon`: o caminho oficial de GPU aqui e `PyTorch MPS` no host.
- `RunPod`: a limitacao acima nao se aplica. O RunPod roda em Linux com GPU NVIDIA, entao Docker GPU continua suportado.

Referencias:

- Docker Desktop GPU: https://docs.docker.com/desktop/features/gpu/
- PyTorch MPS: https://docs.pytorch.org/docs/2.9/notes/mps.html
- RunPod Pods: https://docs.runpod.io/pods/overview

## Matriz de suporte

| Ambiente | GPU | Suporte oficial | Observacao |
|---|---:|---|---|
| Mac host-native MPS | Sim | Sim | Caminho oficial local para Apple Silicon |
| Mac com Docker Desktop GPU | Nao | Nao | Fora do escopo operacional |
| Linux local + NVIDIA Docker | Sim | Sim | Requer RTX 3090 ou superior |
| RunPod + NVIDIA Docker | Sim | Sim | Ambiente oficial de producao |
| Google Colab | Sim | Parcial | Apenas piloto e validacao rapida |

## 1. Estrutura minima esperada

Diretorios relevantes:

- `data/raw/common_voice_pt/`
- `data/processed/common_voice_pt/`
- `data/manifests/`
- `artifacts/embeddings/`
- `artifacts/checkpoints/`
- `artifacts/audio/`
- `artifacts/evaluation/`
- `report_assets/`

Arquivos de controle:

- `configs/speecht5_minimal.yaml`
- `artifacts/run_matrix.csv`
- `artifacts/evaluation/samples.csv`

## 2. Pipeline canônico

1. Aceitar os termos do Common Voice no site do Mozilla Data Collective e exportar o token de API.
2. Baixar e extrair o dataset `pt` para `data/raw/common_voice_pt/`.
3. Preparar metadados do subconjunto `pt-BR`.
4. Preprocessar audio para WAV mono.
5. Selecionar os 4 speakers e validar manifesto.
6. Extrair embeddings de speaker.
7. Gerar matriz e ledger de amostras.
8. Rodar `SpeechT5 zero-shot`.
9. Rodar `SpeechT5 few-shot decoder fine-tune`.
10. Rodar `SpeechT5 LoRA`.
11. Rodar `Parler-TTS` opcionalmente.
12. Rodar `Whisper`.
13. Calcular `WER`, `speaker_similarity`, `NISQA`, `F0 RMSE`.
14. Montar avaliacao humana.
15. Agregar resultados.
16. Gerar `report_assets/`.
17. Abrir demo Gradio.

## 3. Runbook: Mac M2 Pro Max host-native com MPS

Objetivo: execucao local completa em GPU no host, sem Docker, usando `venv` do projeto.

Capacidade assumida:

- `64 GB` de memoria unificada
- `32-core GPU` integrada
- adequado para zero-shot, embeddings, metricas, pilotos de treino e parte relevante do treino principal se couber em memoria

### Preparacao

```bash
bash scripts/bootstrap_macos_mps.sh
source .venv/bin/activate
```

O bootstrap:

- cria `.venv`
- instala `requirements-macos-mps.txt`
- exige `ffmpeg` no host
- valida `torch.backends.mps.is_available()`
- roda smoke tests do projeto

### Execucao completa no host

Antes do download, aceite os termos do dataset no site do Mozilla Data Collective e exporte:

```bash
export MOZILLA_DATA_COLLECTIVE_API_KEY="seu_token_aqui"
```

```bash
python3 scripts/download_common_voice_pt.py \
  --out-dir data/raw/common_voice_pt

# 3m
# Archive: data/raw/common_voice_pt/common-voice-scripted-speech-25-0-portug-0254cce0.tar.gz
# Clips: data/raw/common_voice_pt/clips
# Validated TSV: data/raw/common_voice_pt/validated.tsv
```


```bash
python3 scripts/prepare_common_voice_metadata.py \
  --tsv data/raw/common_voice_pt/validated.tsv \
  --clips-dir data/raw/common_voice_pt/clips \
  --locale pt \
  --variant pt-BR \
  --out data/manifests/common_voice_metadata.csv

# 2m
# Wrote 66183 Common Voice metadata rows to data/manifests/common_voice_curated.csv
```

```bash
python3 scripts/preprocess_audio_dataset.py \
  --metadata data/manifests/common_voice_metadata.csv \
  --out-dir data/processed/common_voice_pt \
  --out-metadata data/manifests/common_voice_curated.csv

# 30m
# Wrote 66193 processed metadata rows to data/manifests/common_voice_curated.csv
```

```bash
python3 scripts/select_speakers.py \
  --metadata data/manifests/common_voice_curated.csv \
  --manifest-out data/manifests/data_manifest.csv \
  --speaker-selection-out data/manifests/speaker_selection.csv

# 2s
# Selected 6 speakers
# Wrote 1200 manifest rows
```

```bash
python3 scripts/validate_manifest.py \
  --manifest data/manifests/data_manifest.csv \
  --prompts data/prompts/ptbr_test_prompts.csv \
  --check-files

# 3s
# OK: data/manifests/data_manifest.csv (1200 rows)
# summary.prompts_commercial_subset_count=8
# summary.prompts_prompt_categories=abreviacoes, expressivas, homografos, interrogativas, neutras, numeros
# summary.prompts_prompt_subexperiment_count=8
# summary.speaker_count=6
# summary.total_duration_s=7214.9039999999995
```

```bash
python3 scripts/extract_speaker_embeddings.py \
  --config configs/speecht5_minimal.yaml \
  --speaker-selection data/manifests/speaker_selection.csv \
  --out-index artifacts/embeddings/speaker_embeddings.csv \
  --out-dir artifacts/embeddings

# o extrator agora usa por padrao `speechbrain/spkrec-xvect-voxceleb`,
# que gera embeddings 512-d compativeis com o SpeechT5.
# o campo `speaker_embedding_path` em `samples.csv` representa esse embedding de sintese.

# 10s
# hyperparams.yaml: 2.04kB [00:00, 4.93MB/s]
# embedding_model.ckpt: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16.9M/16.9M [00:03<00:00, 5.62MB/s]
# mean_var_norm_emb.ckpt: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.20k/3.20k [00:01<00:00, 3.19kB/s]
# classifier.ckpt: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15.9M/15.9M [00:02<00:00, 7.91MB/s]
# label_encoder.txt: 129kB [00:00, 41.9MB/s]
# Wrote 6 speaker embeddings to artifacts/embeddings/speaker_embeddings.csv
```

```bash
python3 scripts/generate_run_matrix.py \
  --config configs/speecht5_minimal.yaml \
  --out artifacts/run_matrix.csv

# 2s
# Generated 400 runs
# Wrote artifacts/run_matrix.csv
```

```bash
python3 scripts/init_samples.py \
  --run-matrix artifacts/run_matrix.csv \
  --speaker-selection data/manifests/speaker_selection.csv \
  --speaker-embeddings artifacts/embeddings/speaker_embeddings.csv \
  --out artifacts/evaluation/samples.csv

# 2s
# Wrote 400 sample rows to artifacts/evaluation/samples.csv
```

```bash
python3 scripts/run_speecht5_zero_shot.py \
  --config configs/speecht5_minimal.yaml \
  --samples artifacts/evaluation/samples.csv

# 8m
# preprocessor_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████| 433/433 [00:00<00:00, 1.29MB/s]
# tokenizer_config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 232/232 [00:00<00:00, 2.76MB/s]
# spm_char.model: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 238k/238k [00:01<00:00, 237kB/s]
# added_tokens.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 40.0/40.0 [00:00<00:00, 311kB/s]
# special_tokens_map.json: 100%|████████████████████████████████████████████████████████████████████████████████████████| 234/234 [00:00<00:00, 1.15MB/s]
# config.json: 2.06kB [00:00, 5.62MB/s]
# pytorch_model.bin: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 585M/585M [00:06<00:00, 86.1MB/s]
# config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 636/636 [00:00<00:00, 2.50MB/s]
# pytorch_model.bin: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 50.7M/50.7M [00:01<00:00, 28.1MB/s]
# model.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 585M/585M [00:05<00:00, 117MB/s]
# model.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 50.6M/50.6M [00:01<00:00, 28.1MB/s]
# SpeechT5 zero-shot inference completed
```

Pilotos de treino recomendados primeiro:

```bash
python3 scripts/run_speecht5_few_shot.py \
  --config configs/speecht5_minimal.yaml \
  --manifest data/manifests/data_manifest.csv \
  --samples artifacts/evaluation/samples.csv \
  --checkpoint-dir artifacts/checkpoints/few_shot \
  --max-steps 50
```

```bash
python3 scripts/run_speecht5_lora.py \
  --config configs/speecht5_minimal.yaml \
  --manifest data/manifests/data_manifest.csv \
  --samples artifacts/evaluation/samples.csv \
  --checkpoint-dir artifacts/checkpoints/lora \
  --max-steps 50
```

Depois do piloto, rode as fases restantes:

```bash
python3 scripts/run_whisper_batch.py --samples artifacts/evaluation/samples.csv --out artifacts/evaluation/samples.csv
python3 scripts/compute_wer.py --samples artifacts/evaluation/samples.csv --out artifacts/evaluation/samples.csv
python3 scripts/compute_speaker_similarity.py --config configs/speecht5_minimal.yaml --samples artifacts/evaluation/samples.csv --out artifacts/evaluation/samples.csv
python3 scripts/compute_nisqa.py --samples artifacts/evaluation/samples.csv --out artifacts/evaluation/samples.csv
python3 scripts/compute_f0_rmse.py --samples artifacts/evaluation/samples.csv --out artifacts/evaluation/samples.csv
python3 scripts/build_human_eval_pack.py --samples artifacts/evaluation/samples.csv --out-dir artifacts/human_eval_pack
python3 scripts/import_human_eval_results.py --results data/evaluation/human_eval_results.csv --out artifacts/evaluation/human_eval_summary.csv
python3 scripts/aggregate_metrics.py --samples artifacts/evaluation/samples.csv --out-dir artifacts/evaluation
python3 scripts/make_report_assets.py --samples artifacts/evaluation/samples.csv --metrics artifacts/evaluation/metrics_summary.csv --costs artifacts/evaluation/cost_summary.csv --human-eval artifacts/evaluation/human_eval_summary.csv --out-dir report_assets
python3 demo/app.py --samples artifacts/evaluation/samples.csv
```

## 4. Runbook: Linux local com NVIDIA + Docker

Objetivo: ambiente oficial local com Docker e GPU NVIDIA.

Requisitos:

- Linux nativo
- `RTX 3090` ou superior
- driver NVIDIA instalado
- `NVIDIA Container Toolkit`

### Preparacao do host

```bash
bash scripts/bootstrap_nvidia_linux.sh
```

O bootstrap:

- instala `requirements-linux-gpu.txt`
- exige `ffmpeg`
- exige `nvidia-smi`
- valida `torch.cuda.is_available()`
- roda smoke tests

### Build e shell

```bash
docker compose -f docker/docker-compose.nvidia.yml build
```

```bash
docker compose -f docker/docker-compose.nvidia.yml run --rm tcc-audio-nvidia bash
```

### Pipeline no container

Dentro do container:

```bash
export MOZILLA_DATA_COLLECTIVE_API_KEY="seu_token_aqui"
python3 scripts/download_common_voice_pt.py --out-dir /workspace/data/raw/common_voice_pt
python3 scripts/prepare_common_voice_metadata.py --tsv /workspace/data/raw/common_voice_pt/validated.tsv --clips-dir /workspace/data/raw/common_voice_pt/clips --locale pt --variant pt-BR --out data/manifests/common_voice_metadata.csv
python3 scripts/preprocess_audio_dataset.py --metadata data/manifests/common_voice_metadata.csv --out-dir data/processed/common_voice_pt --out-metadata data/manifests/common_voice_curated.csv
python3 scripts/select_speakers.py --metadata data/manifests/common_voice_curated.csv --manifest-out data/manifests/data_manifest.csv --speaker-selection-out data/manifests/speaker_selection.csv
python3 scripts/validate_manifest.py --manifest data/manifests/data_manifest.csv --prompts data/prompts/ptbr_test_prompts.csv --check-files
python3 scripts/extract_speaker_embeddings.py --config configs/speecht5_minimal.yaml --speaker-selection data/manifests/speaker_selection.csv --out-index artifacts/embeddings/speaker_embeddings.csv --out-dir artifacts/embeddings
python3 scripts/generate_run_matrix.py --config configs/speecht5_minimal.yaml --out artifacts/run_matrix.csv
python3 scripts/init_samples.py --run-matrix artifacts/run_matrix.csv --speaker-selection data/manifests/speaker_selection.csv --speaker-embeddings artifacts/embeddings/speaker_embeddings.csv --out artifacts/evaluation/samples.csv
python3 scripts/run_speecht5_zero_shot.py --config configs/speecht5_minimal.yaml --samples artifacts/evaluation/samples.csv
python3 scripts/run_speecht5_few_shot.py --config configs/speecht5_minimal.yaml --manifest data/manifests/data_manifest.csv --samples artifacts/evaluation/samples.csv --checkpoint-dir artifacts/checkpoints/few_shot
python3 scripts/run_speecht5_lora.py --config configs/speecht5_minimal.yaml --manifest data/manifests/data_manifest.csv --samples artifacts/evaluation/samples.csv --checkpoint-dir artifacts/checkpoints/lora
python3 scripts/run_parler_reference.py --config configs/speecht5_minimal.yaml --samples artifacts/evaluation/samples.csv
python3 scripts/run_whisper_batch.py --samples artifacts/evaluation/samples.csv --out artifacts/evaluation/samples.csv
python3 scripts/compute_wer.py --samples artifacts/evaluation/samples.csv --out artifacts/evaluation/samples.csv
python3 scripts/compute_speaker_similarity.py --config configs/speecht5_minimal.yaml --samples artifacts/evaluation/samples.csv --out artifacts/evaluation/samples.csv
python3 scripts/compute_nisqa.py --samples artifacts/evaluation/samples.csv --out artifacts/evaluation/samples.csv
python3 scripts/compute_f0_rmse.py --samples artifacts/evaluation/samples.csv --out artifacts/evaluation/samples.csv
python3 scripts/build_human_eval_pack.py --samples artifacts/evaluation/samples.csv --out-dir artifacts/human_eval_pack
python3 scripts/import_human_eval_results.py --results data/evaluation/human_eval_results.csv --out artifacts/evaluation/human_eval_summary.csv
python3 scripts/aggregate_metrics.py --samples artifacts/evaluation/samples.csv --out-dir artifacts/evaluation
python3 scripts/make_report_assets.py --samples artifacts/evaluation/samples.csv --metrics artifacts/evaluation/metrics_summary.csv --costs artifacts/evaluation/cost_summary.csv --human-eval artifacts/evaluation/human_eval_summary.csv --out-dir report_assets
python3 demo/app.py --samples artifacts/evaluation/samples.csv
```

## 5. Runbook: Google Colab

Objetivo: piloto curto de `SpeechT5 zero-shot`, `few-shot` e `LoRA`.

Checklist:

1. Abrir `notebooks/colab_smoke_test.ipynb`.
2. Montar Google Drive.
3. Clonar o projeto ou fazer upload do zip.
4. Instalar `ffmpeg` e `requirements-linux-gpu.txt`.
5. Rodar smoke test e um piloto curto de 1 speaker.

Limites:

- nao usar como ambiente oficial para as 400 amostras
- usar para validar hiperparametros e estabilidade

## 6. Runbook: RunPod com NVIDIA + Docker

Objetivo: ambiente oficial de producao do TCC.

Observacao:

- A limitacao de Docker no macOS nao afeta o RunPod.
- O pod roda em Linux com GPU NVIDIA, entao Docker GPU continua suportado.

### Bootstrap

No host do pod:

```bash
bash scripts/install_runpod_gpu.sh
```

### Build e shell

```bash
docker compose -f docker/docker-compose.nvidia.yml build
```

```bash
docker compose -f docker/docker-compose.nvidia.yml run --rm tcc-audio-nvidia bash
```

### Pipeline oficial

Dentro do container, execute a mesma sequencia do runbook `Linux local com NVIDIA + Docker`.

## 7. Criterios de encerramento

- `artifacts/evaluation/samples.csv` com linhas `status=ok`
- `metrics_summary.csv`, `cost_summary.csv` e `human_eval_summary.csv` preenchidos
- `report_assets/` com tabelas e graficos
- `demo/app.py` abrindo e tocando audios reais
