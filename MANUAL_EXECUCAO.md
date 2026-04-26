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
5. Selecionar globalmente os 4 speakers alvo e validar manifesto.
6. Extrair embeddings de speaker.
7. Gerar matriz e ledger de amostras.
8. Rodar `SpeechT5 zero-shot`.
9. Rodar `SpeechT5 few-shot decoder fine-tune`.
10. Rodar `SpeechT5 LoRA`.
11. Rodar `Whisper`.
12. Calcular `WER`, `speaker_similarity`, `NISQA`, `F0 RMSE`.
13. Agregar resultados.
14. Gerar `report_assets/`.
15. Abrir demo Gradio.

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
- instala `requirements-macos-mps.txt`, incluindo `pytest` para os smoke tests
- clona `NISQA` em `./NISQA`
- exige `ffmpeg` no host
- valida suporte do Python a `lzma/_lzma`
- valida `torch.backends.mps.is_available()`
- roda smoke tests do projeto

Cheque rapido antes de rodar `few-shot` ou `LoRA`:

```bash
python3 -c "import lzma, _lzma; print('lzma ok')"
```

Se esse comando falhar com `No module named '_lzma'`, o problema e do interpretador Python, nao do script `run_speecht5_few_shot.py`. Em `macOS + Homebrew + asdf`, a recuperacao esperada e:

```bash
brew install xz
asdf uninstall python 3.11.12
asdf install python 3.11.12
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-macos-mps.txt
```

### Execucao completa no host

Antes do download, aceite os termos do dataset no site do Mozilla Data Collective e exporte:

```bash
export MOZILLA_DATA_COLLECTIVE_API_KEY="seu_token_aqui"
```

```bash
python3 scripts/download_common_voice_pt.py \
  --out-dir data/raw/common_voice_pt
```


```bash
python3 scripts/prepare_common_voice_metadata.py \
  --tsv data/raw/common_voice_pt/validated.tsv \
  --clips-dir data/raw/common_voice_pt/clips \
  --locale pt \
  --variant pt-BR \
  --out data/manifests/common_voice_metadata.csv
```

```bash
python3 scripts/preprocess_audio_dataset.py \
  --metadata data/manifests/common_voice_metadata.csv \
  --out-dir data/processed/common_voice_pt \
  --out-metadata data/manifests/common_voice_curated.csv
```

```bash
python3 scripts/select_speakers.py \
  --metadata data/manifests/common_voice_curated.csv \
  --speaker-target-count 4 \
  --manifest-out data/manifests/data_manifest.csv \
  --speaker-selection-out data/manifests/speaker_selection.csv
```

```bash
python3 scripts/validate_manifest.py \
  --manifest data/manifests/data_manifest.csv \
  --config configs/speecht5_minimal.yaml \
  --prompts data/prompts/ptbr_test_prompts.csv \
  --check-files
```

```bash
python3 scripts/extract_speaker_embeddings.py \
  --config configs/speecht5_minimal.yaml \
  --speaker-selection data/manifests/speaker_selection.csv \
  --out-index artifacts/embeddings/speaker_embeddings.csv \
  --out-dir artifacts/embeddings
```

```bash
python3 scripts/generate_run_matrix.py \
  --config configs/speecht5_minimal.yaml \
  --out artifacts/run_matrix.csv
```

```bash
python3 scripts/init_samples.py \
  --run-matrix artifacts/run_matrix.csv \
  --speaker-selection data/manifests/speaker_selection.csv \
  --speaker-embeddings artifacts/embeddings/speaker_embeddings.csv \
  --out artifacts/evaluation/samples.csv
```

```bash
python3 scripts/run_speecht5_zero_shot.py \
  --config configs/speecht5_minimal.yaml \
  --samples artifacts/evaluation/samples.csv
```

Pilotos de treino recomendados primeiro:

```bash
python3 scripts/run_speecht5_few_shot.py \
  --config configs/speecht5_minimal.yaml \
  --manifest data/manifests/data_manifest.csv \
  --samples artifacts/evaluation/samples.csv \
  --checkpoint-dir artifacts/checkpoints/few_shot \
  --gpu-hourly-rate 0.0 \
  --max-steps 50
```

```bash
python3 scripts/run_speecht5_lora.py \
  --config configs/speecht5_minimal.yaml \
  --manifest data/manifests/data_manifest.csv \
  --samples artifacts/evaluation/samples.csv \
  --checkpoint-dir artifacts/checkpoints/lora \
  --gpu-hourly-rate 0.0 \
  --max-steps 50
```

Se precisar preencher `train_gpu_hours` e `cost_usd` depois do treino, o backfill usa `total_train_gpu_hours` quando a coluna estiver salva no `samples.csv` e, na falta dela, cai para `run_started_at` e `run_finished_at`:

```bash
python3 scripts/backfill_training_costs.py \
  --samples artifacts/evaluation/samples.csv \
  --gpu-hourly-rate 0.0 \
  --summary-out artifacts/evaluation/manual_training_costs_summary.csv
```

Depois do piloto, rode as fases restantes:

```bash
python3 scripts/run_whisper_batch.py --samples artifacts/evaluation/samples.csv --out artifacts/evaluation/samples.csv
python3 scripts/compute_wer.py --samples artifacts/evaluation/samples.csv --out artifacts/evaluation/samples.csv
python3 scripts/compute_speaker_similarity.py --config configs/speecht5_minimal.yaml --samples artifacts/evaluation/samples.csv --out artifacts/evaluation/samples.csv
git clone https://github.com/gabrielmittag/NISQA.git
python3 scripts/compute_nisqa.py --samples artifacts/evaluation/samples.csv --out artifacts/evaluation/samples.csv
python3 scripts/compute_nisqa.py --samples artifacts/evaluation/samples.csv --out artifacts/evaluation/samples.csv --nisqa-path "$(pwd)/NISQA"
python3 scripts/compute_f0_rmse.py --samples artifacts/evaluation/samples.csv --out artifacts/evaluation/samples.csv
python3 scripts/aggregate_metrics.py --samples artifacts/evaluation/samples.csv --out-dir artifacts/evaluation
python3 scripts/make_report_assets.py --samples artifacts/evaluation/samples.csv --metrics artifacts/evaluation/metrics_summary.csv --costs artifacts/evaluation/cost_summary.csv --out-dir report_assets
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

- instala `requirements-linux-gpu.txt`, incluindo `pytest` para os smoke tests
- clona `NISQA` em `./NISQA`
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
python3 scripts/validate_manifest.py --manifest data/manifests/data_manifest.csv --config configs/speecht5_minimal.yaml --prompts data/prompts/ptbr_test_prompts.csv --check-files
python3 scripts/extract_speaker_embeddings.py --config configs/speecht5_minimal.yaml --speaker-selection data/manifests/speaker_selection.csv --out-index artifacts/embeddings/speaker_embeddings.csv --out-dir artifacts/embeddings
python3 scripts/generate_run_matrix.py --config configs/speecht5_minimal.yaml --out artifacts/run_matrix.csv
python3 scripts/init_samples.py --run-matrix artifacts/run_matrix.csv --speaker-selection data/manifests/speaker_selection.csv --speaker-embeddings artifacts/embeddings/speaker_embeddings.csv --out artifacts/evaluation/samples.csv
python3 scripts/run_speecht5_zero_shot.py --config configs/speecht5_minimal.yaml --samples artifacts/evaluation/samples.csv
python3 scripts/run_speecht5_few_shot.py --config configs/speecht5_minimal.yaml --manifest data/manifests/data_manifest.csv --samples artifacts/evaluation/samples.csv --checkpoint-dir artifacts/checkpoints/few_shot --gpu-hourly-rate 0.0
python3 scripts/run_speecht5_lora.py --config configs/speecht5_minimal.yaml --manifest data/manifests/data_manifest.csv --samples artifacts/evaluation/samples.csv --checkpoint-dir artifacts/checkpoints/lora --gpu-hourly-rate 0.0
python3 scripts/run_whisper_batch.py --samples artifacts/evaluation/samples.csv --out artifacts/evaluation/samples.csv
python3 scripts/compute_wer.py --samples artifacts/evaluation/samples.csv --out artifacts/evaluation/samples.csv
python3 scripts/compute_speaker_similarity.py --config configs/speecht5_minimal.yaml --samples artifacts/evaluation/samples.csv --out artifacts/evaluation/samples.csv
git clone https://github.com/gabrielmittag/NISQA.git
python3 scripts/compute_nisqa.py --samples artifacts/evaluation/samples.csv --out artifacts/evaluation/samples.csv
# ou:
export NISQA_PATH=/caminho/para/NISQA
python3 scripts/compute_nisqa.py --samples artifacts/evaluation/samples.csv --out artifacts/evaluation/samples.csv
python3 scripts/compute_f0_rmse.py --samples artifacts/evaluation/samples.csv --out artifacts/evaluation/samples.csv
python3 scripts/aggregate_metrics.py --samples artifacts/evaluation/samples.csv --out-dir artifacts/evaluation
python3 scripts/make_report_assets.py --samples artifacts/evaluation/samples.csv --metrics artifacts/evaluation/metrics_summary.csv --costs artifacts/evaluation/cost_summary.csv --out-dir report_assets
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
- `metrics_summary.csv` e `cost_summary.csv` preenchidos
- `report_assets/` com tabelas e graficos
- `demo/app.py` abrindo e tocando audios reais
