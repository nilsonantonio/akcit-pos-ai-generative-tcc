# Manual de Execucao do TCC de Audio

Manual operacional oficial para executar o projeto em:

- `Mac M2 Pro Max host-native com MPS`
- `Linux local com NVIDIA + Docker`
- `RunPod com NVIDIA host-native`

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
5. Selecionar globalmente os speakers alvo e validar manifesto.
6. Extrair embeddings de speaker.
7. Gerar matriz e ledger de amostras.
8. Rodar as condicionais `SpeechT5 LoRA` configuradas no YAML.
9. Materializar checkpoints e gerar audios por checkpoint salvo.
10. Rodar `Whisper`.
11. Calcular `WER`, `speaker_similarity`, `NISQA`, `F0 RMSE`.
12. Agregar resultados.
13. Gerar `report_assets/`.
14. Abrir demo Gradio.

## 3. Como usar este manual

Escolha e prepare o ambiente correspondente em [SETUP_AMBIENTES.md](/Users/nilsonantonio/des/datascience/akcit/pos/generative-ai/14-tcc/src-light/SETUP_AMBIENTES.md).

Depois do setup:

- no `Mac` e no `RunPod`, mantenha a `.venv` ativa
- no `Linux NVIDIA + Docker`, entre no container e rode o pipeline a partir da raiz do projeto

## 4. Execução passo a passo do pipeline

Antes do download, aceite os termos do dataset no site do Mozilla Data Collective e exporte o token de acesso.

```bash
export MOZILLA_DATA_COLLECTIVE_API_KEY="seu_token_aqui"
```

Baixe e extraia o subconjunto `pt` do Common Voice.

```bash
python3 scripts/download_common_voice_pt.py \
  --out-dir data/raw/common_voice_pt
```

Converta os metadados brutos em um CSV operacional do projeto.

```bash
python3 scripts/prepare_common_voice_metadata.py \
  --tsv data/raw/common_voice_pt/validated.tsv \
  --clips-dir data/raw/common_voice_pt/clips \
  --locale pt \
  --variant pt-BR \
  --out data/manifests/common_voice_metadata.csv
```

Preprocesse os audios para o formato canonico usado no experimento.

```bash
python3 scripts/preprocess_audio_dataset.py \
  --metadata data/manifests/common_voice_metadata.csv \
  --out-dir data/processed/common_voice_pt \
  --out-metadata data/manifests/common_voice_curated.csv
```

Selecione os speakers alvo e materialize o manifesto principal do experimento.

```bash
python3 scripts/select_speakers.py \
  --metadata data/manifests/common_voice_curated.csv \
  --speaker-target-count 1000 \
  --manifest-out data/manifests/data_manifest.csv \
  --speaker-selection-out data/manifests/speaker_selection.csv
```

Registre um inventario hierarquico dos dados preparados e selecionados.

```bash
python3 scripts/log_dataset_inventory.py \
  --config configs/speecht5_minimal.yaml \
  --json-out artifacts/dataset_inventory.json
```

O comando gera um relatorio no console e grava um JSON com o inventario hierarquico de `data/raw`, `data/processed` e `selected_speakers`.

Valide o manifesto e os prompts antes de iniciar treino e inferência.

```bash
python3 scripts/validate_manifest.py \
  --manifest data/manifests/data_manifest.csv \
  --config configs/speecht5_minimal.yaml \
  --prompts data/prompts/ptbr_test_prompts.csv \
  --check-files
```

Extraia os embeddings de speaker usados pelas etapas de sintese.

```bash
python3 scripts/extract_speaker_embeddings.py \
  --config configs/speecht5_minimal.yaml \
  --speaker-selection data/manifests/speaker_selection.csv \
  --out-index artifacts/embeddings/speaker_embeddings.csv \
  --out-dir artifacts/embeddings
```

Gere a matriz de execucao a partir da configuracao do experimento.

```bash
python3 scripts/generate_run_matrix.py \
  --config configs/speecht5_minimal.yaml \
  --out artifacts/run_matrix.csv
```

Materialize o ledger inicial de amostras que sera enriquecido nas proximas etapas.

```bash
python3 scripts/init_samples.py \
  --run-matrix artifacts/run_matrix.csv \
  --speaker-selection data/manifests/speaker_selection.csv \
  --speaker-embeddings artifacts/embeddings/speaker_embeddings.csv \
  --out artifacts/evaluation/samples.csv
```

Execute o treino LoRA e a materializacao dos checkpoints avaliados.

```bash
python3 scripts/run_speecht5_lora.py \
  --config configs/speecht5_minimal.yaml \
  --manifest data/manifests/data_manifest.csv \
  --samples artifacts/evaluation/samples.csv \
  --checkpoint-dir artifacts/checkpoints/lora \
  --gpu-hourly-rate 0.0
```

Sem `--condition`, o script percorre todas as condicionais LoRA do YAML em sequencia.

Cada checkpoint salvo em `checkpoint-<step>` vira um braco independente de avaliacao e materializa novas linhas no `samples.csv`.

Se ocorrer `CUDA out of memory` em `RTX 4090 24 GB`, reduza `per_device_train_batch_size` para `1` e aumente `gradient_accumulation_steps` para `4`.

Transcreva os audios gerados com Whisper para preparar o calculo de WER.

```bash
python3 scripts/run_whisper_batch.py \
  --samples artifacts/evaluation/samples.csv \
  --out artifacts/evaluation/samples.csv
```

Calcule a taxa de erro de palavras a partir das transcricoes do ASR.

```bash
python3 scripts/compute_wer.py \
  --samples artifacts/evaluation/samples.csv \
  --out artifacts/evaluation/samples.csv
```

Calcule a similaridade entre speaker de referencia e speaker sintetizado.

```bash
python3 scripts/compute_speaker_similarity.py \
  --config configs/speecht5_minimal.yaml \
  --samples artifacts/evaluation/samples.csv \
  --out artifacts/evaluation/samples.csv
```

Calcule a metrica perceptual NISQA usando o checkout local validado no setup.

```bash
python3 scripts/compute_nisqa.py \
  --samples artifacts/evaluation/samples.csv \
  --out artifacts/evaluation/samples.csv \
  --nisqa-path "$(pwd)/NISQA"
```

Calcule a divergencia de frequencia fundamental entre referencia e sintese.

```bash
python3 scripts/compute_f0_rmse.py \
  --samples artifacts/evaluation/samples.csv \
  --out artifacts/evaluation/samples.csv
```

Agregue os resultados por condicao e gere os CSVs sumarizados de metricas e custos.

```bash
python3 scripts/aggregate_metrics.py \
  --samples artifacts/evaluation/samples.csv \
  --out-dir artifacts/evaluation
```

Gere os assets finais que serao usados no relatorio do experimento.

```bash
python3 scripts/make_report_assets.py \
  --samples artifacts/evaluation/samples.csv \
  --metrics artifacts/evaluation/metrics_summary.csv \
  --costs artifacts/evaluation/cost_summary.csv \
  --out-dir report_assets
```

## 5. Comandos utilitários

Reexecute apenas um subconjunto de condicionais LoRA do YAML.

```bash
python3 scripts/run_speecht5_lora.py \
  --config configs/speecht5_minimal.yaml \
  --manifest data/manifests/data_manifest.csv \
  --samples artifacts/evaluation/samples.csv \
  --checkpoint-dir artifacts/checkpoints/lora \
  --condition speecht5_lora_conservative \
  --condition speecht5_lora_unique \
  --gpu-hourly-rate 0.0
```

Preencha custos de treino depois da execucao, quando necessario.

```bash
python3 scripts/backfill_training_costs.py \
  --samples artifacts/evaluation/samples.csv \
  --gpu-hourly-rate 0.0 \
  --summary-out artifacts/evaluation/manual_training_costs_summary.csv
```

Remova os rastros materializados de treino de uma condicional especifica.

```bash
python3 scripts/clear_speecht5_training_results.py \
  --condition speecht5_lora_unique
```

Remova os rastros materializados de varias condicionais especificas.

```bash
python3 scripts/clear_speecht5_training_results.py \
  --condition speecht5_lora_conservative \
  --condition speecht5_lora_unique
```

Remova todos os treinos LoRA materializados do experimento.

```bash
python3 scripts/clear_speecht5_training_results.py \
  --condition all
```

Esse cleanup remove checkpoints, audios materializados, linhas materializadas no `samples.csv`, agregados globais em `artifacts/evaluation/` e o diretorio `report_assets/`.

Remova uma condicional inteira do experimento e invalide a matriz quando aplicavel.

```bash
python3 scripts/remove_speecht5_condition.py \
  --condition speecht5_lora_unique
```

Esse fluxo encadeia o cleanup com `bypass` interno, remove as linhas base da condicional no `samples.csv`, remove a condicional do bloco `conditions:` do config e invalida `deliverables.run_matrix` quando o arquivo existir.

Abra a demo para inspecao manual dos audios gerados.

```bash
python3 demo/app.py \
  --samples artifacts/evaluation/samples.csv
```

## 6. Criterios de encerramento

- `artifacts/evaluation/samples.csv` com linhas `status=ok`
- `metrics_summary.csv` e `cost_summary.csv` preenchidos
- `report_assets/` com tabelas e graficos
- `demo/app.py` abrindo e tocando audios reais
