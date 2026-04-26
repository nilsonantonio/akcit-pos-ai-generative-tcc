# TCC Audio Execution Plan

Base executavel para a parte pratica do TCC em IA generativa de audio.

O projeto implementa o plano operacional atualizado:

- eixo principal em `SpeechT5`: condicionais `LoRA` com avaliacao por checkpoint;
- `Whisper` sem ajuste como avaliador de WER;
- `SpeechBrain ECAPA` para embeddings e similaridade de speaker;
- `BRSpeech-DF` como evidencia contextual, nao como braco principal.

## Estrutura

- `configs/speecht5_minimal.yaml`: configuracao oficial do desenho experimental.
- `data/prompts/ptbr_test_prompts.csv`: 24 prompts fixos em PT-BR, com texto cru e normalizado.
- `data/manifests/common_voice_curated.csv`: metadados curados apos preprocessamento do Common Voice.
- `data/manifests/data_manifest.csv`: manifesto consolidado usado nas etapas de treino e validacao.
- `data/manifests/speaker_selection.csv`: selecao final de speakers usada nas etapas de embeddings e avaliacao.
- `tcc_audio/`: pacote com validacao, matriz de runs, inferencia, metricas e relatorios.
- `scripts/`: wrappers de linha de comando.
- `demo/app.py`: demo Gradio minima para comparar checkpoints LoRA A/B/C.
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

5. Gere o manifesto real a partir do subset curado de Common Voice. A selecao agora e global, limitada por `speaker_target_count`, e nao usa mais balanceamento por genero.

```bash
python3 scripts/select_speakers.py \
  --metadata data/manifests/common_voice_curated.csv \
  --speaker-target-count 4 \
  --manifest-out data/manifests/data_manifest.csv \
  --speaker-selection-out data/manifests/speaker_selection.csv
```

`minutes_per_speaker` continua definindo o teto de audio amostrado por speaker, mas speakers com menos minutos continuam elegiveis.

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

10. Rode o pipeline LoRA:

```bash
python3 scripts/run_speecht5_lora.py \
  --config configs/speecht5_minimal.yaml \
  --manifest data/manifests/data_manifest.csv \
  --samples artifacts/evaluation/samples.csv \
  --checkpoint-dir artifacts/checkpoints/lora \
  --gpu-hourly-rate 0.0
```

Sem `--condition`, todas as condicionais LoRA do YAML sao treinadas sequencialmente. Para rodar apenas um subconjunto:

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

Cada checkpoint salvo vira um braco independente de avaliacao e materializa linhas novas no `samples.csv` com `checkpoint_label=<condicao>@step<k>`.

Se for necessario preencher custos de treino depois da execucao, o backfill usa `total_train_gpu_hours` quando a coluna estiver salva no `samples.csv` e, na falta dela, cai para `run_started_at` e `run_finished_at`:

```bash
python3 scripts/backfill_training_costs.py \
  --samples artifacts/evaluation/samples.csv \
  --gpu-hourly-rate 0.0 \
  --summary-out artifacts/evaluation/manual_training_costs_summary.csv
```

11. Rode Whisper e calcule WER para as amostras materializadas por checkpoint:

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
git clone https://github.com/gabrielmittag/NISQA.git ./NISQA
```

Neste projeto, o checkout canonico fica em `./NISQA`. Os bootstraps oficiais ja fazem esse clone e validam o ambiente. Antes de rodar a metrica fora do bootstrap, confirme que o checkout contem:

- `./NISQA/.git`
- `./NISQA/nisqa/NISQA_model.py`
- `./NISQA/weights/nisqa_tts.tar`

Se o pacote NISQA nao estiver instalado no ambiente ativo, aponte esse checkout local no comando:

```bash
python3 scripts/compute_nisqa.py \
  --samples artifacts/evaluation/samples.csv \
  --out artifacts/evaluation/samples.csv \
  --nisqa-path "$(pwd)/NISQA"
```

Ou exporte:

```bash
export NISQA_PATH="$(pwd)/NISQA"
```

```bash
python3 scripts/compute_f0_rmse.py \
  --samples artifacts/evaluation/samples.csv \
  --out artifacts/evaluation/samples.csv
```

11. Depois de preencher as metricas e marcar linhas concluidas com `status=ok`, agregue resultados:

```bash
python3 scripts/aggregate_metrics.py \
  --samples artifacts/evaluation/samples.csv \
  --out-dir artifacts/evaluation
```

12. Gere assets finais:

```bash
python3 scripts/make_report_assets.py \
  --samples artifacts/evaluation/samples.csv \
  --metrics artifacts/evaluation/metrics_summary.csv \
  --costs artifacts/evaluation/cost_summary.csv \
  --out-dir report_assets
```

13. Rode a demo, se `gradio` estiver instalado:

```bash
python3 demo/app.py --samples artifacts/evaluation/samples.csv
```

## Ambientes

- `MANUAL_EXECUCAO.md`: runbooks completos para `Mac M2 host-native com MPS`, `Linux local com NVIDIA + Docker` e `RunPod com NVIDIA + Docker`.
- `requirements-macos-mps.txt`: dependencia para `macOS Apple Silicon` com `PyTorch MPS`.
- `requirements-linux-gpu.txt`: dependencia para `Linux/RunPod` com `CUDA`.
- `docker/docker-compose.nvidia.yml`: compose oficial para `Linux NVIDIA` e `RunPod`.
- `docker/Dockerfile.cpu`: legado para analise/preprocess, nao e caminho oficial de GPU.

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
sample_id,run_id,condition,speaker_id,prompt_id,text_variant,target_text,audio_path,reference_audio_path,speaker_embedding_path,model_name,checkpoint_label,checkpoint_step,checkpoint_path,checkpoint_run_ts,training_scope,training_unit,run_started_at,run_finished_at,failure_reason,wer,speaker_similarity,nisqa,f0_rmse,rtf,train_gpu_hours,inference_seconds,cost_usd,status,lora_gate_status
```

No `samples.csv`, `speaker_embedding_path` representa apenas o embedding de sintese usado pelo TTS.
`condition` preserva a condicional base do YAML, enquanto `checkpoint_label` e o rotulo analitico principal usado nas comparacoes e agregacoes.

Campos extras sao permitidos. Use `asr_text` para armazenar a transcricao do Whisper antes de rodar `scripts/compute_wer.py`.
Quando disponivel, `total_train_gpu_hours` armazena o tempo total de treino por unidade de treino materializada e e usado pelo backfill de custos.

## Checkpoints LoRA

- Cada condicional LoRA deve declarar subblocos separados `lora` e `training`.
- `training.scope` aceita `per_speaker` e `unique`.
- `save_total_limit` pode aparecer no YAML como metadado de configuracao, mas o pipeline ignora pruning para preservar todos os checkpoints avaliados.
- As agregacoes principais saem por `checkpoint_label`, e o diretório de avaliacao tambem recebe visoes por speaker.
