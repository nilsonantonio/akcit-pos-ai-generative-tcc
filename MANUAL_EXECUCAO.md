# Manual de Execucao do TCC de Audio

Guia operacional principal do projeto para:

- `Mac M2 Pro Max host-native com MPS`
- `Linux local com NVIDIA + Docker`
- `RunPod com NVIDIA host-native`

## Visao curta do pipeline

O fluxo oficial do repositório é:

1. baixar o `Common Voice PT`
2. preparar metadados `pt-BR`
3. preprocessar audio
4. selecionar speakers e validar manifesto
5. extrair embeddings
6. gerar `run_matrix.csv` e `samples.csv`
7. treinar `SpeechT5 LoRA` e materializar checkpoints
8. rodar métricas automáticas
9. agregar resultados
10. gerar `report_assets/`
11. abrir a demo Gradio

Este manual prioriza o caminho feliz: primeiro os comandos mínimos do pipeline canônico, depois uma seção separada para sobrescritas e casos avançados.

## Pre-requisitos e convencoes

- faça o setup do ambiente em [SETUP_AMBIENTES.md](SETUP_AMBIENTES.md)
- execute os comandos a partir da raiz do repositório
- no `Mac` e no `RunPod`, mantenha a `.venv` ativa
- no `Linux NVIDIA + Docker`, execute o pipeline já dentro do container
- aceite os termos do dataset no site do Mozilla Data Collective antes do download

Estrutura esperada ao longo da execução:

- `data/raw/common_voice_pt/`
- `data/processed/common_voice_pt/`
- `data/manifests/`
- `artifacts/embeddings/`
- `artifacts/checkpoints/`
- `artifacts/audio/`
- `artifacts/evaluation/`
- `report_assets/`

Arquivos de controle principais:

- `configs/speecht5_minimal.yaml`
- `artifacts/run_matrix.csv`
- `artifacts/evaluation/samples.csv`

Convencao da interface CLI:

- quando `--config` é informado, os comandos priorizam os paths definidos no YAML
- quando `--config` não é informado, os comandos usam os paths canônicos do repositório
- parâmetros de intenção operacional continuam explícitos no fluxo principal, como `--speaker-target-count`

## Pipeline canônico

Exporte o token do Mozilla Data Collective:

```bash
export MOZILLA_DATA_COLLECTIVE_API_KEY="seu_token_aqui"
```

Baixe e extraia o `Common Voice PT`:

```bash
python3 scripts/download_common_voice_pt.py
```

Prepare o CSV operacional `pt-BR`:

```bash
python3 scripts/prepare_common_voice_metadata.py
```

Preprocesse os audios para WAV mono:

```bash
python3 scripts/preprocess_audio_dataset.py
```

Selecione os speakers da curadoria e gere o manifesto:

```bash
python3 scripts/select_speakers.py --speaker-target-count 1000
```

Esse parâmetro controla a curadoria dos speakers selecionados para o manifesto. Ele é diferente de `data.speaker_target_count` no YAML, que controla quantos `speaker_XX` entram no desenho consumido por `run_matrix.csv`.

Registre o inventário hierárquico dos dados:

```bash
python3 scripts/log_dataset_inventory.py
```

Por padrão, se `artifacts/dataset_inventory.json` já existir, o comando reaproveita esse arquivo e apenas imprime o mesmo relatório no console sem recalcular o inventário.

Para forçar uma nova geração e sobrescrever o JSON:

```bash
python3 scripts/log_dataset_inventory.py --override
```

Valide manifesto e prompts antes do treino:

```bash
python3 scripts/validate_manifest.py --check-files
```

Extraia os embeddings de síntese:

```bash
python3 scripts/extract_speaker_embeddings.py
```

Gere a matriz oficial de execução:

```bash
python3 scripts/generate_run_matrix.py
```

Inicialize o ledger `samples.csv`:

```bash
python3 scripts/init_samples.py
```

Execute o pipeline LoRA e materialize checkpoints:

```bash
python3 scripts/run_speecht5_lora.py
```

Sem `--condition`, o comando percorre todas as condicionais LoRA do YAML em sequência.

Transcreva os audios gerados com Whisper:

```bash
python3 scripts/run_whisper_batch.py
```

Calcule `WER`:

```bash
python3 scripts/compute_wer.py
```

Calcule `speaker_similarity`:

```bash
python3 scripts/compute_speaker_similarity.py
```

Calcule `NISQA`:

```bash
python3 scripts/compute_nisqa.py
```

O comando procura um checkout válido em `./NISQA` por padrão.

Calcule `F0 RMSE`:

```bash
python3 scripts/compute_f0_rmse.py
```

Agregue os resultados:

```bash
python3 scripts/aggregate_metrics.py
```

Gere os assets finais do relatório:

```bash
python3 scripts/make_report_assets.py
```

Abra a demo de inspeção qualitativa:

```bash
python3 demo/app.py
```

## Sobrescritas e Execucoes Avancadas

### Config e paths

Use `--config` quando quiser que os defaults sejam resolvidos a partir de outro YAML:

```bash
python3 scripts/generate_run_matrix.py --config configs/experimento_alternativo.yaml
python3 scripts/run_speecht5_lora.py --config configs/experimento_alternativo.yaml
python3 scripts/make_report_assets.py --config configs/experimento_alternativo.yaml
```

Quando sobrescrever paths manualmente:

- use isso se quiser gravar artefatos fora do layout canônico do repositório
- priorize sobrescrever apenas o que realmente mudou

Exemplos:

```bash
python3 scripts/prepare_common_voice_metadata.py \
  --tsv /mnt/dados/common_voice/validated.tsv \
  --clips-dir /mnt/dados/common_voice/clips \
  --out data/manifests/common_voice_metadata_alt.csv

python3 scripts/init_samples.py \
  --run-matrix artifacts/run_matrix_ablation.csv \
  --out artifacts/evaluation/samples_ablation.csv
```

Impacto esperado:

- você passa a desacoplar a execução do layout padrão do repo
- os próximos comandos precisam apontar para os novos artefatos, ou receber `--config` que resolva esses mesmos caminhos

### Selecao e escopo

Sobrescreva a curadoria dos speakers quando quiser variar o tamanho do subconjunto selecionado:

```bash
python3 scripts/select_speakers.py --speaker-target-count 200
```

Impacto esperado:

- muda o conjunto de speakers elegíveis no manifesto
- não altera, por si só, quantos `speaker_XX` entram na `run_matrix`

Para alterar o desenho consumido no treino e na inferência, edite o YAML:

- `data.speaker_target_count`
- `conditions[].speaker_subset_count`
- `conditions[].prompt_subset_count`

### Treino LoRA

Reexecute apenas um subconjunto de condicionais:

```bash
python3 scripts/run_speecht5_lora.py \
  --condition speecht5_lora_conservative \
  --condition speecht5_lora_unique
```

Sobrescreva globalmente a taxa de GPU da chamada atual:

```bash
python3 scripts/run_speecht5_lora.py --gpu-hourly-rate 1.75
```

Impacto esperado:

- `--condition` reduz o escopo do treino sem editar o YAML
- `--gpu-hourly-rate` só afeta a execução corrente; o valor estrutural continua no bloco `training` de cada condicional

Se ocorrer `CUDA out of memory` em `RTX 4090 24 GB`, o primeiro ajuste recomendado é:

- reduzir `per_device_train_batch_size` para `1`
- aumentar `gradient_accumulation_steps` para `4`

### Metricas

Force reprocessamento do Whisper mesmo com `asr_text` já preenchido:

```bash
python3 scripts/run_whisper_batch.py --all
```

Use outra coluna de transcrição para o cálculo de `WER`:

```bash
python3 scripts/compute_wer.py --asr-column custom_asr_text
```

Aponte explicitamente para outro checkout do `NISQA`:

```bash
python3 scripts/compute_nisqa.py --nisqa-path /opt/NISQA
```

Troque o modelo de similaridade de speaker apenas para a chamada atual:

```bash
python3 scripts/compute_speaker_similarity.py --model-name speechbrain/spkrec-ecapa-voxceleb
```

Impacto esperado:

- essas flags alteram a forma de avaliação, não a estrutura do experimento
- se você mudar uma métrica ou modelo de avaliação, regenere os agregados e `report_assets/`

### Outputs e execucao in-place

Os comandos que atualizam `samples.csv` escrevem no mesmo arquivo quando `--out` é omitido:

- `scripts/run_whisper_batch.py`
- `scripts/compute_wer.py`
- `scripts/compute_speaker_similarity.py`
- `scripts/compute_nisqa.py`
- `scripts/compute_f0_rmse.py`

Se quiser preservar o arquivo atual e gerar uma variante:

```bash
python3 scripts/compute_wer.py \
  --samples artifacts/evaluation/samples.csv \
  --out artifacts/evaluation/samples_wer_reprocessado.csv
```

O mesmo vale para `backfill`:

```bash
python3 scripts/backfill_training_costs.py \
  --gpu-hourly-rate 0.0 \
  --out artifacts/evaluation/samples_costs_manual.csv \
  --summary-out artifacts/evaluation/manual_training_costs_summary_alt.csv
```

Impacto esperado:

- sem `--out`, o fluxo principal permanece simples e incremental
- com `--out`, você cria ramificações de artefatos e precisa carregá-las explicitamente nas etapas seguintes

## Manutencao e cleanup

Preencha custos de treino depois da execução, quando necessário:

```bash
python3 scripts/backfill_training_costs.py --gpu-hourly-rate 0.0
```

Remova os rastros materializados de uma condicional:

```bash
python3 scripts/clear_speecht5_training_results.py \
  --condition speecht5_lora_unique
```

Remova várias condicionais materializadas:

```bash
python3 scripts/clear_speecht5_training_results.py \
  --condition speecht5_lora_conservative \
  --condition speecht5_lora_unique
```

Remova todos os treinos LoRA materializados do experimento:

```bash
python3 scripts/clear_speecht5_training_results.py --condition all
```

Esse cleanup remove checkpoints, audios materializados, linhas materializadas no `samples.csv`, agregados em `artifacts/evaluation/` e `report_assets/`.

Remova uma condicional inteira do experimento:

```bash
python3 scripts/remove_speecht5_condition.py \
  --condition speecht5_lora_unique
```

Esse fluxo encadeia o cleanup com `bypass` interno, remove as linhas base da condicional no `samples.csv`, remove a condicional do bloco `conditions:` do config e invalida `deliverables.run_matrix` quando o arquivo existir.

## Artefatos esperados

Ao final do pipeline canônico, espere encontrar:

- `artifacts/evaluation/samples.csv` com linhas materializadas e métricas preenchidas
- `artifacts/evaluation/metrics_summary.csv`
- `artifacts/evaluation/cost_summary.csv`
- `artifacts/evaluation/metrics_by_speaker.csv`
- `artifacts/evaluation/cost_by_speaker.csv`
- `report_assets/` com tabelas e gráficos
- `demo/app.py` abrindo e servindo audios reais

## Execucao via notebooks no RunPod/Jupyter

O diretório `notebooks/` contém a versão canônica deste mesmo fluxo para execução em pod com Jupyter e repositório clonado em `/workspace`.

Ordem recomendada:

- `notebooks/01_runpod_setup_e_dados.ipynb`
- `notebooks/02_treino_e_inferencia.ipynb`
- `notebooks/03_avaliacao_relatorio_e_demo.ipynb`

Os notebooks espelham o pipeline mínimo deste manual e executam os CLIs oficiais via `.venv/bin/python`.
