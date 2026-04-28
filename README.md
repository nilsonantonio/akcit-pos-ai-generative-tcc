# Framework Reproduzivel para LoRA em TTS PT-BR

Este repositório implementa um framework reproduzível para experimentar `LoRA` sobre `SpeechT5` em síntese de fala em português brasileiro, com avaliação automática por checkpoint, agregação estatística e inspeção qualitativa por app.

O objetivo não é apenas "treinar um modelo", mas criar um fluxo repetível para responder perguntas como:

- qual configuração de `LoRA` adapta melhor um pequeno conjunto de speakers PT-BR
- em que ponto do treino cada condição começa a melhorar ou degradar
- qual tradeoff aparece entre inteligibilidade, similaridade de speaker, custo e latência
- como comparar checkpoints intermediários de forma justa, e não apenas o checkpoint final

O repositório cobre o ciclo completo:

- preparação e curadoria do dataset
- seleção de speakers e montagem de manifesto
- treino `LoRA`
- geração de áudio por checkpoint salvo
- cálculo de métricas automáticas
- agregação de resultados
- geração de assets para relatório
- app Gradio para comparação A/B/C

Para setup e execução operacional:

- veja [SETUP_AMBIENTES.md](SETUP_AMBIENTES.md)
- veja [MANUAL_EXECUCAO.md](MANUAL_EXECUCAO.md)

## Execucao via Jupyter/RunPod

Para execucao no RunPod com Jupyter e o repositorio clonado em `/workspace`, use os notebooks canonicos em `notebooks/`:

- `notebooks/01_runpod_setup_e_dados.ipynb`
- `notebooks/02_treino_e_inferencia.ipynb`
- `notebooks/03_avaliacao_relatorio_e_demo.ipynb`

Eles espelham o pipeline do manual e executam os CLIs oficiais do projeto via `.venv/bin/python`, sem depender do kernel da `.venv`.

Se a `.venv` ainda nao existir no pod, o notebook `01` inclui uma celula opcional para executar `bash scripts/bootstrap_runpod.sh`.

## Execucao operacional

O passo a passo oficial do pipeline está em [MANUAL_EXECUCAO.md](MANUAL_EXECUCAO.md).

Este `README.md` fica focado em:

- explicar o desenho do framework
- esclarecer os contratos de dados e artefatos
- registrar as decisões conceituais do pipeline

O manual operacional fica focado em:

- comandos mínimos do fluxo canônico
- sobrescritas e modos avançados
- manutenção e cleanup

## Como a interface CLI foi pensada

Os CLIs do projeto seguem três regras simples:

1. o pipeline canônico deve funcionar com o menor número possível de flags
2. `--config` existe para resolver defaults a partir do YAML quando isso fizer sentido
3. parâmetros de intenção experimental continuam explícitos, mesmo quando paths triviais viram defaults

Na prática, isso significa:

- comandos de preparação, treino, métricas e agregação usam o layout canônico do repo por padrão
- quando `--config` é informado, os comandos priorizam os paths definidos em `data` e `deliverables`
- flags como `--condition`, `--gpu-hourly-rate` e `--speaker-target-count` continuam sendo sobrescritas deliberadas, não ruído operacional

Uma distinção importante:

- `select_speakers.py --speaker-target-count` controla a curadoria e o tamanho do subconjunto selecionado para o manifesto
- `speaker_selection.csv` controla quais `speaker_XX` entram no desenho consumido por `run_matrix.csv`

## O que este repositório é

Pense neste projeto como um pequeno framework experimental, e não como um único script de treino.

Ele define:

- um contrato de dados para sair do `Common Voice PT` e chegar a um manifesto estável
- um contrato de execução para transformar configurações YAML em uma `run_matrix`
- um contrato de avaliação para comparar amostras geradas por `checkpoint`
- um contrato de artefatos para que treino, métricas, relatórios e demo conversem entre si

Na prática, isso permite:

- trocar condições `LoRA` sem reescrever o pipeline
- repetir experimentos com o mesmo desenho
- comparar múltiplos checkpoints de uma mesma condição
- documentar resultados quantitativos e qualitativos com o mesmo conjunto de arquivos

## Intuição do pipeline

O fluxo ponta a ponta é:

1. `dados brutos`: baixar o `Common Voice PT`
2. `curadoria`: filtrar o subconjunto `pt-BR`, converter áudio e consolidar metadados
3. `manifesto`: transformar o subset curado em um contrato de treino/validação
4. `seleção de speakers`: escolher speakers globais por duração disponível
5. `embeddings`: materializar embeddings de síntese por speaker
6. `run matrix`: expandir a configuração YAML em execuções concretas
7. `ledger de samples`: criar o `samples.csv`, que será enriquecido ao longo do pipeline
8. `treino LoRA`: treinar cada condição e salvar checkpoints
9. `geração por checkpoint`: sintetizar áudio para cada checkpoint materializado
10. `métricas`: rodar `Whisper`, `WER`, `speaker similarity`, `NISQA`, `F0 RMSE`, custo e latência
11. `agregação`: resumir os resultados por condição analítica
12. `report/app`: gerar assets e abrir a comparação qualitativa entre checkpoints

A forma operacional mínima desse fluxo, com os CLIs reduzidos e as sobrescritas separadas, está documentada em [MANUAL_EXECUCAO.md](MANUAL_EXECUCAO.md).

A unidade analítica principal do framework é `checkpoint_label=<condicao>@step<k>`.

Isso é importante porque o projeto não assume que "o último checkpoint é o melhor". Cada checkpoint salvo pode ser tratado como um braço comparável do experimento. Essa escolha permite:

- observar a trajetória de aprendizagem
- detectar overfitting antes do final do treino
- comparar custo incremental contra ganho real de qualidade
- fazer comparação mais fina entre configurações `LoRA`

## Arquitetura do repositório

### `configs/`

Entrada:

- arquivos YAML com o desenho do experimento

Saída:

- não produz artefatos diretamente; governa como o resto do pipeline se comporta

Consumido por:

- geração de `run_matrix`
- treino `LoRA`
- métricas e resolução de defaults

Arquivo principal:

- `configs/speecht5_minimal.yaml`

### `data/raw/`

Entrada:

- dataset bruto baixado do `Common Voice`

Saída:

- base ainda próxima do formato original

Consumido por:

- preparação de metadados
- preprocessamento de áudio

### `data/processed/`

Entrada:

- subset curado e reformatado do áudio

Saída:

- WAV mono e metadados já alinhados ao pipeline

Consumido por:

- seleção de speakers
- treino e referência de síntese

### `data/manifests/`

Entrada:

- metadados preparados e regras do experimento

Saída:

- CSVs que formalizam o contrato de dados do pipeline

Consumido por:

- validação
- embeddings
- treino
- avaliação

Arquivos típicos:

- `common_voice_metadata.csv`
- `common_voice_processed.csv`
- `data_manifest.csv`
- `speaker_selection.csv`

### `artifacts/embeddings/`

Entrada:

- seleção final de speakers

Saída:

- embeddings usados na síntese do `SpeechT5`
- embeddings L2-normalizados com metadado `embedding_normalization=l2`

Consumido por:

- inicialização do `samples.csv`
- inferência de TTS

### `artifacts/checkpoints/`

Entrada:

- saídas do treino `LoRA`

Saída:

- diretórios `checkpoint-*` por condição

Consumido por:

- geração de áudio por checkpoint
- comparação analítica entre estados intermediários do treino

### `artifacts/audio/`

Entrada:

- checkpoints e `samples.csv`

Saída:

- áudios sintetizados

Consumido por:

- `Whisper`
- métricas de speaker e pitch
- demo Gradio

### `artifacts/evaluation/`

Entrada:

- `samples.csv` enriquecido ao longo da execução

Saída:

- resumos agregados de métricas e custo

Consumido por:

- `report_assets`
- app de comparação

### `report_assets/`

Entrada:

- `samples.csv`, `metrics_summary.csv`, `cost_summary.csv`

Saída:

- tabelas Markdown
- overview
- gráficos PNG quando `matplotlib` está disponível

Consumido por:

- escrita de relatório
- apresentação final

### `scripts/`

Entrada:

- comandos de linha de comando

Saída:

- wrappers operacionais do pacote `tcc_audio`

Consumido por:

- setup
- execução do pipeline
- rotinas auxiliares

### `tcc_audio/`

Entrada:

- implementação do framework

Saída:

- lógica de negócio reutilizada pelos scripts

Consumido por:

- todos os CLIs do projeto

### `demo/`

Entrada:

- `samples.csv` concluído

Saída:

- interface Gradio para comparar áudios e métricas

Consumido por:

- inspeção qualitativa final

### Documentos operacionais

- [SETUP_AMBIENTES.md](SETUP_AMBIENTES.md): preparação de ambiente
- [MANUAL_EXECUCAO.md](MANUAL_EXECUCAO.md): execução passo a passo

## Artefatos gerados

### `common_voice_metadata.csv`

Primeiro CSV operacional do pipeline. Representa o subset `pt-BR` extraído do `Common Voice`, ainda próximo da origem, mas já filtrado por locale/variant e alinhado ao projeto.

### `common_voice_processed.csv`

Versão processada após preprocessamento do áudio. Serve de base para a seleção de speakers.

### `data_manifest.csv`

Contrato principal de treino/validação. Cada linha materializa:

- speaker selecionado
- utterance
- split
- caminho de áudio
- texto alvo
- texto normalizado para `SpeechT5`

### `speaker_selection.csv`

Resumo da seleção final de speakers, com duração total, áudio de referência e observações para auditoria manual.

### `dataset_inventory.json`

Inventário hierárquico do estado do dataset e da amostragem. É útil para verificar rapidamente o que existe em `raw`, `processed` e no subconjunto de speakers escolhidos.

O comando `python3 scripts/log_dataset_inventory.py` usa esse arquivo como cache por padrão: se ele já existir, o conteúdo salvo é reapresentado no console sem recalcular o inventário. Use `python3 scripts/log_dataset_inventory.py --override` para reconstruir o inventário e sobrescrever o JSON.

### `run_matrix.csv`

Expansão do YAML em execuções concretas. Aqui o desenho experimental vira linhas reais combinando:

- condição
- speaker
- prompt
- variante de texto

### `samples.csv`

Ledger incremental do experimento. Este é o arquivo central do framework.

Ele começa com linhas base por run e vai sendo enriquecido com:

- `checkpoint_label`
- caminhos de áudio
- timestamps
- transcrição ASR
- métricas
- custo
- status

Se você quiser entender "o estado atual do experimento", normalmente é aqui que deve olhar primeiro.

### `metrics_summary.csv`

Resumo agregado por condição analítica e `text_variant`, com médias, medianas, desvio, intervalos de confiança e testes pareados quando aplicáveis.

### `cost_summary.csv`

Resumo agregado de custo, horas de GPU e latência média de inferência.

### `report_assets/`

Pacote final para comunicação de resultados:

- tabelas Markdown
- overview
- gráficos por métrica
- gráficos por speaker

## Intuição do pipeline canônico

### 1. Baixar o `Common Voice PT`

O pipeline começa com o dataset bruto, sem assumir que a estrutura local já está pronta. Isso reduz ambiguidade e torna o experimento reproduzível desde a origem dos dados.

### 2. Filtrar `pt-BR` e consolidar metadados

O projeto não usa o dataset "como veio". Primeiro ele seleciona o locale `pt` e a variant `pt-BR`, consolidando colunas como `gender`, `locale`, `variant`, texto e duração.

Essa etapa é importante porque o subset linguístico é parte do desenho experimental, não um detalhe operacional.

### 3. Preprocessar áudio

O áudio é convertido para um formato canônico do projeto, com organização estável em disco e cálculo explícito de duração. Isso reduz variabilidade entre etapas e simplifica a seleção e o treino.

### 4. Selecionar speakers e gerar o manifesto

Antes do ranking global, o framework aplica um recorte fixo no metadata:

- mantém apenas clips com duração entre `2s` e `8s`
- exige pelo menos `20` clips elegíveis por speaker
- exige pelo menos `60s` totais por speaker após esse filtro
- limita cada speaker a no máximo `120` clips, mantendo os mais longos

Depois disso, ele ranqueia speakers globalmente por duração disponível e seleciona os speakers que entram no `speaker_selection.csv`. Para cada speaker escolhido:

- o áudio é ordenado por duração
- clips são acumulados até o teto passado para `select_speakers.py`
- um subconjunto pequeno vira `val`
- o restante vira `train`
- o clip de referência é o primeiro selecionado, isto é, o de maior duração dentro do subconjunto escolhido

Essa política favorece speakers com mais material útil e mantém um split simples e reprodutível.

### 5. Extrair embeddings de síntese

Os embeddings extraídos aqui não são para avaliação. Eles são os vetores consumidos pelo `SpeechT5` no momento da síntese.

Esses embeddings de síntese são L2-normalizados no momento da extração e normalizados novamente no carregamento, de forma idempotente, para manter o mesmo contrato no treino e na inferência.

Por isso o `speaker_embedding_path` no `samples.csv` representa o embedding de síntese, não o embedding usado na métrica de similaridade.

Se você tiver artefatos antigos gerados com embeddings crus, trate a migração como breaking: reextraia `artifacts/embeddings/`, recrie `run_matrix` e `samples`, e reexecute treino, síntese e métricas antes de comparar resultados.

### 6. Expandir o YAML em `run_matrix`

Antes de treinar qualquer coisa, o framework materializa todas as combinações planejadas entre:

- condição
- speaker
- prompt
- variante de texto

Isso torna o desenho explícito e auditável. Você consegue inspecionar o experimento antes de gastar GPU.

### 7. Inicializar o ledger de samples

`samples.csv` nasce antes da inferência para funcionar como ledger do experimento. O pipeline não trata samples como efeitos colaterais dispersos; ele os trata como entidades rastreáveis.

### 8. Treinar LoRA e salvar checkpoints

Cada condição `LoRA` é treinada conforme o YAML. O treino salva checkpoints em passos definidos, e cada checkpoint passa a ser um candidato real de avaliação.

`training.gpu_hourly_rate` também é definido por condicional no YAML. O parâmetro `--gpu-hourly-rate` do entrypoint existe apenas como sobrescrita global opcional da execução.

### 9. Materializar áudio por checkpoint

Quando um checkpoint é salvo, o pipeline gera áudio com ele e preenche novas linhas no ledger. Isso transforma o histórico do treino em um conjunto comparável de saídas.

### 10. Rodar ASR e métricas

Depois da síntese:

- `Whisper` gera transcrições
- `WER` mede inteligibilidade
- `speaker_similarity` mede preservação de identidade
- `NISQA` aproxima qualidade perceptual
- `F0 RMSE` mede desvio de pitch
- `RTF`, custo e horas de GPU completam a visão operacional

### 11. Agregar e comparar

O agregador trabalha sobre `analysis_condition`, que prioriza `checkpoint_label` quando ele existe. Na prática, isso significa que a comparação principal não é apenas entre "condições", mas entre checkpoints concretos.

### 12. Gerar relatório e abrir a demo

O fim do pipeline produz:

- resumos agregados para análise quantitativa
- assets de relatório para apresentação
- uma interface para ouvir e comparar saídas

## Normalizações e filtros

### `raw_text` versus `normalized_text`

Os prompts carregam as duas visões:

- `raw_text`: texto original, mais próximo da formulação humana
- `normalized_text`: texto ajustado para ser mais estável para o `SpeechT5`

O framework opera com `text_variant`, então a variação entre texto cru e normalizado faz parte do desenho experimental.

### Auditoria de texto para `SpeechT5`

O manifesto pode carregar `target_text_speecht5`, e a validação confere se a forma normalizada ainda produz `<unk>` no tokenizer do modelo. Isso evita treinar e avaliar entradas linguisticamente mal representadas pelo modelo base.

### Filtros de dataset

O pipeline parte do `Common Voice PT`, mas o experimento usa explicitamente o recorte:

- `locale=pt`
- `variant=pt-BR`

Além disso, ele descarta durações inválidas e exige caminhos de áudio consistentes quando a checagem de arquivos é ativada.

### Restrição de duração

Na preparação de dados e no setup experimental aparecem duas noções de duração:

- duração real de cada clip
- teto de áudio acumulado por speaker na etapa `select_speakers.py`

Além disso, a curadoria compartilhada entre análise e seleção usa o recorte:

- `min_audio_duration_s=2`
- `max_audio_duration_s=8`
- `min_clips_per_speaker=20`
- `max_clips_per_speaker=120`
- `min_duration_per_speaker_s=60`

O objetivo é reduzir outliers, exigir cobertura mínima por speaker e manter um orçamento de adaptação comparável.

## Seleção de speakers e regras de amostragem

### `speaker_selection.csv`

Define quais `speaker_id` entram no experimento e em qual ordem eles são consumidos pelo `run_matrix.csv`.

### `minutes_per_speaker`

Continua existindo apenas na etapa `select_speakers.py`, como teto de áudio acumulado por speaker durante a curadoria do manifesto. Ele não faz mais parte do YAML experimental.

### Seleção global

A seleção atual é global por duração total disponível. O framework não balanceia mais por gênero nessa etapa. Isso simplifica a política de amostragem e torna explícita a prioridade por cobertura útil de áudio.

### Splits

Depois da escolha dos clips:

- uma fração pequena vai para `val`
- o restante vai para `train`

A função interna garante que speakers selecionados preservem ao menos um clip de treino.

## Como configurar LoRA

O coração do experimento está em `configs/speecht5_minimal.yaml`, especialmente no bloco `conditions:`.

Cada condição responde à pergunta: "como quero adaptar o modelo e como quero avaliar essa adaptação?"

Campos importantes:

- `id`: identificador estável da condição
- `label`: nome legível para análise e relatório
- `train_strategy`: indica que a condição usa `lora`
- `uses_speaker_embeddings`: sinaliza que a síntese consome embeddings de speaker
- `normalization_modes`: define se a run usa `raw`, `normalized` ou ambos
- `lora`: hiperparâmetros do adaptador
- `training`: hiperparâmetros do processo de treino

### Bloco `lora`

Campos principais:

- `r`: rank do adaptador
- `lora_alpha`: escala do adaptador
- `lora_dropout`: regularização
- `target_modules`: quais projeções do modelo serão adaptadas
- `bias`: política de bias

Em termos práticos:

- ranks menores tendem a ser mais conservadores
- ranks maiores aumentam capacidade, custo e risco de overfit

### Bloco `training`

Campos principais:

- `scope`
- `max_steps`
- `learning_rate`
- `warmup_ratio`
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `effective_batch_size`
- `fp16`
- `gradient_checkpointing`
- `save_steps`
- `eval_steps`

### `scope: per_speaker` versus `scope: unique`

`per_speaker`:

- treina uma unidade por speaker
- tende a focar adaptação mais localizada
- facilita leitura de comportamento individual

`unique`:

- treina uma unidade única agregando os dados previstos para a condição
- é útil quando a hipótese é aprender um comportamento mais compartilhado

## Dois exemplos de configuração LoRA

Os exemplos abaixo são didáticos. O perfil oficial do repositório continua sendo o que está no YAML versionado.

### Exemplo conservador

Este perfil é próximo da configuração oficial atual. A ideia é adaptar com parcimônia, manter batch efetivo modesto e preservar estabilidade em GPUs como a `RTX 4090 24 GB`.

```yaml
- id: speecht5_lora_conservative_example
  label: SpeechT5 LoRA Conservative Example
  train_strategy: lora
  uses_speaker_embeddings: true
  normalization_modes: [normalized]
  lora:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    bias: none
    target_modules: [q_proj, k_proj, v_proj, out_proj]
  training:
    scope: per_speaker
    gpu_hourly_rate: 0.0
    max_steps: 1500
    learning_rate: 3.0e-5
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 2
    effective_batch_size: 4
    per_device_eval_batch_size: 2
    fp16: true
    gradient_checkpointing: true
    save_steps: 500
    eval_steps: 500
```

Quando usar:

- primeiro experimento reprodutível
- budget de VRAM controlado
- maior preocupação com estabilidade do que com agressividade de adaptação

### Exemplo agressivo

Este perfil aumenta capacidade e pressão de otimização. Deve ser lido como ponto de partida experimental, com risco maior de OOM, instabilidade e overfit.

```yaml
- id: speecht5_lora_aggressive_example
  label: SpeechT5 LoRA Aggressive Example
  train_strategy: lora
  uses_speaker_embeddings: true
  normalization_modes: [normalized]
  lora:
    r: 32
    lora_alpha: 64
    lora_dropout: 0.05
    bias: none
    target_modules: [q_proj, k_proj, v_proj, out_proj]
  training:
    scope: unique
    gpu_hourly_rate: 0.0
    max_steps: 2500
    learning_rate: 5.0e-5
    per_device_train_batch_size: 4
    gradient_accumulation_steps: 2
    effective_batch_size: 8
    per_device_eval_batch_size: 4
    fp16: true
    gradient_checkpointing: false
    save_steps: 500
    eval_steps: 500
```

Tradeoffs esperados:

- maior capacidade de adaptação
- maior risco de overfit em poucos minutos por speaker
- maior pressão de VRAM
- mais chance de ganhos rápidos seguidos de degradação em checkpoints tardios

## Comparação entre checkpoints

Este é um dos pontos mais importantes do framework.

Cada checkpoint salvo:

- gera novas amostras
- recebe métricas com o mesmo conjunto de speakers e prompts
- entra nas agregações com um `checkpoint_label` explícito

Isso é melhor do que avaliar apenas o último checkpoint porque permite observar:

- onde uma condição realmente começa a melhorar
- se ganhos iniciais se sustentam
- se um checkpoint intermediário já é suficiente
- se o custo adicional de treinar mais compensa

No agregador:

- `analysis_condition` usa `checkpoint_label` quando disponível
- comparações pareadas usam `Wilcoxon`
- intervalos de confiança são estimados por bootstrap

Na prática, você consegue comparar:

- `condicao A @ step500` versus `condicao A @ step1000`
- `condicao A @ step500` versus `condicao B @ step500`
- efeitos por speaker
- efeitos por `text_variant`

## Validações, métricas, reports e app

### Validações

Antes do treino, o framework valida:

- colunas obrigatórias dos manifests
- valores válidos de `split` e `text_variant`
- duração positiva
- duplicatas por `speaker_id + utterance_id + split`
- coerência de prompts
- compatibilidade de texto com o tokenizer do `SpeechT5`

Esse passo existe para falhar cedo e evitar gastar GPU com entradas inconsistentes.

### Métricas

#### `WER`

Responde: "o que foi sintetizado é inteligível o suficiente para ser recuperado por ASR?"

#### `speaker_similarity`

Responde: "a voz sintetizada ainda parece pertencer ao speaker pretendido?"

#### `NISQA`

Responde: "qual a qualidade perceptual geral aproximada do áudio?"

#### `F0 RMSE`

Responde: "o comportamento melódico e de pitch se afastou muito da referência?"

#### `RTF`

Responde: "quão cara é a inferência em relação à duração do próprio áudio?"

#### `train_gpu_hours`, `inference_seconds`, `cost_usd`

Respondem: "quanto custou treinar e gerar esse braço do experimento?"

### Reports

O agregador produz:

- `metrics_summary.csv`
- `cost_summary.csv`
- `metrics_by_speaker.csv`
- `cost_by_speaker.csv`

Depois, `report_assets` transforma esses outputs em:

- tabelas Markdown
- overview do experimento
- gráficos por métrica
- gráficos por speaker

### App

O app Gradio é uma camada de inspeção qualitativa, não uma substituição das métricas.

Ele permite:

- escolher `prompt`
- escolher `speaker`
- comparar três condições analíticas em paralelo
- ouvir os três áudios
- visualizar métricas resumidas por amostra

Isso é particularmente útil quando duas condições parecem próximas numericamente, mas soam diferentes.

## Como começar

Se você acabou de baixar o repositório, a ordem recomendada é:

1. leia este `README.md` para entender o framework
2. prepare o ambiente em [SETUP_AMBIENTES.md](SETUP_AMBIENTES.md)
3. execute o pipeline em [MANUAL_EXECUCAO.md](MANUAL_EXECUCAO.md)

Se a sua meta for apenas testar novas condições `LoRA`, pense no ciclo abaixo:

1. editar o bloco `conditions:` do YAML
2. regenerar `run_matrix.csv`
3. inicializar ou reutilizar `samples.csv`
4. treinar e materializar checkpoints
5. comparar checkpoints nas métricas e no app

Para os comandos concretos de cada etapa, use [MANUAL_EXECUCAO.md](MANUAL_EXECUCAO.md) como fonte operacional primária.

Esse é o valor principal do repositório: oferecer um caminho reproduzível para transformar hipóteses de adaptação `LoRA` em experimentos comparáveis, auditáveis e comunicáveis.
