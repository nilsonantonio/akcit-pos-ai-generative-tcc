# Setup de Ambientes do TCC de Audio

Escolha um dos ambientes abaixo, conclua o setup correspondente e depois volte para [MANUAL_EXECUCAO.md](/Users/nilsonantonio/des/datascience/akcit/pos/generative-ai/14-tcc/src-light/MANUAL_EXECUCAO.md) na seção de pipeline.

## 1. Mac M2 Pro Max host-native com MPS

Objetivo: execucao local completa em GPU no host, sem Docker, usando `venv` do projeto.

Capacidade assumida:

- `64 GB` de memoria unificada
- `32-core GPU` integrada
- adequado para embeddings, metricas, pilotos LoRA e parte relevante do treino principal se couber em memoria

### Preparacao

```bash
bash scripts/bootstrap_macos_mps.sh
source .venv/bin/activate
```

O bootstrap:

- cria `.venv`
- instala `requirements-macos-mps.txt`, incluindo `pytest` para os smoke tests
- clona ou recria `NISQA` em `./NISQA`
- valida `./NISQA/.git`, `./NISQA/nisqa/NISQA_model.py` e `./NISQA/weights/nisqa_tts.tar`
- exige `ffmpeg` no host
- valida suporte do Python a `lzma/_lzma`
- valida `torch.backends.mps.is_available()`
- roda smoke tests do projeto

### Cheque rapido de `lzma`

```bash
python3 -c "import lzma, _lzma; print('lzma ok')"
```

Se esse comando falhar com `No module named '_lzma'`, o problema e do interpretador Python, nao do pipeline `run_speecht5_lora.py`. Em `macOS + Homebrew + asdf`, a recuperacao esperada e:

```bash
brew install xz
asdf uninstall python 3.11.12
asdf install python 3.11.12
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-macos-mps.txt
```

Ao final deste setup, volte para [MANUAL_EXECUCAO.md](/Users/nilsonantonio/des/datascience/akcit/pos/generative-ai/14-tcc/src-light/MANUAL_EXECUCAO.md) na seção de pipeline.

## 2. Linux local com NVIDIA + Docker

Objetivo: ambiente oficial local com Docker e GPU NVIDIA.

Requisitos:

- Linux nativo
- `RTX 3090` ou superior
- perfil de referencia validado: `RTX 4090 24 GB VRAM`, `41 GB RAM`, `6 vCPUs`
- driver NVIDIA instalado
- `NVIDIA Container Toolkit`

### Preparacao do host

```bash
bash scripts/bootstrap_nvidia_host.sh
```

O bootstrap:

- exige `docker`
- exige `docker compose`
- exige `nvidia-smi`
- exige `nvidia-ctk`
- valida a versao do driver NVIDIA
- valida o runtime NVIDIA no Docker com `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`

### Build e shell

```bash
docker compose -f docker/docker-compose.nvidia.yml build
```

```bash
docker compose -f docker/docker-compose.nvidia.yml run --rm tcc-audio-nvidia bash
```

### Bootstrap do container

Dentro do container:

```bash
bash scripts/bootstrap_nvidia_container.sh
```

O bootstrap do container:

- valida `python3`, `ffmpeg` e `git`
- valida `torch==2.6.0`
- valida `torch.version.cuda` com prefixo `12.4`
- valida `torch.cuda.is_available()`
- clona ou recria `NISQA` em `./NISQA`
- valida `./NISQA/.git`, `./NISQA/nisqa/NISQA_model.py` e `./NISQA/weights/nisqa_tts.tar`
- roda checagens leves de CLI do projeto

Ao final deste setup, volte para [MANUAL_EXECUCAO.md](/Users/nilsonantonio/des/datascience/akcit/pos/generative-ai/14-tcc/src-light/MANUAL_EXECUCAO.md) na seção de pipeline.

## 3. RunPod com NVIDIA host-native

Objetivo: ambiente oficial de producao do TCC.

Observacao:

- o pipeline oficial no RunPod roda no proprio host do pod
- o ambiente Python/CUDA fica na `.venv` do projeto, sem `docker compose`

### Bootstrap

No host do pod:

```bash
bash scripts/bootstrap_runpod.sh
```

Depois do bootstrap:

```bash
source .venv/bin/activate
```

Ao final deste setup, volte para [MANUAL_EXECUCAO.md](/Users/nilsonantonio/des/datascience/akcit/pos/generative-ai/14-tcc/src-light/MANUAL_EXECUCAO.md) na seção de pipeline.
