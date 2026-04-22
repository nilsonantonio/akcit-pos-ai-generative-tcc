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
# o manifesto agora inclui `target_text_speecht5`, frontend normalizado
# e auditavel consumido pelo treino do SpeechT5.
```

```bash
python3 scripts/validate_manifest.py \
  --manifest data/manifests/data_manifest.csv \
  --config configs/speecht5_minimal.yaml \
  --prompts data/prompts/ptbr_test_prompts.csv \
  --check-files

# 3s
# OK: data/manifests/data_manifest.csv (1200 rows)
# summary.prompts_commercial_subset_count=8
# summary.prompts_prompt_categories=abreviacoes, expressivas, homografos, interrogativas, neutras, numeros
# summary.prompts_prompt_subexperiment_count=8
# summary.speaker_count=6
# summary.speecht5_rows_with_unk_raw=938
# summary.speecht5_rows_with_unk_normalized=0
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

# se `data_manifest.csv` foi gerado antes da coluna `target_text_speecht5`,
# regenere com `scripts/select_speakers.py` e revalide com `--config`
# antes de rodar `few-shot` ou `LoRA`.

# 9m
# max_steps is given, it will override any value given in num_train_epochs
# {'loss': 0.9561, 'grad_norm': 1.2958765029907227, 'learning_rate': 9e-05, 'epoch': 0.5}                                                                                                                
# {'loss': 0.9786, 'grad_norm': 0.6477676033973694, 'learning_rate': 8e-05, 'epoch': 0.11}                                                                                                                
# {'eval_runtime': 4.8378, 'eval_samples_per_second': 4.134, 'eval_steps_per_second': 2.067, 'epoch': 0.11}                                                                                               
#  20%|████████████████████████████████▌                                                                                                                                  | 10/50 [00:22<00:41,  1.04s/it/Users/nilsonantonio/des/datascience/akcit/pos/generative-ai/14-tcc/src/.venv/lib/python3.11/site-packages/transformers/modeling_utils.py:2817: UserWarning: Moving the following attributes in the config to the generation config: {'max_length': 1876}. You are seeing this warning because you've set generation parameters in the model config, as opposed to in the generation config.
#   warnings.warn(
# {'loss': 1.0257, 'grad_norm': 0.5738189220428467, 'learning_rate': 7e-05, 'epoch': 0.16}                                                                                                                
# {'loss': 0.9894, 'grad_norm': 0.7110907435417175, 'learning_rate': 6e-05, 'epoch': 0.22}                                                                                                                
# {'eval_runtime': 0.9345, 'eval_samples_per_second': 21.403, 'eval_steps_per_second': 10.701, 'epoch': 0.22}                                                                                             
# {'loss': 1.0267, 'grad_norm': 0.7073429226875305, 'learning_rate': 5e-05, 'epoch': 0.27}                                                                                                                
# {'loss': 0.9622, 'grad_norm': 0.45031675696372986, 'learning_rate': 4e-05, 'epoch': 0.33}                                                                                                               
# {'eval_runtime': 0.8906, 'eval_samples_per_second': 22.456, 'eval_steps_per_second': 11.228, 'epoch': 0.33}                                                                                             
# {'loss': 0.9357, 'grad_norm': 0.6621122360229492, 'learning_rate': 3e-05, 'epoch': 0.38}                                                                                                                
# {'loss': 1.1497, 'grad_norm': 1.1423869132995605, 'learning_rate': 2e-05, 'epoch': 0.44}                                                                                                                
# {'eval_runtime': 0.9096, 'eval_samples_per_second': 21.987, 'eval_steps_per_second': 10.994, 'epoch': 0.44}                                                                                             
# {'loss': 0.93, 'grad_norm': 0.45452338457107544, 'learning_rate': 1e-05, 'epoch': 0.49}                                                                                                                 
# {'loss': 0.9209, 'grad_norm': 0.5435110926628113, 'learning_rate': 0.0, 'epoch': 0.55}                                                                                                                  
# {'eval_runtime': 0.8721, 'eval_samples_per_second': 22.933, 'eval_steps_per_second': 11.467, 'epoch': 0.55}                                                                                             
# {'train_runtime': 51.8704, 'train_samples_per_second': 1.928, 'train_steps_per_second': 0.964, 'train_loss': 0.9874877071380616, 'epoch': 0.55}                                                         
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:51<00:00,  1.04s/it]
# max_steps is given, it will override any value given in num_train_epochs
# {'loss': 1.0649, 'grad_norm': 0.6341679692268372, 'learning_rate': 9e-05, 'epoch': 0.06}                                                                                                                
# {'loss': 1.0195, 'grad_norm': 0.7519306540489197, 'learning_rate': 8e-05, 'epoch': 0.12}                                                                                                                
# {'eval_runtime': 4.3354, 'eval_samples_per_second': 4.152, 'eval_steps_per_second': 2.076, 'epoch': 0.12}                                                                                               
#  20%|████████████████████████████████▌                                                                                                                                  | 10/50 [00:09<00:21,  1.82it/s/Users/nilsonantonio/des/datascience/akcit/pos/generative-ai/14-tcc/src/.venv/lib/python3.11/site-packages/transformers/modeling_utils.py:2817: UserWarning: Moving the following attributes in the config to the generation config: {'max_length': 1876}. You are seeing this warning because you've set generation parameters in the model config, as opposed to in the generation config.
#   warnings.warn(
# {'loss': 1.0505, 'grad_norm': 0.510928750038147, 'learning_rate': 7e-05, 'epoch': 0.18}                                                                                                                 
# {'loss': 1.0071, 'grad_norm': 0.5775542259216309, 'learning_rate': 6e-05, 'epoch': 0.24}                                                                                                                
# {'eval_runtime': 0.8198, 'eval_samples_per_second': 21.957, 'eval_steps_per_second': 10.979, 'epoch': 0.24}                                                                                             
# {'loss': 0.9617, 'grad_norm': 0.5944191813468933, 'learning_rate': 5e-05, 'epoch': 0.3}                                                                                                                 
# {'loss': 0.9833, 'grad_norm': 0.43759462237358093, 'learning_rate': 4e-05, 'epoch': 0.37}                                                                                                               
# {'eval_runtime': 0.8008, 'eval_samples_per_second': 22.478, 'eval_steps_per_second': 11.239, 'epoch': 0.37}                                                                                             
# {'loss': 0.9145, 'grad_norm': 0.5649120211601257, 'learning_rate': 3e-05, 'epoch': 0.43}                                                                                                                
# {'loss': 1.0253, 'grad_norm': 1.077677607536316, 'learning_rate': 2e-05, 'epoch': 0.49}                                                                                                                 
# {'eval_runtime': 0.7354, 'eval_samples_per_second': 24.478, 'eval_steps_per_second': 12.239, 'epoch': 0.49}                                                                                             
# {'loss': 1.0342, 'grad_norm': 0.44792261719703674, 'learning_rate': 1e-05, 'epoch': 0.55}                                                                                                               
# {'loss': 1.0372, 'grad_norm': 0.8737694621086121, 'learning_rate': 0.0, 'epoch': 0.61}                                                                                                                  
# {'eval_runtime': 0.6935, 'eval_samples_per_second': 25.953, 'eval_steps_per_second': 12.977, 'epoch': 0.61}                                                                                             
# {'train_runtime': 27.188, 'train_samples_per_second': 3.678, 'train_steps_per_second': 1.839, 'train_loss': 1.0098110961914062, 'epoch': 0.61}                                                          
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:27<00:00,  1.84it/s]
# max_steps is given, it will override any value given in num_train_epochs
# {'loss': 0.9866, 'grad_norm': 0.9115402698516846, 'learning_rate': 9e-05, 'epoch': 0.07}                                                                                                                
# {'loss': 0.8643, 'grad_norm': 0.9107004404067993, 'learning_rate': 8e-05, 'epoch': 0.14}                                                                                                                
# {'eval_runtime': 3.1785, 'eval_samples_per_second': 5.034, 'eval_steps_per_second': 2.517, 'epoch': 0.14}                                                                                               
#  20%|████████████████████████████████▌                                                                                                                                  | 10/50 [00:08<00:15,  2.55it/s/Users/nilsonantonio/des/datascience/akcit/pos/generative-ai/14-tcc/src/.venv/lib/python3.11/site-packages/transformers/modeling_utils.py:2817: UserWarning: Moving the following attributes in the config to the generation config: {'max_length': 1876}. You are seeing this warning because you've set generation parameters in the model config, as opposed to in the generation config.
#   warnings.warn(
# {'loss': 0.8373, 'grad_norm': 0.5468085408210754, 'learning_rate': 7e-05, 'epoch': 0.21}                                                                                                                
# {'loss': 1.0219, 'grad_norm': 0.9107628464698792, 'learning_rate': 6e-05, 'epoch': 0.28}                                                                                                                
# {'eval_runtime': 0.6064, 'eval_samples_per_second': 26.386, 'eval_steps_per_second': 13.193, 'epoch': 0.28}                                                                                             
# {'loss': 0.838, 'grad_norm': 0.5631428956985474, 'learning_rate': 5e-05, 'epoch': 0.35}                                                                                                                 
# {'loss': 0.7663, 'grad_norm': 0.525170087814331, 'learning_rate': 4e-05, 'epoch': 0.42}                                                                                                                 
# {'eval_runtime': 0.5969, 'eval_samples_per_second': 26.807, 'eval_steps_per_second': 13.403, 'epoch': 0.42}                                                                                             
# {'loss': 0.7263, 'grad_norm': 0.7765266299247742, 'learning_rate': 3e-05, 'epoch': 0.49}                                                                                                                
# {'loss': 0.84, 'grad_norm': 1.1438319683074951, 'learning_rate': 2e-05, 'epoch': 0.56}                                                                                                                  
# {'eval_runtime': 0.6187, 'eval_samples_per_second': 25.859, 'eval_steps_per_second': 12.929, 'epoch': 0.56}                                                                                             
# {'loss': 1.1965, 'grad_norm': 0.5788520574569702, 'learning_rate': 1e-05, 'epoch': 0.63}                                                                                                                
# {'loss': 0.8994, 'grad_norm': 0.4456580877304077, 'learning_rate': 0.0, 'epoch': 0.7}                                                                                                                   
# {'eval_runtime': 0.6492, 'eval_samples_per_second': 24.647, 'eval_steps_per_second': 12.323, 'epoch': 0.7}                                                                                              
# {'train_runtime': 23.944, 'train_samples_per_second': 4.176, 'train_steps_per_second': 2.088, 'train_loss': 0.8976590013504029, 'epoch': 0.7}                                                           
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:23<00:00,  2.09it/s]
# max_steps is given, it will override any value given in num_train_epochs
# {'loss': 1.9281, 'grad_norm': 1.2863519191741943, 'learning_rate': 9e-05, 'epoch': 0.07}                                                                                                                
# {'loss': 1.8518, 'grad_norm': 1.4000122547149658, 'learning_rate': 8e-05, 'epoch': 0.14}                                                                                                                
# {'eval_runtime': 4.0832, 'eval_samples_per_second': 3.674, 'eval_steps_per_second': 1.959, 'epoch': 0.14}                                                                                               
#  20%|████████████████████████████████▌                                                                                                                                  | 10/50 [00:07<00:14,  2.70it/s/Users/nilsonantonio/des/datascience/akcit/pos/generative-ai/14-tcc/src/.venv/lib/python3.11/site-packages/transformers/modeling_utils.py:2817: UserWarning: Moving the following attributes in the config to the generation config: {'max_length': 1876}. You are seeing this warning because you've set generation parameters in the model config, as opposed to in the generation config.
#   warnings.warn(
# {'loss': 1.4593, 'grad_norm': 0.6593133807182312, 'learning_rate': 7e-05, 'epoch': 0.22}                                                                                                                
# {'loss': 2.3776, 'grad_norm': 0.8642070889472961, 'learning_rate': 6e-05, 'epoch': 0.29}                                                                                                                
# {'eval_runtime': 0.6057, 'eval_samples_per_second': 24.763, 'eval_steps_per_second': 13.207, 'epoch': 0.29}                                                                                             
# {'loss': 2.1365, 'grad_norm': 0.4899252951145172, 'learning_rate': 5e-05, 'epoch': 0.36}                                                                                                                
# {'loss': 1.2824, 'grad_norm': 0.6256305575370789, 'learning_rate': 4e-05, 'epoch': 0.43}                                                                                                                
# {'eval_runtime': 0.6279, 'eval_samples_per_second': 23.888, 'eval_steps_per_second': 12.74, 'epoch': 0.43}                                                                                              
# {'loss': 1.5105, 'grad_norm': 0.6664444804191589, 'learning_rate': 3e-05, 'epoch': 0.51}                                                                                                                
# {'loss': 1.7863, 'grad_norm': 0.8497591614723206, 'learning_rate': 2e-05, 'epoch': 0.58}                                                                                                                
# {'eval_runtime': 0.6527, 'eval_samples_per_second': 22.981, 'eval_steps_per_second': 12.257, 'epoch': 0.58}                                                                                             
# {'loss': 1.2592, 'grad_norm': 0.6392358541488647, 'learning_rate': 1e-05, 'epoch': 0.65}                                                                                                                
# {'loss': 2.4779, 'grad_norm': 0.7126353979110718, 'learning_rate': 0.0, 'epoch': 0.72}                                                                                                                  
# {'eval_runtime': 0.6327, 'eval_samples_per_second': 23.708, 'eval_steps_per_second': 12.644, 'epoch': 0.72}                                                                                             
# {'train_runtime': 21.6388, 'train_samples_per_second': 4.621, 'train_steps_per_second': 2.311, 'train_loss': 1.8069417953491211, 'epoch': 0.72}                                                         
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:21<00:00,  2.31it/s]
# max_steps is given, it will override any value given in num_train_epochs
# {'loss': 0.8522, 'grad_norm': 0.8468495607376099, 'learning_rate': 9e-05, 'epoch': 0.06}                                                                                                                
# {'loss': 0.8512, 'grad_norm': 0.6330819725990295, 'learning_rate': 8e-05, 'epoch': 0.12}                                                                                                                
# {'eval_runtime': 2.2229, 'eval_samples_per_second': 8.098, 'eval_steps_per_second': 4.049, 'epoch': 0.12}                                                                                               
#  20%|████████████████████████████████▌                                                                                                                                  | 10/50 [00:04<00:08,  4.99it/s/Users/nilsonantonio/des/datascience/akcit/pos/generative-ai/14-tcc/src/.venv/lib/python3.11/site-packages/transformers/modeling_utils.py:2817: UserWarning: Moving the following attributes in the config to the generation config: {'max_length': 1876}. You are seeing this warning because you've set generation parameters in the model config, as opposed to in the generation config.
#   warnings.warn(
# {'loss': 0.809, 'grad_norm': 0.5491392016410828, 'learning_rate': 7e-05, 'epoch': 0.18}                                                                                                                 
# {'loss': 0.8825, 'grad_norm': 0.8116652369499207, 'learning_rate': 6e-05, 'epoch': 0.24}                                                                                                                
# {'eval_runtime': 0.8451, 'eval_samples_per_second': 21.299, 'eval_steps_per_second': 10.65, 'epoch': 0.24}                                                                                              
# {'loss': 0.7509, 'grad_norm': 0.7849262952804565, 'learning_rate': 5e-05, 'epoch': 0.3}                                                                                                                 
# {'loss': 0.841, 'grad_norm': 0.7431350350379944, 'learning_rate': 4e-05, 'epoch': 0.37}                                                                                                                 
# {'eval_runtime': 0.8455, 'eval_samples_per_second': 21.289, 'eval_steps_per_second': 10.645, 'epoch': 0.37}                                                                                             
# {'loss': 0.7136, 'grad_norm': 0.7492908239364624, 'learning_rate': 3e-05, 'epoch': 0.43}                                                                                                                
# {'loss': 0.9082, 'grad_norm': 1.3734104633331299, 'learning_rate': 2e-05, 'epoch': 0.49}                                                                                                                
# {'eval_runtime': 0.7719, 'eval_samples_per_second': 23.318, 'eval_steps_per_second': 11.659, 'epoch': 0.49}                                                                                             
# {'loss': 0.7767, 'grad_norm': 0.7403693795204163, 'learning_rate': 1e-05, 'epoch': 0.55}                                                                                                                
# {'loss': 0.8169, 'grad_norm': 0.6594685912132263, 'learning_rate': 0.0, 'epoch': 0.61}                                                                                                                  
# {'eval_runtime': 0.8125, 'eval_samples_per_second': 22.153, 'eval_steps_per_second': 11.077, 'epoch': 0.61}                                                                                             
# {'train_runtime': 22.1237, 'train_samples_per_second': 4.52, 'train_steps_per_second': 2.26, 'train_loss': 0.8202208232879639, 'epoch': 0.61}                                                           
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:22<00:00,  2.26it/s]
# max_steps is given, it will override any value given in num_train_epochs
# {'loss': 1.1416, 'grad_norm': 0.8889824748039246, 'learning_rate': 9e-05, 'epoch': 0.03}                                                                                                                
# {'loss': 0.9642, 'grad_norm': 0.960250198841095, 'learning_rate': 8e-05, 'epoch': 0.07}                                                                                                                 
# {'eval_runtime': 5.554, 'eval_samples_per_second': 5.762, 'eval_steps_per_second': 2.881, 'epoch': 0.07}                                                                                                
#  20%|████████████████████████████████▌                                                                                                                                  | 10/50 [00:15<00:40,  1.01s/it/Users/nilsonantonio/des/datascience/akcit/pos/generative-ai/14-tcc/src/.venv/lib/python3.11/site-packages/transformers/modeling_utils.py:2817: UserWarning: Moving the following attributes in the config to the generation config: {'max_length': 1876}. You are seeing this warning because you've set generation parameters in the model config, as opposed to in the generation config.
#   warnings.warn(
# {'loss': 0.8825, 'grad_norm': 0.9224182367324829, 'learning_rate': 7e-05, 'epoch': 0.1}                                                                                                                 
# {'loss': 1.1032, 'grad_norm': 0.5055694580078125, 'learning_rate': 6e-05, 'epoch': 0.14}                                                                                                                
# {'eval_runtime': 1.1652, 'eval_samples_per_second': 27.463, 'eval_steps_per_second': 13.732, 'epoch': 0.14}                                                                                             
# {'loss': 0.9425, 'grad_norm': 0.8937988877296448, 'learning_rate': 5e-05, 'epoch': 0.17}                                                                                                                
# {'loss': 1.0511, 'grad_norm': 0.4373694062232971, 'learning_rate': 4e-05, 'epoch': 0.21}                                                                                                                
# {'eval_runtime': 1.1363, 'eval_samples_per_second': 28.163, 'eval_steps_per_second': 14.081, 'epoch': 0.21}                                                                                             
# {'loss': 1.0909, 'grad_norm': 0.6484680771827698, 'learning_rate': 3e-05, 'epoch': 0.24}                                                                                                                
# {'loss': 1.0997, 'grad_norm': 0.9060553312301636, 'learning_rate': 2e-05, 'epoch': 0.27}                                                                                                                
# {'eval_runtime': 1.1562, 'eval_samples_per_second': 27.678, 'eval_steps_per_second': 13.839, 'epoch': 0.27}                                                                                             
# {'loss': 1.0801, 'grad_norm': 0.6765602231025696, 'learning_rate': 1e-05, 'epoch': 0.31}                                                                                                                
# {'loss': 0.9424, 'grad_norm': 0.5113322138786316, 'learning_rate': 0.0, 'epoch': 0.34}                                                                                                                  
# {'eval_runtime': 1.1028, 'eval_samples_per_second': 29.016, 'eval_steps_per_second': 14.508, 'epoch': 0.34}                                                                                             
# {'train_runtime': 48.932, 'train_samples_per_second': 2.044, 'train_steps_per_second': 1.022, 'train_loss': 1.02981858253479, 'epoch': 0.34}                                                            
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:48<00:00,  1.02it/s]
# SpeechT5 few-shot pipeline completed
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
python3 scripts/validate_manifest.py --manifest data/manifests/data_manifest.csv --config configs/speecht5_minimal.yaml --prompts data/prompts/ptbr_test_prompts.csv --check-files
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
