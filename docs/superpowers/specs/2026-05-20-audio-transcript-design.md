# Audio Transcript — MVP Design Spec

**Data:** 2026-05-20
**Autor:** Diego de Oliveira (com Claude)
**Status:** Aprovado, pronto pra writing-plans

## Contexto e problema

Diego participa de um pequeno grupo da igreja que se reúne quinzenalmente às quartas. Ele grava os pedidos de oração em áudio e precisa transcrever esse material. APIs hospedadas (OpenAI, AssemblyAI etc.) têm limite de tamanho de arquivo e custo recorrente. O computador dele já tem o app Handy instalado, que baixou localmente o modelo **Parakeet TDT 0.6B v3 (int8)** da NVIDIA (3 arquivos ONNX em `~/Library/Application Support/com.pais.handy/models/parakeet-tdt-0.6b-v3-int8/`, ~640MB).

O objetivo é construir uma ferramenta local que **reaproveite esse modelo** para transcrever gravações longas, quebrando o áudio em chunks automaticamente. Sem custo recorrente, sem dependência de internet, sem limite de tamanho de arquivo.

## Decisões de produto

| Item | Decisão |
|---|---|
| Modelo de STT | Parakeet TDT 0.6B v3 (reaproveitar arquivos do Handy) |
| Idioma alvo | Português brasileiro (PT-BR) |
| Interface | App web rodando em `localhost` |
| Formato de saída | Texto corrido com quebras de parágrafo nos silêncios longos |
| Pós-processamento (LLM cleanup, resumo) | **Fora do MVP** |
| Diarização (identificar quem falou) | Fora do MVP |
| Timestamps no output | Fora do MVP |
| Autenticação | Nenhuma (uso local single-user) |
| Persistência | Arquivos em disco; sem banco de dados |

## Stack escolhida

- **Backend:** Python 3.11+ com FastAPI
- **Inferência:** `sherpa-onnx` (bindings Python)
- **Processamento de áudio:** `ffmpeg` via subprocess
- **Frontend:** HTML + JavaScript vanilla servidos pelo próprio FastAPI (sem build step, sem framework)
- **Streaming de progresso:** Server-Sent Events (SSE)
- **Empacotamento:** `pyproject.toml` + `uv` ou `pip` (a definir no plano)

Alternativas consideradas e descartadas:
- Node/TypeScript com shell-out pro CLI do sherpa-onnx — mais lento por chunk, sem bindings nativos limpos.
- Script Python sem framework — sem SSE nativo, progresso teria que ser via polling.

## Arquitetura

```
Browser  ──HTTP/SSE──>  FastAPI  ──┬──>  ffmpeg (normalize + silencedetect + split)
                                    │
                                    ├──>  sherpa-onnx (Parakeet TDT, 1 chunk por vez)
                                    │
                                    └──>  Assembler (concat com \n\n → transcript.txt)
```

Single-process. Escutando em `localhost:8000`. Estado de cada job é um JSON em `data/uploads/<uuid>/job.json`.

## Endpoints

| Método | Rota | Função |
|---|---|---|
| `GET` | `/` | Serve `index.html` |
| `POST` | `/upload` | Recebe arquivo de áudio, salva, cria job em background, responde `{"job_id": "..."}` |
| `GET` | `/jobs/{id}/stream` | SSE com eventos de progresso (`stage`, `processed`, `total`, `eta_seconds`, `error`, `done`) |
| `GET` | `/jobs/{id}/result` | Download do `transcript.txt` |
| `GET` | `/jobs/{id}` | (Opcional, para debug) JSON com estado atual do job |

## Componentes (módulos Python)

```
audio_transcript/
├── app.py           # FastAPI: endpoints, SSE, background tasks
├── transcriber.py   # Singleton sherpa-onnx; transcribe(wav_path) -> str
├── chunker.py       # ffmpeg normalize + silencedetect + split
├── job.py           # JobState (dataclass) + persistência JSON
├── assembler.py     # Combina trechos em parágrafos
└── static/index.html
```

Cada módulo tem uma responsabilidade única e dá pra testar isolado:

- **`transcriber.py`** carrega o modelo uma única vez no boot do app (singleton). Expõe `transcribe(wav_path: Path) -> str`. Configura o sherpa-onnx com `--lang pt`. Localização do modelo configurável via env var `HANDY_MODELS_DIR` (default: `~/Library/Application Support/com.pais.handy/models/parakeet-tdt-0.6b-v3-int8/`).
- **`chunker.py`** expõe três funções: `normalize(src_path) -> wav_path`, `detect_silences(wav_path) -> list[(start, end)]`, `split(wav_path, silences) -> list[chunk_path]`.
- **`job.py`** define o dataclass `JobState` com campos `id`, `status` (`pending|normalizing|chunking|transcribing|assembling|done|failed`), `stage_progress`, `total_chunks`, `processed_chunks`, `error_message`, `source_path`, `result_path`. Métodos `save()` e `load(id)` lêem/escrevem `job.json`.
- **`assembler.py`** recebe lista de strings de chunks e produz o texto final, separando com `\n\n` e fazendo trim. Cada chunk vira um parágrafo.
- **`app.py`** orquestra: recebe upload, dispara `asyncio.create_task` rodando o pipeline, mantém broadcaster SSE por job.

## Fluxo de dados (do upload ao .txt)

1. **Upload (browser → `POST /upload`)**
   - Multipart form com o arquivo. Aceita qualquer formato que o ffmpeg leia (m4a, mp3, wav, ogg, flac, opus etc.).
   - Server cria `data/uploads/<uuid>/`, salva o arquivo como `source.<ext>`, cria `job.json` com status `pending`, dispara background task.
   - Responde imediatamente `200 {"job_id": "<uuid>"}`.

2. **Normalize (`chunker.normalize`)**
   - `ffmpeg -i source.<ext> -ac 1 -ar 16000 -c:a pcm_s16le normalized.wav`
   - Mono, 16kHz, PCM 16-bit — formato esperado pelo sherpa-onnx.
   - Atualiza `job.json`: `status = "normalizing"`.

3. **Detect silences (`chunker.detect_silences`)**
   - `ffmpeg -i normalized.wav -af silencedetect=noise=-30dB:duration=1.5 -f null -` (parse do stderr)
   - Extrai pares `(silence_start, silence_end)` dos logs do filtro.
   - Calcula pontos de corte preferindo silêncios > 1.5s.
   - Se algum trecho ficar > 90s sem silêncio, força corte adicional no meio (chunks muito grandes degradam qualidade do Parakeet TDT).
   - Retorna lista de intervalos `[(start_sec, end_sec), ...]` cobrindo o áudio completo.

4. **Split (`chunker.split`)**
   - Pra cada intervalo, gera `chunks/chunk_NNN.wav` via `ffmpeg -ss X -to Y -c copy` (corte sem reencode).
   - Atualiza `job.json`: `status = "chunking"`, `total_chunks = N`.

5. **Transcribe (`transcriber.transcribe`)**
   - Loop sequencial pelos chunks. Cada chunk é alimentado no sherpa-onnx, retorna string.
   - Após cada chunk, atualiza `processed_chunks++` e publica evento SSE com progresso e ETA estimado.
   - Sequencial no MVP. Paralelismo é otimização futura.

6. **Assemble (`assembler.combine`)**
   - Junta os textos dos chunks com `\n\n`.
   - Normalização leve: trim de espaços, capitalização do início de cada parágrafo se faltar, garantia de pontuação final.
   - Escreve `transcript.txt`.

7. **Done**
   - `job.json`: `status = "done"`, `result_path` setado.
   - SSE emite `event: done` e fecha a conexão.
   - Browser troca a UI pra mostrar botão "Baixar transcrição" (que aponta pra `/jobs/{id}/result`).

## UI (frontend)

`static/index.html` — uma página, ~150 linhas de JS vanilla:

- **Estado vazio:** zona drag-and-drop ocupando 70% da viewport, com texto "Arraste o áudio aqui ou clique pra escolher".
- **Estado em progresso:** barra de progresso (`processed/total`), nome do estágio atual ("Normalizando…", "Dividindo em chunks…", "Transcrevendo chunk 5 de 23…"), ETA estimado.
- **Estado concluído:** botão grande "Baixar transcrição (.txt)" + preview com primeiros 500 caracteres + botão "Nova transcrição" que recarrega a página.
- **Estado de erro:** caixa vermelha com a mensagem do erro vinda do SSE + botão "Tentar de novo".

EventSource (SSE nativo do browser) consome `/jobs/{id}/stream`. Reconexão automática em caso de drop.

## Tratamento de erros

| Falha | Comportamento |
|---|---|
| Arquivo não-áudio ou corrompido | ffmpeg falha na normalização → job vira `failed`, SSE emite `event: error` com mensagem `"Não consegui ler o arquivo de áudio. Confere se é um formato suportado."` |
| Modelo Handy ausente no path esperado | App falha no boot com erro claro: `"Modelo Parakeet não encontrado em <path>. Instale o Handy ou configure HANDY_MODELS_DIR."` |
| sherpa-onnx levanta exceção em um chunk | Loga warning, insere `[TRECHO NÃO TRANSCRITO: 00:12:34–00:13:01]` no lugar, continua os próximos chunks. Não derruba o job. |
| Disco cheio ou erro de IO | Job vira `failed` com a mensagem do erro original. |
| Cliente desconecta o SSE | Server continua processando. Reconexão lê o último estado do `job.json` e retoma envio de eventos. |
| Upload > 1GB | HTTP 413 com mensagem `"Arquivo grande demais. Limite atual: 1GB."` (configurável via env). |
| Áudio com 0s ou silêncio total | Detecção retorna 0 chunks → job vira `failed` com `"Não detectei fala no áudio."` |

Arquivos source/normalized/chunks **nunca** são apagados automaticamente. O usuário decide quando limpar `data/uploads/`. Botão "Limpar histórico" fica pra versão pós-MVP.

## Estratégia de testes (TDD)

Ordem de implementação:

1. **`test_assembler.py`** (pura lógica, sem deps externas)
   - Input: 3 strings de chunk → output esperado com 3 parágrafos separados por `\n\n`, trims aplicados.
   - Edge cases: chunks vazios filtrados; chunk único; whitespace excessivo.

2. **`test_chunker.py`** (depende de ffmpeg instalado)
   - Fixture: `tests/fixtures/two_speakers_10s.wav` (10s, duas falas curtas separadas por 2s de silêncio).
   - Asserções: `detect_silences` retorna 1 intervalo de silêncio; `split` produz 2 chunks; chunks cobrem o áudio inteiro.
   - Edge case: áudio sem silêncios suficientemente longos → corte forçado a cada 90s.

3. **`test_transcriber.py`**
   - CI: usa stub que retorna texto fake. Garante interface estável.
   - Local: roda fixture PT-BR de 3s com palavra-chave esperada (`"Senhor"` ou similar) e faz assert que aparece no output. Marcado `@pytest.mark.local`.

4. **`test_app.py`** (integração, transcriber stubado)
   - `TestClient` do FastAPI.
   - Upload de fixture pequena → consome SSE até `event: done` → baixa `/jobs/{id}/result` → assert que `.txt` é não-vazio e tem o conteúdo do stub.
   - Testa cenário de erro: upload de arquivo inválido → SSE emite `event: error`.

## Layout do repositório

```
audio-transcript/
├── audio_transcript/         # código da aplicação
│   ├── __init__.py
│   ├── app.py
│   ├── transcriber.py
│   ├── chunker.py
│   ├── job.py
│   ├── assembler.py
│   └── static/index.html
├── tests/
│   ├── fixtures/
│   ├── test_app.py
│   ├── test_assembler.py
│   ├── test_chunker.py
│   └── test_transcriber.py
├── data/                     # gitignored, criado em runtime
│   └── uploads/<uuid>/...
├── docs/superpowers/specs/   # este spec
├── pyproject.toml
├── README.md
└── .gitignore
```

## Risco aberto: compatibilidade dos ONNX do Handy com sherpa-onnx

Os arquivos ONNX que o Handy exportou (`encoder-model.int8.onnx`, `decoder_joint-model.int8.onnx`, `nemo128.onnx`) seguem o padrão de export que o time do sherpa-onnx usa para o Parakeet TDT, então **devem** funcionar direto. Mas existem riscos pequenos:

- Naming dos arquivos pode divergir do que a CLI/bindings do sherpa-onnx espera (ex: `encoder.onnx` vs `encoder-model.int8.onnx`).
- Metadata do modelo (vocab embutido vs externo, language tokens) pode estar configurada de forma diferente.

**Plano de mitigação:** A primeira tarefa do plano de implementação será uma spike de 30-60 min validando que o sherpa-onnx CLI consegue transcrever um wav de teste usando os arquivos do Handy. Se não funcionar:

- **Fallback 1:** Renomear/symlink os arquivos pro naming esperado pelo sherpa-onnx.
- **Fallback 2:** Baixar o pacote oficial do sherpa-onnx Parakeet TDT v3 do Hugging Face (~600MB, mesmo modelo subjacente, naming garantido). Custo de disco extra; produto final continua local e zero-cost.

Essa validação acontece antes de qualquer outra coisa no plano de implementação.

## Fora de escopo (explicitamente)

- Diarização de quem falou
- Timestamps no output
- Pós-processamento por LLM (limpeza, resumo)
- Histórico/UI de listar jobs anteriores
- Autenticação
- Deploy hospedado
- Suporte a múltiplos idiomas além de PT-BR
- Paralelização da inferência (chunks rodam sequencial)

## Critérios de aceitação do MVP

1. Subir um `.m4a` de ~30 min de uma reunião do grupo no browser.
2. Ver progresso em tempo real (chunks processados / total).
3. Baixar `transcript.txt` ao final, com parágrafos refletindo as pausas longas da gravação.
4. Custo: zero (nada de API paga, nada de download adicional além do que o Handy já trouxe — modulo o fallback).
5. Funciona offline.
