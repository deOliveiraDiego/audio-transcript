# Backlog — pós-MVP

Lista de melhorias deliberadamente fora do escopo do MVP. Cada item tem nota de motivação e onde encostar no código.

**Prioridade sugerida para a próxima sessão:**
1. UI: feedbacks visuais mais ricos

## ✅ Paralelização da inferência — entregue (commit fea1f07)

Bench em áudio real (2h25min, 98 chunks): 11min 44s → 5min 17s, 2.22x speedup, output byte-idêntico ao sequencial. Configurável via `AUDIO_TRANSCRIPT_PARALLELISM` (default 4). Validação de thread-safety em `docs/superpowers/spike/parallel_thread_safety.py`.

---

## UI: feedbacks visuais mais ricos

**Gap atual:** `audio_transcript/static/index.html` tem 3 estados (drop / progress / done) com barra simples + label de texto. Funciona, mas é pobre pra entender o que está acontecendo num job de 10-20 min (transcrição de 2h+).

**Ideias concretas (UX wishlist):**

- **Linha do tempo dos estágios** com checks verdes/spinners — `normalizing → chunking → transcribing → assembling → done`. Cada estágio com tempo decorrido.
- **Visualização da waveform** do áudio uploaded (Web Audio API via `<canvas>`), com chunks plotados como blocos sobrepostos. À medida que cada chunk é transcrito, o bloco correspondente "preenche".
- **Preview ao vivo da transcrição**: à medida que chunks chegam, mostrar o texto acumulado num `<pre>` que cresce, em vez de só preencher no final.
- **ETA real** (não só "transcrevendo 5 de 98"): calcular tempo médio por chunk processado e mostrar "faltam ~3 min".
- **Animação no progress bar** (shimmer / pulse) pra não parecer que travou em chunks longos.
- **Som de notificação** opcional quando o job completa (importante pra usuário que dispara e vai fazer outra coisa).
- **Estado de erro mais expressivo**: mostrar qual chunk falhou, com timestamp, e oferecer "transcrever só esse trecho de novo" (precisa endpoint novo).
- **Modo dark** (cosmético).
- **Acessibilidade**: ARIA labels nos botões, anúncio do progresso por screen reader.

**Onde encostar no código:**
- `audio_transcript/static/index.html` (frontend)
- `audio_transcript/pipeline.py` — talvez emitir mais granularidade nos eventos SSE (ex: `chunk_started` + `chunk_finished` separados, em vez de só `progress`)
- `audio_transcript/app.py` — endpoint pra streaming do texto acumulado se quisermos preview ao vivo

**Esforço estimado:** 1-2 dias pra pegar as 3-4 melhorias mais impactantes (timeline + ETA + preview ao vivo + animação).

---

## Histórico de jobs

**Gap:** Cada job é um diretório em `data/uploads/<uuid>/`. Não tem UI pra listar/abrir/deletar jobs anteriores.

**Ideia:** `/history` endpoint + tela que lista jobs com data, duração do áudio, status, link pro `.txt`.

---

## Limpeza automática de arquivos intermediários

**Gap:** `normalized.wav` (centenas de MB) e `chunks/` (várias dezenas de wavs) ficam em disco indefinidamente.

**Ideia:** Botão "limpar este job" que deleta tudo exceto `transcript.txt` + `job.json`. Ou auto-clean N dias após `done`.

---

## Modo CLI

**Gap:** Tudo via web. Útil ter `audio-transcript-cli reuniao.m4a` que pega o arquivo e cospe `reuniao.txt` direto, sem subir server.

**Onde encostar:** novo entrypoint em `pyproject.toml` apontando pra função que chama `run_pipeline` direto, sem broker/SSE.

---

## Diarização (quem falou)

**Gap:** Todos os parágrafos são anônimos. Pro caso de uso (pedidos de oração), seria ótimo ter `[Pessoa A] ...`.

**Ideia:** pyannote-audio rodando local. Mas: outro modelo (~500MB-1GB), processamento bem mais pesado, qualidade mediana em áudio casual com sobreposição.

**Status:** intencionalmente excluído do MVP. Volta a entrar se a transcrição crua se mostrar útil mas frustrante de ler.

---

## Pós-processamento via LLM

**Gap:** Texto bruto de fala casual tem muito "ééé", "tipo", "é né". Pra leitura humana, valeria limpar.

**Ideias:**
- Botão "limpar transcrição" que manda o texto pra Claude/OpenAI API. Custo: ~R$ 0,05-0,20 por reunião quinzenal.
- Ou: Ollama local com Llama 3.x ou similar. Sem custo, mais lento.

**Status:** intencionalmente excluído do MVP. Pode ser primeira feature pós-MVP se a leitura do texto cru se mostrar irritante.
