# audio-transcript

Aplicativo web local que transcreve gravações de áudio em um `.txt` formatado por parágrafos, usando o modelo Parakeet TDT 0.6B v3 da NVIDIA via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). Pensado para quem quer transcrição self-hosted com privacidade — o áudio nunca sai da sua máquina. Tradeoff: custo zero por minuto, mas você precisa rodar o modelo em hardware próprio (CPU já basta, mas mais núcleos = mais rápido).

## Quick start com Docker (recomendado)

```bash
cp .env.example .env
docker compose up -d
# abra http://localhost
```

> HTTPS via Caddy só funciona com um domínio real apontando para o host. Para testar localmente, deixe `DOMAIN=localhost` no `.env` e acesse pela porta 80 sem TLS.

No primeiro boot o container baixa ~670 MB dos arquivos ONNX do Parakeet para o volume `models` — a UI demora alguns minutos até responder. Os downloads ficam persistidos no volume, então reinicializações são instantâneas.

## Quick start para desenvolvimento local (Python)

Requer macOS ou Linux (x86_64 / arm64), Python 3.11+ (testado em 3.13) e [ffmpeg](https://ffmpeg.org/) no `$PATH`.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

O app espera os quatro arquivos do modelo em `~/Library/Application Support/audio-transcript/models/sherpa-onnx-parakeet-v3-int8/` (ou no caminho de `PARAKEET_MODEL_DIR`). Baixe uma vez:

```bash
MODEL_DIR="$HOME/Library/Application Support/audio-transcript/models/sherpa-onnx-parakeet-v3-int8"
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"
BASE="https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/main"
curl -L -o encoder.int8.onnx "$BASE/encoder.int8.onnx"
curl -L -o decoder.int8.onnx "$BASE/decoder.int8.onnx"
curl -L -o joiner.int8.onnx  "$BASE/joiner.int8.onnx"
curl -L -o tokens.txt        "$BASE/tokens.txt"
```

Depois:

```bash
audio-transcript
# servidor sobe em http://127.0.0.1:8000
```

Abra a URL, arraste a gravação para a página, espere a barra de progresso encher, baixe o `.txt`.

## Deploy em VPS

O `docker-compose.yml` + `Caddyfile` deste repositório já incluem TLS automático via Let's Encrypt. Para passo a passo (DNS, firewall, dimensionamento de VPS, backups), veja [`DEPLOY.md`](./DEPLOY.md).

## Configuração

Todas as opções vêm de variáveis de ambiente (definidas em `audio_transcript/config.py` e `.env.example`).

| Variável | Default | Descrição |
|---|---|---|
| `DOMAIN` | `transcrever.example.com` | Domínio usado pelo Caddy para TLS. Use `localhost` para teste local. |
| `ACME_EMAIL` | `you@example.com` | Email da conta Let's Encrypt. |
| `PARAKEET_MODEL_DIR` | `~/Library/Application Support/audio-transcript/models/sherpa-onnx-parakeet-v3-int8` | Pasta com os arquivos ONNX do Parakeet. No Docker já vem apontando para `/var/lib/audio-transcript/models/...`. |
| `AUDIO_TRANSCRIPT_DATA_DIR` | `./data` | Onde ficam uploads e resultados. |
| `AUDIO_TRANSCRIPT_MAX_UPLOAD_BYTES` | `524288000` (500 MB no `.env.example`; `1073741824` / 1 GiB se a env var não estiver setada) | Rejeita uploads maiores que isto. |
| `AUDIO_TRANSCRIPT_PARALLELISM` | `4` | Quantos chunks são transcritos em paralelo. Aumente em VPSes maiores. |
| `AUDIO_TRANSCRIPT_RETENTION_DAYS` | `7` | Apaga jobs com mais de N dias automaticamente. `0` desativa a limpeza. |
| `AUDIO_TRANSCRIPT_HOST` | `127.0.0.1` (no Docker: `0.0.0.0`) | Host de bind. |
| `AUDIO_TRANSCRIPT_PORT` | `8000` | Porta de bind. |
| `AUDIO_TRANSCRIPT_SILENCE_DB` | `-30` | Limite em dB abaixo do qual considera-se silêncio. |
| `AUDIO_TRANSCRIPT_SILENCE_DUR` | `1.5` | Duração mínima (s) de silêncio para virar quebra de parágrafo. |
| `AUDIO_TRANSCRIPT_MAX_CHUNK_S` | `90` | Força corte após N segundos mesmo sem silêncio. |
| `HCAPTCHA_SITEKEY` | `""` (vazio = captcha desativado) | Sitekey do hCaptcha (tier gratuito em hcaptcha.com). |
| `HCAPTCHA_SECRET` | `""` | Secret correspondente. Precisa estar setado em par com a sitekey. |
| `ADSENSE_CLIENT_ID` | `""` (vazio = slots vazios) | Publisher ID do Google AdSense no formato `ca-pub-XXXXXXXXXXXXXXXX`. |

## Testes

```bash
pytest -m "not local"   # CI-safe (não usa modelo real)
pytest -m local         # Roda contra o Parakeet real (exige PARAKEET_MODEL_DIR populado)
```

## Troubleshooting

- **`Parakeet model directory not found`** no boot: rode o download da seção de desenvolvimento local, ou aponte `PARAKEET_MODEL_DIR` para um diretório com os quatro arquivos do modelo. No Docker, espere o entrypoint terminar o download na primeira inicialização.
- **Upload trava ou dá timeout**: confirme que `ffmpeg` está no `$PATH` (`which ffmpeg`).
- **Transcrição vazia**: o áudio pode estar inteiro abaixo do threshold de silêncio. Tente baixar `AUDIO_TRANSCRIPT_SILENCE_DB` (ex.: `-40`).
- **Chunks saindo como `[TRECHO NÃO TRANSCRITO]`**: erro de inferência naquele chunk. Rodar o mesmo arquivo de novo normalmente resolve.
- **HTTP 429 no upload**: rate limit por IP (5 uploads/hora). Espere uma hora ou suba de outro IP.
- **"Captcha inválido"**: `HCAPTCHA_SITEKEY` e `HCAPTCHA_SECRET` precisam ser um par válido do mesmo site no painel do hCaptcha. Se quiser desativar o captcha, deixe os dois vazios.
- **Anúncios não aparecem**: `ADSENSE_CLIENT_ID` está vazio ou a conta do AdSense ainda não foi aprovada. Com o ID vazio, os slots de ad são renderizados como placeholders vazios.

## Como funciona

1. Upload escreve o arquivo em `data/uploads/<uuid>/source.<ext>`.
2. `ffmpeg` normaliza para WAV mono 16 kHz PCM.
3. O filtro `silencedetect` do `ffmpeg` localiza pausas ≥ 1.5 s (configurável).
4. O áudio é dividido em chunks nessas pausas (forçando corte após 90 s se não houver pausa).
5. Cada chunk é transcrito pelo sherpa-onnx (Parakeet TDT). Chunks rodam em paralelo (4 workers por padrão, via `AUDIO_TRANSCRIPT_PARALLELISM` + `asyncio.Semaphore`). Se um chunk falha, o pipeline insere `[TRECHO NÃO TRANSCRITO: hh:mm:ss–hh:mm:ss]` e continua.
6. Os textos dos chunks são unidos por linhas em branco — cada chunk vira um parágrafo.

## Fora de escopo (deliberado)

- Diarização de falantes
- Timestamps dentro do `.txt`
- Pós-processamento com LLM (limpeza / resumo)
- Multi-tenancy / contas de usuário

Veja `docs/superpowers/specs/2026-05-20-audio-transcript-design.md` para o design completo e `docs/superpowers/spike/2026-05-20-onnx-compat-result.md` para a investigação de compatibilidade do modelo.
