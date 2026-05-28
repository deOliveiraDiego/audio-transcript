# Deploy co-locado na VPS da daltonlab (IP:porta, HTTP)

Guia para subir o `audio-transcript` **na mesma VPS** que já roda a stack do
Obsidian (Caddy + CouchDB + Dozzle, em `/opt/daltonlab-infra`), acessível por
`http://<IP_DA_VPS>:8080` — sem subdomínio, sem TLS.

> Para o deploy "produto" com domínio próprio + HTTPS via Caddy, veja
> [`../DEPLOY.md`](../DEPLOY.md). Este guia é o caminho interno/rápido.

## 1. Como ele convive com a stack existente

- **Não sobe Caddy.** O Caddy da daltonlab já ocupa as portas 80/443. Este
  deploy publica o app direto numa porta do host (8080) e o acesso é por
  IP:porta. Os dois Caddys brigariam pelas portas — por isso usamos
  `docker-compose.vps.yml`, que **não** inclui o serviço `caddy`.
- **Stacks independentes.** É um `docker compose` próprio (projeto
  `audio-transcript`), separado do `/opt/daltonlab-infra`. Atualiza, reinicia
  e derruba sem tocar no Obsidian.
- **Dozzle pega de graça.** O Dozzle lê o `docker.sock` do host, então o
  container `audio-transcript` aparece sozinho em
  `https://dozzle.daltonlab.ai` (login `diego`) assim que sobe. É ali que você
  observa logs e uso de CPU/RAM da transcrição.

## 2. ⚠️ Trade-offs deste modo

- **HTTP puro, sem TLS.** O upload do áudio trafega sem criptografia. Aceitável
  para uso interno; se um dia virar público, migre para o caminho com domínio +
  Caddy do `DEPLOY.md`.
- **Porta aberta = qualquer um com o IP:porta acessa.** Mitigação recomendada:
  no Cloud Firewall do Hetzner, restrinja o inbound da porta 8080 aos IPs
  conhecidos (casa/escritório) em vez de `0.0.0.0/0`. Alternativa: ligar
  hCaptcha (`HCAPTCHA_*` no `.env`).
- **Recurso é apertado.** A VPS é 2 vCPU / 3.7 GB e divide com o CouchDB. O
  compose limita o app a `mem_limit=2g` / `cpus=1.5` e `PARALLELISM=2` para não
  sufocar o sync do Obsidian. Áudios longos vão demorar; acompanhe pelo Dozzle.

## 3. Passo a passo

```bash
# 1. SSH na VPS (mesma da daltonlab)
ssh root@178.105.240.207

# 2. Clonar o repo do audio-transcript (separado da infra do Obsidian)
git clone <repo url do audio-transcript> /opt/audio-transcript
cd /opt/audio-transcript

# 3. Criar .env a partir do template deste modo
cp .env.vps.example .env
nano .env            # confira AT_HOST_PORT, limites e PARALLELISM

# 4. Subir (usa o compose SEM Caddy)
docker compose -f docker-compose.vps.yml up -d

# 5. Acompanhar o download do modelo Parakeet (~670MB, só na 1ª vez)
docker compose -f docker-compose.vps.yml logs -f app
#    Pronto quando aparecer "Uvicorn running on http://0.0.0.0:8000".
```

### 4. Abrir a porta no Cloud Firewall do Hetzner

No painel do Hetzner Cloud → **Firewalls → Inbound Rules**, adicione:

| Protocolo | Porta | Origem |
|---|---|---|
| TCP | 8080 | seus IPs conhecidos (recomendado) ou `0.0.0.0/0` |

Sem essa regra a porta fica bloqueada de fora, mesmo com o container no ar.

### 5. Acessar

```
http://178.105.240.207:8080
```

Arraste um áudio para a área de drop e baixe o `.txt`. Verifique no Dozzle
(`https://dozzle.daltonlab.ai`) que o container `audio-transcript` aparece e
que o CouchDB segue saudável durante uma transcrição pesada.

## 6. Operação

```bash
cd /opt/audio-transcript

# Logs
docker compose -f docker-compose.vps.yml logs -f app

# Restart
docker compose -f docker-compose.vps.yml restart app

# Atualizar para o último commit
git pull
docker compose -f docker-compose.vps.yml build app
docker compose -f docker-compose.vps.yml up -d

# Derrubar (não afeta a stack do Obsidian)
docker compose -f docker-compose.vps.yml down
```

> Os volumes `audio-transcript_models` e `audio-transcript_data` são próprios
> deste projeto e não colidem com os volumes da daltonlab. O modelo baixado
> fica persistido em `models`, então restarts são instantâneos.
