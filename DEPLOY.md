# Deploy — VPS (Hetzner + Caddy + Let's Encrypt)

## 1. Visão geral

Este guia coloca o `audio-transcript` rodando numa VPS Linux, atrás de um domínio próprio, com HTTPS automático via Let's Encrypt. O stack é Docker Compose: dois containers (`app` FastAPI/sherpa-onnx e `caddy` como reverse proxy). No final você tem `https://seu-dominio` servindo a UI de drag-and-drop, transcrição funcionando offline (modelo Parakeet local) e ad slots prontos para receber código real do AdSense quando a conta for aprovada.

## 2. Custo estimado

| Item | Custo | Observação |
|---|---|---|
| VPS Hetzner CX22 (2 vCPU / 4 GB RAM) | ~€4,5/mês (~R$ 25/mês) | Suficiente para uso pessoal. Para 2-3 usuários simultâneos vá de CCX13 (~€8,5/mês). |
| Domínio (`.com` / `.com.br`) | ~R$ 60/ano | Qualquer registrar (Registro.br, Cloudflare, Namecheap). |
| Serviços externos | R$ 0 | hCaptcha free tier; AdSense é pay-per-impression (revenue, não custo). |

## 3. Pré-requisitos

- VPS Linux (Ubuntu 22.04+ recomendado) com IPv4 público.
- Acesso SSH como root ou usuário com `sudo`.
- Docker Engine + plugin `docker compose` instalados.
- Domínio com registro A apontando para o IP da VPS (propagação pode levar até 1h).
- Portas 80 e 443 liberadas no firewall do Hetzner Cloud (Cloud Firewall do painel).
- Opcional: conta no [hCaptcha](https://www.hcaptcha.com) (free tier) para bloquear bot.
- Opcional: conta no [Google AdSense](https://adsense.google.com) (aprovação leva 1-4 semanas).

## 4. Passo a passo do deploy

```bash
# 1. SSH na VPS
ssh root@<IP_DA_VPS>

# 2. Instalar Docker + plugin compose
curl -fsSL https://get.docker.com | sh

# 3. Clonar o repo
git clone <repo url> /opt/audio-transcript
cd /opt/audio-transcript

# 4. Criar .env a partir do template
cp .env.example .env
nano .env
```

No `.env` preencha pelo menos:

```ini
DOMAIN=transcrever.seudominio.com
ACME_EMAIL=voce@seudominio.com
# Opcionais (deixe em branco se ainda não tem):
HCAPTCHA_SITEKEY=
HCAPTCHA_SECRET=
ADSENSE_CLIENT_ID=
```

```bash
# 5. Apontar DNS — no painel do registrar, criar registro A:
#    transcrever  →  <IP_DA_VPS>
#    Confirme com: dig +short transcrever.seudominio.com

# 6. Subir os containers
docker compose up -d

# 7. Acompanhar download do modelo Parakeet (~670 MB, só na primeira vez)
docker compose logs -f app
#    Quando aparecer "Uvicorn running on http://0.0.0.0:8000" o app está pronto.

# 8. Acompanhar o Caddy obter o certificado Let's Encrypt
docker compose logs -f caddy
#    Procure por "certificate obtained successfully".

# 9. Abrir no navegador
#    https://transcrever.seudominio.com
```

Pronto. Arraste um arquivo de áudio para a área de drop e veja a transcrição rodar.

## 5. Configurando hCaptcha (opcional, recomendado)

Sem hCaptcha qualquer bot pode martelar `/upload`. O rate limit por IP segura abuso básico, mas captcha é a primeira linha de defesa.

1. Criar conta em https://www.hcaptcha.com (free tier basta).
2. **Add a new site** com seu domínio. Copiar a **Site Key** e a **Secret Key**.
3. No `.env` da VPS:
   ```ini
   HCAPTCHA_SITEKEY=10000000-ffff-ffff-ffff-000000000001
   HCAPTCHA_SECRET=0xSEU_SECRET_AQUI
   ```
4. Recriar o container do app:
   ```bash
   docker compose up -d --force-recreate app
   ```
5. O widget aparece automaticamente abaixo da área de drop na home.

## 6. Configurando AdSense (opcional, demora pra aprovar)

Os ad slots já estão no HTML (topo e rodapé da home). Sem `ADSENSE_CLIENT_ID` eles renderizam como placeholder "Espaço para anúncio". Para ativar anúncios reais:

1. Aplicar em https://adsense.google.com.
2. Adicionar o site. **Requisito:** páginas de privacidade e termos públicas — este projeto já serve em `/privacy` e `/terms`.
3. Aguardar aprovação (1-4 semanas). Primeira tentativa é frequentemente rejeitada por "site com pouco conteúdo" — adicione algumas postagens/exemplos no domínio principal antes de reaplicar.
4. Quando aprovado, copiar o **Publisher ID** (formato `ca-pub-XXXXXXXXXXXXXXXX`).
5. No `.env`:
   ```ini
   ADSENSE_CLIENT_ID=ca-pub-XXXXXXXXXXXXXXXX
   ```
6. Recriar:
   ```bash
   docker compose up -d --force-recreate app
   ```
7. Os slots passam a carregar o script do AdSense e renderizar anúncios reais.

## 7. Operação dia-a-dia

```bash
# Ver logs do app em tempo real
docker compose logs -f app

# Restart só do app (sem mexer no Caddy / certificados)
docker compose restart app

# Atualizar para o último commit do repo
git pull
docker compose build app
docker compose up -d

# Backup dos uploads e jobs
docker run --rm \
  -v audio-transcript_data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/data.tar.gz /data

# Uso de disco dos volumes Docker
docker system df
du -sh /var/lib/docker/volumes/audio-transcript_data/_data/uploads/
```

> Os jobs antigos são deletados automaticamente após `AUDIO_TRANSCRIPT_RETENTION_DAYS` dias (default 7). Para desabilitar, defina como `0`.

## 8. Troubleshooting

- **Caddy não consegue obter certificado.** DNS ainda não propagou (`dig +short seu-dominio` deve retornar o IP da VPS) ou portas 80/443 estão bloqueadas pelo Hetzner Cloud Firewall. Cheque o painel do Hetzner em *Firewalls → Inbound Rules*.
- **Modelo nunca termina de baixar.** Sem acesso de saída para HuggingFace, sem espaço em disco, ou HF está fora. Cheque com `docker compose exec app df -h /var/lib/audio-transcript/models` e `docker compose logs app | grep -i download`.
- **Upload trava ou retorna 413.** O `Caddyfile` já está com `max_size 500MB`. Se precisa subir arquivo maior, ajuste lá **e** `AUDIO_TRANSCRIPT_MAX_UPLOAD_BYTES` no `.env`, depois `docker compose up -d --force-recreate`.
- **HTTP 429 sem motivo aparente.** Rate limit é 5 uploads/hora por IP. Espera 1h ou reinicia o container (o contador é in-memory).
- **Captcha sempre falha.** Confira se `HCAPTCHA_SITEKEY` e `HCAPTCHA_SECRET` não estão trocadas no `.env` — Site Key é pública (visível no HTML), Secret é privada (só backend).

## 9. Limites e quando escalar

O setup é single-instance: um worker FastAPI, rate limit e jobs em memória/disco local. Aguenta confortavelmente 1-3 usuários simultâneos numa VPS de 4-8 GB. Se virar produto público sério você vai precisar de: (a) Redis para rate limit e fila distribuídos, (b) múltiplos workers atrás de um load balancer, (c) provavelmente GPU para reduzir o tempo de transcrição abaixo de tempo-real em áudios longos, (d) storage compartilhado (S3/equivalente) no lugar do volume `data` local. Até lá, esse compose roda sozinho por meses sem manutenção.
