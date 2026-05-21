SHELL := /bin/bash

.DEFAULT_GOAL := help

COMPOSE := docker compose
DATA_VOLUME := audio-transcript_data
BACKUP_DIR := backups

.PHONY: help up down restart build logs logs-app logs-caddy update shell ps disk clean-old-jobs backup setup-env

## help: Lista todos os targets disponíveis com descrições
help:
	@echo "audio-transcript — targets disponíveis:"
	@echo ""
	@grep -E '^## ' Makefile | sed 's/^## /  /'
	@echo ""

## up: Sobe os containers em background (build se imagem estiver desatualizada)
up: require-env
	$(COMPOSE) up -d

## down: Para os containers preservando volumes
down:
	$(COMPOSE) down

## restart: Reinicia apenas o container app (mantém o Caddy rodando)
restart:
	$(COMPOSE) restart app

## build: Força rebuild da imagem do app após mudanças de código
build:
	$(COMPOSE) build app

## logs: Acompanha logs do app e do caddy (últimas 200 linhas)
logs:
	$(COMPOSE) logs -f --tail=200 app caddy

## logs-app: Acompanha logs apenas do app (últimas 200 linhas)
logs-app:
	$(COMPOSE) logs -f --tail=200 app

## logs-caddy: Acompanha logs apenas do caddy (últimas 200 linhas)
logs-caddy:
	$(COMPOSE) logs -f --tail=200 caddy

## update: Deploy completo — git pull, rebuild do app e up -d
update: require-env
	git pull && $(COMPOSE) build app && $(COMPOSE) up -d

## shell: Abre shell interativo (bash) dentro do container app
shell:
	$(COMPOSE) exec app bash

## ps: Mostra status dos containers do projeto
ps:
	$(COMPOSE) ps

## disk: Mostra uso de disco de uploads/ e models/ dentro do volume de dados
disk:
	$(COMPOSE) exec app du -sh /var/lib/audio-transcript/data/uploads /var/lib/audio-transcript/models

## clean-old-jobs: Dispara manualmente a limpeza de jobs antigos (normalmente roda a cada hora)
clean-old-jobs:
	$(COMPOSE) exec app python -c "from audio_transcript.cleanup import sweep_once; from audio_transcript import config; print(f'Deleted: {sweep_once(config.UPLOADS_DIR, config.RETENTION_DAYS)} job(s)')"

## backup: Cria backup tar.gz do volume de dados em backups/data-YYYY-MM-DD-HHMM.tar.gz
backup:
	@mkdir -p $(BACKUP_DIR)
	@stamp=$$(date +%Y-%m-%d-%H%M); \
	file="data-$${stamp}.tar.gz"; \
	echo "Criando backup em $(BACKUP_DIR)/$${file}..."; \
	docker run --rm \
		-v $(DATA_VOLUME):/data:ro \
		-v "$$(pwd)/$(BACKUP_DIR)":/backup \
		alpine \
		tar czf "/backup/$${file}" -C /data . ; \
	echo "Backup concluído: $(BACKUP_DIR)/$${file}"

## setup-env: Copia .env.example para .env se ainda não existir (não sobrescreve)
setup-env:
	@if [ -f .env ]; then \
		echo ".env já existe — nada a fazer."; \
	else \
		cp .env.example .env; \
		echo ".env criado a partir de .env.example."; \
		echo "Lembre-se de editar .env antes de subir o stack."; \
	fi

# Internal helper — verifica que .env existe antes de targets que dependem dele
.PHONY: require-env
require-env:
	@if [ ! -f .env ]; then \
		echo "ERRO: arquivo .env não encontrado."; \
		echo "Rode 'make setup-env' e edite o .env antes de continuar."; \
		exit 1; \
	fi
