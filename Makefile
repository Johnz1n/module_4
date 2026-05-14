PYTHON       := python3
VENV         := .venv
PIP          := $(VENV)/bin/pip
PYTHON_VENV  := $(VENV)/bin/python
UVICORN      := $(VENV)/bin/uvicorn
PYTEST       := $(VENV)/bin/pytest
IMAGE_NAME   := bbas3-lstm
IMAGE_TAG    := latest
PORT         := 8000

.PHONY: help install train run test build run-docker build-local run-local clean

help:
	@echo ""
	@echo "  BBAS3 LSTM — Comandos disponíveis"
	@echo "  ─────────────────────────────────────────"
	@echo "  make install      Cria venv e instala dependências (com torch MPS)"
	@echo "  make train        Executa o pipeline completo de treino"
	@echo "  make run          Inicia a API FastAPI localmente (com reload)"
	@echo "  make test         Executa a suite de testes com pytest"
	@echo "  make build        Build da imagem Docker de produção (linux/amd64)"
	@echo "  make run-docker   Executa a imagem de produção localmente"
	@echo "  make build-local  Build da imagem Docker de desenvolvimento"
	@echo "  make run-local    Inicia a stack local via docker-compose"
	@echo "  make clean        Remove artefactos gerados pelo treino"
	@echo ""

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

train:
	PYTHONPATH=. PYTHONUNBUFFERED=1 $(PYTHON_VENV) train_pipeline.py

run:
	PYTHONPATH=. $(UVICORN) api.main:app --reload --host 0.0.0.0 --port $(PORT)

test:
	PYTHONPATH=. $(PYTEST) tests/ -v --tb=short

# --platform linux/amd64 garante compatibilidade com Railway (AMD64)
build:
	docker build --platform linux/amd64 -t $(IMAGE_NAME):$(IMAGE_TAG) -f Dockerfile .

run-docker:
	docker run --rm -p $(PORT):$(PORT) $(IMAGE_NAME):$(IMAGE_TAG)

build-local:
	docker build -t $(IMAGE_NAME):dev -f Dockerfile.local .

run-local:
	docker compose up

clean:
	rm -f artifacts/model.pt artifacts/model_best.pt \
	      artifacts/scaler.pkl artifacts/evaluation_plot.png
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
