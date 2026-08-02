# Atajos para las operaciones habituales. Todo funciona igual con `docker compose`
# directamente; esto solo ahorra teclas.

.PHONY: help up down logs ingest reindex chat analytics health test lint shell clean

help:  ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up:  ## Levanta todo el stack (qdrant, postgres, api, ui)
	docker compose up -d --build
	@echo ""
	@echo "  API  -> http://localhost:8000/docs"
	@echo "  UI   -> http://localhost:8501"
	@echo ""
	@echo "  Siguiente paso:  make ingest"

down:  ## Detiene el stack
	docker compose down

logs:  ## Sigue los logs de la API
	docker compose logs -f api

ingest:  ## Scraping + indexación del sitio
	docker compose run --rm ingest

reindex:  ## Reindexa el corpus ya descargado (sin volver a rastrear)
	docker compose run --rm ingest rag-assistant index --recreate --reclean

chat:  ## Chat por consola dentro del contenedor
	docker compose run --rm ingest rag-assistant chat

analytics:  ## Informe de métricas del histórico
	docker compose run --rm ingest rag-assistant analytics

health:  ## Estado del sistema
	@curl -s http://localhost:8000/health | python -m json.tool

test:  ## Ejecuta la batería de tests
	pytest -q

lint:  ## Linter
	ruff check src tests

shell:  ## Shell dentro del contenedor de la API
	docker compose exec api bash

clean:  ## Elimina contenedores y volúmenes (BORRA el índice y el histórico)
	docker compose down -v
