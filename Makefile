PY ?= python3
COMPOSE ?= docker compose

.PHONY: venv test lint eval brief ask seed backfill serve web clean

venv:
	python3 -m venv .venv && .venv/bin/pip install --upgrade pip
	.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu || true
	.venv/bin/pip install -e .[dev] || .venv/bin/pip install -r requirements.txt

test:
	OMP_NUM_THREADS=4 $(PY) -m pytest -q

lint:
	ruff check app tests

eval:
	$(PY) -m app.eval.harness --dataset eval/datasets/atlas_qa.jsonl --out perf/results/eval-$$(date +%Y%m%d).json

brief:
	curl -s 'http://127.0.0.1:8000/brief?topic=reranking&since=2025-10-01&min_citations=80' | python3 -m json.tool

ask:
	curl -s -X POST http://127.0.0.1:8000/ask -H 'content-type: application/json' -d '{"question":"new science today?"}' | python3 -m json.tool

seed:
	$(PY) scripts/seed_topics.py --topics reranking:80,retrieval:50 --papers 30

backfill:
	$(PY) scripts/backfill.py --topic reranking --since 2024-01-01 --min-citations 80 --limit 50

serve:
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

web:
	cd web && npm install && npm run dev

clean:
	rm -rf .venv __pycache__ .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
