.PHONY: install mlflow-server profile train retrain test lint serve-api serve-gradio

install:
	pip install -e ".[dev]"

mlflow-server:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000

profile:
	smuba-profile --data data/raw/social_media_user_behaviour.csv

# usage: make train MODEL=random_forest
train:
	smuba-train --data data/raw/social_media_user_behaviour.csv --model $(MODEL) --promote

# usage: make retrain MODEL=random_forest DATA=data/raw/new_batch.csv
retrain:
	smuba-retrain --data $(DATA) --model $(MODEL)

test:
	pytest -q

lint:
	ruff check src tests

serve-api:
	uvicorn smuba.serving.api:app --reload --port 8000

serve-gradio:
	python -m smuba.serving.gradio_app
