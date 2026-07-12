# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean autobuild marimo

# Clean build directories
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf $(SOURCEDIR)/.doctrees
	rm -rf $(SOURCEDIR)/__pycache__
	find $(SOURCEDIR) -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Autobuild
autobuild:
	sphinx-autobuild $(SOURCEDIR) $(BUILDDIR)/html

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Social media user behaviour analysis

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
