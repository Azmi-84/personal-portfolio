# Minimal makefile for Sphinx documentation
# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS     ?=
SPHINXBUILD    ?= sphinx-build
SOURCEDIR      = source
BUILDDIR       = build

PROJECT_DIR          = $(SOURCEDIR)/project
PERSONAL_PROJECT_DIR = $(PROJECT_DIR)/personal_project
SOFTWARE_DIR         = $(PERSONAL_PROJECT_DIR)/software
PROGRAMMING_DIR      = $(SOFTWARE_DIR)/programming
PYTHON_DIR           = $(PROGRAMMING_DIR)/python
AI_ML_DIR            = $(PYTHON_DIR)/ai_ml
MACHINE_LEARNING_DIR = $(AI_ML_DIR)/machine_learning
SMUBA_DIR            = $(MACHINE_LEARNING_DIR)/social_media_user_behaviour_analysis

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean autobuild install mlflow-server profile train retrain test lint serve-api serve-gradio

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

# Install docs dependencies
install:
	pip install -r requirements.txt

# Machine Learning

# Social Media User Behaviour Analysis

mlflow-server:
	mlflow server --backend-store-uri sqlite:///$(SMUBA_DIR)/mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000

profile:
	cd $(SMUBA_DIR) && PYTHONPATH=src python -m smuba.data.profiling --data data/raw/instagram_usage_lifestyle.csv --output reports/eda/profiling_report.html

# usage: make train MODEL=random_forest
train:
	cd $(SMUBA_DIR) && PYTHONPATH=src python -m smuba.pipelines.training_pipeline --data data/raw/instagram_usage_lifestyle.csv --model $(MODEL) --promote

# usage: make retrain MODEL=random_forest DATA=data/raw/new_batch.csv
retrain:
	cd $(SMUBA_DIR) && PYTHONPATH=src python -m smuba.pipelines.retrain --data $(DATA) --model $(MODEL)

test:
	cd $(SMUBA_DIR) && PYTHONPATH=src pytest -q

lint:
	cd $(SMUBA_DIR) && ruff check src tests

serve-api:
	cd $(SMUBA_DIR) && PYTHONPATH=src uvicorn smuba.serving.api:app --reload --port 8000

serve-gradio:
	cd $(SMUBA_DIR) && PYTHONPATH=src python -m smuba.serving.gradio_app
