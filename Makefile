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

.PHONY: help Makefile clean autobuild install

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
