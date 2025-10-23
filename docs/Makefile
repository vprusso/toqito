# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   = python3 -m sphinx
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help clean Makefile fast-docs run-one-docstring

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	rm -rf auto_examples
	mkdir -p auto_examples

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# `make fast-docs` if you do not want to run any of the docstring examples
fast-docs:
	TOQITO_DOCS_FAST=1 make html

# `make run-one-docstring ONLY=toqito.channel_ops.apply_channel` if you only want to run the docstring in apply_channel
run-one-docstring:
	@if [ -z "$$ONLY" ]; then \
		echo "Usage: make run-one ONLY=toqito.module.func"; \
		exit 1; \
	fi; \
	TOQITO_DOCS_FAST=1 TOQITO_DOCS_ONLY=$$ONLY make html
