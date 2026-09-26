# AI Coding Harness - standard interface required by the hackathon.
#   make setup   install / configure everything (no API key needed)
#   make run     launch the harness (reads AI_API_KEY from the environment)
#   make test    run the harness's own automated tests (offline, no API key needed)
#   make clean   remove generated artefacts
#
# Optional variables for make run:
#   REPO=<path or git URL>   ISSUE=<text | file | GitHub issue URL>   ISSUE_FILE=<file>
#   AI_MODEL=<model id>      AI_PROVIDER=<deepseek|qwen|openrouter|custom|...>  AI_BASE_URL=<url>

SHELL := /bin/bash
VENV  := .venv
PY    := $(shell if [ -x $(VENV)/bin/python ]; then echo $(VENV)/bin/python; else echo python3; fi)
export PATH := $(CURDIR)/$(VENV)/bin:$(PATH)
export PYTHONUNBUFFERED := 1

.PHONY: setup run test clean demo check report help

help:
	@grep -E "^#" Makefile | head -12

setup:
	@echo "==> Checking requirements"
	@command -v python3 >/dev/null || { echo "ERROR: python3 is required"; exit 1; }
	@python3 -c "import sys; assert sys.version_info >= (3, 8), \"Python 3.8+ required\"" 
	@command -v git >/dev/null || echo "WARNING: git not found (only needed to clone repos from URLs)"
	@echo "==> Creating virtual environment (falls back to system Python if unavailable)"
	@if [ ! -x $(VENV)/bin/python ]; then python3 -m venv --system-site-packages $(VENV) 2>/dev/null || echo "   venv unavailable - using system python3"; fi
	@if [ -x $(VENV)/bin/python ]; then $(VENV)/bin/python -m pip install --quiet --disable-pip-version-check -r requirements.txt 2>/dev/null || echo "   (optional pytest install skipped)"; fi
	@mkdir -p runs workspace
	@$(PY) -c "import harness; print(\"==> harness\", harness.__version__, \"ready\")"
	@if [ -z "$$AI_API_KEY" ]; then echo "NOTE: set AI_API_KEY before make run"; fi
	@echo "==> Setup complete"

run:
	@$(PY) -m harness $(if $(REPO),--repo "$(REPO)") $(if $(ISSUE),--issue "$(ISSUE)") $(if $(ISSUE_FILE),--issue-file "$(ISSUE_FILE)")

demo:
	@rm -rf runs/demo_repo && cp -r examples/sample_repo runs/demo_repo
	@$(PY) -m harness --repo runs/demo_repo --issue-file examples/sample_issue.md

report:
	@$(PY) -m harness.report

check:
	@$(PY) -m harness --check

test:
	@$(PY) -m unittest discover -s tests -t . -v

clean:
	@rm -rf runs workspace .pytest_cache
	@find . -name "__pycache__" -type d -prune -not -path "./$(VENV)/*" -exec rm -rf {} +
	@echo "cleaned"
