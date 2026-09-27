# AI Coding Harness - standard interface required by the hackathon.
#   make setup   install / configure everything (no API key needed)
#   make run     launch the harness (reads AI_API_KEY from the environment)
#   make test    run the harness's own automated tests (offline, no API key needed)
#   make clean   remove generated artefacts
#   make ui      optional web UI: watch runs live, replay them, start new ones
#   make bench   offline token benchmark on the practice tasks (no API key)
#
# Optional variables for make run:
#   REPO=<path or git URL>   ISSUE=<text | file | GitHub issue URL>   ISSUE_FILE=<file>
#   TEST_CMD=<test command that must pass once the issue is fixed>
#   AI_MODEL=<model id>      AI_PROVIDER=<deepseek|qwen|openrouter|custom|...>  AI_BASE_URL=<url>

SHELL := /bin/bash
VENV  := .venv
PY    := $(shell if [ -x $(VENV)/bin/python ]; then echo $(VENV)/bin/python; else echo python3; fi)
export PATH := $(CURDIR)/$(VENV)/bin:$(PATH)
export PYTHONUNBUFFERED := 1
# Task inputs reach the harness through the environment, so quotes inside
# ISSUE or TEST_CMD survive intact (harness/cli.py reads these variables).
export REPO ISSUE ISSUE_FILE TEST_CMD

.PHONY: setup run test clean demo check report eval ui bench help

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
	@$(PY) -m harness

demo:
	@mkdir -p runs && rm -rf runs/demo_repo && cp -r examples/sample_repo runs/demo_repo
	@$(PY) -m harness --repo runs/demo_repo --issue-file examples/sample_issue.md

bench:
	@$(PY) -m harness.tokenbench

ui:
	@$(PY) -m harness.web

eval:
	@$(PY) -m harness.eval $(if $(TASK),--task "$(TASK)")

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
