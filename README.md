# AI Coding Harness

An autonomous coding agent. Give it a repository and a GitHub issue; it finds the
relevant code, fixes the root cause, runs the tests, and only reports success when
the harness itself has verified the fix.

Built for the LCC × DevClub AI Coding Harness Hackathon 2026, and tuned for the
evaluation models (DeepSeek and Qwen).

## Quick start

```bash
export AI_API_KEY="<your key>"
make setup      # checks Python/git, creates .venv (optional), installs pytest
make run        # interactive: asks for a repo and an issue
make test       # the harness's own test suite (offline, no key needed)
make clean      # removes runs/ and workspace/
```

Non-interactive:

```bash
make run REPO=/path/to/repo ISSUE_FILE=issue.md
make run REPO=/path/to/repo ISSUE_FILE=issue.md TEST_CMD="python3 -m pytest tests/test_x.py::test_bug"
make run REPO=https://github.com/owner/repo ISSUE="Title: ... description ..."
make run ISSUE=https://github.com/owner/repo/issues/42     # repo is cloned automatically
cat issue.md | make run REPO=/path/to/repo
make demo                                                   # bundled buggy repo
make check                                                  # test the API key / model connection
make report                                                 # table of all runs: status, tokens, steps, errors
make eval                                                   # run every practice task, print a score table
```

`TEST_CMD` is the evaluator's test case: a command that must pass once the issue is
fixed. It is optional. When it is given, the harness runs it before any change (to show
the bug) and after the fix (as the definition of "done"). The interactive prompt also
asks for it.

`make run` with no arguments starts the interactive prompt. Pressing Enter at both prompts
runs the bundled demo.

Requirements: Python 3.8+ and git. The harness itself uses **only the Python standard library**,
so `make setup` has nothing that can fail to install.

## Model configuration

The API key is read **only** from the `AI_API_KEY` environment variable. It is never written to
disk, never put in the config file, and it is removed from the environment of every command the
agent runs.

Model settings live in [`config/harness.json`](config/harness.json). Any of them can be overridden
by an environment variable:

| Setting | Env var | Default |
|---|---|---|
| provider | `AI_PROVIDER` | `auto` |
| model | `AI_MODEL` | picked from the provider's `/models` list (see below) |
| base URL | `AI_BASE_URL` | provider default |
| tool calling | `AI_TOOL_MODE` | `auto` (`native` or `text`) |
| thinking | `AI_THINKING` | `auto` (`off` adds `/no_think` for hybrid Qwen models; faster locally) |
| step budget | `HARNESS_MAX_STEPS` | 40 |
| token budget | `HARNESS_MAX_TOKENS` | 800,000 |
| time limit | `HARNESS_MAX_MINUTES` | 30 |

**Model auto-selection.** Model names change often (DeepSeek and Qwen both release new
versions regularly), so the harness does not rely on a hard-coded name. When `AI_MODEL` is
not set, it reads the provider's `/models` list for the given key and picks the best chat
model: preferred coding models first, and never embedding, vision or audio models. If the
API later says a configured model does not exist, the client switches to an available one
and carries on.

**Provider auto-detection.** DeepSeek and Qwen (DashScope) keys both start with `sk-`, so in `auto`
mode the harness sends the key to each provider's `/models` endpoint and uses the first one that
accepts it. It then picks the best available coding model from a preference list. OpenRouter
(`sk-or-`), Groq, OpenAI, Anthropic and Gemini are also supported, and so is any OpenAI-compatible
server (vLLM, Ollama, a hackathon gateway) with `AI_PROVIDER=custom AI_BASE_URL=... AI_MODEL=...`.

Examples:

```bash
AI_API_KEY=ollama make run                        # free local Qwen via Ollama (ollama pull qwen3:8b)
AI_PROVIDER=deepseek make run                     # model picked from the key's /models list
AI_PROVIDER=qwen     AI_MODEL=qwen3-coder-plus make run
AI_PROVIDER=custom   AI_BASE_URL=http://localhost:8000/v1 AI_MODEL=Qwen2.5-Coder-32B make run
```

Temperature defaults to 0 for reproducible runs.

## How it works

```
 issue ──► [1] context gathering ──► [2] agent loop ─────────────► [3] verification gate ──► report
            (no LLM calls)            model ⇄ tools                    target test must pass
            repo tree                 search / read / edit / revert    no NEW failures vs
            test command              run tests / diff                  baseline suite run
            issue identifiers         compaction, recovery,             rejects unverified
            baseline + target run     step/token/time budgets           "finish" calls
```

### 1. Context gathering (`harness/context.py`)
Before the first model call, the harness collects a compact overview for free:
the repo tree, the detected test command, recent commits, a **baseline test run** (and
the target test, if one was given, so the model sees the bug fail before it starts), and
**where each identifier named in the issue appears in the code** (function names, file
paths, `code spans`, CamelCase/snake_case words). This usually saves the model 3-5
exploratory steps, which is the biggest single token saving.

### 2. Agent loop (`harness/agent.py`)
A single tool-calling loop with a structured workflow in the system prompt:
understand → locate → reproduce → plan → fix → verify → review → finish.

Tools (`harness/tools.py`): `list_dir`, `find_files`, `search_code` (ripgrep with a
pure-Python fallback), `read_file` (line-numbered, paged), `edit_file` (exact unique
search/replace), `create_file`, `revert_file` (undo all changes to a file), `run_command`,
`run_tests` (runs the target test, or auto-detects pytest, unittest, npm, go, cargo, maven,
gradle, make), `git_diff`, `finish`.

**Recovery**, so the model can fix its own mistakes:
- Tool errors come back as text with a concrete next step, and are never raised.
- A failed `edit_file` shows the closest matching block in the file.
- Every Python edit is syntax-checked right away.
- `revert_file` lets the model throw away a broken attempt and start that file again.
- A missing file suggests similar paths.
- The same call repeated 3 times gets a warning to change approach.
- 3 errors in a row get a "re-check your assumptions" hint.
- If the model replies without a tool call, the harness nudges it; after 3 such replies it treats the text as a finish attempt.
- If a reasoning model spends its whole output budget thinking and returns nothing, the request is retried with a larger budget (up to 16k tokens).
- API errors are retried with exponential backoff and `Retry-After`. The client also adapts to endpoint quirks (`max_completion_tokens`, unsupported `temperature`, DeepSeek `reasoning_content`).

**Robust tool calling for open models.** In `auto` mode the harness uses native function
calling, and it also parses tool calls the model writes as text (Qwen-style
`<tool_call>{...}</tool_call>` or fenced JSON). If an endpoint rejects the `tools` parameter,
it switches to a text tool protocol for the rest of the run. `<think>` blocks are stripped.

### 3. Verification gate: "evidence over claims" (`harness/verify.py`)
The harness never takes the model's word that a fix works.

- `finish` is **rejected** if files changed but the tests have not passed since the last edit.
- When the model finishes, the **harness verifies the change itself**:
  1. **Target check:** the target test (`TEST_CMD`), if given, must pass. It was also run
     before any change, so the report shows it failing before and passing after.
  2. **Regression check:** the full suite runs again and is compared with the baseline run
     from before the first edit. Failing test ids are parsed from pytest and unittest output.
     Tests that were **already failing** before the change are reported but don't block the
     fix. Any **new** failure rejects the finish, and the model is told exactly which tests broke.
- The status is `RESOLVED` only when this check passes; otherwise the run ends as
  `FINISHED_UNVERIFIED`. If the loop stops early (step, token or time budget), the harness
  still runs the check, so the report always says whether the final code works.

### Efficiency
- Every tool output is truncated head+tail (8k chars max; file reads max 300 lines).
- **Context compaction:** tool outputs older than the 6 most recent are replaced with one-line stubs, with more aggressive trimming above a 100k-character budget. The issue is never trimmed.
- A status line after every tool result (step, changed files, test state) keeps the model oriented without re-reading.
- Hard step, token and wall-clock budgets, plus a "wrap up" warning near the end.
- Token usage is tracked and reported for every run.

### Safety
- All paths are confined to the target repository.
- Destructive commands are blocked (`git push`, `git reset --hard`, `rm -rf /`, `sudo`, `curl | sh`, …).
- Commands run non-interactively with timeouts.
- `AI_API_KEY` is stripped from the environment of every command the agent runs.

## Telemetry and reporting

Each run writes to `runs/<timestamp>/`:

- `report.md`: human-readable report with status, model, steps, tokens, time, a telemetry table, the agent's summary, the harness verification result (target and regression check), test evidence and the patch
- `telemetry.json`: counters for the run, covering:
  - model calls and time spent waiting on the model
  - tokens per step
  - tool calls, errors and time, per tool
  - edits, and test runs vs passes
  - finish rejections and why each happened
  - context compactions and characters saved
  - recovery nudges (no-tool, repeat, error-streak)
  - tool calls recovered from text output
- `patch.diff`: unified diff of all changes
- `summary.json`: machine-readable results, with the telemetry included
- `trajectory.jsonl`: every model turn, tool call, compaction and verification, with timestamps

`make report` gathers every run into one table (status, model, steps, tokens, time, tool
errors, finish rejections) and prints the resolve rate and average tokens/steps. Use it
to compare prompt or tool changes across the same set of issues.

## Evaluation set

`examples/tasks.json` lists practice tasks, each with a repo, an issue and a target test:

| Task | Bug | What it exercises |
|---|---|---|
| `cart-discount` | a percentage used as a fraction | issue-only repo: the target check comes from the issue's example |
| `slugify` | repeated and trailing hyphens | the repo has an **unrelated test that already fails**, to exercise the regression-aware verification |
| `median` | wrong result for even-length lists | a specific failing test as the target |

`make eval` runs the harness on a fresh copy of each task and prints a table (status,
verified, steps, tokens, time), saved to `runs/eval-<time>/eval_summary.md`.
`python3 -m harness.eval --check` confirms, without a model, that every task's bug really
reproduces. `make test` runs this check too.

## Project layout

```
Makefile              setup / run / test / clean (+ demo, check, report, eval)
config/harness.json   model + budget configuration (no secrets)
harness/
  cli.py              entry point, interactive prompts, GitHub issue fetch, repo clone
  config.py           config loading, provider detection
  llm.py              provider-agnostic client (OpenAI-compatible + Anthropic), text tool fallback
  agent.py            the loop, finish gate, recovery, budgets, reports
  verify.py           target check + regression check against the baseline run
  eval.py             runs the practice task set and prints a score table
  report.py           aggregates all runs (make report)
  context.py          repo overview, issue keyword search, context compaction
  tools.py            the tools and their JSON schemas
  prompts.py          system prompt and harness messages
  ui.py               terminal output
tests/                45 offline tests (tools, parsing, wire formats, verification, full loop with a scripted model)
examples/             demo repo + practice tasks (tasks.json), each with a real bug and an issue
```

## Testing

`make test` runs fully offline. It includes end-to-end runs of the whole loop with a
scripted fake model. The tests check that:

- the finish gate rejects an unverified finish
- a pre-existing failing test does not block a correct fix
- a fix that breaks another test is rejected, naming the broken test
- step and time budgets stop the loop
- repeat warnings fire
- the model is switched when the configured one does not exist
- the request bodies sent to each API style are well-formed
