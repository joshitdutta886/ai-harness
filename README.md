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
make run REPO=https://github.com/owner/repo ISSUE="Title: ... description ..."
make run ISSUE=https://github.com/owner/repo/issues/42     # repo is cloned automatically
cat issue.md | make run REPO=/path/to/repo
make demo                                                   # bundled buggy repo
make check                                                  # test the API key / model connection
```

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
| model | `AI_MODEL` | provider default (`deepseek-chat`, `qwen-plus`, …) |
| base URL | `AI_BASE_URL` | provider default |
| tool calling | `AI_TOOL_MODE` | `auto` (`native` or `text`) |
| step budget | `HARNESS_MAX_STEPS` | 40 |
| token budget | `HARNESS_MAX_TOKENS` | 800,000 |

**Provider auto-detection.** DeepSeek and Qwen (DashScope) keys both start with `sk-`, so in `auto`
mode the harness sends the key to each provider's `/models` endpoint and uses the first one that
accepts it. It then picks the best available coding model from a preference list. OpenRouter
(`sk-or-`), Groq, OpenAI, Anthropic and Gemini are also supported, and so is any OpenAI-compatible
server (vLLM, Ollama, a hackathon gateway) with `AI_PROVIDER=custom AI_BASE_URL=... AI_MODEL=...`.

Examples:

```bash
AI_PROVIDER=deepseek AI_MODEL=deepseek-chat make run
AI_PROVIDER=qwen     AI_MODEL=qwen3-coder-plus make run
AI_PROVIDER=custom   AI_BASE_URL=http://localhost:8000/v1 AI_MODEL=Qwen2.5-Coder-32B make run
```

Temperature defaults to 0 for reproducible runs.

## How it works

```
 issue ──► [1] context gathering ──► [2] agent loop ─────────────► [3] verification gate ──► report
            (no LLM calls)            model ⇄ tools                    harness re-runs tests
            repo tree                 search / read / edit              rejects unverified
            test command              run tests / diff                  "finish" calls
            issue identifiers         compaction, recovery,
            baseline test run         budgets
```

### 1. Context gathering (`harness/context.py`)
Before the first model call, the harness collects a compact overview for free:
the repo tree, the detected test command, recent commits, a **baseline test run**, and
**where each identifier named in the issue appears in the code** (function names, file
paths, `code spans`, CamelCase/snake_case words). This usually saves the model 3-5
exploratory steps, which is the biggest single token saving.

### 2. Agent loop (`harness/agent.py`)
A single tool-calling loop with a structured workflow in the system prompt:
understand → locate → reproduce → plan → fix → verify → review → finish.

Tools (`harness/tools.py`): `list_dir`, `find_files`, `search_code` (ripgrep with a
pure-Python fallback), `read_file` (line-numbered, paged), `edit_file` (exact unique
search/replace), `create_file`, `run_command`, `run_tests` (auto-detects pytest, unittest,
npm, go, cargo, maven, gradle, make), `git_diff`, `finish`.

**Recovery**, so the model can fix its own mistakes:
- Tool errors come back as text with a concrete next step, and are never raised.
- A failed `edit_file` shows the closest matching block in the file.
- Every Python edit is syntax-checked right away.
- A missing file suggests similar paths.
- The same call repeated 3 times gets a warning to change approach.
- 3 errors in a row get a "re-check your assumptions" hint.
- If the model replies without a tool call, the harness nudges it; after 3 such replies it treats the text as a finish attempt.
- API errors are retried with exponential backoff and `Retry-After`. The client also adapts to endpoint quirks (`max_completion_tokens`, unsupported `temperature`, DeepSeek `reasoning_content`).

**Robust tool calling for open models.** In `auto` mode the harness uses native function
calling, and it also parses tool calls the model writes as text (Qwen-style
`<tool_call>{...}</tool_call>` or fenced JSON). If an endpoint rejects the `tools` parameter,
it switches to a text tool protocol for the rest of the run. `<think>` blocks are stripped.

### 3. Verification gate: "evidence over claims"
`finish` is **rejected** if files changed but the tests have not passed since the last edit.
When the model does finish, the **harness runs the test suite itself** and rejects the
finish (showing the failure output) if the tests fail. The status is `RESOLVED` only when
the harness has seen the tests pass after the final edit; otherwise it is reported as
`FINISHED_UNVERIFIED`.

### Efficiency
- Every tool output is truncated head+tail (8k chars max; file reads max 300 lines).
- **Context compaction:** tool outputs older than the 6 most recent are replaced with one-line stubs, with more aggressive trimming above a 100k-character budget. The issue is never trimmed.
- A status line after every tool result (step, changed files, test state) keeps the model oriented without re-reading.
- Hard step and token budgets, plus a "wrap up" warning near the end.
- Token usage is tracked and reported for every run.

### Safety
- All paths are confined to the target repository.
- Destructive commands are blocked (`git push`, `git reset --hard`, `rm -rf /`, `sudo`, `curl | sh`, …).
- Commands run non-interactively with timeouts.
- `AI_API_KEY` is stripped from the environment of every command the agent runs.

## Output

Each run writes to `runs/<timestamp>/`:

- `report.md`: status, model, steps, tokens, time, the agent's summary, test evidence and the patch
- `patch.diff`: unified diff of all changes
- `summary.json`: machine-readable results
- `trajectory.jsonl`: every model turn and tool call, for debugging and evaluation

## Project layout

```
Makefile              setup / run / test / clean (+ demo, check)
config/harness.json   model + budget configuration (no secrets)
harness/
  cli.py              entry point, interactive prompts, GitHub issue fetch, repo clone
  config.py           config loading, provider detection
  llm.py              provider-agnostic client (OpenAI-compatible + Anthropic), text tool fallback
  agent.py            the loop, verification gate, recovery, budgets, reports
  context.py          repo overview, issue keyword search, context compaction
  tools.py            the tools and their JSON schemas
  prompts.py          system prompt and harness messages
  ui.py               terminal output
tests/                31 offline tests (tools, parsing, wire formats, full loop with a scripted model)
examples/             demo repo with a real bug + matching issue
```

## Testing

`make test` runs fully offline. It includes end-to-end runs of the whole loop with a
scripted fake model, checking that the verification gate rejects an unverified finish,
that step budgets stop the loop, that repeat warnings fire, and that the request bodies
sent to each API style are well-formed.
