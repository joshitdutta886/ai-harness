# AI Coding Harness

An autonomous coding agent. Give it a repository and a GitHub issue; it finds the
relevant code, fixes the root cause, runs the tests, and only reports success when
the harness itself has verified the fix.

It is built to use **as few tokens as possible**. On the bundled practice tasks it resolves
each bug in **one model call, about 1,500 tokens per task**. An earlier version of this same
harness used about 17,000 tokens and 6 calls per task (see [Token efficiency](#token-efficiency)).

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
make bench                                                  # offline token benchmark (no API key)
```

`TEST_CMD` is the evaluator's test case: a command that must pass once the issue is
fixed. It is optional. When it is given, the harness runs it before any change (to show
the bug) and after the fix (as the definition of "done"). The interactive prompt also
asks for it.

`make run` with no arguments starts the interactive prompt. Pressing Enter at both prompts
runs the bundled demo.

Requirements: Python 3.8+ and git. The harness itself uses **only the Python standard library**,
so `make setup` has nothing that can fail to install.

## Token efficiency

Every model call re-sends the system prompt, the tool schemas and the conversation so far,
so the total cost is roughly *calls × context size*. The harness cuts both.

| What changed | Why it saves tokens |
|---|---|
| Short system prompt (~210 tokens) and tool schemas (~600) | These are re-sent on every call. Together they were ~1,600 tokens per call and are now ~815. |
| **The harness runs the tests after every edit** and attaches the result to the edit's reply | The model never spends a call on "run the tests". |
| **The run stops as soon as the fix is proven**: the target test goes from failing to passing and the regression check shows no new failures | No extra "finish" or "review" calls. |
| **"Likely relevant code" in the first message**: the definitions of names from the issue, plus functions they call (one hop), up to 60 lines | The model can often edit right away, with no search and no read calls. |
| The first message keeps only high-value facts: file tree (capped), code locations ranked source → tests → docs, test verdicts, and failing test names | No commit log, no passing test logs, and no duplicate failure output. |
| Only the last 3 tool outputs stay in full; older ones become one-line stubs | The history stops growing with every step. |
| Output caps: 4k characters per tool result, 200 lines per file read, 30 search hits; lock files and minified files are never searched | One tool call can't flood the context. |
| **Adaptive thinking**: reasoning starts off (`/no_think`, plus `enable_thinking: false` on DashScope). It turns on only after two failed test runs or a rejected finish | Easy bugs don't pay for long hidden "thinking" output; hard ones still get full reasoning. |
| No `git_diff` tool or review step | The edit reply already shows the changed lines. |
| The model may send **edit and finish in the same reply**; the harness runs the tests before accepting the finish | With no target test, a confident fix takes one call instead of two (a real Qwen run was 2 calls and ~4,100 tokens). |
| The **target test's source code** goes in the first message | The model sees the expected values without opening the test file. |
| **Code already in the first message is never sent again**: a `read_file` of unchanged lines already shown returns a one-line note, and only the missing lines are sent | Real runs showed models re-reading code they already had. |
| Test output uses repo-relative paths and drops decorative separator lines | Shorter failure messages on every call. |

Measured with `make bench`. This runs the same scripted agent on the three practice tasks
and counts tokens as characters / 4 of every request and reply:

| Version | Calls per task | Tokens per task | Resolved |
|---|---|---|---|
| Before the token work | 6 | ~17,300 | 3/3 |
| **Now** | **1** | **~1,500** | **3/3** |

The scripted agent behaves like a sensible model: it skips searching or reading code that
is already visible, and it never calls a tool the harness doesn't offer. `tests/test_tokens.py`
fails if a practice task needs more than 3 calls or 6,000 tokens, or if the fixed cost per
call goes above 900 tokens, so the savings can't quietly regress.

## Design decisions

**One agent, with a supervisor that does not use AI.** The harness runs a single tool-calling
agent. The work a planner or reviewer agent would do is handled by plain code around it.
Three reasons:

- **Context.** One agent keeps the whole task in view. Handing work between agents loses detail, and open models like DeepSeek and Qwen are especially sensitive to that.
- **Tokens.** Every extra agent re-reads the same context, and efficiency is scored.
- **Checking.** The job of a "reviewer" is done by the test results, compared against a run from before the change. Another model can be talked into approving a bad fix; a failing test cannot.

**The harness does the cheap work, the model does the thinking.** Anything that doesn't
need judgement is done by code, before or around the model:

- reading the repo tree and finding where the issue's names appear in the code
- running the baseline tests
- detecting the test command
- verifying the result

This saves steps and tokens, and it means these steps work the same way every time.

**Every failure becomes the model's next input.** Nothing in the loop raises an exception
at the model. A failed edit, a missing file, a syntax error or a rejected finish all come
back as plain text that says what went wrong and what to try next. That is how the agent
recovers without human help.

**Built for the evaluation models, not for one API.** DeepSeek and Qwen differ in API
details: model names, tool-call format, and "thinking" output. The harness adapts to each
of these at run time rather than assuming one, so the evaluator only has to set
`AI_API_KEY`.

**Honest reporting.** A run is only called `RESOLVED` when the harness itself has seen the
target test pass and no new test failures. Everything else is reported as it is:
`FINISHED_UNVERIFIED`, `BUDGET_EXHAUSTED` or `TIME_EXHAUSTED`. Each report includes the
verification result and full telemetry.

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
| thinking | `AI_THINKING` | `auto`: adaptive, off until the model struggles (`on` / `off` to force) |
| max output tokens per reply | `AI_MAX_OUTPUT_TOKENS` | 4096 (lowered automatically if the provider reports a smaller limit) |
| step budget | `HARNESS_MAX_STEPS` | 30 |
| token budget | `HARNESS_MAX_TOKENS` | 400,000 |
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
                                                  # start Ollama with OLLAMA_CONTEXT_LENGTH=16384 ollama serve:
                                                  # its default 4k context is too small for the first prompt
AI_PROVIDER=deepseek make run                     # model picked from the key's /models list
AI_PROVIDER=qwen     AI_MODEL=qwen3-coder-plus make run
AI_PROVIDER=custom   AI_BASE_URL=http://localhost:8000/v1 AI_MODEL=Qwen2.5-Coder-32B make run
```

Temperature defaults to 0 for reproducible runs.

## How it works

```
 issue ──► [1] context gathering ──► [2] agent loop ──────────────► [3] verification ──► report
            (no model calls)          model ⇄ tools                     target test must pass
            file tree, test command    search / read / edit / revert     no NEW failures vs the
            code locations +          harness auto-runs tests           baseline suite run
            likely relevant code        after every edit                stops the run the moment
            baseline + target run     compaction, recovery, budgets     the fix is proven
```

### 1. Context gathering (`harness/context.py`)
Before the first model call, the harness collects a compact overview, with no model calls:

- the file tree (capped) and the detected test command
- **where each name from the issue appears in the code**: function names, file paths, `code spans`, and CamelCase, snake_case and kebab-case words, ranked source → tests → docs
- **the likely relevant code**: the definitions of those names, plus the functions they call (one hop), up to 60 lines
- the **target test's source code**, when the target command names a test (`path::test` or `pkg.module.Class.test`)
- the **target test result before any change**, including its assertion error
- a one-line **baseline suite verdict** with the names of tests that were already failing

For typical bugs the model can edit right away from this message.

### 2. Agent loop (`harness/agent.py`)
A single tool-calling loop. The system prompt asks for: locate → minimal fix → (the harness
tests it) → finish. After every edit, **the harness runs the target test (or the suite)
itself** and attaches the result to the edit's reply. When the target test flips from
failing to passing, the harness runs its regression check and, if that passes, **ends the
run as resolved without another model call**.

Tools (`harness/tools.py`): `list_dir`, `find_files`, `search_code` (ripgrep with a
pure-Python fallback), `read_file` (line-numbered, paged), `edit_file` (exact unique
search/replace), `create_file`, `revert_file` (undo all changes to a file), `run_command`,
`run_tests` (runs the target test, or auto-detects pytest, unittest, npm, go, cargo, maven,
gradle, make), `finish`.

**Recovery**, so the model can fix its own mistakes:
- Tool errors come back as text with a concrete next step, and are never raised.
- A failed `edit_file` shows the closest matching block in the file.
- Every Python edit is syntax-checked right away.
- `revert_file` lets the model throw away a broken attempt and start that file again.
- A missing file suggests similar paths. Paths written as `/app/x.ts` are read as repo-relative (`app/x.ts`), because models often write them that way.
- `search_code` never dead-ends silently. If an exact search finds nothing, it retries as a regex (when the query looks like one), ignoring case, and word by word, and it says which fallback found the results. If nothing matches at all, it suggests `find_files` or `list_dir`.
- The same call repeated 3 times gets a warning to change approach. A call that returned nothing or an error gets the warning on its second repeat.
- Common alternative argument names are accepted and translated, such as `line_start`→`start_line`, `file`→`path` and `old_string`→`old_str`. An unknown argument gets an error that lists the real ones, so a model's naming habit can't cause a loop.
- 3 errors in a row get a "re-check your assumptions" hint.
- If the model replies without a tool call, the harness nudges it; after 3 such replies it treats the text as a finish attempt.
- If a reasoning model spends its whole output budget thinking and returns nothing, the request is retried with a larger budget (up to 16k tokens).
- If the provider rejects a request because `max_tokens` is above its output limit (for example Groq's free tier: "OTPM limit 1000"), the harness reads the limit from the error, lowers `max_tokens` and retries. Later retries never go above that limit.
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
     from before the first edit. Failing test ids are parsed from pytest, unittest, vitest
     and jest output.
     Tests that were **already failing** before the change are reported but don't block the
     fix. Any **new** failure rejects the finish, and the model is told exactly which tests broke.
- The same check runs automatically after an edit makes the target test pass; if it holds,
  the run ends there.
- The status is `RESOLVED` only when this check passes; otherwise the run ends as
  `FINISHED_UNVERIFIED`. If the loop stops early (step, token or time budget), the harness
  still runs the check, so the report always says whether the final code works.

### Efficiency
See [Token efficiency](#token-efficiency). In short:

- auto-tests after edits and an early verified stop
- likely relevant code in the first message
- short prompts and tool schemas, and adaptive thinking
- truncated outputs (4k characters, 200-line reads) and compaction (the last 3 outputs stay in full)
- a one-line status after each tool result
- hard step, token and wall-clock budgets

Token usage is tracked and reported for every run.

### Safety
- All paths are confined to the target repository.
- Destructive commands are blocked (`git push`, `git reset --hard`, `rm -rf /`, `sudo`, `curl | sh`, …).
- Commands run non-interactively with timeouts.
- `AI_API_KEY` is stripped from the environment of every command the agent runs.

## Web UI (optional): `make ui`

`make ui` opens a local dashboard at http://localhost:8787 (bound to 127.0.0.1 only). It
adds nothing to `make run`, which stays the evaluation entry point. The dashboard shows:

- **Live runs.** A pipeline (Context → Explore → Edit → Test → Verify → Done) lights up as the agent works, each tool call slides into a timeline with its result, token and tool counters count up, and a "thinking" indicator shows while the model is working.
- **Replay** of any finished run at 0.5–5× speed, including the runs made from the terminal. `examples/recordings/` has a scripted recording, so the UI can be shown with no model at all.
- **Results:** the harness verification (target test and regression check), the agent's summary and a colour-coded diff.
- **Start runs** on the demo, a practice task, or any local repo with your own issue and target test.

It reads the same `runs/*/trajectory.jsonl` files the harness always writes, and it uses
only the Python standard library.

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
Makefile              setup / run / test / clean (+ demo, check, report, eval, ui, bench)
config/harness.json   model + budget configuration (no secrets)
harness/
  cli.py              entry point, interactive prompts, GitHub issue fetch, repo clone
  config.py           config loading, provider detection
  llm.py              provider-agnostic client (OpenAI-compatible + Anthropic), text tool fallback
  agent.py            the loop, finish gate, recovery, budgets, reports
  verify.py           target check + regression check against the baseline run
  eval.py             runs the practice task set and prints a score table
  report.py           aggregates all runs (make report)
  tokenbench.py       offline token benchmark (make bench)
  web.py, web/        optional local dashboard (make ui): live view, replay, start runs
  context.py          repo overview, issue keyword search, context compaction
  tools.py            the tools and their JSON schemas
  prompts.py          system prompt and harness messages
  ui.py               terminal output
tests/                71 offline tests (tools, parsing, wire formats, verification, web UI,
                      full loop with a scripted model, and regressions from a real-repo run)
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
- token use stays low: `tests/test_tokens.py` caps calls and tokens per practice task
- every HTTP request carries a harness User-Agent, because Cloudflare-fronted APIs such as Groq block Python's default one (error 1010)

`tests/test_realworld.py` replays problems seen when a small local model ran the harness on
a real Next.js repository (github.com/nst-sdc/Open-Source-Tracker-NST, issue #76):

- repo paths written with a leading `/`
- regex-style searches treated as plain text
- repeated dead-end searches
- test log lines mistaken for failures
- JavaScript test output that wasn't parsed

Each one is fixed and covered by a test.
