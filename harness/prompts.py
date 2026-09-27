"""Everything the harness says to the model.

Kept deliberately short: the system prompt and tool schemas are re-sent on every
model call, so each word here is paid for once per step.
"""

SYSTEM_PROMPT = """You fix the issue in this repository using the tools. Nobody will answer questions: decide and act.

1. Locate: start from the overview (code locations, likely code, test results). Use search_code and read_file with line ranges; read only what you need and never re-read unchanged code.
2. Fix the root cause with minimal edit_file changes. old_str must match the file exactly. Keep the code style. Never weaken or delete tests.
3. The harness re-runs the tests after every edit and shows you the result. If they fail, fix and edit again.
4. Finish in the same turn as your final edit: send edit_file and finish together. The harness runs the tests first and rejects finish if they fail, so this is safe and saves a turn.

Rules: batch independent tool calls in one turn. Paths are relative to the repo root. Keep your messages to one short line. If an approach fails twice, try another. Install missing third-party test dependencies with run_command."""


def initial_user_message(issue: str, overview: str) -> str:
    return "# Issue\n%s\n\n# Repository overview\n%s" % (issue.strip(), overview.strip())


NUDGE_NO_TOOL = (
    "You did not call a tool. Call one now, e.g. search_code {\"query\": \"name from the issue\"} or "
    "read_file {\"path\": \"src/file.py\"}. If the fix is done and tests pass, call finish."
)

NUDGE_WRAP_UP = "[harness] {left} steps left. Stop exploring: make the fix, check the tests, call finish."

REPEAT_WARNING = "\n[harness] You made this exact call {n} times; the result will not change. Try something else."

ERROR_STREAK_HINT = ("\n[harness] Several calls failed in a row. Re-check exact paths (find_files) and file "
                     "content (read_file) before retrying.")

FINISH_NO_TESTS = ("[harness] finish rejected: tests have not passed since your last edit. Run run_tests "
                   "(or a reproduction script), fix any failure, then call finish.")

FINISH_VERIFY_FAILED = ("[harness] finish rejected: the harness checked your change and it did not pass.\n"
                        "{verdict}\n{output}\n"
                        "Only the target test and NEW failures count. Fix it, then call finish.")

FINISH_NO_CHANGES = ("[harness] finish rejected: no file was changed. Implement the fix, or call finish again "
                     "explaining why no change is needed.")

AUTO_TEST_PASS = "\n[harness] Tests after your edit: PASSED (`{cmd}`)."
AUTO_TEST_FAIL = "\n[harness] Tests after your edit: FAILED (`{cmd}`):\n{tail}"
AUTO_VERIFY_FAIL = "\n[harness] The target test passes, but the full check found a problem:\n{verdict}"
