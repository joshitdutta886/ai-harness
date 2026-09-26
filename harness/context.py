"""Context engineering: what goes into the model's window.

1. build_overview(): before the first model call, the harness gathers a cheap
   repo summary (tree, test command, baseline test result, and code locations
   for identifiers named in the issue). This saves the model several
   exploratory steps, which is the biggest single token saving.
2. compact(): as the conversation grows, old tool outputs are replaced with
   one-line stubs so the prompt stays under a fixed character budget.
"""
import json
import re
import subprocess
from typing import List

from . import tools as T
from . import verify as V

_STOP = {
    "the", "and", "for", "that", "this", "with", "when", "from", "should", "would", "which", "there",
    "error", "issue", "value", "return", "returns", "function", "method", "class", "file", "code",
    "expected", "actual", "none", "true", "false", "self", "test", "tests", "python", "import",
    "print", "bug", "fix", "call", "calls", "called", "using", "used", "into", "does", "not", "but",
    "have", "has", "will", "can", "are", "was", "were", "then", "than", "also", "only", "more",
    "some", "what", "why", "how", "def", "string", "list", "dict", "int", "str", "object", "type",
}


def extract_keywords(issue: str, limit: int = 8) -> List[str]:
    """Pull identifiers likely to exist in the code: `code spans`, snake_case, CamelCase, dotted names, paths."""
    found: List[str] = []

    def add(tok: str) -> None:
        tok = tok.strip("`'\"()[]{}.,:;")
        if len(tok) < 3 or tok.lower() in _STOP or tok in found:
            return
        found.append(tok)

    for span in re.findall(r"`([^`\n]{3,60})`", issue):
        for part in re.findall(r"[A-Za-z_][A-Za-z0-9_./]*", span):
            add(part.split(".")[-1] if "/" not in part else part)
    for tok in re.findall(r"\b[A-Za-z0-9_/\-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|c|cpp|h|json|yaml|yml|toml)\b", issue):
        add(tok)
    for tok in re.findall(r"\b[a-z]+(?:_[a-z0-9]+)+\b", issue):        # snake_case
        add(tok)
    for tok in re.findall(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b", issue):  # CamelCase
        add(tok)
    for tok in re.findall(r"\b[a-z]+[A-Z][A-Za-z0-9]+\b", issue):      # camelCase
        add(tok)
    for tok in re.findall(r"\b(\w+)\(\)", issue):                       # func()
        add(tok)
    return found[:limit]


def _git(ws: T.Workspace, args: str) -> str:
    try:
        p = subprocess.run("git " + args, shell=True, cwd=ws.root, capture_output=True, text=True, timeout=20)
        return p.stdout.strip() if p.returncode == 0 else ""
    except Exception:
        return ""


def build_overview(ws: T.Workspace, issue: str, run_baseline: bool = True, baseline_timeout: int = 150) -> str:
    parts = []
    parts.append("## Layout\n```\n%s\n```" % T.truncate(T.list_dir(ws, ".", 2), 3500))

    test_cmd = ws.detect_test_command()
    parts.append("## Test command\n%s" % (test_cmd or "(not detected - find how tests are run)"))

    log = _git(ws, "log --oneline -5")
    if log:
        parts.append("## Recent commits\n```\n%s\n```" % log)

    kws = extract_keywords(issue)
    if kws:
        hits = []
        for kw in kws:
            res = T.search_code(ws, kw, max_results=8)
            if not res.startswith("No matches"):
                hits.append("### `%s`\n```\n%s\n```" % (kw, T.truncate(res, 1200)))
        if hits:
            parts.append("## Where identifiers from the issue appear\n" + "\n".join(hits[:6]))

    if ws.target_test_command:
        parts.append("## Target test for this task\n`%s`\nThis test must pass when you are done." % ws.target_test_command)
    if run_baseline and ws.target_test_command:
        snap = V.snapshot(ws, ws.target_test_command, baseline_timeout)
        ws.target_baseline = snap
        state = "PASSES already" if snap.passed else "FAILS right now - this is the bug to fix"
        parts.append("## Target test before any change: %s\n```\n%s\n```" % (state, _tail(snap.output)))
    suite = V.suite_command(test_cmd)
    if run_baseline and suite:
        snap = V.snapshot(ws, suite, baseline_timeout)
        ws.baseline = snap
        ws.baseline_test_output = snap.output
        note = ""
        if snap.failing:
            note = ("\nAlready failing before your change (%d): %s" %
                    (len(snap.failing), ", ".join(sorted(snap.failing)[:15])))
        parts.append("## Baseline test run (before any change)\n```\n%s\n```%s" % (_tail(snap.output), note))
    return "\n\n".join(parts)


def _tail(res: str) -> str:
    # keep the tail: that is where the failure summary lives
    return res if len(res) < 2500 else res[:300] + "\n...\n" + res[-2200:]


def _msg_chars(m: dict) -> int:
    n = len(m.get("content") or "")
    for tc in m.get("tool_calls") or []:
        n += len(json.dumps(tc.get("args", {})))
    return n


def total_chars(messages: List[dict]) -> int:
    return sum(_msg_chars(m) for m in messages)


def compact(messages: List[dict], keep_recent: int, char_budget: int) -> int:
    """Shrink old tool outputs in place. Returns number of messages compacted.

    Always stubs tool results older than `keep_recent` once they are large;
    if still over budget, keeps shrinking from the oldest, and finally trims
    long assistant texts. The first user message (the issue) is never touched.
    """
    tool_idx = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    old = tool_idx[:-keep_recent] if keep_recent else tool_idx
    n = 0
    for i in old:
        m = messages[i]
        c = m.get("content") or ""
        if len(c) > 400 and not m.get("_compacted"):
            first = c.strip().splitlines()[0][:200] if c.strip() else ""
            m["content"] = "%s\n[older output compacted by harness (%d chars); re-run the tool if you need it again]" % (first, len(c))
            m["_compacted"] = True
            n += 1
    if total_chars(messages) > char_budget:
        for i in tool_idx:
            if total_chars(messages) <= char_budget:
                break
            m = messages[i]
            if not m.get("_compacted") and len(m.get("content") or "") > 400 and i != tool_idx[-1]:
                c = m["content"]
                m["content"] = c.strip().splitlines()[0][:200] + "\n[compacted by harness; re-run if needed]"
                m["_compacted"] = True
                n += 1
        for m in messages[1:-4]:
            if total_chars(messages) <= char_budget:
                break
            if m["role"] == "assistant" and len(m.get("content") or "") > 600:
                m["content"] = m["content"][:500] + " [...]"
                m.pop("reasoning", None)
    return n
