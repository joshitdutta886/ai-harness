"""Context engineering: what goes into the model's window.

1. build_overview(): before the first model call, the harness gathers a cheap
   repo summary (tree, test command, baseline test result, and code locations
   for identifiers named in the issue). This saves the model several
   exploratory steps, which is the biggest single token saving.
2. compact(): as the conversation grows, old tool outputs are replaced with
   one-line stubs so the prompt stays under a fixed character budget.
"""
import json
import os
import time
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
    for tok in re.findall(r"\b[A-Za-z0-9_/\-][A-Za-z0-9_/\-.]*\.(?:py|js|ts|tsx|jsx|mjs|cjs|mts|cts|go|rs|java|rb|c|cpp|h|json|yaml|yml|toml)\b", issue):
        add(tok)
    for tok in re.findall(r"\b[a-z]+(?:-[a-z0-9]+){1,4}\b", issue):   # kebab-case (rule / package names)
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


_DEF = re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:def|class|function|func|fn|"
                  r"const|let|var|public|private|protected|static|interface|type)\b")


_DOC_EXT = (".md", ".rst", ".txt", ".html", ".json", ".yml", ".yaml", ".toml", ".csv", ".ipynb")


def _rank(path: str) -> int:
    """Sort order for search hits: source code, then tests, then docs/config."""
    if path.lower().endswith(_DOC_EXT):
        return 2
    return 1 if _is_test_path(path) else 0


def _is_test_path(path: str) -> bool:
    p = path.replace("\\", "/").lower()
    base = p.rsplit("/", 1)[-1]
    return ("/test" in "/" + p or base.startswith("test_") or "_test." in base or ".test." in base
            or ".spec." in base or "/__tests__/" in p)


_NOT_CALLS = {"if", "for", "while", "return", "print", "len", "range", "str", "int", "float", "list",
              "dict", "set", "tuple", "round", "sum", "min", "max", "sorted", "isinstance", "super",
              "open", "enumerate", "zip", "map", "filter", "any", "all", "abs", "type", "bool",
              "ValueError", "TypeError", "KeyError", "Exception", "RuntimeError", "require",
              "function", "catch", "switch", "new", "typeof", "await", "async", "def", "class"}


def _def_range(ws: T.Workspace, rel: str, line_no: int, max_lines: int = 25):
    """(rel, first, last, lines) for the definition starting at line_no."""
    try:
        with open(os.path.join(ws.root, rel), "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    i = line_no - 1
    if i < 0 or i >= len(lines):
        return None
    indent = len(lines[i]) - len(lines[i].lstrip())
    j = i + 1
    while j < len(lines) and j - i < max_lines:
        cur = lines[j]
        if cur.strip() and (len(cur) - len(cur.lstrip())) <= indent and _DEF.match(cur):
            break
        j += 1
    while j > i + 1 and not lines[j - 1].strip():
        j -= 1
    return (rel, i + 1, j, lines)


def _render(rng) -> str:
    rel, a, b, lines = rng
    body = "\n".join("%6d\t%s" % (k, lines[k - 1]) for k in range(a, b + 1))
    return "%s (lines %d-%d of %d)\n%s" % (rel, a, b, len(lines), body)


def _called_defs(ws: T.Workspace, rng, limit: int = 2):
    """Definitions (outside tests) of functions called inside a snippet: one hop of the call graph."""
    rel, a, b, lines = rng
    code = "\n".join(lines[a:b])            # skip the def line itself
    names = []
    for n in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\s*\(", code):
        if n not in _NOT_CALLS and n not in names:
            names.append(n)
    found = []
    for n in names[:4]:
        pat = r"^\s*(?:export\s+)?(?:async\s+)?(?:def|function|func|fn)\s+%s\b" % re.escape(n)
        try:
            rows = T._search(ws, ws.root, pat, True, False, "")
        except Exception:
            continue
        for r in rows:
            m = re.match(r"^(.+?):(\d+):", r)
            if m and not _is_test_path(m.group(1)):
                found.append((m.group(1), int(m.group(2))))
                break
        if len(found) >= limit:
            break
    return found


def _likely_code(ws: T.Workspace, defs, max_snippets: int = 3, max_lines: int = 60) -> str:
    ranges = [r for r in (_def_range(ws, p, n) for p, n in defs[:2]) if r]
    for r in list(ranges):
        for p, n in _called_defs(ws, r):
            extra = _def_range(ws, p, n)
            if extra:
                ranges.append(extra)
    kept = []
    for r in ranges:   # drop duplicates and ranges contained in another one
        if any(o is not r and o[0] == r[0] and o[1] <= r[1] and r[2] <= o[2] and (o[1], o[2]) != (r[1], r[2])
               for o in ranges):
            continue
        if any(k[0] == r[0] and (k[1], k[2]) == (r[1], r[2]) for k in kept):
            continue
        kept.append(r)
    out, used = [], 0
    for r in kept[:max_snippets]:
        n = r[2] - r[1] + 1
        if used + n > max_lines:
            break
        used += n
        ws.files_read.add(r[0])
        ws.mark_shown(r[0], r[1], r[2], "\n".join(r[3]))
        out.append(_render(r))
    return "\n\n".join(out)


def _target_test_code(ws: T.Workspace, cmd: str, max_lines: int = 30) -> str:
    """Source of the test(s) named in the target command, e.g. `tests/test_x.py::TestA::test_b`
    or `tests.test_x.TestA.test_b` - so the model sees the expected behaviour directly."""
    targets = []                                   # (rel path, [names])
    for path, names in re.findall(r"([\w./-]+\.py)::([\w:\[\]-]+)", cmd):
        targets.append((path, [n.split("[")[0] for n in names.split("::") if n]))
    if not targets:
        for dotted in re.findall(r"\b((?:[A-Za-z_]\w*\.){1,}[A-Za-z_]\w*)\b", cmd):
            parts = dotted.split(".")
            for k in range(len(parts) - 1, 0, -1):
                rel = "/".join(parts[:k]) + ".py"
                if os.path.isfile(os.path.join(ws.root, rel)):
                    targets.append((rel, parts[k:]))
                    break
    ranges = []
    for rel, names in targets[:2]:
        full = os.path.join(ws.root, rel)
        if not names or not os.path.isfile(full):
            continue
        name = names[-1]
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    if re.match(r"\s*(?:async\s+)?(?:def|class)\s+%s\b" % re.escape(name), line):
                        r = _def_range(ws, rel, i, max_lines)
                        if r:
                            ranges.append(r)
                        break
        except OSError:
            continue
    out, used = [], 0
    for r in ranges:
        n = r[2] - r[1] + 1
        if used + n > max_lines:
            break
        used += n
        ws.mark_shown(r[0], r[1], r[2], "\n".join(r[3]))
        out.append(_render(r))
    return "\n\n".join(out)


def build_overview(ws: T.Workspace, issue: str, run_baseline: bool = True, baseline_timeout: int = 150) -> str:
    """Cheap, model-free context. It is sent with every call, so only high-value facts go in."""
    parts = ["## Files\n```\n%s\n```" % T.truncate(T.list_dir(ws, ".", 2), 1500)]
    test_cmd = ws.detect_test_command()
    parts.append("## Test command: %s" % ("`%s`" % test_cmd if test_cmd else "not detected"))

    hits, defs, seen = [], [], set()
    for kw in extract_keywords(issue)[:6]:
        res = T.search_code(ws, kw, max_results=25)   # rank these, keep the best 5
        if res.startswith("No "):          # no match, or only fuzzy fallback matches: skip the noise
            continue
        rows = [r for r in res.splitlines()
                if re.match(r"^[^:\s]+:\d+:", r) and "[Omitted long matching line]" not in r]
        rows = sorted(rows, key=lambda r: _rank(r.split(":", 1)[0]))[:5]
        if not rows:
            continue
        hits.append("# %s\n%s" % (kw, "\n".join(r[:160] for r in rows)))
        name = re.escape(kw.split("/")[-1].split(".")[0])
        for r in rows:
            m = re.match(r"^(.+?):(\d+):(.*)$", r)
            if (m and _DEF.match(m.group(3)) and not _is_test_path(m.group(1))
                    and re.search(r"\b%s\b" % name, m.group(3))):
                key = (m.group(1), int(m.group(2)))
                if key not in seen:
                    seen.add(key)
                    defs.append(key)
    if hits:
        parts.append("## Where the issue's names appear\n```\n%s\n```" % "\n".join(hits[:4]))
    likely = _likely_code(ws, defs)
    if likely:
        parts.append("## Likely relevant code (line numbers are not part of the file)\n```\n%s\n```" % likely)

    if ws.target_test_command:
        code = _target_test_code(ws, ws.target_test_command)
        if code:
            parts.append("## Target test code\n```\n%s\n```" % code)
    if ws.target_test_command and run_baseline:
        snap = V.snapshot(ws, ws.target_test_command, baseline_timeout)
        ws.target_baseline = snap
        if snap.passed:
            parts.append("## Target test `%s`: PASSES already (exit code 0)" % ws.target_test_command)
        else:
            parts.append("## Target test `%s`: FAILS right now (exit code %s) - make it pass\n```\n%s\n```"
                         % (ws.target_test_command, snap.code, _tail(snap.output, 900)))
    elif ws.target_test_command:
        parts.append("## Target test: `%s` must pass when you are done." % ws.target_test_command)

    suite = V.suite_command(test_cmd)
    if run_baseline and suite:
        t0 = time.time()
        snap = V.snapshot(ws, suite, baseline_timeout)
        ws.baseline_seconds = time.time() - t0
        ws.baseline = snap
        ws.baseline_test_output = snap.output
        if snap.passed:
            parts.append("## Baseline test run: PASSED before any change (log messages are normal output)")
        else:
            failing = sorted(snap.failing)
            listed = ("\nAlready failing (%d): %s" % (len(failing), ", ".join(failing[:8]))) if failing else ""
            target_shown = ws.target_baseline is not None and not ws.target_baseline.passed
            log = "" if (target_shown and failing) else "\n```\n%s\n```" % _tail(snap.output, 700)
            parts.append("## Baseline test run: FAILED before any change (exit code %s)%s%s" % (
                snap.code, listed, log))
    return "\n\n".join(parts)


_NOISE_LINE = re.compile(r"^(?:[=\-_*~]{8,}|\[stderr\]|exit code: -?\d+.*)$")


def _tail(res: str, n: int = 900) -> str:
    # drop decorative separator lines, keep the end: that is where the assertion lives
    res = "\n".join(l for l in res.strip().splitlines() if not _NOISE_LINE.match(l.strip())).strip()
    return res if len(res) <= n else "...\n" + res[-n:]


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
