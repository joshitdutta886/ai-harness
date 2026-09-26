"""Tools the model can call. All paths are confined to the target repository.

Design choices that matter for weaker open models:
  * Every result is short, plain text, and truncated (head + tail) so one
    tool call can never flood the context window.
  * Errors are returned as text with a concrete next step, never raised,
    so the model can recover by itself.
  * Edits are exact search/replace (unique match required) and Python files
    are syntax-checked right after every edit.
"""
import difflib
import fnmatch
import os
import py_compile
import re
import shlex
import shutil
import subprocess
import time
from typing import Callable, Dict, List, Optional

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env", ".tox", ".mypy_cache",
             ".pytest_cache", "dist", "build", ".idea", ".vscode", "target", ".next", "site-packages",
             ".eggs", "runs"}
MAX_OUTPUT = 8000
MAX_READ_LINES = 300

BLOCKED_PATTERNS = [
    r"\brm\s+-rf?\s+(/|~|\$HOME)(\s|$)", r"\bgit\s+push\b", r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\b", r"\bgit\s+checkout\s+\.\s*$", r"\bshutdown\b", r"\breboot\b",
    r"\bmkfs\b", r":\(\)\s*\{", r"\bcurl\b.*\|\s*(ba)?sh", r"\bwget\b.*\|\s*(ba)?sh",
    r"\bsudo\b",
]


def truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.4)
    tail = limit - head
    omitted = len(text) - limit
    return text[:head] + "\n\n... [%d chars omitted] ...\n\n" % omitted + text[-tail:]


class Workspace:
    """State shared by all tools for one task."""

    def __init__(self, root: str, command_timeout: int = 180):
        self.root = os.path.realpath(root)
        self.command_timeout = command_timeout
        self.originals: Dict[str, Optional[str]] = {}  # rel path -> original text (None = new file)
        self.edit_counter = 0            # increments on every successful change
        self.last_test_edit_counter = -1  # edit_counter value when tests last passed
        self.last_test_passed: Optional[bool] = None
        self.last_test_command: str = ""
        self.last_test_output: str = ""
        self.test_command: Optional[str] = None
        self.files_read: set = set()
        self.baseline_test_output: str = ""
        self.target_test_command: Optional[str] = None   # evaluator-supplied "test case"
        self.baseline = None                              # verify.Snapshot of the suite before edits
        self.target_baseline = None                       # verify.Snapshot of the target test before edits

    # --------------------------------------------------------------- paths
    def resolve(self, path: str) -> str:
        path = (path or ".").strip()
        full = os.path.realpath(os.path.join(self.root, path))
        if full != self.root and not full.startswith(self.root + os.sep):
            raise ValueError("Path '%s' is outside the repository." % path)
        return full

    def rel(self, full: str) -> str:
        r = os.path.relpath(full, self.root)
        return "." if r == "." else r.replace(os.sep, "/")

    def _remember(self, full: str) -> None:
        rel = self.rel(full)
        if rel not in self.originals:
            if os.path.exists(full):
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    self.originals[rel] = f.read()
            else:
                self.originals[rel] = None

    # --------------------------------------------------------------- diff
    def diff(self) -> str:
        chunks = []
        for rel, before in sorted(self.originals.items()):
            full = os.path.join(self.root, rel)
            after = None
            if os.path.exists(full):
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    after = f.read()
            if before == after:
                continue
            a = (before or "").splitlines(keepends=True)
            b = (after or "").splitlines(keepends=True)
            chunks.append("".join(difflib.unified_diff(
                a, b,
                fromfile="/dev/null" if before is None else "a/" + rel,
                tofile="/dev/null" if after is None else "b/" + rel)))
        return "".join(chunks)

    def changed_files(self) -> List[str]:
        out = []
        for rel, before in self.originals.items():
            full = os.path.join(self.root, rel)
            after = None
            if os.path.exists(full):
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                    after = f.read()
            if before != after:
                out.append(rel)
        return sorted(out)

    # ---------------------------------------------------- test detection
    def detect_test_command(self) -> Optional[str]:
        if self.test_command:
            return self.test_command
        r = self.root
        has = lambda p: os.path.exists(os.path.join(r, p))
        py_markers = ["pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini", "setup.py", "conftest.py"]
        has_py_tests = has("tests") or has("test") or any(
            fnmatch.fnmatch(f, "test_*.py") or fnmatch.fnmatch(f, "*_test.py") for f in os.listdir(r))
        if any(has(m) for m in py_markers) or has_py_tests:
            if _python_has("pytest"):
                self.test_command = "python3 -m pytest -x -q"
            elif shutil.which("pytest"):
                self.test_command = "pytest -x -q"
            else:
                self.test_command = "python3 -m unittest discover -v"
            return self.test_command
        if has("package.json"):
            return "npm test --silent"
        if has("go.mod"):
            return "go test ./..."
        if has("Cargo.toml"):
            return "cargo test"
        if has("pom.xml"):
            return "mvn -q test"
        if has("build.gradle") or has("build.gradle.kts"):
            return "./gradlew test"
        if has("Makefile"):
            return "make test"
        return None


def _python_has(mod: str) -> bool:
    try:
        return subprocess.run(["python3", "-c", "import " + mod], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


# =========================================================================
# Tool implementations. Each takes (ws, **args) and returns a string.
# =========================================================================

def list_dir(ws: Workspace, path: str = ".", depth: int = 2) -> str:
    base = ws.resolve(path)
    if not os.path.isdir(base):
        return "Error: '%s' is not a directory." % path
    depth = max(1, min(int(depth or 2), 4))
    lines = []
    base_depth = base.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info"))
        level = dirpath.rstrip(os.sep).count(os.sep) - base_depth
        if level >= depth:
            dirnames[:] = []
            continue
        indent = "  " * level
        if level > 0:
            lines.append("%s%s/" % ("  " * (level - 1), os.path.basename(dirpath)))
        for f in sorted(filenames)[:60]:
            lines.append("%s%s" % (indent, f))
        if len(filenames) > 60:
            lines.append("%s... (%d more files)" % (indent, len(filenames) - 60))
        if len(lines) > 400:
            lines.append("... (listing truncated; list a subdirectory)")
            break
    return "\n".join(lines) or "(empty directory)"


def find_files(ws: Workspace, pattern: str) -> str:
    matches = []
    for dirpath, dirnames, filenames in os.walk(ws.root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            rel = ws.rel(os.path.join(dirpath, f))
            if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(rel, pattern):
                matches.append(rel)
                if len(matches) >= 200:
                    break
    if not matches:
        return "No files match '%s'." % pattern
    return "\n".join(sorted(matches))


def search_code(ws: Workspace, query: str, path: str = ".", regex: bool = False,
                file_glob: str = "", max_results: int = 60) -> str:
    base = ws.resolve(path)
    max_results = max(1, min(int(max_results or 60), 200))
    if shutil.which("rg"):
        cmd = ["rg", "-n", "--no-heading", "--color", "never", "-M", "300"]
        if not regex:
            cmd.append("-F")
        if file_glob:
            cmd += ["-g", file_glob]
        for d in SKIP_DIRS:
            cmd += ["-g", "!%s/" % d]
        cmd += ["-e", query, base]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            lines = [l.replace(ws.root + os.sep, "") for l in p.stdout.splitlines()]
        except subprocess.TimeoutExpired:
            return "Error: search timed out. Narrow the path or query."
    else:
        lines = []
        pat = re.compile(query if regex else re.escape(query))
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if file_glob and not fnmatch.fnmatch(fn, file_glob):
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    with open(full, "r", encoding="utf-8", errors="strict") as fh:
                        for i, line in enumerate(fh, 1):
                            if pat.search(line):
                                lines.append("%s:%d:%s" % (ws.rel(full), i, line.rstrip()[:300]))
                except (UnicodeDecodeError, OSError):
                    continue
    if not lines:
        return "No matches for '%s'." % query
    total = len(lines)
    out = "\n".join(lines[:max_results])
    if total > max_results:
        out += "\n... (%d more matches; narrow the search)" % (total - max_results)
    return out


def read_file(ws: Workspace, path: str, start_line: int = 1, end_line: Optional[int] = None) -> str:
    full = ws.resolve(path)
    if not os.path.isfile(full):
        hint = find_files(ws, "*" + os.path.basename(path))
        return "Error: file '%s' not found.%s" % (
            path, ("\nSimilar files:\n" + hint) if not hint.startswith("No files") else "")
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        return "Error reading '%s': %s" % (path, e)
    n = len(lines)
    start = max(1, int(start_line or 1))
    end = int(end_line) if end_line else min(n, start + MAX_READ_LINES - 1)
    end = min(end, n, start + MAX_READ_LINES - 1)
    ws.files_read.add(ws.rel(full))
    body = "".join("%6d\t%s" % (i, lines[i - 1]) for i in range(start, end + 1))
    header = "%s (lines %d-%d of %d)\n" % (ws.rel(full), start, end, n)
    footer = ""
    if end < n:
        footer = "\n... %d more lines. Call read_file with start_line=%d to continue." % (n - end, end + 1)
    return header + (body if body else "(empty file)\n") + footer


def _syntax_check(full: str) -> str:
    if not full.endswith(".py"):
        return ""
    try:
        py_compile.compile(full, doraise=True)
        return ""
    except py_compile.PyCompileError as e:
        return "\nWARNING: the file now has a Python syntax error:\n%s\nFix it before doing anything else." % (
            str(e.msg)[-800:])


def edit_file(ws: Workspace, path: str, old_str: str, new_str: str, replace_all: bool = False) -> str:
    full = ws.resolve(path)
    if not os.path.isfile(full):
        return "Error: '%s' does not exist. Use create_file for new files." % path
    with open(full, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if old_str == new_str:
        return "Error: old_str and new_str are identical; nothing to change."
    count = text.count(old_str) if old_str else 0
    if count == 0:
        # Try a whitespace-tolerant match to help the model recover.
        hint = _closest_block(text, old_str)
        return ("Error: old_str was not found in %s. It must match the file exactly, including "
                "indentation. Re-read the exact lines with read_file and try again.%s" % (path, hint))
    if count > 1 and not replace_all:
        return ("Error: old_str appears %d times in %s. Include more surrounding lines so it is "
                "unique, or set replace_all=true." % (count, path))
    ws._remember(full)
    new_text = text.replace(old_str, new_str) if replace_all else text.replace(old_str, new_str, 1)
    with open(full, "w", encoding="utf-8") as f:
        f.write(new_text)
    ws.edit_counter += 1
    # show the edited region so the model can confirm without another read
    line_no = text[:text.find(old_str)].count("\n") + 1
    lines = new_text.splitlines()
    lo, hi = max(1, line_no - 3), min(len(lines), line_no + new_str.count("\n") + 3)
    snippet = "\n".join("%6d\t%s" % (i, lines[i - 1]) for i in range(lo, hi + 1))
    return "Edited %s (%d replacement%s). Region now:\n%s%s" % (
        path, count if replace_all else 1, "s" if (replace_all and count > 1) else "", snippet, _syntax_check(full))


def _closest_block(text: str, old: str) -> str:
    old_lines = [l.strip() for l in old.strip().splitlines() if l.strip()]
    if not old_lines:
        return ""
    lines = text.splitlines()
    first = old_lines[0]
    best, best_i = 0.0, -1
    for i, l in enumerate(lines):
        r = difflib.SequenceMatcher(None, l.strip(), first).ratio()
        if r > best:
            best, best_i = r, i
    if best < 0.6:
        return ""
    lo, hi = max(0, best_i - 2), min(len(lines), best_i + len(old_lines) + 2)
    snippet = "\n".join("%6d\t%s" % (i + 1, lines[i]) for i in range(lo, hi))
    return "\nThe closest matching text is around line %d:\n%s" % (best_i + 1, snippet)


def create_file(ws: Workspace, path: str, content: str) -> str:
    full = ws.resolve(path)
    existed = os.path.exists(full)
    ws._remember(full)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    ws.edit_counter += 1
    return "%s %s (%d lines).%s" % ("Overwrote" if existed else "Created", path,
                                    content.count("\n") + 1, _syntax_check(full))


def run_command(ws: Workspace, command: str, timeout: Optional[int] = None) -> str:
    for pat in BLOCKED_PATTERNS:
        if re.search(pat, command):
            return "Error: this command is blocked for safety: %s" % command
    timeout = min(int(timeout or ws.command_timeout), 600)
    t0 = time.time()
    env = dict(os.environ)
    env.pop("AI_API_KEY", None)  # never expose the key to repository code
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env["CI"] = "1"
    env["GIT_PAGER"] = "cat"
    env["PAGER"] = "cat"
    try:
        p = subprocess.run(command, shell=True, cwd=ws.root, capture_output=True, text=True,
                           timeout=timeout, env=env, stdin=subprocess.DEVNULL, errors="replace")
    except subprocess.TimeoutExpired as e:
        partial = (e.stdout or "") if isinstance(e.stdout, str) else ""
        return "Error: command timed out after %ds.\n%s" % (timeout, truncate(partial, 3000))
    out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
    return "exit code: %d  (%.1fs)\n%s" % (p.returncode, time.time() - t0, truncate(out.strip() or "(no output)"))


def run_tests(ws: Workspace, command: str = "", timeout: Optional[int] = None) -> str:
    cmd = (command or "").strip() or ws.target_test_command or ws.detect_test_command()
    if not cmd:
        return ("Error: could not detect how to run tests in this repository. Pass an explicit "
                "command, e.g. run_tests(command=\"python3 -m pytest tests/test_x.py -q\").")
    result = run_command(ws, cmd, timeout)
    m = re.match(r"exit code: (-?\d+)", result)
    passed = bool(m) and m.group(1) == "0"
    ws.last_test_passed = passed
    ws.last_test_command = cmd
    ws.last_test_output = result
    if passed:
        ws.last_test_edit_counter = ws.edit_counter
    verdict = "TESTS PASSED" if passed else "TESTS FAILED (or did not run)"
    return "%s\ncommand: %s\n%s" % (verdict, cmd, result)


def revert_file(ws: Workspace, path: str) -> str:
    """Undo every change to one file (restores the original, or deletes a file you created)."""
    full = ws.resolve(path)
    rel = ws.rel(full)
    if rel not in ws.originals:
        return "Error: you have not changed '%s', so there is nothing to revert." % path
    original = ws.originals[rel]
    if original is None:
        if os.path.exists(full):
            os.remove(full)
        msg = "Deleted %s (it did not exist before your changes)." % rel
    else:
        with open(full, "w", encoding="utf-8") as f:
            f.write(original)
        msg = "Restored %s to its original content." % rel
    ws.edit_counter += 1
    return msg


def git_diff(ws: Workspace) -> str:
    d = ws.diff()
    return truncate(d, 12000) if d else "No changes made yet."


# =========================================================================
# Tool schemas (JSON Schema, shared by all providers)
# =========================================================================

def _schema(props: dict, required: List[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


TOOL_SPECS = [
    {"name": "list_dir",
     "description": "List files and folders (tree view). Use first to learn the repository layout.",
     "parameters": _schema({"path": {"type": "string", "description": "Directory relative to repo root. Default '.'"},
                            "depth": {"type": "integer", "description": "How many levels deep (1-4). Default 2"}}, [])},
    {"name": "find_files",
     "description": "Find files by name or glob, e.g. '*.py', 'test_*.py', 'src/**/config*'.",
     "parameters": _schema({"pattern": {"type": "string"}}, ["pattern"])},
    {"name": "search_code",
     "description": "Search file contents (like grep). Returns path:line:text. Use it to locate functions, "
                    "error messages and names mentioned in the issue.",
     "parameters": _schema({"query": {"type": "string", "description": "Text to find"},
                            "path": {"type": "string", "description": "Folder or file to search. Default '.'"},
                            "regex": {"type": "boolean", "description": "Treat query as a regex. Default false"},
                            "file_glob": {"type": "string", "description": "Only files matching this glob, e.g. '*.py'"}},
                           ["query"])},
    {"name": "read_file",
     "description": "Read a file with line numbers (max 300 lines per call). Use start_line/end_line for large files.",
     "parameters": _schema({"path": {"type": "string"},
                            "start_line": {"type": "integer"},
                            "end_line": {"type": "integer"}}, ["path"])},
    {"name": "edit_file",
     "description": "Replace an exact block of text in a file. old_str must match the file exactly "
                    "(including indentation) and be unique; include 2-3 surrounding lines to make it unique. "
                    "Do NOT include line numbers from read_file output.",
     "parameters": _schema({"path": {"type": "string"},
                            "old_str": {"type": "string", "description": "Exact existing text"},
                            "new_str": {"type": "string", "description": "Replacement text"},
                            "replace_all": {"type": "boolean"}}, ["path", "old_str", "new_str"])},
    {"name": "create_file",
     "description": "Create a new file (or fully overwrite a small one). Prefer edit_file for existing files.",
     "parameters": _schema({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"])},
    {"name": "run_command",
     "description": "Run a shell command in the repository root (non-interactive, with timeout). Use for "
                    "reproducing bugs, running scripts, git log/status, installing a missing test dependency.",
     "parameters": _schema({"command": {"type": "string"},
                            "timeout": {"type": "integer", "description": "Seconds. Default 180"}}, ["command"])},
    {"name": "run_tests",
     "description": "Run tests. With no command it runs the target test for this task if one was given, "
                    "otherwise the auto-detected suite. Pass an explicit command to run a specific file or "
                    "the whole suite. You MUST run tests successfully after your last edit before calling finish.",
     "parameters": _schema({"command": {"type": "string", "description": "Optional explicit test command"},
                            "timeout": {"type": "integer"}}, [])},
    {"name": "revert_file",
     "description": "Undo all your changes to one file, restoring the original. Use it when an edit "
                    "went wrong and you want a clean start on that file.",
     "parameters": _schema({"path": {"type": "string"}}, ["path"])},
    {"name": "git_diff",
     "description": "Show the diff of every change you have made so far. Review it before finishing.",
     "parameters": _schema({}, [])},
    {"name": "finish",
     "description": "Call when the task is fully done and verified. Give a short summary of the root cause, "
                    "the fix, and the test evidence.",
     "parameters": _schema({"summary": {"type": "string"}}, ["summary"])},
]

TOOL_FUNCS: Dict[str, Callable] = {
    "list_dir": list_dir,
    "find_files": find_files,
    "search_code": search_code,
    "read_file": read_file,
    "edit_file": edit_file,
    "create_file": create_file,
    "run_command": run_command,
    "run_tests": run_tests,
    "revert_file": revert_file,
    "git_diff": git_diff,
}


def execute(ws: Workspace, name: str, args: dict) -> str:
    """Run a tool and always return text (errors included)."""
    if "__invalid_json__" in args:
        return ("Error: your tool arguments were not valid JSON. Send them again as a JSON object. "
                "Received: %s" % str(args["__invalid_json__"])[:300])
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return "Error: unknown tool '%s'. Available: %s" % (name, ", ".join(list(TOOL_FUNCS) + ["finish"]))
    try:
        return fn(ws, **args)
    except TypeError as e:
        return "Error: bad arguments for %s: %s" % (name, e)
    except ValueError as e:
        return "Error: %s" % e
    except Exception as e:  # never crash the loop
        return "Error: %s failed: %s: %s" % (name, type(e).__name__, e)
