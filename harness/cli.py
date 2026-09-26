"""Command-line / interactive entry point (what `make run` launches).

Ways to give it a task:
  make run                                   -> interactive prompts
  make run REPO=path/or/git-url ISSUE="text"  -> non-interactive
  make run ISSUE=https://github.com/o/r/issues/12   (repo is cloned automatically)
  python3 -m harness --repo ../proj --issue-file issue.md
  cat issue.md | python3 -m harness --repo ../proj
"""
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request

from .config import ROOT, load_config
from .ui import UI

GH_ISSUE = re.compile(r"https?://github\.com/([^/\s]+)/([^/\s]+)/issues/(\d+)")
GH_REPO = re.compile(r"^(https?://github\.com/[^/\s]+/[^/\s]+?)(?:\.git)?/?$")
DEMO_REPO = os.path.join(ROOT, "examples", "sample_repo")
DEMO_ISSUE = os.path.join(ROOT, "examples", "sample_issue.md")


def fetch_github_issue(url: str) -> str:
    m = GH_ISSUE.search(url)
    owner, repo, num = m.group(1), m.group(2), m.group(3)
    api = "https://api.github.com/repos/%s/%s/issues/%s" % (owner, repo, num)
    headers = {"accept": "application/vnd.github+json", "user-agent": "ai-harness"}
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        headers["authorization"] = "Bearer " + tok
    with urllib.request.urlopen(urllib.request.Request(api, headers=headers), timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    text = "Title: %s\n\n%s" % (data.get("title", ""), data.get("body") or "")
    try:
        req = urllib.request.Request(api + "/comments?per_page=10", headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            comments = json.loads(r.read().decode("utf-8"))
        if comments:
            text += "\n\n## Comments\n" + "\n\n".join(
                "- %s: %s" % (c.get("user", {}).get("login", "?"), (c.get("body") or "")[:1500]) for c in comments)
    except Exception:
        pass
    return text


def resolve_issue(value: str) -> str:
    value = (value or "").strip()
    if GH_ISSUE.search(value) and len(value) < 300:
        return fetch_github_issue(value)
    if value and len(value) < 500 and "\n" not in value and os.path.isfile(os.path.expanduser(value)):
        with open(os.path.expanduser(value), "r", encoding="utf-8") as f:
            return f.read()
    return value


def resolve_repo(value: str, issue_hint: str, ui: UI) -> str:
    value = (value or "").strip()
    if not value:
        m = GH_ISSUE.search(issue_hint or "")
        if m:
            value = "https://github.com/%s/%s" % (m.group(1), m.group(2))
    if not value:
        return ""
    if value.startswith(("http://", "https://", "git@")) or value.endswith(".git"):
        name = re.sub(r"\.git$", "", value.rstrip("/").split("/")[-1]) or "repo"
        dest = os.path.join(ROOT, "workspace", name)
        if os.path.isdir(os.path.join(dest, ".git")):
            ui.info("using existing clone at %s" % dest)
        else:
            ui.phase("Cloning %s" % value)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            p = subprocess.run(["git", "clone", "--quiet", value, dest], capture_output=True, text=True)
            if p.returncode != 0:
                raise SystemExit("git clone failed:\n" + p.stderr)
        return dest
    path = os.path.realpath(os.path.expanduser(value))
    if path == os.path.realpath(DEMO_REPO):
        # never modify the bundled example; work on a fresh copy
        import shutil
        dest = os.path.join(ROOT, "runs", "demo_repo")
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(path, dest)
        ui.info("working on a fresh copy of the demo repo: %s" % dest)
        return dest
    if not os.path.isdir(path):
        raise SystemExit("Repository path not found: %s" % value)
    return path


def read_multiline(prompt: str) -> str:
    print(prompt, flush=True)
    lines, blank = [], 0
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        if line.strip() == "":
            blank += 1
            if blank >= 2 and lines:
                break
        else:
            blank = 0
        lines.append(line)
    return "\n".join(lines).strip()


def interactive_inputs(ui: UI):
    ui.p("Give me a repository and an issue to fix.\n")
    repo = input("Repository - local path or git URL\n  (press Enter to use the demo repo)\n> ").strip()
    if not repo:
        repo = DEMO_REPO
        with open(DEMO_ISSUE, "r", encoding="utf-8") as f:
            default_issue = f.read()
        ui.info("using demo repo: %s" % repo)
    else:
        default_issue = ""
    issue = read_multiline(
        "\nIssue - paste the text, a GitHub issue URL, or a file path.\n"
        "  (finish with two empty lines, a line with END, or Ctrl-D%s)" %
        ("; just press Enter twice for the demo issue" if default_issue else ""))
    test_cmd = ""
    try:
        test_cmd = input("\nTest command that should pass once fixed (optional - press Enter to skip)\n> ").strip()
    except EOFError:
        pass
    return repo, issue or default_issue, test_cmd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="harness", description="Autonomous coding-agent harness")
    ap.add_argument("--repo", default=os.environ.get("REPO", ""), help="path or git URL of the target repository")
    ap.add_argument("--issue", default=os.environ.get("ISSUE", ""), help="issue text, file path, or GitHub issue URL")
    ap.add_argument("--issue-file", default=os.environ.get("ISSUE_FILE", ""))
    ap.add_argument("--test-cmd", default=os.environ.get("TEST_CMD", ""),
                    help="test command that must pass when the issue is fixed (the evaluator's test case)")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--no-baseline", action="store_true", help="skip the baseline test run")
    ap.add_argument("--demo", action="store_true", help="run on the bundled sample repo")
    ap.add_argument("--check", action="store_true", help="check the API key / model connection and exit")
    args = ap.parse_args(argv)

    ui = UI()
    ui.banner()
    cfg = load_config()
    if args.max_steps:
        cfg.max_steps = args.max_steps

    if args.check:
        from .llm import LLMClient
        ui.header(cfg, "-")
        r = LLMClient(cfg).chat("Reply with the single word OK.", [{"role": "user", "content": "ping"}], [])
        ui.p("model replied: %r  (%d in / %d out tokens)" % (r.text[:80], r.input_tokens, r.output_tokens))
        return 0

    issue_raw = args.issue
    if args.issue_file:
        issue_raw = args.issue_file
    if args.demo:
        args.repo, issue_raw = DEMO_REPO, DEMO_ISSUE

    if not issue_raw and not sys.stdin.isatty():
        issue_raw = sys.stdin.read()
    if not args.repo and not issue_raw and sys.stdin.isatty():
        args.repo, issue_raw, typed_test = interactive_inputs(ui)
        args.test_cmd = args.test_cmd or typed_test
    elif not issue_raw and sys.stdin.isatty():
        issue_raw = read_multiline("Issue - paste text, a GitHub issue URL, or a file path (end with two empty lines):")

    issue = resolve_issue(issue_raw)
    if not issue:
        raise SystemExit("No issue given. Pass ISSUE=... or run interactively.")
    repo = resolve_repo(args.repo, issue_raw, ui) or os.getcwd()

    from .agent import Agent
    result = Agent(cfg, repo, ui=ui, run_baseline=not args.no_baseline, test_cmd=args.test_cmd,
                   run_dir=os.environ.get("HARNESS_RUN_DIR") or None).run(issue)
    return 0 if result.status in ("resolved", "finished_unverified") else 1


if __name__ == "__main__":
    sys.exit(main())
