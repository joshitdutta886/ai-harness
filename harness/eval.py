"""Run the harness on every task in examples/tasks.json and print a score table.

    make eval                 # all tasks (needs AI_API_KEY)
    make eval TASK=median     # one task
    python3 -m harness.eval --check   # no model: confirm every task's bug is real

Each task runs on a fresh copy of its repo, so the examples are never modified.
"""
import argparse
import json
import os
import shutil
import sys
import time

from .config import ROOT, load_config
from .ui import UI

TASKS_FILE = os.path.join(ROOT, "examples", "tasks.json")


def load_tasks(only: str = ""):
    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if only:
        tasks = [t for t in tasks if t["name"] == only]
    return tasks


def fresh_copy(task: dict, dest_root: str) -> str:
    dest = os.path.join(dest_root, task["name"], "repo")
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(os.path.join(ROOT, task["repo"]), dest)
    return dest


def check_tasks(tasks) -> int:
    """Without a model: make sure each task really has a failing test to fix."""
    from . import tools as T
    from . import verify as V
    import tempfile
    tmp = tempfile.mkdtemp(prefix="harness_eval_check_")
    bad = 0
    for t in tasks:
        repo = fresh_copy(t, tmp)
        ws = T.Workspace(repo)
        cmd = t.get("test_cmd") or ws.detect_test_command()
        snap = V.snapshot(ws, cmd, 120)
        ok = not snap.passed
        bad += 0 if ok else 1
        print("%-14s %-6s %s" % (t["name"], "ok" if ok else "BROKEN", "bug reproduces" if ok else "tests already pass"))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="harness.eval")
    ap.add_argument("--task", default="")
    ap.add_argument("--check", action="store_true", help="only confirm the tasks reproduce (no model)")
    args = ap.parse_args(argv)
    tasks = load_tasks(args.task)
    if not tasks:
        print("No matching tasks in %s" % TASKS_FILE)
        return 1
    if args.check:
        return check_tasks(tasks)

    from .agent import Agent
    cfg = load_config()
    ui = UI()
    ui.banner()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = os.path.join(cfg.runs_dir, "eval-" + stamp)
    rows = []
    for t in tasks:
        ui.phase("TASK %s" % t["name"])
        with open(os.path.join(ROOT, t["issue"]), "r", encoding="utf-8") as f:
            issue = f.read()
        repo = fresh_copy(t, root)
        run_dir = os.path.join(root, t["name"])
        r = Agent(cfg, repo, ui=ui, test_cmd=t.get("test_cmd") or None, run_dir=run_dir).run(issue)
        rows.append((t["name"], r.status, "yes" if r.verified else "no", r.steps,
                     r.input_tokens + r.output_tokens, round(r.seconds)))

    hdr = ("task", "status", "verified", "steps", "tokens", "secs")
    widths = [max(len(str(x)) for x in col) for col in zip(hdr, *rows)]
    fmt = lambda r: "  ".join(str(v).ljust(w) for v, w in zip(r, widths))
    lines = [fmt(hdr), "  ".join("-" * w for w in widths)] + [fmt(r) for r in rows]
    solved = sum(1 for r in rows if r[1] == "resolved")
    lines.append("")
    lines.append("resolved %d/%d | model %s (%s) | total tokens %d" % (
        solved, len(rows), cfg.model, cfg.provider, sum(r[4] for r in rows)))
    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(root, "eval_summary.md"), "w", encoding="utf-8") as f:
        f.write("# Evaluation run %s\n\n```\n%s\n```\n" % (stamp, table))
    print("\nsaved: %s" % os.path.join(root, "eval_summary.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
