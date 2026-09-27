"""Token benchmark: `make bench`  (offline, no API key).

Runs the harness on every practice task with a scripted "typical agent" that behaves
the same way every time (search -> read -> edit -> test -> review -> finish), and
counts the tokens the harness would send and receive on each model call. Tokens are
estimated as characters / 4 of the full request (system prompt + tool schemas +
conversation), which is close to what DeepSeek and Qwen tokenizers give for code
and English.

This measures the HARNESS's cost, since the model's behaviour is fixed. Use it to
compare harness changes: lower is better, at the same resolve rate.
"""
import json
import os
import shutil
import sys
import tempfile

from .config import ROOT, Config
from .llm import LLMResponse, ToolCall

# The scripted agent: for each task, the calls a competent model typically makes.
# Like a real model, it skips search/read calls when the code it must change is already
# visible in the conversation, and it never calls a tool the harness does not offer.
FIXES = {
    "cart-discount": ("apply_discount", "shop/pricing.py",
                      "    return amount - amount * percent", "    return amount - amount * percent / 100"),
    "slugify": ("slugify", "textkit/text.py",
                '    out = []\n    for ch in title.lower():\n        out.append(ch if ch.isalnum() else "-")\n    return "".join(out)',
                '    import re\n    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")'),
    "median": ("median", "numkit/stats.py", "    return data[n // 2]",
               "    mid = n // 2\n    return data[mid] if n % 2 else (data[mid - 1] + data[mid]) / 2"),
}


def est(obj) -> int:
    return max(1, len(obj if isinstance(obj, str) else json.dumps(obj)) // 4)


class MeterLLM:
    """Scripted model that records the size of every request it receives."""

    def __init__(self, task: str):
        sym, path, old, new = FIXES[task]
        self.plan = [
            ("search_code", {"query": sym}),
            ("read_file", {"path": path}),
            ("edit_file", {"path": path, "old_str": old, "new_str": new}),
            ("run_tests", {}),
            ("git_diff", {}),
            ("finish", {"summary": "Fixed %s in %s; tests pass." % (sym, path)}),
        ]
        self.calls = []
        self.fix_line = old.strip().splitlines()[-1].strip()

    def chat(self, system, messages, tools):
        names = {t["name"] for t in tools}
        tin = est(system) + est(tools) + est(messages)
        seen = "\n".join(m.get("content") or "" for m in messages)
        while self.plan and (
                (self.plan[0][0] not in names and self.plan[0][0] != "finish")      # tool not offered
                or (self.plan[0][0] in ("search_code", "read_file") and self.fix_line in seen)):  # code already visible
            self.plan.pop(0)
        if not self.plan:
            name, args = "finish", {"summary": "done"}
        else:
            name, args = self.plan.pop(0)
        text = "Plan: %s." % name
        tout = est(text) + est({"name": name, "arguments": args})
        self.calls.append((tin, tout))
        return LLMResponse(text=text, tool_calls=[ToolCall("c%d" % len(self.calls), name, args)],
                           input_tokens=tin, output_tokens=tout)


def run_bench(quiet: bool = True):
    from .agent import Agent
    from .ui import UI
    with open(os.path.join(ROOT, "examples", "tasks.json"), "r", encoding="utf-8") as f:
        tasks = json.load(f)
    rows = []
    for t in tasks:
        if t["name"] not in FIXES:
            continue
        tmp = tempfile.mkdtemp(prefix="harness_bench_")
        repo = os.path.join(tmp, "repo")
        shutil.copytree(os.path.join(ROOT, t["repo"]), repo)
        cfg = Config(api_key="bench", provider="custom", base_url="http://bench.invalid/v1", model="bench",
                     runs_dir=os.path.join(tmp, "runs")).resolve(probe=False)
        with open(os.path.join(ROOT, t["issue"]), "r", encoding="utf-8") as f:
            issue = f.read()
        llm = MeterLLM(t["name"])
        r = Agent(cfg, repo, llm=llm, ui=UI(quiet=quiet), test_cmd=t.get("test_cmd") or None).run(issue)
        rows.append({"task": t["name"], "status": r.status, "calls": len(llm.calls),
                     "in": sum(c[0] for c in llm.calls), "out": sum(c[1] for c in llm.calls),
                     "first_call_in": llm.calls[0][0] if llm.calls else 0})
        shutil.rmtree(tmp, ignore_errors=True)
    return rows


def main() -> int:
    rows = run_bench()
    print("%-14s %-10s %5s %9s %8s %9s %12s" % ("task", "status", "calls", "in", "out", "total", "1st call in"))
    for r in rows:
        print("%-14s %-10s %5d %9d %8d %9d %12d" % (r["task"], r["status"], r["calls"], r["in"], r["out"],
                                                   r["in"] + r["out"], r["first_call_in"]))
    tot = sum(r["in"] + r["out"] for r in rows)
    ok = sum(1 for r in rows if r["status"] == "resolved")
    print("\nresolved %d/%d | total %d tokens (estimated: chars/4) | %.0f tokens per task" % (
        ok, len(rows), tot, tot / max(1, len(rows))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
