"""Aggregate every run in runs/ into one table: `make report`."""
import glob
import json
import os
import sys

from .config import ROOT


def main() -> int:
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "summary.json"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                s = json.load(f)
        except Exception:
            continue
        t = s.get("telemetry", {})
        rows.append((os.path.basename(os.path.dirname(path)), s.get("status", "?"), s.get("model", "?"),
                     s.get("steps", 0), s.get("input_tokens", 0) + s.get("output_tokens", 0),
                     round(s.get("seconds", 0)), t.get("tool_errors", 0), len(t.get("finish_rejections", []))))
    if not rows:
        print("No runs yet. Try: make demo")
        return 0
    hdr = ("run", "status", "model", "steps", "tokens", "secs", "tool_err", "rejections")
    widths = [max(len(str(x)) for x in col) for col in zip(hdr, *rows)]
    line = lambda r: "  ".join(str(v).ljust(w) for v, w in zip(r, widths))
    print(line(hdr))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(line(r))
    resolved = sum(1 for r in rows if r[1] == "resolved")
    print("\n%d runs | %d resolved (%.0f%%) | avg %.0f tokens | avg %.1f steps" % (
        len(rows), resolved, 100.0 * resolved / len(rows),
        sum(r[4] for r in rows) / len(rows), sum(r[3] for r in rows) / len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
