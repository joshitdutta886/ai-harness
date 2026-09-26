"""Terminal output. Plain ANSI, no dependencies. Respects NO_COLOR and non-TTY."""
import itertools
import json
import os
import sys
import threading
import time


def _color_enabled() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


class UI:
    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.c = _color_enabled()

    def _s(self, code: str, text: str) -> str:
        return "\033[%sm%s\033[0m" % (code, text) if self.c else text

    def p(self, text: str = "") -> None:
        if not self.quiet:
            print(text, flush=True)

    def waiting(self, label: str = "model thinking"):
        """Context manager: animated spinner + elapsed seconds (only on a real terminal)."""
        ui = self

        class _Spin:
            def __enter__(self):
                self.stop = threading.Event()
                self.t = None
                if ui.quiet or not ui.c:
                    return self
                def run():
                    t0 = time.time()
                    for ch in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
                        if self.stop.wait(0.1):
                            break
                        sys.stdout.write("\r   \033[36m%s\033[0m \033[2m%s... %.0fs\033[0m " % (ch, label, time.time() - t0))
                        sys.stdout.flush()
                    sys.stdout.write("\r\033[K")
                    sys.stdout.flush()
                self.t = threading.Thread(target=run, daemon=True)
                self.t.start()
                return self

            def __exit__(self, *exc):
                self.stop.set()
                if self.t:
                    self.t.join(timeout=1)
                return False

        return _Spin()

    def banner(self) -> None:
        self.p(self._s("1;36", "=" * 64))
        self.p(self._s("1;36", "  AI Coding Harness  -  autonomous issue-to-verified-fix agent"))
        self.p(self._s("1;36", "=" * 64))

    def header(self, cfg, repo: str) -> None:
        self.p(self._s("2", "model: %s  |  provider: %s  |  tool mode: %s" % (cfg.model, cfg.provider, cfg.tool_mode)))
        for n in getattr(cfg, "notes", []):
            self.p(self._s("2", n))
        self.p(self._s("2", "repo:  %s" % repo))
        self.p(self._s("2", "budget: %d steps, %d tokens" % (cfg.max_steps, cfg.max_total_tokens)))

    def phase(self, text: str) -> None:
        self.p(self._s("1;35", "\n>> " + text))

    def info(self, text: str) -> None:
        self.p(self._s("2", "   " + text))

    def error(self, text: str) -> None:
        self.p(self._s("1;31", "!! " + text))

    def step(self, n: int, total: int, tin: int, tout: int) -> None:
        self.p(self._s("1;34", "\n-- step %d/%d " % (n, total)) + self._s("2", "(tokens: %d in / %d out)" % (tin, tout)))

    def thought(self, text: str) -> None:
        t = text.strip()
        if len(t) > 600:
            t = t[:600] + " ..."
        for line in t.splitlines():
            self.p(self._s("37", "   " + line))

    def tool_call(self, name: str, args: dict) -> None:
        short = {}
        for k, v in args.items():
            s = v if isinstance(v, str) else json.dumps(v)
            s = s.replace("\n", "\\n")
            short[k] = s if len(s) <= 70 else s[:67] + "..."
        arg_s = ", ".join("%s=%s" % (k, v) for k, v in short.items())
        self.p(self._s("33", "   > %s(%s)" % (name, arg_s)))

    def tool_result(self, out: str, is_error: bool) -> None:
        body = out.split("\n[harness]")[0].strip().splitlines()
        first = body[0] if body else ""
        extra = " (+%d lines)" % (len(body) - 1) if len(body) > 1 else ""
        color = "31" if is_error else "32"
        self.p(self._s(color, "     %s" % first[:150]) + self._s("2", extra))

    def final(self, r) -> None:
        self.p("")
        self.p(self._s("1;36", "=" * 64))
        colors = {"resolved": "1;32", "finished_unverified": "1;33"}
        self.p(self._s(colors.get(r.status, "1;31"), "  STATUS: %s" % r.status.upper()))
        if getattr(r, "verification", ""):
            self.p("  " + r.verification.replace("\n", "\n  "))
        self.p("  steps: %d   tokens: %d in / %d out   time: %.0fs" % (r.steps, r.input_tokens, r.output_tokens, r.seconds))
        self.p("  changed files: %s" % (", ".join(r.changed_files) or "none"))
        tel = getattr(r, "telemetry", None)
        if tel:
            self.p("  tool calls: %d (%d errors)   edits: %d   test runs: %d (%d passed)   finish rejections: %d" % (
                tel["tool_calls"], tel["tool_errors"], tel["edits"], tel["test_runs"], tel["test_passes"],
                len(tel["finish_rejections"])))
        if r.summary:
            self.p("\n" + self._s("1", "Summary:"))
            self.p(r.summary.strip())
        if r.diff:
            self.p("\n" + self._s("1", "Patch:"))
            for line in r.diff.splitlines()[:120]:
                col = "32" if line.startswith("+") else "31" if line.startswith("-") else "2"
                self.p(self._s(col, line))
        self.p("\n  report: %s" % os.path.join(r.run_dir, "report.md"))
        self.p(self._s("1;36", "=" * 64))
