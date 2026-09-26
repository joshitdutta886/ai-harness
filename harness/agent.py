"""The agent loop: orchestration, verification gate, recovery and budgets."""
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

from . import context as C
from . import prompts as P
from . import tools as T
from . import verify as V
from .config import Config
from .llm import LLMClient, LLMError, LLMResponse
from .ui import UI


@dataclass
class RunResult:
    status: str = "incomplete"          # resolved | finished_unverified | budget_exhausted | error
    summary: str = ""
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    diff: str = ""
    changed_files: List[str] = field(default_factory=list)
    tests_passed: Optional[bool] = None
    test_command: str = ""
    test_output: str = ""
    run_dir: str = ""
    error: str = ""
    telemetry: dict = field(default_factory=dict)
    verified: Optional[bool] = None
    verification: str = ""


class Agent:
    def __init__(self, cfg: Config, repo: str, llm=None, ui: Optional[UI] = None,
                 run_baseline: bool = True, test_cmd: Optional[str] = None, run_dir: Optional[str] = None):
        self.cfg = cfg
        self.ws = T.Workspace(repo, command_timeout=cfg.command_timeout)
        self.ws.target_test_command = (test_cmd or "").strip() or None
        self.verdict: Optional[V.Verdict] = None
        self.t_start = time.time()
        self.llm = llm or LLMClient(cfg)
        self.ui = ui or UI()
        self.run_baseline = run_baseline
        self.messages: List[dict] = []
        self.result = RunResult()
        self.call_counts = {}
        self.error_streak = 0
        self.finish_rejections = 0
        self.no_tool_streak = 0
        self.warned_wrap_up = False
        self.telemetry = {
            "model_calls": 0, "model_seconds": 0.0, "tool_calls": 0, "tool_errors": 0,
            "tools": {},                 # name -> {"calls", "errors", "seconds"}
            "tokens_per_step": [],       # [input, output] per model call
            "compactions": 0, "chars_saved_by_compaction": 0,
            "finish_rejections": [],     # reasons
            "nudges_no_tool": 0, "repeat_warnings": 0, "recovery_hints": 0,
            "text_mode_tool_calls": 0, "edits": 0, "test_runs": 0, "test_passes": 0,
            "verifications": 0, "reverts": 0,
        }
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = run_dir or os.path.join(cfg.runs_dir, stamp)
        os.makedirs(self.run_dir, exist_ok=True)
        self.result.run_dir = self.run_dir
        self._traj = open(os.path.join(self.run_dir, "trajectory.jsonl"), "w", encoding="utf-8")

    # ---------------------------------------------------------------- logging
    def log(self, kind: str, **data) -> None:
        data.update({"t": round(time.time(), 2), "kind": kind})
        self._traj.write(json.dumps(data, default=str) + "\n")
        self._traj.flush()

    # ------------------------------------------------------------------- main
    def run(self, issue: str) -> RunResult:
        t0 = self.t_start = time.time()
        self.ui.header(self.cfg, self.ws.root)
        self.log("start", issue=issue, repo=self.ws.root, model=self.cfg.model, provider=self.cfg.provider,
                 max_steps=self.cfg.max_steps, target_test=self.ws.target_test_command or "")
        self.ui.phase("Gathering repository context")
        overview = C.build_overview(self.ws, issue, run_baseline=self.run_baseline)
        self.log("overview", text=overview)
        self.ui.info("test command: %s" % (self.ws.detect_test_command() or "not detected"))
        self.messages = [{"role": "user", "content": P.initial_user_message(issue, overview)}]

        try:
            self._loop()
        except LLMError as e:
            self.result.status = "error"
            self.result.error = str(e)
            self.ui.error(str(e))
        except KeyboardInterrupt:
            self.result.status = "interrupted"
            self.ui.error("Interrupted by user.")

        if self.verdict is None and self.ws.changed_files() and self.result.status not in ("interrupted",):
            try:
                self._verify()   # evidence for the report even when the loop ended early
            except Exception as e:  # never lose the report because of this
                self.log("verify_error", error=str(e))
        r = self.result
        if self.verdict is not None:
            r.verified = self.verdict.ok
            r.verification = self.verdict.describe()
        r.seconds = time.time() - t0
        r.diff = self.ws.diff()
        r.changed_files = self.ws.changed_files()
        r.tests_passed = self.ws.last_test_passed
        r.test_command = self.ws.last_test_command
        r.test_output = self.ws.last_test_output
        r.telemetry = self.telemetry
        self._write_artifacts(issue)
        self.log("end", status=r.status, verified=r.verified, verification=r.verification, summary=r.summary,
                 steps=r.steps, input_tokens=r.input_tokens, output_tokens=r.output_tokens,
                 seconds=round(r.seconds, 1), diff=r.diff, changed_files=r.changed_files)
        self._traj.close()
        self.ui.final(r)
        return r

    def _loop(self) -> None:
        cfg = self.cfg
        for step in range(1, cfg.max_steps + 1):
            self.result.steps = step
            used = self.result.input_tokens + self.result.output_tokens
            if used >= cfg.max_total_tokens:
                self.result.status = "budget_exhausted"
                self.ui.error("Token budget exhausted (%d)." % used)
                return

            elapsed_min = (time.time() - self.t_start) / 60.0
            if cfg.max_minutes and elapsed_min >= cfg.max_minutes:
                self.result.status = "time_exhausted"
                self.ui.error("Time limit reached (%.0f min)." % cfg.max_minutes)
                return
            left = cfg.max_steps - step
            time_short = cfg.max_minutes and elapsed_min >= 0.8 * cfg.max_minutes
            if (left <= max(3, cfg.max_steps // 6) or time_short) and not self.warned_wrap_up:
                self.warned_wrap_up = True
                self._add_user(P.NUDGE_WRAP_UP.format(left=left))

            before = C.total_chars(self.messages)
            n = C.compact(self.messages, cfg.keep_recent_tool_results, cfg.context_char_budget)
            if n:
                after = C.total_chars(self.messages)
                self.telemetry["compactions"] += n
                self.telemetry["chars_saved_by_compaction"] += before - after
                self.log("compact", count=n, chars=after)

            self.ui.step(step, cfg.max_steps, self.result.input_tokens, self.result.output_tokens)
            self.log("step", step=step, input_tokens=self.result.input_tokens, output_tokens=self.result.output_tokens)
            tm = time.time()
            wasted0 = getattr(self.llm, "wasted_tokens", 0)
            with self.ui.waiting("model thinking"):
                resp: LLMResponse = self.llm.chat(P.SYSTEM_PROMPT, self.messages, T.TOOL_SPECS)
            self.result.input_tokens += getattr(self.llm, "wasted_tokens", 0) - wasted0  # count retried calls too
            tel = self.telemetry
            tel["model_calls"] += 1
            tel["model_seconds"] += time.time() - tm
            tel["tokens_per_step"].append([resp.input_tokens, resp.output_tokens])
            tel["text_mode_tool_calls"] += sum(1 for tc in resp.tool_calls if tc.id.startswith("txt_"))
            self.result.input_tokens += resp.input_tokens
            self.result.output_tokens += resp.output_tokens
            self.log("model", text=resp.text, tool_calls=[tc.__dict__ for tc in resp.tool_calls],
                     usage=[resp.input_tokens, resp.output_tokens], stop=resp.stop_reason)
            if resp.text:
                self.ui.thought(resp.text)

            msg = {"role": "assistant", "content": resp.text,
                   "tool_calls": [{"id": tc.id, "name": tc.name, "args": tc.args} for tc in resp.tool_calls]}
            if resp.reasoning:
                msg["reasoning"] = resp.reasoning
            self.messages.append(msg)

            if not resp.tool_calls:
                self.no_tool_streak += 1
                if self.no_tool_streak >= 3:
                    # model keeps talking without acting: treat its text as a finish attempt
                    if self._try_finish(resp.text or "(no summary)", None):
                        return
                else:
                    self.telemetry["nudges_no_tool"] += 1
                    self._add_user(P.NUDGE_NO_TOOL)
                continue
            self.no_tool_streak = 0

            finish_call = None
            for tc in resp.tool_calls:
                if tc.name == "finish":
                    finish_call = tc
                    continue
                self._run_tool(tc)
            if finish_call is not None:
                summary = str(finish_call.args.get("summary", "")) or resp.text
                if self._try_finish(summary, finish_call.id):
                    return
        self.result.status = "budget_exhausted"
        self.ui.error("Step budget exhausted.")

    # ------------------------------------------------------------------ tools
    def _run_tool(self, tc) -> None:
        sig = tc.name + json.dumps(tc.args, sort_keys=True)
        self.call_counts[sig] = self.call_counts.get(sig, 0) + 1
        self.ui.tool_call(tc.name, tc.args)
        t0 = time.time()
        out = T.execute(self.ws, tc.name, tc.args)
        dt = time.time() - t0
        is_error = out.startswith("Error") or out.startswith("TESTS FAILED")
        self.error_streak = self.error_streak + 1 if out.startswith("Error") else 0
        tel = self.telemetry
        tel["tool_calls"] += 1
        t = tel["tools"].setdefault(tc.name, {"calls": 0, "errors": 0, "seconds": 0.0})
        t["calls"] += 1
        t["seconds"] = round(t["seconds"] + dt, 2)
        if out.startswith("Error"):
            t["errors"] += 1
            tel["tool_errors"] += 1
        if tc.name in ("edit_file", "create_file") and not out.startswith("Error"):
            tel["edits"] += 1
        if tc.name == "revert_file" and not out.startswith("Error"):
            tel["reverts"] += 1
        if tc.name == "run_tests":
            tel["test_runs"] += 1
            tel["test_passes"] += int(out.startswith("TESTS PASSED"))
        repeats = self.call_counts[sig]
        if repeats >= 3 and tc.name not in ("run_tests", "git_diff"):
            out += P.REPEAT_WARNING.format(n=repeats)
            tel["repeat_warnings"] += 1
        if self.error_streak >= 3:
            out += P.ERROR_STREAK_HINT
            tel["recovery_hints"] += 1
            self.error_streak = 0
        out += self._status_line()
        self.ui.tool_result(out, is_error)
        self.log("tool", name=tc.name, args=tc.args, seconds=round(dt, 2), output=out[:4000])
        self.messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": out})

    def _status_line(self) -> str:
        changed = self.ws.changed_files()
        if self.ws.last_test_passed is None:
            tests = "not run yet"
        elif self.ws.last_test_passed and self.ws.last_test_edit_counter == self.ws.edit_counter:
            tests = "PASSED (up to date)"
        elif self.ws.last_test_passed:
            tests = "passed before your latest edit - re-run"
        else:
            tests = "FAILED"
        return "\n[harness] step %d/%d | changed: %s | tests: %s" % (
            self.result.steps, self.cfg.max_steps, ", ".join(changed) or "none", tests)

    def _add_user(self, text: str) -> None:
        # Tool results must directly follow the assistant turn; appending a user
        # message after them is fine for both wire formats.
        self.messages.append({"role": "user", "content": text})

    def _reply_to_finish(self, call_id: Optional[str], text: str) -> None:
        if call_id:
            self.messages.append({"role": "tool", "tool_call_id": call_id, "name": "finish", "content": text})
        else:
            self._add_user(text)

    # ----------------------------------------------------------------- finish
    def _verify(self) -> Optional[V.Verdict]:
        ws = self.ws
        suite = V.suite_command(ws.detect_test_command())
        if not suite and not ws.target_test_command:
            return None
        self.ui.phase("Harness verification (target test + regression check)")
        verdict = V.verify(ws, ws.target_test_command, suite, ws.baseline, self.cfg.command_timeout)
        self.verdict = verdict
        self.telemetry["verifications"] += 1
        self.log("verify", ok=verdict.ok, reason=verdict.reason, detail=verdict.describe(),
                 new_failures=sorted(verdict.new_failures), preexisting=sorted(verdict.preexisting))
        self.ui.info(verdict.describe().replace("\n", "\n   "))
        return verdict

    def _try_finish(self, summary: str, call_id: Optional[str]) -> bool:
        ws = self.ws
        changed = ws.changed_files()
        max_rejections = 2

        if not changed and self.finish_rejections < 1:
            self.finish_rejections += 1
            self.ui.info("finish rejected: no changes yet")
            self.telemetry["finish_rejections"].append("no_changes")
            self._reply_to_finish(call_id, P.FINISH_NO_CHANGES)
            return False

        if changed and self.finish_rejections < max_rejections:
            tests_fresh = ws.last_test_passed and ws.last_test_edit_counter == ws.edit_counter
            no_way_to_test = ws.detect_test_command() is None and not ws.target_test_command
            if not tests_fresh and not no_way_to_test:
                self.finish_rejections += 1
                self.ui.info("finish rejected: tests not run after last edit")
                self.telemetry["finish_rejections"].append("tests_not_run")
                self._reply_to_finish(call_id, P.FINISH_NO_TESTS)
                return False
            # Independent verification by the harness ("evidence over claims").
            verdict = self._verify()
            if verdict is not None and not verdict.ok:
                self.finish_rejections += 1
                self.ui.info("finish rejected: harness verification failed")
                self.telemetry["finish_rejections"].append("verification_failed")
                failing = verdict.target if (verdict.target and not verdict.target.passed) else verdict.suite
                detail = T.truncate(failing.output, 3000) if failing else ""
                self._reply_to_finish(call_id, P.FINISH_VERIFY_FAILED.format(
                    verdict=verdict.describe(), output=detail))
                return False
        elif changed and self.verdict is None:
            self._verify()   # out of retries: still record the evidence

        self.result.status = "resolved" if (changed and self.verdict is not None and self.verdict.ok) \
            else "finished_unverified"
        self.result.summary = summary
        self._reply_to_finish(call_id, "Finished.")
        return True

    # -------------------------------------------------------------- artifacts
    def _write_artifacts(self, issue: str) -> None:
        r = self.result
        with open(os.path.join(self.run_dir, "patch.diff"), "w", encoding="utf-8") as f:
            f.write(r.diff)
        summary = {k: v for k, v in r.__dict__.items() if k not in ("diff", "test_output", "telemetry")}
        summary.update({"provider": self.cfg.provider, "model": self.cfg.model, "repo": self.ws.root})
        tel = dict(self.telemetry)
        tel["model_seconds"] = round(tel["model_seconds"], 2)
        summary["telemetry"] = tel
        with open(os.path.join(self.run_dir, "telemetry.json"), "w", encoding="utf-8") as f:
            json.dump(tel, f, indent=2)
        with open(os.path.join(self.run_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        lines = [
            "# Harness run report", "",
            "- **Status:** %s" % r.status,
            "- **Model:** %s (%s)" % (self.cfg.model, self.cfg.provider),
            "- **Repository:** %s" % self.ws.root,
            "- **Steps:** %d / %d" % (r.steps, self.cfg.max_steps),
            "- **Tokens:** %d in / %d out" % (r.input_tokens, r.output_tokens),
            "- **Time:** %.1fs" % r.seconds,
            "- **Changed files:** %s" % (", ".join(r.changed_files) or "none"),
            "- **Verified by harness:** %s" % {True: "yes", False: "no", None: "not run"}[r.verified],
            "- **Target test:** %s" % (self.ws.target_test_command or "none given"), "",
            "## Issue", "", issue.strip(), "",
            "## Telemetry", "",
            "| Metric | Value |", "|---|---|",
            "| Model calls | %d (%.1fs waiting on the model) |" % (self.telemetry["model_calls"], self.telemetry["model_seconds"]),
            "| Tool calls | %d (%d returned an error to the model) |" % (self.telemetry["tool_calls"], self.telemetry["tool_errors"]),
            "| Edits | %d (reverts: %d) |" % (self.telemetry["edits"], self.telemetry["reverts"]),
            "| Harness verifications | %d |" % self.telemetry["verifications"],
            "| Test runs | %d (%d passed) |" % (self.telemetry["test_runs"], self.telemetry["test_passes"]),
            "| Finish rejections | %s |" % (", ".join(self.telemetry["finish_rejections"]) or "none"),
            "| Context compactions | %d (%d chars saved) |" % (self.telemetry["compactions"], self.telemetry["chars_saved_by_compaction"]),
            "| Recovery nudges | no-tool %d, repeat %d, error-streak %d |" % (self.telemetry["nudges_no_tool"], self.telemetry["repeat_warnings"], self.telemetry["recovery_hints"]),
            "| Text-format tool calls parsed | %d |" % self.telemetry["text_mode_tool_calls"], "",
            "Per tool:", "",
            "| Tool | Calls | Errors | Seconds |", "|---|---|---|---|",
        ] + ["| %s | %d | %d | %.1f |" % (k, v["calls"], v["errors"], v["seconds"])
             for k, v in sorted(self.telemetry["tools"].items())] + [
            "",
            "## Agent summary", "", r.summary or r.error or "(none)", "",
            "## Harness verification", "",
            "```", r.verification or "(not run: no test command available)", "```", "",
            "## Test evidence", "",
            "Command: `%s`" % (r.test_command or "n/a"), "",
            "```", T.truncate(r.test_output or "(tests not run)", 4000), "```", "",
            "## Patch", "", "```diff", r.diff or "(no changes)", "```", "",
        ]
        with open(os.path.join(self.run_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
