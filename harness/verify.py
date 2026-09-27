"""Regression-aware verification ("evidence over claims").

The harness never trusts the model's word that a fix works. It verifies on
its own, in two parts:

1. Target check - if a target test command was supplied (the evaluator's
   "test case"), it must pass after the fix. The harness also runs it before
   any change, so the report shows it failing before and passing after.
2. Regression check - the full suite is run before the first edit (baseline)
   and again at the end. The fix is accepted when the suite passes, or when
   every test still failing was already failing in the baseline, so no new
   failures were introduced. Test ids are parsed from pytest and unittest
   output; for other runners only the exit code is used.
"""
import re
from dataclasses import dataclass, field
from typing import Optional, Set

from . import tools as T

_PYTEST_FAIL = re.compile(r"^(?:FAILED|ERROR)\s+(\S+?)(?:\s+-\s+.*)?$", re.M)
_UNITTEST_FAIL = re.compile(r"^(?:FAIL|ERROR):\s+(\S+)\s+\(([^)]+)\)", re.M)
# vitest: " FAIL  lib/a.test.ts > suite > case"   jest: "FAIL src/a.test.js"   also " × case 5ms"
_JS_FAIL = re.compile(r"^\s*(?:FAIL|\u00d7|\u2715)\s+(\S.*?)\s*$", re.M)
_DURATION = re.compile(r"\s+\d+(?:\.\d+)?\s*m?s$")
_EXIT = re.compile(r"^exit code: (-?\d+)")


def exit_code(output: str) -> Optional[int]:
    m = _EXIT.search(output or "")
    return int(m.group(1)) if m else None


def failing_tests(output: str) -> Set[str]:
    """Extract ids of failing tests from pytest (-rfE) or unittest output."""
    ids = set(_PYTEST_FAIL.findall(output or ""))
    for name, where in _UNITTEST_FAIL.findall(output or ""):
        # py3.11+: "test_x (pkg.mod.Class.test_x)"; older: "test_x (pkg.mod.Class)"
        ids.add(where if where.endswith("." + name) else "%s.%s" % (where, name))
    for raw in _JS_FAIL.findall(output or ""):
        ids.add(_DURATION.sub("", raw).strip())
    return ids


def suite_command(cmd: Optional[str]) -> Optional[str]:
    """Turn the agent's quick test command into a full, parseable suite run."""
    if not cmd:
        return None
    if "pytest" in cmd:
        base = cmd.replace(" -x", "").replace(" -q", "")
        return base + " -q -rfE -p no:cacheprovider"
    return cmd


@dataclass
class Snapshot:
    command: str = ""
    output: str = ""
    code: Optional[int] = None
    failing: Set[str] = field(default_factory=set)

    @property
    def passed(self) -> bool:
        return self.code == 0


def snapshot(ws: T.Workspace, cmd: Optional[str], timeout: int) -> Optional[Snapshot]:
    if not cmd:
        return None
    out = T.run_command(ws, cmd, timeout=timeout)
    return Snapshot(cmd, out, exit_code(out), failing_tests(out))


@dataclass
class Verdict:
    ok: bool
    reason: str
    target: Optional[Snapshot] = None
    suite: Optional[Snapshot] = None
    new_failures: Set[str] = field(default_factory=set)
    preexisting: Set[str] = field(default_factory=set)

    def describe(self) -> str:
        lines = [("VERIFIED: " if self.ok else "NOT VERIFIED: ") + self.reason]
        if self.target:
            lines.append("- target test `%s`: %s" % (self.target.command, "PASS" if self.target.passed else "FAIL"))
        if self.suite:
            lines.append("- full suite `%s`: %s" % (
                self.suite.command, "PASS" if self.suite.passed else "exit %s" % self.suite.code))
        if self.preexisting:
            lines.append("- failing before and after (pre-existing, not caused by the fix): %s" %
                         ", ".join(sorted(self.preexisting)[:10]))
        if self.new_failures:
            lines.append("- NEW failures introduced: %s" % ", ".join(sorted(self.new_failures)[:10]))
        return "\n".join(lines)


def verify(ws: T.Workspace, target_cmd: Optional[str], suite_cmd: Optional[str],
           baseline: Optional[Snapshot], timeout: int) -> Verdict:
    target = snapshot(ws, target_cmd, timeout)
    if target is not None and not target.passed:
        return Verdict(False, "the target test still fails", target=target)

    suite = snapshot(ws, suite_cmd, timeout)
    if suite is None:
        if target is not None:
            return Verdict(True, "target test passes (no suite detected)", target=target)
        return Verdict(False, "no test command available to verify the change")
    if suite.passed:
        return Verdict(True, "all tests pass", target=target, suite=suite)

    if baseline is not None and baseline.code not in (0, None) and suite.failing and baseline.failing:
        new = suite.failing - baseline.failing
        pre = suite.failing & baseline.failing
        if not new:
            return Verdict(True, "no new failures; %d test(s) were already failing before the change" % len(pre),
                           target=target, suite=suite, preexisting=pre)
        return Verdict(False, "the change breaks %d test(s)" % len(new), target=target, suite=suite,
                       new_failures=new, preexisting=pre)
    return Verdict(False, "the test suite fails", target=target, suite=suite, new_failures=suite.failing)
