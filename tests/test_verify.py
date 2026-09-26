"""Verification, target tests, revert, time limit and the eval task set."""
import json
import os
import shutil
import tempfile
import unittest

from harness import tools as T
from harness import verify as V
from harness.agent import Agent
from harness.eval import check_tasks, load_tasks
from harness.ui import UI
from tests.helpers import ROOT, fake_config
from tests.test_agent import ScriptedLLM


def task_copy(name):
    d = tempfile.mkdtemp(prefix="harness_task_")
    dest = os.path.join(d, "repo")
    shutil.copytree(os.path.join(ROOT, "examples", "tasks", name, "repo"), dest)
    return dest


PYTEST_OUT = """exit code: 1  (0.3s)
..F.
=========================== short test summary info ============================
FAILED tests/test_text.py::TitleCaseTests::test_small_words - AssertionError: 'a' != 'b'
ERROR tests/test_io.py::test_read
1 failed, 3 passed in 0.12s"""

UNITTEST_OUT = """exit code: 1  (0.1s)
======================================================================
FAIL: test_small_words (tests.test_text.TitleCaseTests.test_small_words)
----------------------------------------------------------------------
ERROR: test_x (tests.test_old.OldTests)
"""


class ParseTests(unittest.TestCase):
    def test_pytest_ids(self):
        self.assertEqual(V.failing_tests(PYTEST_OUT),
                         {"tests/test_text.py::TitleCaseTests::test_small_words", "tests/test_io.py::test_read"})

    def test_unittest_ids(self):
        self.assertEqual(V.failing_tests(UNITTEST_OUT),
                         {"tests.test_text.TitleCaseTests.test_small_words", "tests.test_old.OldTests.test_x"})

    def test_suite_command(self):
        self.assertEqual(V.suite_command("python3 -m pytest -x -q"), "python3 -m pytest -q -rfE -p no:cacheprovider")
        self.assertEqual(V.suite_command("npm test"), "npm test")
        self.assertIsNone(V.suite_command(None))


class VerificationLoopTests(unittest.TestCase):
    def run_agent(self, repo, script, test_cmd=None, **cfg_over):
        cfg = fake_config(tempfile.mkdtemp())
        for k, v in cfg_over.items():
            setattr(cfg, k, v)
        llm = ScriptedLLM(script)
        agent = Agent(cfg, repo, llm=llm, ui=UI(quiet=True), test_cmd=test_cmd)
        return agent.run("issue"), llm

    def test_preexisting_failure_does_not_block(self):
        repo = task_copy("slugify")
        fix = ("edit_file", {"path": "textkit/text.py",
                             "old_str": '''    out = []
    for ch in title.lower():
        out.append(ch if ch.isalnum() else "-")
    return "".join(out)''',
                             "new_str": '''    import re
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")'''})
        r, _ = self.run_agent(repo, [fix, ("run_tests", {}), ("finish", {"summary": "fixed"})],
                              test_cmd="python3 -m unittest tests.test_text.SlugifyTests")
        self.assertEqual(r.status, "resolved", r.verification)
        self.assertTrue(r.verified)
        self.assertIn("already failing", r.verification)
        self.assertIn("test_small_words", r.verification)

    def test_new_failure_is_rejected(self):
        repo = task_copy("median")
        fix = ("edit_file", {"path": "numkit/stats.py", "old_str": "    return data[n // 2]",
                             "new_str": "    mid = n // 2\n    return data[mid] if n % 2 else (data[mid - 1] + data[mid]) / 2"})
        breaks_mean = ("edit_file", {"path": "numkit/stats.py", "old_str": "return sum(values) / len(values)",
                                     "new_str": "return sum(values) // len(values)"})
        target = "python3 -m unittest tests.test_stats.MedianTests.test_even_length"
        r, llm = self.run_agent(repo, [fix, breaks_mean, ("run_tests", {}), ("finish", {"summary": "done"})],
                                test_cmd=target)
        self.assertIn("verification_failed", r.telemetry["finish_rejections"])
        self.assertIn("NEW failures", json.dumps(llm.seen[-1]))
        self.assertIn("test_mean", json.dumps(llm.seen[-1]))

    def test_target_test_resolves_and_is_shown_failing_first(self):
        repo = task_copy("median")
        fix = ("edit_file", {"path": "numkit/stats.py", "old_str": "    return data[n // 2]",
                             "new_str": "    mid = n // 2\n    return data[mid] if n % 2 else (data[mid - 1] + data[mid]) / 2"})
        target = "python3 -m unittest tests.test_stats.MedianTests.test_even_length"
        r, llm = self.run_agent(repo, [fix, ("run_tests", {}), ("finish", {"summary": "done"})], test_cmd=target)
        self.assertEqual(r.status, "resolved", r.verification)
        self.assertIn("FAILS right now", llm.seen[0][0]["content"])
        self.assertIn("target test", r.verification)

    def test_time_limit(self):
        repo = task_copy("median")
        r, _ = self.run_agent(repo, [("list_dir", {})] * 10, max_minutes=0.0001)
        self.assertEqual(r.status, "time_exhausted")


class RevertTests(unittest.TestCase):
    def test_revert_restores_and_deletes(self):
        repo = task_copy("median")
        ws = T.Workspace(repo)
        T.edit_file(ws, "numkit/stats.py", "data[n // 2]", "data[0]")
        T.create_file(ws, "new.py", "x = 1\n")
        self.assertEqual(ws.changed_files(), ["new.py", "numkit/stats.py"])
        self.assertIn("Restored", T.revert_file(ws, "numkit/stats.py"))
        self.assertIn("Deleted", T.revert_file(ws, "new.py"))
        self.assertEqual(ws.changed_files(), [])
        self.assertIn("nothing to revert", T.revert_file(ws, "numkit/__init__.py"))


class EvalSetTests(unittest.TestCase):
    def test_every_task_reproduces(self):
        tasks = load_tasks()
        self.assertGreaterEqual(len(tasks), 3)
        self.assertEqual(check_tasks(tasks), 0)


if __name__ == "__main__":
    unittest.main()
