"""Token budget guard: the harness must stay cheap. Uses the offline token benchmark."""
import json
import unittest

from harness import prompts as P
from harness import tools as T
from harness.tokenbench import run_bench


class TokenBudgetTests(unittest.TestCase):
    def test_fixed_cost_per_call_is_small(self):
        # system prompt + tool schemas are re-sent on every model call
        per_call = (len(P.SYSTEM_PROMPT) + len(json.dumps(T.TOOL_SPECS))) // 4
        self.assertLess(per_call, 900, per_call)

    def test_practice_tasks_resolve_cheaply(self):
        rows = run_bench()
        self.assertEqual([r["status"] for r in rows], ["resolved"] * len(rows))
        for r in rows:
            self.assertLessEqual(r["calls"], 3, r)
            self.assertLess(r["in"] + r["out"], 6000, r)   # was ~17,000 before the token work


if __name__ == "__main__":
    unittest.main()


class ContextReuseTests(unittest.TestCase):
    """From a real Qwen run on the slugify task: the model re-read code it already had
    and opened the test file to learn the expected values."""

    def setUp(self):
        import os, shutil, tempfile
        from tests.helpers import ROOT
        d = tempfile.mkdtemp()
        self.repo = os.path.join(d, "repo")
        shutil.copytree(os.path.join(ROOT, "examples", "tasks", "slugify", "repo"), self.repo)
        with open(os.path.join(ROOT, "examples", "tasks", "slugify", "issue.md")) as f:
            self.issue = f.read()

    def overview(self, target):
        from harness import context as C
        ws = T.Workspace(self.repo)
        ws.target_test_command = target
        return ws, C.build_overview(ws, self.issue, run_baseline=False)

    def test_target_test_code_is_shown(self):
        _, ov = self.overview("python3 -m unittest tests.test_text.SlugifyTests")
        self.assertIn("## Target test code", ov)
        self.assertIn('"hello-world"', ov)
        _, ov2 = self.overview("python3 -m pytest tests/test_text.py::SlugifyTests::test_collapses_and_strips")
        self.assertIn("def test_collapses_and_strips", ov2)

    def test_reading_code_already_shown_is_not_resent(self):
        ws, ov = self.overview("python3 -m unittest tests.test_text.SlugifyTests")
        self.assertIn("def slugify", ov)
        out = T.read_file(ws, "textkit/text.py")
        self.assertIn("already in the overview", out)
        self.assertNotIn('out.append(ch if ch.isalnum() else "-")', out)   # shown lines omitted
        self.assertIn("def title_case", out)                                 # unseen lines still sent
        T.edit_file(ws, "textkit/text.py", '    return "".join(out)', '    return "".join(out).strip("-")')
        self.assertIn('out.append(ch if ch.isalnum() else "-")', T.read_file(ws, "textkit/text.py"))  # changed file: full read
