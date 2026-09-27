"""Regressions taken from a real run on github.com/nst-sdc/Open-Source-Tracker-NST (issue #76)
with a small local model. Each test recreates one thing that went wrong in that log."""
import json
import os
import tempfile
import unittest

from harness import tools as T
from harness import verify as V
from harness.agent import Agent
from harness.ui import UI
from tests.helpers import fake_config
from tests.test_agent import ScriptedLLM


def js_repo():
    root = os.path.join(tempfile.mkdtemp(prefix="harness_js_"), "repo")
    os.makedirs(os.path.join(root, "app", "api", "agent"))
    os.makedirs(os.path.join(root, "lib"))
    with open(os.path.join(root, "app", "api", "agent", "route.ts"), "w") as f:
        f.write("import { runAgent } from '@/lib/agent-loop';\nexport async function POST(req) {\n  return runAgent(req);\n}\n")
    with open(os.path.join(root, "lib", "llm-budget.ts"), "w") as f:
        f.write("export function reserveProviderCalls(n: number) {\n  return n <= 8;\n}\n")
    return root


class RealWorldRegressions(unittest.TestCase):
    def setUp(self):
        self.ws = T.Workspace(js_repo())

    def test_leading_slash_path_is_repo_relative(self):
        # log: read_file(path=/app/api/agent/route.ts) -> "outside the repository"
        out = T.read_file(self.ws, "/app/api/agent/route.ts")
        self.assertIn("runAgent", out)
        self.assertIn("route.ts", T.list_dir(self.ws, "/app/api/agent"))

    def test_real_escape_still_blocked(self):
        self.assertIn("outside the repository", T.execute(self.ws, "read_file", {"path": "../../../etc/passwd"}))

    def test_regex_looking_query_falls_back_to_regex(self):
        # log: search_code(query=route.*agent.*) -> "No matches" (searched as plain text)
        out = T.search_code(self.ws, "run.*Agent")
        self.assertIn("as a regex", out)
        self.assertIn("app/api/agent/route.ts", out)

    def test_case_insensitive_fallback(self):
        out = T.search_code(self.ws, "reserveprovidercalls")
        self.assertIn("ignoring case", out)
        self.assertIn("lib/llm-budget.ts", out)

    def test_multi_word_query_searches_each_word(self):
        # log: search_code(query=route agent) -> "No matches"
        out = T.search_code(self.ws, "POST runAgent handler")
        self.assertIn("separate words", out)
        self.assertIn("route.ts", out)

    def test_no_match_message_suggests_next_step(self):
        out = T.search_code(self.ws, "zzz_nothing_here")
        self.assertTrue(out.startswith("No matches"))
        self.assertIn("find_files", out)

    def test_dead_end_repeat_warned_on_second_try(self):
        # log: the same empty search was repeated many times without a warning
        cfg = fake_config(tempfile.mkdtemp())
        cfg.max_steps = 3
        llm = ScriptedLLM([("search_code", {"query": "zzz_nothing_here"})] * 3)
        Agent(cfg, self.ws.root, llm=llm, ui=UI(quiet=True), run_baseline=False).run("x")
        self.assertIn("exact call 2 times", json.dumps(llm.seen[-1]))

    def test_single_file_search_without_ripgrep(self):
        # log: on a Mac without ripgrep, search_code(path="lib/github.ts") returned "No matches"
        from unittest import mock
        with mock.patch("harness.tools.shutil.which", return_value=None):
            out = T.search_code(self.ws, "reserveProviderCalls", path="lib/llm-budget.ts")
            self.assertIn("reserveProviderCalls", out)
            self.assertFalse(out.startswith("No matches"))
            self.assertIn("lib/llm-budget.ts:1:", T.search_code(self.ws, "reserveProviderCalls", path="lib"))

    def test_common_argument_aliases_are_accepted(self):
        # log: gpt-oss-120b looped for 30 steps on read_file(line_start=..., line_end=...)
        out = T.execute(self.ws, "read_file", {"path": "lib/llm-budget.ts", "line_start": 1, "line_end": 2})
        self.assertIn("lines 1-2 of", out)
        out = T.execute(self.ws, "read_file", {"file_path": "lib/llm-budget.ts", "start": 2, "limit": 1})
        self.assertIn("lines 2-2 of", out)
        self.assertIn("reserveProviderCalls",
                      T.execute(self.ws, "search_code", {"pattern": "reserveProviderCalls", "dir": "lib"}))
        out = T.execute(self.ws, "edit_file", {"file": "lib/llm-budget.ts", "old_string": "n <= 8",
                                               "new_string": "n < 9"})
        self.assertTrue(out.startswith("Edited"), out)

    def test_unknown_argument_error_lists_the_real_names(self):
        out = T.execute(self.ws, "read_file", {"path": "lib/kv.ts", "colour": 1})
        self.assertIn("has no argument(s) 'colour'", out)
        self.assertIn("start_line, end_line", out)

    def test_keywords_from_lint_issue(self):
        from harness.context import extract_keywords
        kws = extract_keywords("Fix react-hooks errors and no-unescaped-entities; update eslint.config.mjs "
                               "and `npm run lint`.")
        for k in ("eslint.config.mjs", "react-hooks", "no-unescaped-entities"):
            self.assertIn(k, kws)

    def test_js_test_failures_are_parsed(self):
        out = ("exit code: 1\n FAIL  lib/llm-budget.test.ts > budget > caps daily calls\n"
               "   × caps daily calls 12ms\nFAIL src/app.test.js\n")
        ids = V.failing_tests(out)
        self.assertIn("lib/llm-budget.test.ts > budget > caps daily calls", ids)
        self.assertIn("caps daily calls", ids)
        self.assertIn("src/app.test.js", ids)
        self.assertFalse(any(i.startswith("ED") for i in V.failing_tests("FAILED tests/t.py::x")))


if __name__ == "__main__":
    unittest.main()
