import os
import unittest

from harness import tools as T
from tests.helpers import sample_copy


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.repo = sample_copy()
        self.ws = T.Workspace(self.repo)

    def test_list_and_find(self):
        self.assertIn("shop/", T.list_dir(self.ws))
        self.assertIn("shop/pricing.py", T.find_files(self.ws, "*.py"))

    def test_search(self):
        out = T.search_code(self.ws, "def apply_discount")
        self.assertIn("shop/pricing.py:1:", out)
        self.assertTrue(T.search_code(self.ws, "zzz_not_there").startswith("No matches"))

    def test_read_file_line_numbers(self):
        out = T.read_file(self.ws, "shop/pricing.py", 1, 2)
        self.assertIn("     1\tdef apply_discount", out)
        self.assertIn("lines 1-2 of 5", out)

    def test_read_missing_file_suggests(self):
        out = T.read_file(self.ws, "pricing.py")
        self.assertTrue(out.startswith("Error"))
        self.assertIn("shop/pricing.py", out)

    def test_path_escape_blocked(self):
        out = T.execute(self.ws, "read_file", {"path": "../../etc/passwd"})
        self.assertIn("outside the repository", out)

    def test_edit_exact_and_diff(self):
        out = T.edit_file(self.ws, "shop/pricing.py", "amount * percent", "amount * percent / 100")
        self.assertTrue(out.startswith("Edited"), out)
        self.assertEqual(self.ws.changed_files(), ["shop/pricing.py"])
        self.assertIn("+    return amount - amount * percent / 100", self.ws.diff())

    def test_edit_not_found_gives_hint(self):
        out = T.edit_file(self.ws, "shop/pricing.py", "return amount - amount *  percent", "x")
        self.assertTrue(out.startswith("Error"))
        self.assertIn("closest matching text", out)

    def test_edit_ambiguous(self):
        T.create_file(self.ws, "dup.py", "x = 1\nx = 1\n")
        out = T.edit_file(self.ws, "dup.py", "x = 1", "x = 2")
        self.assertIn("appears 2 times", out)

    def test_syntax_error_reported(self):
        out = T.edit_file(self.ws, "shop/pricing.py", "return amount - amount * percent", "return (amount")
        self.assertIn("syntax error", out)

    def test_run_tests_and_state(self):
        out = T.run_tests(self.ws, "python3 -m unittest discover -s tests -t .")
        self.assertTrue(out.startswith("TESTS PASSED"), out)
        self.assertTrue(self.ws.last_test_passed)

    def test_blocked_command(self):
        self.assertIn("blocked", T.run_command(self.ws, "git push origin main"))

    def test_api_key_not_exposed(self):
        os.environ["AI_API_KEY"] = "secret-123"
        out = T.run_command(self.ws, "echo key=$AI_API_KEY")
        self.assertNotIn("secret-123", out)

    def test_truncate(self):
        s = "a" * 20000
        t = T.truncate(s, 1000)
        self.assertLess(len(t), 1100)
        self.assertIn("omitted", t)

    def test_unknown_tool_and_bad_args(self):
        self.assertIn("unknown tool", T.execute(self.ws, "nope", {}))
        self.assertIn("has no argument(s) 'wrong'", T.execute(self.ws, "read_file", {"wrong": 1}))


if __name__ == "__main__":
    unittest.main()
