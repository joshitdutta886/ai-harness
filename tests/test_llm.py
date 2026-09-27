import unittest

from harness.config import Config, provider_from_key_shape
from harness.context import compact, extract_keywords
from harness.llm import parse_text_tool_calls, strip_think


class ParsingTests(unittest.TestCase):
    known = {"read_file", "search_code"}

    def test_qwen_style_tool_call(self):
        text = 'Let me look.\n<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
        calls, cleaned = parse_text_tool_calls(text, self.known)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "read_file")
        self.assertEqual(calls[0].args, {"path": "a.py"})
        self.assertNotIn("tool_call", cleaned)

    def test_fenced_json_tool_call(self):
        text = '```json\n{"name": "search_code", "arguments": {"query": "foo",}}\n```'
        calls, _ = parse_text_tool_calls(text, self.known)
        self.assertEqual(calls[0].args["query"], "foo")

    def test_unknown_tool_ignored(self):
        calls, _ = parse_text_tool_calls('<tool_call>{"name": "rm", "arguments": {}}</tool_call>', self.known)
        self.assertEqual(calls, [])

    def test_strip_think(self):
        self.assertEqual(strip_think("<think>hmm</think>answer").strip(), "answer")


class ConfigTests(unittest.TestCase):
    def test_key_shapes(self):
        self.assertEqual(provider_from_key_shape("sk-or-abc"), "openrouter")
        self.assertEqual(provider_from_key_shape("sk-ant-abc"), "anthropic")
        self.assertIsNone(provider_from_key_shape("sk-abc"))

    def test_explicit_provider(self):
        c = Config(api_key="sk-x", provider="qwen").resolve(probe=False)
        self.assertIn("dashscope", c.base_url)
        self.assertEqual(c.model, "qwen-plus")

    def test_pick_model_prefers_coding_chat_models(self):
        from harness.config import pick_model
        avail = ["text-embedding-v3", "qwen-vl-max", "qwen3-coder-plus", "qwen-turbo"]
        self.assertEqual(pick_model(avail, ["qwen3-coder-plus"], "x"), "qwen3-coder-plus")
        self.assertEqual(pick_model(avail, ["nope"], "x"), "qwen3-coder-plus")
        self.assertEqual(pick_model(["qwen3:8b", "llama3:8b"], ["qwen3"], "x"), "qwen3:8b")
        self.assertEqual(pick_model(["qwen3.5:0.8b", "qwen3:8b"], ["qwen3"], "x"), "qwen3:8b")
        self.assertEqual(pick_model(["deepseek-flash", "deepseek-v4-pro"], ["deepseek-v4-pro"], "x"), "deepseek-v4-pro")

    def test_ollama_key(self):
        self.assertEqual(provider_from_key_shape("ollama"), "ollama")

    def test_custom_requires_url(self):
        with self.assertRaises(ValueError):
            Config(api_key="k", provider="custom").resolve(probe=False)


class UserAgentTests(unittest.TestCase):
    """Cloudflare-fronted APIs (e.g. Groq) block Python's default agent with 403 / error 1010."""

    def test_requests_send_harness_user_agent(self):
        import urllib.request
        from harness import config as CF
        from harness import llm as L
        seen = []

        class FakeResp:
            def __init__(self, body): self.body = body
            def read(self): return self.body
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake_urlopen(req, timeout=None):
            seen.append(req.get_header("User-agent"))
            return FakeResp(b'{"data": [{"id": "m"}], "choices": [{"message": {"content": "ok"}}]}')
        orig = urllib.request.urlopen
        urllib.request.urlopen = fake_urlopen
        try:
            L._post("https://example.invalid/v1/chat/completions", {"authorization": "Bearer x"}, {}, 5)
            CF.list_models("https://example.invalid/v1", "x")
        finally:
            urllib.request.urlopen = orig
        self.assertEqual(len(seen), 2)
        for ua in seen:
            self.assertTrue(ua and ua.startswith("ai-coding-harness/"), ua)


class ContextTests(unittest.TestCase):
    def test_keywords(self):
        kws = extract_keywords("`Cart.total()` fails in shop/cart.py when apply_discount runs; see CartTotals")
        for k in ("total", "shop/cart.py", "apply_discount", "CartTotals"):
            self.assertIn(k, kws)

    def test_compaction(self):
        msgs = [{"role": "user", "content": "issue"}]
        for i in range(10):
            msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": str(i), "name": "x", "args": {}}]})
            msgs.append({"role": "tool", "tool_call_id": str(i), "name": "x", "content": "line\n" + "y" * 5000})
        n = compact(msgs, keep_recent=3, char_budget=100000)
        self.assertEqual(n, 7)
        self.assertIn("compacted", msgs[2]["content"])
        self.assertNotIn("compacted", msgs[-1]["content"])


if __name__ == "__main__":
    unittest.main()
