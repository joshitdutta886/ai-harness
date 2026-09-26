"""End-to-end tests of the agent loop with a scripted (fake) model - no network."""
import json
import tempfile
import unittest

from harness import llm as L
from harness.agent import Agent
from harness.llm import LLMResponse, ToolCall
from harness.ui import UI
from tests.helpers import fake_config, sample_copy


class ScriptedLLM:
    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def chat(self, system, messages, tools):
        self.seen.append([dict(m) for m in messages])
        if not self.script:
            return LLMResponse(text="nothing left", input_tokens=10, output_tokens=1)
        name, args = self.script.pop(0)
        if name is None:
            return LLMResponse(text=args, input_tokens=100, output_tokens=10)
        return LLMResponse(text="", tool_calls=[ToolCall("c%d" % len(self.seen), name, args)],
                           input_tokens=100, output_tokens=10)


FIX = ("edit_file", {"path": "shop/pricing.py", "old_str": "amount - amount * percent",
                     "new_str": "amount - amount * percent / 100"})


class AgentLoopTests(unittest.TestCase):
    def make(self, script, **kw):
        repo = sample_copy()
        cfg = fake_config(tempfile.mkdtemp())
        llm = ScriptedLLM(script)
        return Agent(cfg, repo, llm=llm, ui=UI(quiet=True), **kw), llm

    def test_resolves_with_verification_gate(self):
        agent, llm = self.make([
            ("search_code", {"query": "apply_discount"}),
            FIX,
            ("finish", {"summary": "fixed"}),       # rejected: tests not run after edit
            ("run_tests", {}),
            ("finish", {"summary": "percent was not divided by 100; tests pass"}),
        ])
        r = agent.run("Cart.total gives -1800 with 10% discount")
        self.assertEqual(r.status, "resolved", r)
        self.assertEqual(r.changed_files, ["shop/pricing.py"])
        self.assertIn("/ 100", r.diff)
        # the rejection message reached the model
        flat = json.dumps(llm.seen[-1])
        self.assertIn("finish rejected", flat)
        tel = r.telemetry
        self.assertEqual(tel["finish_rejections"], ["tests_not_run"])
        self.assertEqual(tel["edits"], 1)
        self.assertEqual(tel["test_passes"], 1)
        self.assertEqual(tel["tools"]["search_code"]["calls"], 1)
        import os
        self.assertTrue(os.path.exists(os.path.join(r.run_dir, "telemetry.json")))
        with open(os.path.join(r.run_dir, "report.md")) as fh:
            self.assertIn("## Telemetry", fh.read())

    def test_overview_contains_issue_identifiers(self):
        agent, llm = self.make([FIX, ("run_tests", {}), ("finish", {"summary": "ok"})])
        agent.run("`apply_discount` is wrong")
        first = llm.seen[0][0]["content"]
        self.assertIn("shop/pricing.py:1:def apply_discount", first)
        self.assertIn("Baseline test run", first)

    def test_nudges_when_no_tool_called(self):
        agent, llm = self.make([(None, "I think the bug is in pricing."), FIX, ("run_tests", {}),
                                ("finish", {"summary": "ok"})])
        r = agent.run("discount bug")
        self.assertEqual(r.status, "resolved")
        self.assertIn("You did not call a tool", json.dumps(llm.seen[1]))

    def test_step_budget(self):
        agent, _ = self.make([("list_dir", {})] * 50)
        agent.cfg.max_steps = 5
        r = agent.run("anything")
        self.assertEqual(r.status, "budget_exhausted")
        self.assertEqual(r.steps, 5)

    def test_repeat_warning(self):
        agent, llm = self.make([("read_file", {"path": "shop/cart.py"})] * 4)
        agent.cfg.max_steps = 4
        agent.run("x")
        self.assertIn("exact call 3 times", json.dumps(llm.seen[-1]))


class WireFormatTests(unittest.TestCase):
    """Check the request bodies we send to each API style."""

    def setUp(self):
        self.captured = []
        self._orig = L._post

    def tearDown(self):
        L._post = self._orig

    def convo(self):
        return [
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": "looking", "tool_calls": [
                {"id": "a", "name": "read_file", "args": {"path": "x.py"}},
                {"id": "b", "name": "list_dir", "args": {}}]},
            {"role": "tool", "tool_call_id": "a", "name": "read_file", "content": "code"},
            {"role": "tool", "tool_call_id": "b", "name": "list_dir", "content": "files"},
            {"role": "user", "content": "[harness] hurry"},
        ]

    def test_openai_native(self):
        def fake(url, headers, body, timeout):
            self.captured.append(body)
            return {"choices": [{"message": {"content": None, "tool_calls": [
                {"id": "z", "type": "function", "function": {"name": "finish", "arguments": "{\"summary\": \"s\"}"}}]},
                "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}
        L._post = fake
        cfg = fake_config(tempfile.mkdtemp())
        r = L.LLMClient(cfg).chat("sys", self.convo(), [{"name": "finish", "description": "d", "parameters": {"type": "object", "properties": {}}}])
        body = self.captured[0]
        roles = [m["role"] for m in body["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "tool", "user"])
        self.assertEqual(r.tool_calls[0].args, {"summary": "s"})
        self.assertEqual(r.input_tokens, 5)

    def test_text_mode_parses_qwen_output(self):
        def fake(url, headers, body, timeout):
            self.captured.append(body)
            return {"choices": [{"message": {"content": "<think>x</think>ok <tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"a\"}}</tool_call>"}}]}
        L._post = fake
        cfg = fake_config(tempfile.mkdtemp())
        cfg.tool_mode = "text"
        tools = [{"name": "read_file", "description": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}]
        r = L.LLMClient(cfg).chat("sys", self.convo(), tools)
        body = self.captured[0]
        self.assertNotIn("tools", body)
        self.assertIn("<tool_call>", body["messages"][0]["content"])
        self.assertTrue(all(m["role"] in ("system", "user", "assistant") for m in body["messages"]))
        self.assertEqual(r.tool_calls[0].name, "read_file")

    def test_switches_model_when_missing(self):
        import io
        import urllib.error
        from harness import config as CF
        calls = []

        def fake(url, headers, body, timeout):
            calls.append(body["model"])
            if body["model"] == "old-model":
                raise urllib.error.HTTPError(url, 400, "bad", {}, io.BytesIO(
                    b'{"error": {"message": "Model Not Exist"}}'))
            return {"choices": [{"message": {"content": "hi"}}], "usage": {}}
        L._post = fake
        orig = CF.list_models
        CF.list_models = lambda base, key, timeout=12: ["text-embedding", "deepseek-v4-pro"]
        try:
            cfg = fake_config(tempfile.mkdtemp())
            cfg.model = "old-model"
            r = L.LLMClient(cfg).chat("sys", [{"role": "user", "content": "x"}], [])
        finally:
            CF.list_models = orig
        self.assertEqual(calls, ["old-model", "deepseek-v4-pro"])
        self.assertEqual(r.text, "hi")

    def test_anthropic_merges_tool_results(self):
        def fake(url, headers, body, timeout):
            self.captured.append(body)
            return {"content": [{"type": "text", "text": "hi"}], "usage": {"input_tokens": 1, "output_tokens": 1}}
        L._post = fake
        cfg = fake_config(tempfile.mkdtemp())
        cfg.provider = "anthropic"
        L.LLMClient(cfg).chat("sys", self.convo(), [])
        msgs = self.captured[0]["messages"]
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "user"])
        types = [b["type"] for b in msgs[2]["content"]]
        self.assertEqual(types, ["tool_result", "tool_result", "text"])


if __name__ == "__main__":
    unittest.main()
