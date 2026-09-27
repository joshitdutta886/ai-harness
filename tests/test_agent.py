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

    def enable_thinking(self):
        self.thinking = True
        return True

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

    def test_tests_run_automatically_after_edit(self):
        agent, llm = self.make([
            ("search_code", {"query": "apply_discount"}),
            FIX,
            ("finish", {"summary": "percent was not divided by 100; tests pass"}),
        ])
        r = agent.run("Cart.total gives -1800 with 10% discount")
        self.assertEqual(r.status, "resolved", r)
        self.assertEqual(r.changed_files, ["shop/pricing.py"])
        self.assertIn("/ 100", r.diff)
        flat = json.dumps(llm.seen[-1])
        self.assertIn("Tests after your edit: PASSED", flat)
        self.assertNotIn("tests: not run", flat.split("Tests after your edit")[1])
        tel = r.telemetry
        self.assertEqual(tel["finish_rejections"], [])
        self.assertEqual(tel["auto_checks"], 1)
        self.assertEqual(tel["edits"], 1)
        self.assertEqual(len(llm.seen), 3)          # search, edit, finish: no separate run_tests call
        import os
        self.assertTrue(os.path.exists(os.path.join(r.run_dir, "telemetry.json")))
        with open(os.path.join(r.run_dir, "report.md")) as fh:
            self.assertIn("## Telemetry", fh.read())

    def test_edit_and_finish_in_one_turn(self):
        agent, _ = self.make([])
        class OneShot:
            def __init__(self): self.calls = 0
            def chat(self, system, messages, tools):
                self.calls += 1
                return LLMResponse(text="Fix percent.", tool_calls=[
                    ToolCall("a", FIX[0], FIX[1]), ToolCall("b", "finish", {"summary": "divide by 100"})])
        agent.llm = OneShot()
        r = agent.run("discount bug")
        self.assertEqual(r.status, "resolved")
        self.assertEqual(agent.llm.calls, 1)          # edit + finish together: one model call

    def test_edit_and_finish_in_one_turn_rejected_if_tests_fail(self):
        agent, _ = self.make([])
        breaks = ("edit_file", {"path": "shop/cart.py", "old_str": "return sum(price * qty",
                                "new_str": "return 1 + sum(price * qty"})
        class OneShot:
            def __init__(self): self.calls = 0
            def chat(self, system, messages, tools):
                self.calls += 1
                if self.calls == 1:
                    return LLMResponse(tool_calls=[ToolCall("a", breaks[0], breaks[1]),
                                                   ToolCall("b", "finish", {"summary": "done"})])
                return LLMResponse(tool_calls=[ToolCall("c%d" % self.calls, "revert_file", {"path": "shop/cart.py"})])
        agent.llm = OneShot()
        agent.cfg.max_steps = 2
        r = agent.run("discount bug")
        self.assertNotEqual(r.status, "resolved")
        self.assertIn("tests_not_run", r.telemetry["finish_rejections"])

    def test_finish_rejected_when_tests_fail_after_edit(self):
        breaks = ("edit_file", {"path": "shop/cart.py", "old_str": "return sum(price * qty",
                                "new_str": "return 1 + sum(price * qty"})
        agent, llm = self.make([breaks, ("finish", {"summary": "done"}), FIX, ("finish", {"summary": "done"})])
        r = agent.run("discount bug")
        self.assertIn("Tests after your edit: FAILED", json.dumps(llm.seen[1]))
        self.assertIn("tests_not_run", r.telemetry["finish_rejections"])
        self.assertIn("thinking_enabled_at_step", r.telemetry)

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

    def test_retries_when_thinking_uses_all_tokens(self):
        sent = []

        def fake(url, headers, body, timeout):
            sent.append(body["max_tokens"])
            if len(sent) == 1:
                return {"choices": [{"message": {"content": "", "reasoning": "hmm..."},
                                     "finish_reason": "length"}], "usage": {}}
            return {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}], "usage": {}}
        L._post = fake
        cfg = fake_config(tempfile.mkdtemp())
        r = L.LLMClient(cfg).chat("sys", [{"role": "user", "content": "x"}], [])
        self.assertEqual(sent, [4096, 8192])
        self.assertEqual(r.text, "OK")

    def test_no_think_switch(self):
        sent = []

        def fake(url, headers, body, timeout):
            sent.append(body["messages"][0]["content"])
            return {"choices": [{"message": {"content": "OK"}}], "usage": {}}
        L._post = fake
        cfg = fake_config(tempfile.mkdtemp())
        cfg.thinking = "off"
        L.LLMClient(cfg).chat("sys", [{"role": "user", "content": "x"}], [])
        self.assertTrue(sent[0].endswith("/no_think"))

    def test_adaptive_thinking(self):
        sent = []

        def fake(url, headers, body, timeout):
            sent.append((body["messages"][0]["content"].endswith("/no_think"), body.get("enable_thinking")))
            return {"choices": [{"message": {"content": "OK"}}], "usage": {}}
        L._post = fake
        cfg = fake_config(tempfile.mkdtemp())
        cfg.provider = "qwen"
        c = L.LLMClient(cfg)
        c.chat("sys", [{"role": "user", "content": "x"}], [])
        self.assertTrue(c.enable_thinking())
        c.chat("sys", [{"role": "user", "content": "x"}], [])
        self.assertEqual(sent, [(True, False), (False, None)])
        cfg2 = fake_config(tempfile.mkdtemp()); cfg2.thinking = "on"
        self.assertFalse(L.LLMClient(cfg2).no_think)

    def test_adapts_to_provider_output_limit(self):
        import io
        import urllib.error
        sent = []

        def fake(url, headers, body, timeout):
            sent.append(body["max_tokens"])
            if body["max_tokens"] > 1000:
                raise urllib.error.HTTPError(url, 429, "limit", {}, io.BytesIO(
                    b'{"error":{"message":"Request too large for model on output tokens per minute (OTPM): '
                    b'Limit 1000, Requested 1028. reduce max_tokens","code":"rate_limit_exceeded"}}'))
            return {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}], "usage": {}}
        L._post = fake
        cfg = fake_config(tempfile.mkdtemp())
        r = L.LLMClient(cfg).chat("sys", [{"role": "user", "content": "x"}], [])
        self.assertEqual(r.text, "OK")
        self.assertEqual(sent[0], 4096)
        self.assertEqual(len(sent), 2)
        self.assertLessEqual(sent[1], 1000)
        c = L.LLMClient(cfg)                      # the "ran out of room" retry never exceeds the cap
        self.assertLessEqual(c.max_tokens_cap, 16384)

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
