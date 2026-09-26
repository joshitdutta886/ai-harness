"""Provider-agnostic chat client with native tool calling.

Uses only the Python standard library (urllib), so `make setup` has nothing to
install. Two wire formats are supported:
  * Anthropic Messages API
  * OpenAI-compatible Chat Completions (OpenAI, Gemini, Groq, OpenRouter, ...)

Internal (canonical) message format used by the rest of the harness:
  {"role": "user", "content": str}
  {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "args"}]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}
"""
import json
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import Config


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""
    reasoning: str = ""


class LLMError(Exception):
    pass


RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def _post(url: str, headers: Dict[str, str], body: dict, timeout: int) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class LLMClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.is_anthropic = cfg.provider == "anthropic"
        # Some OpenAI models reject `max_tokens` / `temperature`; we learn and adapt.
        self._use_max_completion_tokens = False
        self._send_temperature = True
        self._send_reasoning = False
        mode = (cfg.tool_mode or "auto").lower()
        self.text_mode = mode == "text"
        self._allow_mode_switch = mode == "auto" and not self.is_anthropic

    # ------------------------------------------------------------------ public
    def chat(self, system: str, messages: List[dict], tools: List[dict]) -> LLMResponse:
        attempt = 0
        while True:
            attempt += 1
            try:
                if self.is_anthropic:
                    return self._anthropic(system, messages, tools)
                return self._openai(system, messages, tools)
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:800]
                except Exception:
                    pass
                if e.code == 400 and not self.is_anthropic and self._adapt(detail):
                    continue  # retry immediately with adjusted params
                if e.code in RETRY_STATUS and attempt < 6:
                    self._sleep(attempt, e)
                    continue
                raise LLMError("HTTP %s from %s: %s" % (e.code, self.cfg.provider, detail))
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                if attempt < 5:
                    self._sleep(attempt, e)
                    continue
                raise LLMError("Network error talking to %s: %s" % (self.cfg.provider, e))

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _sleep(attempt: int, err: Exception) -> None:
        wait = None
        headers = getattr(err, "headers", None)
        if headers is not None:
            ra = headers.get("retry-after")
            if ra:
                try:
                    wait = float(ra)
                except ValueError:
                    wait = None
        if wait is None:
            wait = min(60.0, (2 ** attempt) + random.random())
        time.sleep(min(wait, 90.0))

    def _adapt(self, detail: str) -> bool:
        d = detail.lower()
        changed = False
        if "max_tokens" in d and "max_completion_tokens" in d and not self._use_max_completion_tokens:
            self._use_max_completion_tokens = True
            changed = True
        if "temperature" in d and self._send_temperature:
            self._send_temperature = False
            changed = True
        if "reasoning_content" in d and not self._send_reasoning:
            self._send_reasoning = True
            changed = True
        if (not changed and self._allow_mode_switch and not self.text_mode
                and any(w in d for w in ("tool", "function"))):
            # endpoint does not support native tool calling -> use text protocol
            self.text_mode = True
            changed = True
        return changed

    # --------------------------------------------------------------- anthropic
    def _anthropic(self, system, messages, tools) -> LLMResponse:
        out: List[dict] = []
        for m in messages:
            if m["role"] == "user":
                blocks = [{"type": "text", "text": m["content"] or "(empty)"}]
                role = "user"
            elif m["role"] == "assistant":
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls") or []:
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["args"]})
                if not blocks:
                    blocks = [{"type": "text", "text": "(no content)"}]
                role = "assistant"
            else:  # tool
                blocks = [{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"] or "(empty)"}]
                role = "user"
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": blocks})
        # tool_result blocks must come before text in a user turn
        for m in out:
            if m["role"] == "user":
                m["content"].sort(key=lambda b: 0 if b["type"] == "tool_result" else 1)

        body = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_output_tokens,
            "system": system,
            "messages": out,
            "tools": [
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                for t in tools
            ],
        }
        if self._send_temperature:
            body["temperature"] = self.cfg.temperature
        headers = {
            "content-type": "application/json",
            "x-api-key": self.cfg.api_key,
            "anthropic-version": "2023-06-01",
        }
        data = _post(self.cfg.base_url + "/messages", headers, body, self.cfg.request_timeout)
        resp = LLMResponse(stop_reason=data.get("stop_reason", ""))
        texts = []
        for b in data.get("content", []):
            if b.get("type") == "text":
                texts.append(b.get("text", ""))
            elif b.get("type") == "tool_use":
                resp.tool_calls.append(ToolCall(b["id"], b["name"], b.get("input") or {}))
        resp.text = "\n".join(texts).strip()
        u = data.get("usage", {})
        resp.input_tokens = int(u.get("input_tokens", 0)) + int(u.get("cache_read_input_tokens", 0) or 0)
        resp.output_tokens = int(u.get("output_tokens", 0))
        return resp

    # ------------------------------------------------------------------ openai
    def _openai(self, system, messages, tools) -> LLMResponse:
        text_mode = self.text_mode
        sys_prompt = system + (text_tool_instructions(tools) if text_mode else "")
        out = [{"role": "system", "content": sys_prompt}]

        def push(role, content):
            if out[-1]["role"] == role and role == "user":
                out[-1]["content"] += "\n\n" + content
            else:
                out.append({"role": role, "content": content})

        for m in messages:
            if m["role"] == "user":
                push("user", m["content"])
            elif m["role"] == "assistant":
                if text_mode:
                    content = m.get("content") or ""
                    for tc in m.get("tool_calls") or []:
                        if "<tool_call>" not in content:
                            content += "\n<tool_call>%s</tool_call>" % json.dumps(
                                {"name": tc["name"], "arguments": tc["args"]})
                    out.append({"role": "assistant", "content": content or "(no content)"})
                    continue
                msg = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {"id": tc["id"], "type": "function",
                         "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
                        for tc in m["tool_calls"]
                    ]
                if self._send_reasoning and m.get("reasoning"):
                    msg["reasoning_content"] = m["reasoning"]
                if msg["content"] is None and "tool_calls" not in msg:
                    msg["content"] = "(no content)"
                out.append(msg)
            else:
                if text_mode:
                    push("user", "<tool_result name=\"%s\">\n%s\n</tool_result>" % (m.get("name", ""), m["content"]))
                else:
                    out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"] or "(empty)"})

        body = {"model": self.cfg.model, "messages": out}
        if not text_mode:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
            body["tool_choice"] = "auto"
        if self._use_max_completion_tokens:
            body["max_completion_tokens"] = self.cfg.max_output_tokens
        else:
            body["max_tokens"] = self.cfg.max_output_tokens
        if self._send_temperature:
            body["temperature"] = self.cfg.temperature
        headers = {"content-type": "application/json", "authorization": "Bearer " + self.cfg.api_key}
        data = _post(self.cfg.base_url + "/chat/completions", headers, body, self.cfg.request_timeout)
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("Empty response from model: %s" % json.dumps(data)[:500])
        choice = choices[0]
        msg = choice.get("message") or {}
        raw_text = msg.get("content") or ""
        resp = LLMResponse(stop_reason=choice.get("finish_reason") or "")
        resp.reasoning = msg.get("reasoning_content") or ""
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError:
                args = {"__invalid_json__": raw}
            resp.tool_calls.append(ToolCall(tc.get("id") or "call_%d" % i, fn.get("name", ""), args or {}))
        # Open models sometimes write tool calls as text even in native mode.
        if not resp.tool_calls:
            parsed, cleaned = parse_text_tool_calls(raw_text, {t["name"] for t in tools})
            if parsed:
                resp.tool_calls = parsed
                raw_text = cleaned if not text_mode else raw_text
        resp.text = strip_think(raw_text).strip()
        u = data.get("usage") or {}
        resp.input_tokens = int(u.get("prompt_tokens", 0) or 0)
        resp.output_tokens = int(u.get("completion_tokens", 0) or 0)
        return resp


# ---------------------------------------------------------------------------
# Text-mode tool calling (fallback for endpoints without native tool support)
# ---------------------------------------------------------------------------
_TOOL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_FENCE = re.compile(r"```(?:json|tool_call)?\s*(\{.*?\})\s*```", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)


def strip_think(text: str) -> str:
    return _THINK.sub("", text or "")


def text_tool_instructions(tools: List[dict]) -> str:
    lines = [
        "",
        "",
        "# How to call tools",
        "You call a tool by writing exactly one block like this (JSON inside the tags):",
        '<tool_call>{"name": "read_file", "arguments": {"path": "src/app.py"}}</tool_call>',
        "Write at most 3 tool calls per reply. After you write them, STOP and wait: the results",
        "come back in <tool_result> blocks. Never invent tool results.",
        "",
        "Available tools:",
    ]
    for t in tools:
        props = t["parameters"].get("properties", {})
        req = set(t["parameters"].get("required", []))
        args = ", ".join("%s%s: %s" % (k, "" if k in req else "?", v.get("type", "any")) for k, v in props.items())
        lines.append("- %s(%s): %s" % (t["name"], args, t["description"].split("\n")[0]))
    return "\n".join(lines)


def _loads_loose(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # tolerate trailing commas / single trailing garbage
        s2 = re.sub(r",\s*([}\]])", r"\1", s)
        return json.loads(s2)


def parse_text_tool_calls(text: str, known: set):
    """Extract tool calls written as text. Returns (calls, text_without_calls)."""
    if not text:
        return [], text
    calls = []
    blobs = _TOOL_TAG.findall(text)
    cleaned = _TOOL_TAG.sub("", text)
    if not blobs:
        blobs = _FENCE.findall(text)
        if blobs:
            cleaned = _FENCE.sub("", text)
    for i, blob in enumerate(blobs):
        try:
            obj = _loads_loose(blob)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool") or obj.get("function")
        args = obj.get("arguments", obj.get("args", obj.get("parameters", {})))
        if isinstance(args, str):
            try:
                args = _loads_loose(args)
            except Exception:
                args = {}
        if name in known and isinstance(args, dict):
            calls.append(ToolCall("txt_%d_%d" % (int(time.time() * 1000) % 10**8, i), name, args))
    return calls, cleaned
