"""Configuration loading.

Order of precedence (highest first):
  1. Environment variables (AI_PROVIDER, AI_MODEL, AI_BASE_URL, HARNESS_MAX_STEPS, ...)
  2. config/harness.json
  3. Built-in defaults below

The API key is ONLY ever read from the AI_API_KEY environment variable.

Evaluation uses DeepSeek and Qwen models. Both expose OpenAI-compatible APIs
and both hand out keys that start with "sk-", so when the provider is "auto"
we probe each candidate's /models endpoint with the key and keep the first
one that accepts it.
"""
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Sent on every HTTP request. Some API gateways (Cloudflare in front of Groq, for
# example) reject Python's default "Python-urllib/x.y" agent with a 403 / error 1010.
USER_AGENT = "ai-coding-harness/0.2 (+https://github.com/joshitdutta886/ai-harness)"
CONFIG_PATH = os.path.join(ROOT, "config", "harness.json")

# provider -> (base url, default model, preferred models in order)
# Model names change often, so the default is only a fallback: when the key is
# checked we read the provider's /models list and pick the best match there
# (see pick_model). The client also switches model automatically if the API
# says the configured one does not exist.
PROVIDERS = {
    "deepseek": ("https://api.deepseek.com/v1", "deepseek-chat",
                 ["deepseek-v4-pro", "deepseek-pro", "deepseek-flash", "deepseek-v4.1-flash",
                  "deepseek-chat", "deepseek-v3"]),
    "qwen": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "qwen-plus",
             ["qwen3-coder-plus", "qwen-plus", "qwen-max", "qwen-flash", "qwen-turbo"]),
    "qwen-cn": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus",
                ["qwen3-coder-plus", "qwen-plus", "qwen-max", "qwen-flash", "qwen-turbo"]),
    "openrouter": ("https://openrouter.ai/api/v1", "deepseek/deepseek-v4.1-flash",
                   ["deepseek/deepseek-pro-latest", "deepseek/deepseek-v4.1-flash",
                    "deepseek/deepseek-flash-latest", "qwen/qwen3.8-flash", "qwen/qwen3-coder"]),
    "siliconflow": ("https://api.siliconflow.cn/v1", "deepseek-ai/DeepSeek-V3",
                    ["deepseek-ai/DeepSeek-V3", "Qwen/Qwen3-Coder-480B-A35B-Instruct"]),
    "together": ("https://api.together.xyz/v1", "deepseek-ai/DeepSeek-V3",
                 ["deepseek-ai/DeepSeek-V3", "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"]),
    "groq": ("https://api.groq.com/openai/v1", "qwen/qwen3-32b", ["qwen/qwen3-32b"]),
    # Free local models via Ollama (https://ollama.com): AI_API_KEY=ollama
    "ollama": ("http://localhost:11434/v1", "qwen3:8b",
               ["qwen3-coder", "qwen2.5-coder", "qwen3", "deepseek"]),
    "openai": ("https://api.openai.com/v1", "gpt-4.1", ["gpt-4.1"]),
    "anthropic": ("https://api.anthropic.com/v1", "claude-sonnet-4-5", ["claude-sonnet-4-5"]),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash",
               ["gemini-2.5-flash"]),
    # Any other OpenAI-compatible server (vLLM, LM Studio, a hackathon gateway):
    # set AI_PROVIDER=custom and AI_BASE_URL=... (AI_MODEL optional: picked from /models)
    "custom": ("", "", []),
}

# Order in which "auto" probes a plain "sk-..." key.
PROBE_ORDER = ["deepseek", "qwen", "qwen-cn", "siliconflow", "together", "openai"]

_NOT_CHAT = ("embed", "rerank", "tts", "whisper", "audio", "image", "vision", "-vl", "ocr",
             "omni", "moderation", "guard", "realtime", "search", "batch", "asr", "wanx")


def pick_model(available: List[str], preferred: List[str], fallback: str) -> str:
    """Choose a chat model from what the API actually offers."""
    chat = [m for m in available if m and not any(b in m.lower() for b in _NOT_CHAT)]
    if not chat:
        return fallback
    avail = set(chat)
    for p in preferred:                      # exact preferred id
        if p in avail:
            return p
    def size_b(m: str) -> float:             # "qwen3:8b" -> 8.0, "qwen3.5:0.8b" -> 0.8
        mm = re.search(r"(\d+(?:\.\d+)?)b\b", m.lower().split(":")[-1])
        return float(mm.group(1)) if mm else 0.0
    for p in preferred:                      # preferred as a prefix (e.g. ollama "qwen3:8b")
        hits = [m for m in chat if m.startswith(p)]
        if hits:
            return sorted(hits, key=size_b, reverse=True)[0]
    def score(m: str) -> int:
        l = m.lower()
        s = 0
        s += 6 if "coder" in l else 0
        s += 4 if ("deepseek" in l or "qwen" in l) else 0
        s += 2 if any(k in l for k in ("pro", "max", "plus", "chat")) else 0
        s -= 3 if any(k in l for k in ("mini", "tiny", "nano", "0.5b", "1.5b", "3b")) else 0
        return s
    return sorted(chat, key=score, reverse=True)[0]


def provider_from_key_shape(api_key: str) -> Optional[str]:
    k = (api_key or "").strip()
    if k.startswith("sk-ant-"):
        return "anthropic"
    if k.startswith("sk-or-"):
        return "openrouter"
    if k.startswith("gsk_"):
        return "groq"
    if k.startswith("AIza"):
        return "gemini"
    if k.lower() == "ollama":
        return "ollama"
    return None


def list_models(base_url: str, api_key: str, timeout: int = 12) -> Optional[List[str]]:
    """Return model ids from GET {base}/models, or None if the key is rejected."""
    req = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"authorization": "Bearer " + api_key, "user-agent": USER_AGENT,
                 "accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)]
    except Exception:
        return None


@dataclass
class Config:
    api_key: str = ""
    provider: str = "auto"
    model: str = ""
    base_url: str = ""
    tool_mode: str = "auto"              # auto | native | text
    thinking: str = "auto"               # auto (adaptive: off until the model struggles) | off | on
    temperature: float = 0.0
    max_output_tokens: int = 4096
    max_steps: int = 30
    max_total_tokens: int = 400_000      # hard budget for a single task
    context_char_budget: int = 60_000    # compaction kicks in above this (~15k tokens)
    keep_recent_tool_results: int = 3    # tool outputs kept in full
    command_timeout: int = 180
    max_minutes: float = 30.0            # wall-clock limit for one task
    request_timeout: int = 240
    runs_dir: str = field(default_factory=lambda: os.path.join(ROOT, "runs"))
    notes: List[str] = field(default_factory=list)

    def resolve(self, probe: bool = True) -> "Config":
        if self.provider in ("", "auto"):
            self.provider = provider_from_key_shape(self.api_key) or ""
            if not self.provider and self.base_url:
                self.provider = "custom"
            if not self.provider and probe:
                for cand in PROBE_ORDER:
                    models = list_models(PROVIDERS[cand][0], self.api_key)
                    if models is not None:
                        self.provider = cand
                        self.notes.append("Detected provider '%s' from API key." % cand)
                        if not self.model:
                            self.model = pick_model(models, PROVIDERS[cand][2], PROVIDERS[cand][1])
                        break
            if not self.provider:
                self.provider = "deepseek"
                self.notes.append("Could not verify key with any provider; defaulting to deepseek.")
        if self.provider not in PROVIDERS:
            raise ValueError("Unknown provider '%s'. Use one of: %s" % (self.provider, ", ".join(PROVIDERS)))
        base, default_model, prefs = PROVIDERS[self.provider]
        self.base_url = (self.base_url or base).rstrip("/")
        if not self.model and probe:
            models = list_models(self.base_url, self.api_key)
            if models:
                self.model = pick_model(models, prefs, default_model)
                self.notes.append("Picked model '%s' from the %d models this key can use." % (self.model, len(models)))
        self.model = self.model or default_model
        if not self.base_url or not self.model:
            raise ValueError("Provider 'custom' needs AI_BASE_URL and AI_MODEL to be set.")
        return self



_ENV_MAP = {
    "AI_PROVIDER": ("provider", str),
    "AI_MODEL": ("model", str),
    "AI_BASE_URL": ("base_url", str),
    "AI_TOOL_MODE": ("tool_mode", str),
    "AI_THINKING": ("thinking", str),
    "AI_MAX_OUTPUT_TOKENS": ("max_output_tokens", int),
    "AI_TEMPERATURE": ("temperature", float),
    "HARNESS_MAX_STEPS": ("max_steps", int),
    "HARNESS_MAX_TOKENS": ("max_total_tokens", int),
    "HARNESS_CMD_TIMEOUT": ("command_timeout", int),
    "HARNESS_MAX_MINUTES": ("max_minutes", float),
}


def load_config(path: Optional[str] = None, require_key: bool = True, probe: bool = True) -> Config:
    cfg = Config()
    path = path or CONFIG_PATH
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if k.startswith("_") or k == "api_key":
                continue  # a key in the config file is never used
            if hasattr(cfg, k) and v not in (None, ""):
                setattr(cfg, k, v)
    for env, (attr, cast) in _ENV_MAP.items():
        val = os.environ.get(env)
        if val:
            setattr(cfg, attr, cast(val))
    cfg.api_key = os.environ.get("AI_API_KEY", "").strip()
    if require_key and not cfg.api_key:
        raise SystemExit('AI_API_KEY is not set.\n  export AI_API_KEY="<your key>"  and run again.')
    return cfg.resolve(probe=probe)
