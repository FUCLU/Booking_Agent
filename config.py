"""
config.py - Đọc cấu hình từ config.yaml và biến môi trường

Xuất các hằng số: DEFAULT_MODEL, LLM_KWARGS, TODAY, PRICE_CAP, MAX_TOOL_CALLS,
REACT_MAX_NOT_DONE, HYBRID_MAX_REPLANS, HYBRID_MAX_RETRIES, HYBRID_RETRYABLE,
ALLOWED_TOOLS (dict tên agent → danh sách tool được phép).
"""
from __future__ import annotations

import os
from datetime import date as Date
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import yaml

_CFG_PATH = Path(__file__).with_name("config.yaml")


def _load() -> dict:
    with open(_CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = _load()

# LLM
_llm = cfg["llm"]
_provider = os.getenv("LLM_PROVIDER", _llm["provider"])
_model_name = os.getenv("LLM_MODEL_NAME", _llm["model"])
LLM_TEMPERATURE = _llm.get("temperature", 0)
LLM_MAX_TOKENS = _llm.get("max_tokens", 4096)
LLM_TOP_P = _llm.get("top_p", 1.0)
LLM_TIMEOUT = _llm.get("timeout", 60)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", _llm.get("base_url", "")) or None  # None = default
LLM_API_VERSION = os.getenv("LLM_API_VERSION", _llm.get("api_version", "")) or None
LLM_API_KEY_ENV = _llm.get("api_key_env", "OPENAI_API_KEY")
LLM_EXTRA_HEADERS = _llm.get("extra_headers", {}) or {}

# Dạng "provider:model" cho init_chat_model()
DEFAULT_MODEL = os.getenv("LLM_MODEL", f"{_provider}:{_model_name}")

# Keyword arguments truyền cho init_chat_model() (ngoài model name)
LLM_KWARGS: dict = {
    "temperature": LLM_TEMPERATURE,
    "max_tokens": LLM_MAX_TOKENS,
}
if LLM_BASE_URL:
    LLM_KWARGS["base_url"] = LLM_BASE_URL
if LLM_API_VERSION:
    LLM_KWARGS["api_version"] = LLM_API_VERSION
if LLM_EXTRA_HEADERS:
    LLM_KWARGS["default_headers"] = LLM_EXTRA_HEADERS
if LLM_API_KEY_ENV and os.getenv(LLM_API_KEY_ENV):
    LLM_KWARGS["api_key"] = os.getenv(LLM_API_KEY_ENV)

# Ngày hệ thống
TODAY = Date.fromisoformat(cfg["system"]["today"])

# Harness
PRICE_CAP = cfg["harness"]["price_cap"]
MAX_TOOL_CALLS = cfg["harness"]["max_tool_calls"]

# Agent - React
REACT_MAX_NOT_DONE = cfg["agent"]["react_agent"]["max_not_done"]

# Agent - Hybrid
HYBRID_MAX_REPLANS = cfg["agent"]["hybrid_agent"]["max_replans"]
HYBRID_MAX_RETRIES = cfg["agent"]["hybrid_agent"]["max_retries"]
HYBRID_RETRYABLE = set(cfg["agent"]["hybrid_agent"]["retryable_codes"])

# Allowed tools per agent (tên agent khớp tên file)
ALLOWED_TOOLS: dict[str, list[str]] = {
    name: agent_cfg["allowed_tools"]
    for name, agent_cfg in cfg["agent"].items()
}
