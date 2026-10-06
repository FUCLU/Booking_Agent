"""
tests/test_config.py - Kiểm thử toàn diện cho module config.py và config.yaml.

Bao gồm:
- POSITIVE TESTS: Kiểm tra đọc cấu hình thành công, kiểu dữ liệu và giá trị hợp lệ.
- NEGATIVE TESTS: Kiểm tra xử lý lỗi khi file thiếu, YAML sai cú pháp, định dạng ngày sai.
- REGRESSION & OVERRIDE TESTS: Kiểm tra biến môi trường ghi đè và tính nhất quán với YAML.
"""
from __future__ import annotations

import importlib
from datetime import date as Date
from pathlib import Path

import pytest
import yaml

import config


# ==============================================================================
# POSITIVE TESTS (1 - 10)
# ==============================================================================

def test_config_loads_successfully():
    """1. config.yaml tải thành công và trả về một dict cấu hình hợp lệ."""
    assert config.cfg is not None
    assert isinstance(config.cfg, dict)
    assert len(config.cfg) > 0

    # Hàm _load() đọc trực tiếp từ file cũng phải trả về dict không rỗng
    loaded = config._load()
    assert isinstance(loaded, dict)
    assert len(loaded) > 0


def test_today_is_date():
    """2. TODAY được parse đúng thành đối tượng datetime.date."""
    assert isinstance(config.TODAY, Date)
    assert isinstance(config.TODAY.year, int)
    assert isinstance(config.TODAY.month, int)
    assert isinstance(config.TODAY.day, int)
    assert config.TODAY.year >= 2026


def test_default_model_format():
    """3. DEFAULT_MODEL tuân theo định dạng 'provider:model'."""
    assert isinstance(config.DEFAULT_MODEL, str)
    assert ":" in config.DEFAULT_MODEL

    parts = config.DEFAULT_MODEL.split(":")
    assert len(parts) == 2, f"DEFAULT_MODEL phải có định dạng 'provider:model', nhận được: {config.DEFAULT_MODEL}"
    assert len(parts[0].strip()) > 0, "Provider không được để trống"
    assert len(parts[1].strip()) > 0, "Model name không được để trống"


def test_llm_temperature_is_number():
    """4. LLM_TEMPERATURE là số (int hoặc float) và nằm trong khoảng hợp lệ [0, 2]."""
    assert isinstance(config.LLM_TEMPERATURE, (int, float))
    assert 0 <= config.LLM_TEMPERATURE <= 2


def test_price_cap_positive():
    """5. PRICE_CAP là số dương lớn hơn 0 (ngân sách hợp lý)."""
    assert isinstance(config.PRICE_CAP, (int, float))
    assert config.PRICE_CAP > 0


def test_max_tool_calls_positive():
    """6. MAX_TOOL_CALLS là số nguyên dương lớn hơn 0."""
    assert isinstance(config.MAX_TOOL_CALLS, int)
    assert config.MAX_TOOL_CALLS > 0


def test_react_max_not_done_positive():
    """7. REACT_MAX_NOT_DONE là số nguyên dương lớn hơn 0."""
    assert isinstance(config.REACT_MAX_NOT_DONE, int)
    assert config.REACT_MAX_NOT_DONE > 0


def test_hybrid_max_replans_positive():
    """8. HYBRID_MAX_REPLANS là số nguyên dương lớn hơn 0."""
    assert isinstance(config.HYBRID_MAX_REPLANS, int)
    assert config.HYBRID_MAX_REPLANS > 0


def test_hybrid_retryable_is_set():
    """9. HYBRID_RETRYABLE là một set chứa các chuỗi mã lỗi hợp lệ."""
    assert isinstance(config.HYBRID_RETRYABLE, set)
    assert len(config.HYBRID_RETRYABLE) > 0
    assert all(isinstance(code, str) for code in config.HYBRID_RETRYABLE)

    # Đảm bảo các mã lỗi cơ bản của hệ thống có mặt
    expected_codes = {"BAD_ARGS", "UNKNOWN_FLIGHT", "TIMEOUT"}
    assert expected_codes.issubset(config.HYBRID_RETRYABLE)


def test_config_yaml_all_keys_present():
    """10. Tất cả các khóa bắt buộc đều hiện diện đầy đủ trong dict cấu hình cfg."""
    cfg = config.cfg

    # Kiểm tra các khóa cấp cao nhất (top-level keys)
    required_top_keys = {"llm", "system", "harness", "agent"}
    assert required_top_keys.issubset(cfg.keys()), f"Thiếu khóa cấp cao nhất trong config.yaml: {required_top_keys - cfg.keys()}"

    # Kiểm tra cấu hình LLM
    llm_keys = {"provider", "model", "temperature"}
    assert llm_keys.issubset(cfg["llm"].keys()), f"Thiếu khóa trong mục llm: {llm_keys - cfg['llm'].keys()}"

    # Kiểm tra cấu hình System
    assert "today" in cfg["system"], "Thiếu khóa 'today' trong mục system"

    # Kiểm tra cấu hình Harness
    harness_keys = {"price_cap", "max_tool_calls"}
    assert harness_keys.issubset(cfg["harness"].keys()), f"Thiếu khóa trong mục harness: {harness_keys - cfg['harness'].keys()}"

    # Kiểm tra cấu hình Agent
    agent_keys = {"react", "hybrid"}
    assert agent_keys.issubset(cfg["agent"].keys()), f"Thiếu khóa trong mục agent: {agent_keys - cfg['agent'].keys()}"

    # Kiểm tra cấu hình Agent React & Hybrid
    assert "max_not_done" in cfg["agent"]["react"]
    hybrid_keys = {"max_replans", "max_retries", "retryable_codes"}
    assert hybrid_keys.issubset(cfg["agent"]["hybrid"].keys()), f"Thiếu khóa trong mục agent.hybrid: {hybrid_keys - cfg['agent']['hybrid'].keys()}"


# ==============================================================================
# NEGATIVE TESTS (11 - 13)
# ==============================================================================

def test_missing_config_yaml(monkeypatch, tmp_path):
    """11. Monkeypatch _CFG_PATH sang file không tồn tại, xác minh ném FileNotFoundError."""
    nonexistent_path = tmp_path / "nonexistent_config.yaml"
    monkeypatch.setattr(config, "_CFG_PATH", nonexistent_path)

    with pytest.raises(FileNotFoundError):
        config._load()


def test_invalid_yaml_content(monkeypatch, tmp_path):
    """12. Đọc file YAML có nội dung không hợp lệ, xác minh ném lỗi yaml.YAMLError."""
    invalid_yaml_file = tmp_path / "broken_config.yaml"
    invalid_yaml_file.write_text("invalid_yaml: [unclosed_bracket", encoding="utf-8")

    monkeypatch.setattr(config, "_CFG_PATH", invalid_yaml_file)

    with pytest.raises(yaml.YAMLError):
        config._load()


def test_today_invalid_format():
    """13. Kiểm tra Date.fromisoformat ném ValueError khi chuỗi ngày không đúng chuẩn ISO."""
    with pytest.raises(ValueError):
        Date.fromisoformat("not-a-date")

    with pytest.raises(ValueError):
        Date.fromisoformat("2026/10/06")  # Dùng gạch chéo thay vì gạch nối ISO 8601

    with pytest.raises(ValueError):
        Date.fromisoformat("")


# ==============================================================================
# REGRESSION & OVERRIDE TESTS (14 - 16)
# ==============================================================================

def test_env_override_model(monkeypatch):
    """14. Đặt biến môi trường LLM_MODEL, reload config, xác minh biến môi trường được ưu tiên."""
    custom_model = "mock-provider:mock-custom-model"
    monkeypatch.setenv("LLM_MODEL", custom_model)

    try:
        importlib.reload(config)
        assert config.DEFAULT_MODEL == custom_model
    finally:
        # Khôi phục trạng thái ban đầu của module config
        monkeypatch.delenv("LLM_MODEL", raising=False)
        importlib.reload(config)


def test_default_model_matches_yaml(monkeypatch):
    """15. Xác minh DEFAULT_MODEL khớp với f'{provider}:{model}' từ config.yaml khi không bị env var ghi đè."""
    # Đảm bảo không có biến môi trường nào đang can thiệp
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL_NAME", raising=False)

    try:
        importlib.reload(config)
        expected_model = f"{config.cfg['llm']['provider']}:{config.cfg['llm']['model']}"
        assert config.DEFAULT_MODEL == expected_model
    finally:
        importlib.reload(config)


def test_today_matches_yaml():
    """16. Xác minh TODAY.isoformat() khớp chính xác với giá trị cấu hình trong config.yaml."""
    expected_today = config.cfg["system"]["today"]
    assert config.TODAY.isoformat() == expected_today


# ==============================================================================
# ADDITIONAL EDGE CASE TESTS
# ==============================================================================

def test_hybrid_max_retries_positive():
    """Kiểm tra HYBRID_MAX_RETRIES là số nguyên dương."""
    assert isinstance(config.HYBRID_MAX_RETRIES, int)
    assert config.HYBRID_MAX_RETRIES > 0


def test_env_override_provider_and_model_name_separately(monkeypatch):
    """Kiểm tra ghi đè từng phần qua LLM_PROVIDER và LLM_MODEL_NAME khi LLM_MODEL không được đặt."""
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL_NAME", "claude-3-5-sonnet")

    try:
        importlib.reload(config)
        assert config._provider == "anthropic"
        assert config._model_name == "claude-3-5-sonnet"
        assert config.DEFAULT_MODEL == "anthropic:claude-3-5-sonnet"
    finally:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("LLM_MODEL_NAME", raising=False)
        importlib.reload(config)
