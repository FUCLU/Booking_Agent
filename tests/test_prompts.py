"""
tests/test_prompts.py - Test suite cho prompts.py

Kiểm thử toàn diện các hàm tạo prompt và định dạng dữ liệu:
- system_prompt, parse_prompt, parse_user
- goal_prompt, plan_prompt, fill_args_prompt
- format_result, render_history

Bao gồm:
- Positive tests (kịch bản thông thường)
- Negative & Edge tests (cắt ngắn dữ liệu dài, handoff ưu tiên hơn error, v.v.)
- Regression tests (kiểm tra kiểu trả về, exclude_none của Intent JSON)
"""
from __future__ import annotations

import json
import pytest

from config import TODAY
from harness import Intent, Obs, ToolResult
from prompts import (
    fill_args_prompt,
    format_result,
    goal_prompt,
    parse_prompt,
    parse_user,
    plan_prompt,
    render_history,
    system_prompt,
)
from tools import TOOLS


# ==============================================================================
# POSITIVE TESTS
# ==============================================================================

def test_system_prompt_contains_today():
    """Kiểm tra system_prompt() chứa ngày hôm nay (YYYY-MM-DD)."""
    prompt = system_prompt()
    assert TODAY.isoformat() in prompt, f"Expected {TODAY.isoformat()} in system_prompt"


def test_system_prompt_contains_rules():
    """Kiểm tra system_prompt() chứa các quy tắc cốt lõi: flight_id, IATA, tool."""
    prompt = system_prompt()
    assert "flight_id" in prompt, "system_prompt phải chứa quy tắc về flight_id"
    assert "IATA" in prompt, "system_prompt phải chứa quy tắc về mã sân bay IATA"
    assert "tool" in prompt.lower(), "system_prompt phải đề cập đến việc gọi tool"
    # Kiểm tra thêm một số quy tắc quan trọng khác
    assert "handoff_to_human" in prompt
    assert "COMPLETION_CHECK_FAILED" in prompt


def test_parse_prompt_contains_today():
    """Kiểm tra parse_prompt() chứa ngày hôm nay để LLM trích xuất ngày tương đối."""
    prompt = parse_prompt()
    assert TODAY.isoformat() in prompt, f"Expected {TODAY.isoformat()} in parse_prompt"


def test_parse_prompt_contains_kinds():
    """Kiểm tra parse_prompt() đề cập đầy đủ 3 loại Intent: book_cheapest, cancel, rebook."""
    prompt = parse_prompt()
    assert "book_cheapest" in prompt
    assert "cancel" in prompt
    assert "rebook" in prompt


def test_parse_user_formats_correctly():
    """Kiểm tra parse_user(name, task) định dạng đúng câu nói của khách."""
    result = parse_user("Nguyen Van A", "Tôi muốn đặt vé từ SGN đi HAN")
    expected = "Khách tên 'Nguyen Van A' nói: Tôi muốn đặt vé từ SGN đi HAN"
    assert result == expected

    # Kiểm tra kịch bản tên hoặc yêu cầu ngắn gọn
    assert parse_user("A", "book") == "Khách tên 'A' nói: book"


def test_goal_prompt_contains_intent_json(intent_book, intent_cancel, intent_rebook):
    """Kiểm tra goal_prompt(intent) chứa chuỗi JSON đại diện cho intent."""
    # Test với intent_book
    goal_book = goal_prompt(intent_book)
    assert "Mục tiêu của khách (đã cấu trúc hoá):" in goal_book
    assert "book_cheapest" in goal_book
    assert "SGN" in goal_book
    assert "HAN" in goal_book
    assert "Nguyen Van A" in goal_book

    # Test với intent_cancel
    goal_cancel = goal_prompt(intent_cancel)
    assert "cancel" in goal_cancel
    assert "BK0001" in goal_cancel

    # Test với intent_rebook
    goal_rebook = goal_prompt(intent_rebook)
    assert "rebook" in goal_rebook
    assert "BK0001" in goal_rebook


def test_plan_prompt_with_empty_history():
    """Kiểm tra plan_prompt([]) khi chưa thực hiện bước nào trả về '(chưa làm gì)'."""
    prompt = plan_prompt([])
    assert "(chưa làm gì)" in prompt
    # Kiểm tra catalog các tool có mặt trong plan_prompt
    for tool_name in TOOLS:
        assert tool_name in prompt
    assert "handoff" in prompt


def test_plan_prompt_with_history():
    """Kiểm tra plan_prompt với danh sách Obs chứa thông tin các tool đã gọi."""
    obs_list = [
        Obs(
            tool="search_flights",
            args={"origin": "SGN", "destination": "HAN", "depart_date": "2026-10-20"},
            res=ToolResult(ok=True, data=[{"flight_id": "VN123", "price": 1500000}]),
        )
    ]
    prompt = plan_prompt(obs_list)
    assert "search_flights" in prompt
    assert "SGN" in prompt
    assert "VN123" in prompt
    assert "(chưa làm gì)" not in prompt


def test_fill_args_prompt_contains_tool_name():
    """Kiểm tra fill_args_prompt(tool) chứa tên tool cần điền tham số."""
    prompt = fill_args_prompt("search_flights")
    assert "search_flights" in prompt
    assert "Hãy gọi tool `search_flights`" in prompt

    prompt_book = fill_args_prompt("book_flight")
    assert "book_flight" in prompt_book


def test_format_result_success():
    """Kiểm tra format_result trả về chuỗi JSON khi kết quả thành công (ok=True)."""
    payload = {"flight_id": "VN123", "price": 1500000, "status": "confirmed"}
    res = ToolResult(ok=True, data=payload)
    formatted = format_result(res)

    # Đảm bảo kết quả là JSON parse được và khớp nội dung
    parsed = json.loads(formatted)
    assert parsed == payload


def test_format_result_error():
    """Kiểm tra format_result trả về format 'LỖI <code>: <message>' khi thất bại."""
    res = ToolResult(ok=False, code="BAD_ARGS", error="flight_id không hợp lệ")
    formatted = format_result(res)
    assert formatted == "LỖI BAD_ARGS: flight_id không hợp lệ"

    res_timeout = ToolResult(ok=False, code="TIMEOUT", error="TIMEOUT")
    assert format_result(res_timeout) == "LỖI TIMEOUT: TIMEOUT"


def test_render_history_empty():
    """Kiểm tra render_history([]) khi lịch sử rỗng trả về chuỗi '(chưa làm gì)'."""
    assert render_history([]) == "(chưa làm gì)"


def test_render_history_with_obs():
    """Kiểm tra render_history định dạng đúng danh sách observation."""
    obs_list = [
        Obs(
            tool="search_flights",
            args={"origin": "SGN", "destination": "HAN", "depart_date": "2026-10-20"},
            res=ToolResult(ok=True, data=[{"flight_id": "VN123", "price": 1000000}]),
        ),
        Obs(
            tool="book_flight",
            args={"flight_id": "VN123"},
            res=ToolResult(ok=False, code="SOLD_OUT", error="Hết chỗ"),
        ),
    ]
    rendered = render_history(obs_list)
    lines = rendered.split("\n")
    assert len(lines) == 2
    assert lines[0].startswith("- search_flights(")
    assert "VN123" in lines[0]
    assert lines[1].startswith("- book_flight(")
    assert "LỖI SOLD_OUT: Hết chỗ" in lines[1]


# ==============================================================================
# NEGATIVE & EDGE TESTS
# ==============================================================================

def test_format_result_truncates_long_data():
    """Kiểm tra kết quả thành công dài hơn 2000 ký tự sẽ bị cắt ngắn tối đa 2000 ký tự."""
    # Tạo dữ liệu lớn vượt quá 2000 ký tự JSON
    long_list = [{"id": f"FL_{i:04d}", "details": "A" * 50} for i in range(100)]
    res = ToolResult(ok=True, data=long_list)

    raw_json = json.dumps(long_list, ensure_ascii=False)
    assert len(raw_json) > 2000, "Dữ liệu mẫu phải lớn hơn 2000 ký tự để kiểm tra cắt ngắn"

    formatted = format_result(res)
    assert len(formatted) == 2000
    assert formatted == raw_json[:2000]


def test_format_result_error_with_handoff():
    """Kiểm tra format_result ưu tiên sử dụng trường handoff khi có lỗi cần bàn giao."""
    res = ToolResult(
        ok=False,
        code="NO_PERMISSION",
        error="Không đủ quyền thực hiện hành động",
        handoff="Người dùng thiếu quyền 'flights:book' cho book_flight",
    )
    formatted = format_result(res)
    assert formatted == "LỖI NO_PERMISSION: Người dùng thiếu quyền 'flights:book' cho book_flight"
    # handoff phải được ưu tiên trước error
    assert "Người dùng thiếu quyền 'flights:book'" in formatted


def test_format_result_preserves_unicode():
    """Kiểm tra format_result giữ nguyên tiếng Việt Unicode (ensure_ascii=False)."""
    vietnamese_data = {"thông_báo": "Đặt vé thành công", "hành_khách": "Nguyễn Văn A"}
    res = ToolResult(ok=True, data=vietnamese_data)
    formatted = format_result(res)
    assert "thông_báo" in formatted
    assert "Đặt vé thành công" in formatted
    assert "\\u" not in formatted


# ==============================================================================
# REGRESSION TESTS
# ==============================================================================

def test_system_prompt_returns_string():
    """Đảm bảo system_prompt() luôn trả về kiểu str và không rỗng."""
    prompt = system_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_all_prompt_functions_return_str(intent_book):
    """Đảm bảo tất cả các hàm tạo prompt đều trả về str."""
    res_system = system_prompt()
    assert isinstance(res_system, str)

    res_parse = parse_prompt()
    assert isinstance(res_parse, str)

    res_user = parse_user("Test Name", "Test Task")
    assert isinstance(res_user, str)

    res_goal = goal_prompt(intent_book)
    assert isinstance(res_goal, str)

    res_plan_empty = plan_prompt([])
    assert isinstance(res_plan_empty, str)

    res_plan_hist = plan_prompt([
        Obs(tool="search_flights", args={}, res=ToolResult(ok=True, data=[]))
    ])
    assert isinstance(res_plan_hist, str)

    res_fill = fill_args_prompt("search_flights")
    assert isinstance(res_fill, str)

    res_fmt_ok = format_result(ToolResult(ok=True, data={"key": "val"}))
    assert isinstance(res_fmt_ok, str)

    res_fmt_err = format_result(ToolResult(ok=False, code="ERR", error="msg"))
    assert isinstance(res_fmt_err, str)

    res_history_empty = render_history([])
    assert isinstance(res_history_empty, str)

    res_history_items = render_history([
        Obs(tool="test_tool", args={"x": 1}, res=ToolResult(ok=True, data="ok"))
    ])
    assert isinstance(res_history_items, str)


def test_goal_prompt_excludes_none():
    """Kiểm tra goal_prompt loại bỏ các trường mang giá trị None trong Intent JSON."""
    # Intent cancel chỉ cần booking_id và passenger, các trường origin/destination/depart_date là None
    intent_sparse = Intent(kind="cancel", booking_id="BK9999", passenger="Tran Van B")
    prompt = goal_prompt(intent_sparse)

    assert '"booking_id"' in prompt
    assert '"passenger"' in prompt
    assert '"kind"' in prompt
    # Các trường None tuyệt đối không xuất hiện trong output
    assert '"origin"' not in prompt
    assert '"destination"' not in prompt
    assert '"depart_date"' not in prompt
