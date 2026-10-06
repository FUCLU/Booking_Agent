"""
test_harness.py - Kiểm thử toàn diện cho module harness.py (BTVN#3)

Bao gồm 30 ca kiểm thử chia làm 3 nhóm:
  - POSITIVE TESTS (1-9): Luồng thành công, ghi nhận dữ liệu, audit log, langchain tools
  - NEGATIVE TESTS (10-23): Chặn vi phạm 4 lớp harness (schema, quyền sở hữu, ngân sách, giá trần, timeout)
  - REGRESSION TESTS (24-30): Chế độ tắt enforce (ablation), các bộ đếm rejected/blocked/calls, đổi vé (rebook)
"""
from __future__ import annotations

import json
import pytest

from config import MAX_TOOL_CALLS, PRICE_CAP
from harness import Harness, Intent, Obs, ToolResult, UserCtx
from scenarios import base_db
from tools import MockAirline


# ==============================================================================
# POSITIVE TESTS (1 - 9)
# ==============================================================================

def test_run_tool_search_success(harness: Harness):
    """
    1. Tìm kiếm chuyến bay qua harness thành công, trả về ok=True cùng danh sách chuyến bay hợp lệ.
    """
    args = {"origin": "SGN", "destination": "HAN", "depart_date": "2026-10-20"}
    res = harness.run_tool("search_flights", args)

    assert res.ok is True
    assert res.code == "OK"
    assert res.error is None
    assert res.handoff is None
    assert isinstance(res.data, list)
    assert len(res.data) == 3
    assert all(f["origin"] == "SGN" and f["destination"] == "HAN" for f in res.data)


def test_run_tool_book_success(harness: Harness):
    """
    2. Sau khi tìm kiếm, đặt vé thành công qua harness, trả về ok=True kèm mã booking.
    """
    # Bước 1: Tìm kiếm để flight_id được lưu vào harness.seen
    search_args = {"origin": "SGN", "destination": "HAN", "depart_date": "2026-10-20"}
    harness.run_tool("search_flights", search_args)

    # Bước 2: Đặt vé chuyến bay đã thấy
    book_args = {"flight_id": "VJ150-20261020", "passenger": "Nguyen Van A"}
    res = harness.run_tool("book_flight", book_args)

    assert res.ok is True
    assert res.code == "OK"
    assert res.data["flight_id"] == "VJ150-20261020"
    assert "booking_id" in res.data
    assert res.data["booking_id"].startswith("BK")
    # Kiểm tra trạng thái trong DB
    booking_in_db = harness.db.bookings[res.data["booking_id"]]
    assert booking_in_db["owner"] == harness.user.user_id
    assert booking_in_db["status"] == "confirmed"


def test_run_tool_cancel_own_booking(harness: Harness, db: MockAirline):
    """
    3. Huỷ booking của chính người dùng hiện tại thành công, trả về ok=True.
    """
    db.add_booking("BK0001", "u1", "VN210-20261020")
    res = harness.run_tool("cancel_booking", {"booking_id": "BK0001"})

    assert res.ok is True
    assert res.code == "OK"
    assert res.data["booking_id"] == "BK0001"
    assert res.data["status"] == "cancelled"
    assert db.bookings["BK0001"]["status"] == "cancelled"


def test_goal_met_after_booking(harness: Harness, intent_book: Intent):
    """
    4. Sau khi đặt chuyến bay rẻ nhất, goal_met() trả về True cho intent book_cheapest.
    """
    # Tìm kiếm và đặt chuyến bay rẻ nhất (VJ150: 1.490.000đ)
    harness.run_tool("search_flights", {"origin": "SGN", "destination": "HAN", "depart_date": "2026-10-20"})
    harness.run_tool("book_flight", {"flight_id": "VJ150-20261020", "passenger": "Nguyen Van A"})

    met, msg = harness.goal_met(intent_book)
    assert met is True
    assert msg == ""


def test_goal_met_after_cancel(harness: Harness, db: MockAirline, intent_cancel: Intent):
    """
    5. Sau khi huỷ booking thành công, goal_met() trả về True cho intent cancel.
    """
    db.add_booking("BK0001", "u1", "VN210-20261020")
    res = harness.run_tool("cancel_booking", {"booking_id": "BK0001"})
    assert res.ok is True

    met, msg = harness.goal_met(intent_cancel)
    assert met is True
    assert msg == ""


def test_check_done_with_enforce(harness: Harness, intent_book: Intent):
    """
    6. Khi enforce=True, check_done() uỷ quyền trực tiếp cho goal_met() để kiểm chứng.
    """
    # Trước khi đặt: cả hai đều trả về False
    assert harness.check_done(intent_book) == harness.goal_met(intent_book)
    assert harness.check_done(intent_book)[0] is False

    # Đặt vé hợp lệ
    harness.run_tool("search_flights", {"origin": "SGN", "destination": "HAN", "depart_date": "2026-10-20"})
    harness.run_tool("book_flight", {"flight_id": "VJ150-20261020", "passenger": "Nguyen Van A"})

    # Sau khi đặt: cả hai đều trả về (True, "")
    assert harness.check_done(intent_book) == harness.goal_met(intent_book)
    assert harness.check_done(intent_book) == (True, "")


def test_audit_log_records_calls(harness: Harness):
    """
    7. Nhật ký audit ghi nhận đúng cấu trúc và chi tiết các tool call đã thực hiện.
    """
    search_args = {"origin": "SGN", "destination": "HAN", "depart_date": "2026-10-20"}
    harness.run_tool("search_flights", search_args)
    book_args = {"flight_id": "VJ150-20261020", "passenger": "Nguyen Van A"}
    harness.run_tool("book_flight", book_args)

    assert len(harness.audit) == 2
    entry0 = harness.audit[0]
    assert entry0["tool"] == "search_flights"
    assert entry0["args"] == search_args
    assert entry0["code"] == "OK"
    assert entry0["detail"] == "ok"

    entry1 = harness.audit[1]
    assert entry1["tool"] == "book_flight"
    assert entry1["args"] == book_args
    assert entry1["code"] == "OK"
    assert entry1["detail"] == "ok"


def test_seen_tracks_search_results(harness: Harness):
    """
    8. harness.seen theo dõi chính xác các chuyến bay trả về từ search_flights.
    """
    assert harness.seen == {}
    harness.run_tool("search_flights", {"origin": "SGN", "destination": "HAN", "depart_date": "2026-10-20"})

    assert "VJ150-20261020" in harness.seen
    assert "VN210-20261020" in harness.seen
    assert "QH320-20261020" in harness.seen
    assert harness.seen["VJ150-20261020"]["price"] == 1_490_000


def test_harness_langchain_tools_callable(harness: Harness):
    """
    9. harness.langchain_tools() tạo danh sách StructuredTool có thể invoke và chạy qua harness.
    """
    tools = harness.langchain_tools()
    assert len(tools) == 3

    search_tool = next(t for t in tools if t.name == "search_flights")
    res_str = search_tool.invoke({"origin": "SGN", "destination": "HAN", "depart_date": "2026-10-20"})

    parsed = json.loads(res_str)
    assert isinstance(parsed, list)
    assert len(parsed) == 3
    # Xác nhận chuyến bay đã được cập nhật vào seen của harness
    assert "VJ150-20261020" in harness.seen


# ==============================================================================
# NEGATIVE TESTS (10 - 23)
# ==============================================================================

def test_run_tool_unknown_tool(harness: Harness):
    """
    10. Gọi tool không tồn tại trong danh mục TOOLS bị từ chối với mã BAD_TOOL.
    """
    res = harness.run_tool("unknown_tool", {"dummy": 123})

    assert res.ok is False
    assert res.code == "BAD_TOOL"
    assert "unknown_tool" in res.error


def test_run_tool_bad_args_schema(harness: Harness):
    """
    11. Tham số sai schema Pydantic (origin='sg' không đủ 3 chữ in hoa) trả về BAD_ARGS.
    """
    res = harness.run_tool("search_flights", {"origin": "sg", "destination": "HAN", "depart_date": "2026-10-20"})

    assert res.ok is False
    assert res.code == "BAD_ARGS"
    assert res.error is not None


def test_run_tool_bad_date_past(harness: Harness):
    """
    12. Ngày khởi hành ở quá khứ vi phạm validator Pydantic, trả về BAD_ARGS.
    """
    res = harness.run_tool("search_flights", {"origin": "SGN", "destination": "HAN", "depart_date": "2020-01-01"})

    assert res.ok is False
    assert res.code == "BAD_ARGS"
    assert "quá khứ" in res.error


def test_run_tool_book_unknown_flight_id(harness: Harness):
    """
    13. Đặt vé với flight_id chưa từng xuất hiện trong kết quả search (bịa mã) trả về UNKNOWN_FLIGHT.
    """
    res = harness.run_tool("book_flight", {"flight_id": "VN999-20261020", "passenger": "Nguyen Van A"})

    assert res.ok is False
    assert res.code == "UNKNOWN_FLIGHT"
    assert "kết quả search" in res.error


def test_run_tool_no_permission_book(db: MockAirline, guest: UserCtx):
    """
    14. Guest (không có quyền flights:book) thử đặt vé bị chặn với NO_PERMISSION và yêu cầu handoff.
    """
    h = Harness(db, guest, enforce=True)
    # Search thành công vì guest có quyền flights:read
    h.run_tool("search_flights", {"origin": "SGN", "destination": "HAN", "depart_date": "2026-10-20"})

    res = h.run_tool("book_flight", {"flight_id": "VJ150-20261020", "passenger": "Khach Vang Lai"})
    assert res.ok is False
    assert res.code == "NO_PERMISSION"
    assert res.handoff is not None
    assert "flights:book" in res.handoff


def test_run_tool_no_permission_cancel(db: MockAirline, guest: UserCtx):
    """
    15. Guest (không có quyền bookings:cancel) thử huỷ vé bị chặn với NO_PERMISSION và handoff.
    """
    db.add_booking("BK0001", "u1", "VN210-20261020")
    h = Harness(db, guest, enforce=True)

    res = h.run_tool("cancel_booking", {"booking_id": "BK0001"})
    assert res.ok is False
    assert res.code == "NO_PERMISSION"
    assert res.handoff is not None
    assert "bookings:cancel" in res.handoff


def test_run_tool_cancel_others_booking(harness: Harness, db: MockAirline):
    """
    16. Khách hàng u1 cố gắng huỷ booking của u2 bị chặn với NOT_OWNER và handoff.
    """
    db.add_booking("BK0002", "u2", "VN210-20261020")
    res = harness.run_tool("cancel_booking", {"booking_id": "BK0002"})

    assert res.ok is False
    assert res.code == "NOT_OWNER"
    assert res.handoff is not None
    assert "BK0002" in res.handoff
    # Booking của u2 vẫn phải ở trạng thái confirmed
    assert db.bookings["BK0002"]["status"] == "confirmed"


def test_run_tool_book_exceeds_price_cap(harness: Harness):
    """
    17. Đặt vé có giá vượt hạn mức tự động PRICE_CAP (8.5M > 5M) trả về NEEDS_APPROVAL và handoff.
    """
    # Tìm kiếm chuyến bay quốc tế SGN-CDG có giá 8.500.000đ
    harness.run_tool("search_flights", {"origin": "SGN", "destination": "CDG", "depart_date": "2026-11-02"})
    res = harness.run_tool("book_flight", {"flight_id": "AF255-20261102", "passenger": "Nguyen Van A"})

    assert res.ok is False
    assert res.code == "NEEDS_APPROVAL"
    assert res.handoff is not None
    assert "hạn mức" in res.handoff.lower()


def test_run_tool_budget_exceeded(harness: Harness):
    """
    18. Gọi tool vượt quá số lần tối đa MAX_TOOL_CALLS trả về lỗi BUDGET.
    """
    search_args = {"origin": "SGN", "destination": "HAN", "depart_date": "2026-10-20"}
    for _ in range(MAX_TOOL_CALLS):
        harness.run_tool("search_flights", search_args)

    # Lần gọi thứ MAX_TOOL_CALLS + 1 sẽ bị chặn
    res = harness.run_tool("search_flights", search_args)
    assert res.ok is False
    assert res.code == "BUDGET"
    assert "ngân sách" in res.error.lower()


def test_run_tool_timeout_auto_retry(harness: Harness, db: MockAirline):
    """
    19. Khi backend gặp timeout tạm thời 1 lần, harness tự động thử lại lần 2 và thành công.
    """
    db.transient["search_flights"] = 1
    res = harness.run_tool("search_flights", {"origin": "SGN", "destination": "HAN", "depart_date": "2026-10-20"})

    assert res.ok is True
    assert res.code == "OK"
    assert len(res.data) > 0
    assert db.transient["search_flights"] == 0


def test_run_tool_timeout_both_fail(harness: Harness, db: MockAirline):
    """
    20. Khi backend timeout 3 lần (vượt quá 2 lần thử của harness), trả về TIMEOUT.
    """
    db.transient["search_flights"] = 3
    res = harness.run_tool("search_flights", {"origin": "SGN", "destination": "HAN", "depart_date": "2026-10-20"})

    assert res.ok is False
    assert res.code == "TIMEOUT"
    assert res.error == "TIMEOUT"
    assert db.transient["search_flights"] == 1  # Đã thử 2 lần (3 -> 2 -> 1)


def test_goal_met_not_booked(harness: Harness, intent_book: Intent):
    """
    21. Khi chưa có bất kỳ booking nào trong DB, goal_met() trả về False.
    """
    met, msg = harness.goal_met(intent_book)
    assert met is False
    assert "chưa có booking" in msg


def test_goal_met_wrong_route(harness: Harness, db: MockAirline, intent_book: Intent):
    """
    22. Đặt vé sai tuyến bay (SGN-CDG thay vì SGN-HAN), goal_met() trả về False.
    """
    db.add_booking("BK0099", "u1", "AF255-20261102")
    met, msg = harness.goal_met(intent_book)

    assert met is False
    assert "chưa có booking mới đúng tuyến/ngày" in msg


def test_goal_met_not_cheapest(harness: Harness, db: MockAirline, intent_book: Intent):
    """
    23. Đặt chuyến đắt hơn chuyến rẻ nhất còn chỗ, goal_met() trả về False cho book_cheapest.
    """
    # Đặt VN210 (1.850.000đ), trong khi VJ150 (1.490.000đ) vẫn còn chỗ
    db.add_booking("BK0099", "u1", "VN210-20261020")
    met, msg = harness.goal_met(intent_book)

    assert met is False
    assert "không phải chuyến rẻ nhất" in msg


# ==============================================================================
# REGRESSION TESTS (24 - 30)
# ==============================================================================

def test_enforce_off_skips_validation(harness_no_enforce: Harness):
    """
    24. Khi enforce=False, harness bỏ qua kiểm tra schema và ràng buộc seen (đặt vé không cần search trước).
    """
    # Chuyến bay có sẵn trong DB nhưng chưa search
    res = harness_no_enforce.run_tool("book_flight", {"flight_id": "VJ150-20261020", "passenger": "Nguyen Van A"})

    assert res.ok is True
    assert res.code == "OK"
    assert harness_no_enforce.rejected == 0


def test_enforce_off_skips_auth(db: MockAirline, guest: UserCtx):
    """
    25. Khi enforce=False, harness bỏ qua kiểm quyền, cho phép guest đặt vé.
    """
    h = Harness(db, guest, enforce=False)
    res = h.run_tool("book_flight", {"flight_id": "VJ150-20261020", "passenger": "Khach Vang Lai"})

    assert res.ok is True
    assert res.code == "OK"
    assert h.blocked == 0
    assert "booking_id" in res.data


def test_check_done_enforce_off_always_true(harness_no_enforce: Harness, intent_book: Intent):
    """
    26. Khi enforce=False, check_done() luôn trả về (True, '') mà không kiểm tra DB.
    """
    done, msg = harness_no_enforce.check_done(intent_book)
    assert done is True
    assert msg == ""


def test_rejected_counter_increments(harness: Harness):
    """
    27. Mỗi lần vi phạm schema hoặc bịa mã chuyến bay, bộ đếm harness.rejected tăng thêm 1.
    """
    assert harness.rejected == 0

    # Lần 1: Sai định dạng IATA sân bay đi
    harness.run_tool("search_flights", {"origin": "INVALID", "destination": "HAN", "depart_date": "2026-10-20"})
    assert harness.rejected == 1

    # Lần 2: Mã chuyến bay hợp lệ regex nhưng chưa từng xuất hiện trong seen
    harness.run_tool("book_flight", {"flight_id": "VN999-20261020", "passenger": "Nguyen Van A"})
    assert harness.rejected == 2


def test_blocked_counter_increments(db: MockAirline, guest: UserCtx):
    """
    28. Khi bị từ chối quyền thực thi tool, bộ đếm harness.blocked tăng thêm 1.
    """
    db.add_booking("BK0001", "u1", "VN210-20261020")
    h = Harness(db, guest, enforce=True)
    assert h.blocked == 0

    h.run_tool("cancel_booking", {"booking_id": "BK0001"})
    assert h.blocked == 1


def test_calls_counter_tracks_total(harness: Harness):
    """
    29. harness.calls theo dõi chính xác tổng số lần gọi tool (cả thành công lẫn thất bại).
    """
    assert harness.calls == 0

    harness.run_tool("search_flights", {"origin": "SGN", "destination": "HAN", "depart_date": "2026-10-20"})
    harness.run_tool("unknown_tool", {})
    harness.run_tool("search_flights", {"origin": "sg", "destination": "HAN", "depart_date": "2026-10-20"})

    assert harness.calls == 3


def test_cancel_then_rebook_goal_met(harness: Harness, db: MockAirline, intent_rebook: Intent):
    """
    30. Intent rebook: Huỷ booking cũ BK0001 sau đó đặt chuyến rẻ nhất mới, goal_met() trả về True.
    """
    # Khởi tạo booking ban đầu cho khách hàng u1
    db.add_booking("BK0001", "u1", "VN210-20261020")

    # Trước khi thao tác, goal_met trả về False
    met_init, _ = harness.goal_met(intent_rebook)
    assert met_init is False

    # Bước 1: Huỷ booking cũ BK0001
    res_cancel = harness.run_tool("cancel_booking", {"booking_id": "BK0001"})
    assert res_cancel.ok is True

    # Huỷ xong nhưng chưa đặt vé mới -> vẫn chưa đạt mục tiêu
    met_after_cancel, _ = harness.goal_met(intent_rebook)
    assert met_after_cancel is False

    # Bước 2: Tìm kiếm các chuyến bay
    res_search = harness.run_tool("search_flights", {"origin": "SGN", "destination": "HAN", "depart_date": "2026-10-20"})
    assert res_search.ok is True

    # Bước 3: Đặt chuyến rẻ nhất (VJ150: 1.490.000đ)
    res_book = harness.run_tool("book_flight", {"flight_id": "VJ150-20261020", "passenger": "Nguyen Van A"})
    assert res_book.ok is True

    # Bước 4: Kiểm tra lại mục tiêu hoàn thành
    met, msg = harness.goal_met(intent_rebook)
    assert met is True
    assert msg == ""
