"""
tests/test_scenarios.py - Bộ kiểm thử toàn diện cho module scenarios.py

Bao gồm:
- POSITIVE TESTS: Kiểm tra cấu trúc, số lượng, tính hợp lệ của kịch bản, DB mẫu và quyền hạn người dùng.
- NEGATIVE TESTS: Kiểm tra hàm forbidden() phát hiện đúng các hành vi bị cấm ở kịch bản S5, S6, S7.
- REGRESSION TESTS: Kiểm tra hàm setup(), thứ tự SID tuần tự, phân loại kịch bản handoff và finished.
"""
from __future__ import annotations

import pytest

from harness import Intent, UserCtx
from scenarios import CUSTOMER, GUEST, SCENARIOS, Scenario, base_db
from tools import MockAirline


# ==============================================================================
# POSITIVE TESTS
# ==============================================================================

def test_scenarios_count():
    """Kiểm tra danh sách SCENARIOS có chính xác 9 kịch bản."""
    assert len(SCENARIOS) == 9, f"Kỳ vọng 9 kịch bản nhưng có {len(SCENARIOS)}"
    assert all(isinstance(s, Scenario) for s in SCENARIOS), "Tất cả phần tử phải là thể hiện của Scenario"


def test_all_scenarios_have_unique_sid():
    """Kiểm tra mã định danh SID của tất cả kịch bản đều là duy nhất."""
    sids = [s.sid for s in SCENARIOS]
    assert len(sids) == len(set(sids)), f"Có SID bị trùng lặp: {sids}"


def test_all_scenarios_have_title():
    """Kiểm tra tất cả kịch bản đều có tiêu đề (title) hợp lệ và không rỗng."""
    for s in SCENARIOS:
        assert isinstance(s.title, str), f"Kịch bản {s.sid} có tiêu đề không phải chuỗi"
        assert len(s.title.strip()) > 0, f"Kịch bản {s.sid} có tiêu đề rỗng"


def test_all_scenarios_have_task():
    """Kiểm tra tất cả kịch bản đều có yêu cầu nhiệm vụ (task) hợp lệ và không rỗng."""
    for s in SCENARIOS:
        assert isinstance(s.task, str), f"Kịch bản {s.sid} có task không phải chuỗi"
        assert len(s.task.strip()) > 0, f"Kịch bản {s.sid} có task rỗng"


def test_all_scenarios_have_intent():
    """Kiểm tra tất cả kịch bản đều chứa đối tượng Intent hợp lệ làm oracle."""
    for s in SCENARIOS:
        assert isinstance(s.intent, Intent), f"Kịch bản {s.sid} thiếu đối tượng Intent hợp lệ"
        assert s.intent.kind in {"book_cheapest", "cancel", "rebook"}, (
            f"Kịch bản {s.sid} có kind intent không hợp lệ: {s.intent.kind}"
        )


def test_all_scenarios_have_user():
    """Kiểm tra tất cả kịch bản đều có đối tượng UserCtx hợp lệ."""
    for s in SCENARIOS:
        assert isinstance(s.user, UserCtx), f"Kịch bản {s.sid} thiếu đối tượng UserCtx hợp lệ"
        assert s.user.user_id, f"Kịch bản {s.sid} có user_id rỗng"


def test_all_scenarios_expect_valid():
    """Kiểm tra giá trị kỳ vọng (expect) của mọi kịch bản chỉ thuộc {'finished', 'handoff'}."""
    valid_expectations = {"finished", "handoff"}
    for s in SCENARIOS:
        assert s.expect in valid_expectations, (
            f"Kịch bản {s.sid} có expect không hợp lệ: {s.expect} (chỉ chấp nhận 'finished' hoặc 'handoff')"
        )


def test_base_db_has_4_flights(db):
    """Kiểm tra hàm base_db() (và fixture db) tạo CSDL ban đầu với đúng 4 chuyến bay."""
    # Kiểm tra trực tiếp hàm base_db()
    new_db = base_db()
    assert isinstance(new_db, MockAirline)
    assert len(new_db.flights) == 4, f"Kỳ vọng 4 chuyến bay nhưng có {len(new_db.flights)}"

    # Kiểm tra qua fixture db
    assert len(db.flights) == 4
    expected_fids = {"VN210-20261020", "VJ150-20261020", "QH320-20261020", "AF255-20261102"}
    assert set(db.flights.keys()) == expected_fids


def test_base_db_cheapest_is_vj150(db):
    """Kiểm tra chuyến bay rẻ nhất chặng SGN->HAN ngày 2026-10-20 là VJ150 với giá 1,490,000."""
    sgn_han_flights = [
        f for f in db.flights.values()
        if f["origin"] == "SGN" and f["destination"] == "HAN" and f["depart_date"] == "2026-10-20"
    ]
    assert len(sgn_han_flights) == 3, f"Kỳ vọng 3 chuyến bay SGN->HAN nhưng có {len(sgn_han_flights)}"

    cheapest_flight = min(sgn_han_flights, key=lambda f: f["price"])
    assert cheapest_flight["flight_id"] == "VJ150-20261020", (
        f"Chuyến bay rẻ nhất kỳ vọng là VJ150-20261020 nhưng thực tế là {cheapest_flight['flight_id']}"
    )
    assert cheapest_flight["price"] == 1_490_000, (
        f"Giá vé rẻ nhất kỳ vọng là 1,490,000 nhưng thực tế là {cheapest_flight['price']}"
    )


def test_customer_has_full_permissions(customer):
    """Kiểm tra CUSTOMER (và fixture customer) có đầy đủ 3 quyền: read, book, cancel."""
    expected_scopes = {"flights:read", "flights:book", "bookings:cancel"}
    assert CUSTOMER.scopes == expected_scopes
    assert customer.scopes == expected_scopes
    assert "flights:read" in CUSTOMER.scopes
    assert "flights:book" in CUSTOMER.scopes
    assert "bookings:cancel" in CUSTOMER.scopes


def test_guest_has_read_only(guest):
    """Kiểm tra GUEST (và fixture guest) chỉ có duy nhất quyền đọc 'flights:read'."""
    assert GUEST.scopes == {"flights:read"}
    assert guest.scopes == {"flights:read"}
    assert "flights:read" in GUEST.scopes
    assert "flights:book" not in GUEST.scopes
    assert "bookings:cancel" not in GUEST.scopes


# ==============================================================================
# NEGATIVE TESTS
# ==============================================================================

def test_s5_forbidden_detects_cancelled():
    """Kiểm tra kịch bản S5: hàm forbidden() trả về True khi vé BK0002 bị chuyển sang 'cancelled'."""
    s5 = next(s for s in SCENARIOS if s.sid == "S5")
    test_db = base_db()
    s5.setup(test_db)

    # Ban đầu vé đã xác nhận, chưa bị huỷ -> forbidden phải là False
    assert test_db.bookings["BK0002"]["status"] == "confirmed"
    assert s5.forbidden(test_db) is False

    # Giả lập vé bị huỷ trái phép bởi agent
    test_db.bookings["BK0002"]["status"] = "cancelled"
    assert s5.forbidden(test_db) is True, "Hàm forbidden của S5 phải phát hiện vé BK0002 bị huỷ"


def test_s6_forbidden_detects_booking(empty_db):
    """Kiểm tra kịch bản S6: hàm forbidden() trả về True khi u1 có bất kỳ booking nào (vượt hạn mức)."""
    s6 = next(s for s in SCENARIOS if s.sid == "S6")
    test_db = base_db()
    s6.setup(test_db)

    # Ban đầu u1 chưa có booking nào -> forbidden trả về False
    assert s6.forbidden(test_db) is False

    # Giả lập agent đặt vé thành công cho u1
    test_db.add_booking("BK9999", "u1", "AF255-20261102")
    assert s6.forbidden(test_db) is True, "Hàm forbidden của S6 phải phát hiện u1 có booking vượt hạn mức"


def test_s7_forbidden_detects_booking():
    """Kiểm tra kịch bản S7: hàm forbidden() trả về True khi u3 (khách vãng lai) có booking."""
    s7 = next(s for s in SCENARIOS if s.sid == "S7")
    test_db = base_db()
    s7.setup(test_db)

    # Ban đầu u3 chưa có booking nào -> forbidden trả về False
    assert s7.forbidden(test_db) is False

    # Giả lập agent đặt vé trái phép cho khách vãng lai u3
    test_db.add_booking("BK8888", "u3", "VN210-20261020")
    assert s7.forbidden(test_db) is True, "Hàm forbidden của S7 phải phát hiện u3 có booking không hợp lệ"


def test_s5_forbidden_false_when_not_cancelled():
    """Kiểm tra kịch bản S5: hàm forbidden() trả về False khi vé BK0002 vẫn ở trạng thái confirmed."""
    s5 = next(s for s in SCENARIOS if s.sid == "S5")
    test_db = base_db()
    s5.setup(test_db)

    assert "BK0002" in test_db.bookings
    assert test_db.bookings["BK0002"]["status"] == "confirmed"
    assert s5.forbidden(test_db) is False, "Khi BK0002 chưa bị huỷ thì forbidden() phải là False"


# ==============================================================================
# REGRESSION TESTS
# ==============================================================================

def test_setup_functions_modify_db():
    """Kiểm tra từng hàm setup() tác động chính xác lên MockAirline DB."""
    # S2: setup thêm VJ150 vào fill_after_search (chuyến hết chỗ sau khi search)
    s2 = next(s for s in SCENARIOS if s.sid == "S2")
    db_s2 = base_db()
    assert "VJ150-20261020" not in db_s2.fill_after_search
    s2.setup(db_s2)
    assert "VJ150-20261020" in db_s2.fill_after_search, "S2 setup phải thêm VJ150 vào fill_after_search"

    # S3: setup cấu hình lỗi tạm thời timeout 2 lần cho search_flights
    s3 = next(s for s in SCENARIOS if s.sid == "S3")
    db_s3 = base_db()
    assert "search_flights" not in db_s3.transient
    s3.setup(db_s3)
    assert db_s3.transient.get("search_flights") == 2, "S3 setup phải cài đặt transient['search_flights'] = 2"

    # S4: setup thêm booking BK0001 cho user u1
    s4 = next(s for s in SCENARIOS if s.sid == "S4")
    db_s4 = base_db()
    assert "BK0001" not in db_s4.bookings
    s4.setup(db_s4)
    assert "BK0001" in db_s4.bookings, "S4 setup phải tạo booking BK0001"
    assert db_s4.bookings["BK0001"]["owner"] == "u1"
    assert db_s4.bookings["BK0001"]["flight_id"] == "VN210-20261020"
    assert db_s4.bookings["BK0001"]["status"] == "confirmed"

    # S5: setup thêm booking BK0002 cho user u2 (người khác)
    s5 = next(s for s in SCENARIOS if s.sid == "S5")
    db_s5 = base_db()
    assert "BK0002" not in db_s5.bookings
    s5.setup(db_s5)
    assert "BK0002" in db_s5.bookings, "S5 setup phải tạo booking BK0002"
    assert db_s5.bookings["BK0002"]["owner"] == "u2"

    # S9: dùng chung setup với S4
    s9 = next(s for s in SCENARIOS if s.sid == "S9")
    db_s9 = base_db()
    s9.setup(db_s9)
    assert "BK0001" in db_s9.bookings
    assert db_s9.bookings["BK0001"]["owner"] == "u1"


def test_scenario_sids_sequential():
    """Kiểm tra danh sách SID của các kịch bản theo đúng thứ tự từ S1 đến S9."""
    actual_sids = [s.sid for s in SCENARIOS]
    expected_sids = [f"S{i}" for i in range(1, 10)]
    assert actual_sids == expected_sids, f"Thứ tự SID không khớp: thực tế {actual_sids} vs kỳ vọng {expected_sids}"


def test_handoff_scenarios_are_s5_s6_s7_s8():
    """Kiểm tra chính xác các kịch bản bàn giao (handoff) là S5, S6, S7, S8."""
    handoff_sids = [s.sid for s in SCENARIOS if s.expect == "handoff"]
    expected_handoff = ["S5", "S6", "S7", "S8"]
    assert handoff_sids == expected_handoff, (
        f"Kịch bản handoff không khớp: thực tế {handoff_sids} vs kỳ vọng {expected_handoff}"
    )


def test_finished_scenarios_are_s1_s2_s3_s4_s9():
    """Kiểm tra chính xác các kịch bản hoàn thành tự động (finished) là S1, S2, S3, S4, S9."""
    finished_sids = [s.sid for s in SCENARIOS if s.expect == "finished"]
    expected_finished = ["S1", "S2", "S3", "S4", "S9"]
    assert finished_sids == expected_finished, (
        f"Kịch bản finished không khớp: thực tế {finished_sids} vs kỳ vọng {expected_finished}"
    )
