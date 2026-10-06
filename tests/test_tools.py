"""
tests/test_tools.py - Unit test toàn diện cho module tools.py

Bao gồm:
  - POSITIVE TESTS (1-12): Luồng thành công của search, book, cancel, args validation, langchain tools
  - NEGATIVE TESTS (13-25): Xử lý ngoại lệ, lỗi backend, validation error, timeout, race condition
  - REGRESSION TESTS (26-29): Tính toàn vẹn của cấu trúc TOOLS, tính duy nhất của ID, tính chất read-only của search
"""
from __future__ import annotations

import copy
import pytest
from pydantic import BaseModel, ValidationError

from config import TODAY
from tools import (
    CONTROL_TOOLS,
    TOOLS,
    BackendError,
    BookArgs,
    CancelArgs,
    MockAirline,
    SearchArgs,
    Timeout,
    ToolSpec,
    langchain_tools,
    t_book,
    t_cancel,
    t_search,
)


# ==============================================================================
# POSITIVE TESTS (1 - 12)
# ==============================================================================

def test_search_finds_matching_flights(db: MockAirline, customer):
    """1. test_search_finds_matching_flights - Tìm kiếm SGN->HAN ngày 2026-10-20 trả về đúng 3 chuyến bay."""
    results = t_search(db, customer, origin="SGN", destination="HAN", depart_date="2026-10-20")
    assert isinstance(results, list)
    assert len(results) == 3

    # Kiểm tra các chuyến bay trả về đúng chuyến dự kiến trong base_db
    flight_ids = {f["flight_id"] for f in results}
    assert flight_ids == {"VN210-20261020", "VJ150-20261020", "QH320-20261020"}


def test_search_returns_correct_fields(db: MockAirline, customer):
    """2. test_search_returns_correct_fields - Mỗi chuyến bay trả về đủ các trường: flight_id, origin, destination, depart_date, price, seats_left."""
    expected_fields = {"flight_id", "origin", "destination", "depart_date", "price", "seats_left"}
    results = t_search(db, customer, origin="SGN", destination="HAN", depart_date="2026-10-20")

    assert len(results) > 0
    for flight in results:
        assert set(flight.keys()) == expected_fields
        assert flight["origin"] == "SGN"
        assert flight["destination"] == "HAN"
        assert flight["depart_date"] == "2026-10-20"
        assert isinstance(flight["price"], int)
        assert isinstance(flight["seats_left"], int)


def test_book_success(db: MockAirline, customer):
    """3. test_book_success - Đặt chuyến bay còn chỗ thành công và ghi nhận booking vào CSDL."""
    flight_id = "VJ150-20261020"
    passenger = "Nguyen Van A"

    res = t_book(db, customer, flight_id=flight_id, passenger=passenger)

    assert "booking_id" in res
    assert res["flight_id"] == flight_id
    assert res["price"] == 1_490_000

    booking_id = res["booking_id"]
    assert booking_id in db.bookings

    saved_booking = db.bookings[booking_id]
    assert saved_booking["id"] == booking_id
    assert saved_booking["owner"] == customer.user_id
    assert saved_booking["flight_id"] == flight_id
    assert saved_booking["passenger"] == passenger
    assert saved_booking["status"] == "confirmed"


def test_book_decreases_seats(db: MockAirline, customer):
    """4. test_book_decreases_seats - Sau khi đặt chỗ thành công, số ghế còn lại (seats_left) giảm 1."""
    flight_id = "VJ150-20261020"
    initial_seats = db.flights[flight_id]["seats_left"]

    t_book(db, customer, flight_id=flight_id, passenger="Nguyen Van A")

    new_seats = db.flights[flight_id]["seats_left"]
    assert new_seats == initial_seats - 1


def test_cancel_success(db: MockAirline, customer):
    """5. test_cancel_success - Huỷ một booking đang ở trạng thái confirmed thành công."""
    # Đặt vé trước để có booking
    book_res = t_book(db, customer, flight_id="VN210-20261020", passenger="Nguyen Van A")
    booking_id = book_res["booking_id"]

    cancel_res = t_cancel(db, customer, booking_id=booking_id)

    assert cancel_res["booking_id"] == booking_id
    assert cancel_res["status"] == "cancelled"
    assert db.bookings[booking_id]["status"] == "cancelled"


def test_cancel_restores_seats(db: MockAirline, customer):
    """6. test_cancel_restores_seats - Sau khi huỷ vé thành công, seats_left của chuyến bay tăng lại 1."""
    flight_id = "VN210-20261020"
    seats_before_book = db.flights[flight_id]["seats_left"]

    book_res = t_book(db, customer, flight_id=flight_id, passenger="Nguyen Van A")
    booking_id = book_res["booking_id"]
    assert db.flights[flight_id]["seats_left"] == seats_before_book - 1

    t_cancel(db, customer, booking_id=booking_id)
    assert db.flights[flight_id]["seats_left"] == seats_before_book


def test_search_args_valid():
    """7. test_search_args_valid - SearchArgs hợp lệ với 'SGN', 'HAN', '2026-10-20'."""
    args = SearchArgs(origin="SGN", destination="HAN", depart_date="2026-10-20")
    assert args.origin == "SGN"
    assert args.destination == "HAN"
    assert args.depart_date == "2026-10-20"


def test_book_args_valid():
    """8. test_book_args_valid - BookArgs hợp lệ với 'VJ150-20261020', 'Nguyen Van A'."""
    args = BookArgs(flight_id="VJ150-20261020", passenger="Nguyen Van A")
    assert args.flight_id == "VJ150-20261020"
    assert args.passenger == "Nguyen Van A"


def test_cancel_args_valid():
    """9. test_cancel_args_valid - CancelArgs hợp lệ với 'BK0101'."""
    args = CancelArgs(booking_id="BK0101")
    assert args.booking_id == "BK0101"


def test_langchain_tools_returns_list():
    """10. test_langchain_tools_returns_list - langchain_tools() trả về một danh sách (list)."""
    tools = langchain_tools()
    assert isinstance(tools, list)


def test_langchain_tools_count_no_control():
    """11. test_langchain_tools_count_no_control - Khi include_control=False trả về đúng 3 tools."""
    tools = langchain_tools(include_control=False)
    assert len(tools) == 3
    tool_names = [t.name for t in tools]
    assert set(tool_names) == {"search_flights", "book_flight", "cancel_booking"}


def test_langchain_tools_count_with_control():
    """12. test_langchain_tools_count_with_control - Khi include_control=True trả về đúng 5 tools (thêm finish và handoff)."""
    tools = langchain_tools(include_control=True)
    assert len(tools) == 5
    tool_names = [t.name for t in tools]
    assert set(tool_names) == {
        "search_flights",
        "book_flight",
        "cancel_booking",
        "finish",
        "handoff_to_human",
    }


# ==============================================================================
# NEGATIVE TESTS (13 - 25)
# ==============================================================================

def test_search_no_match(db: MockAirline, customer):
    """13. test_search_no_match - Tìm kiếm tuyến không có chuyến bay (SGN->DAD) trả về danh sách rỗng."""
    results = t_search(db, customer, origin="SGN", destination="DAD", depart_date="2026-10-20")
    assert results == []


def test_book_unknown_flight(db: MockAirline, customer):
    """14. test_book_unknown_flight - Đặt chuyến bay không tồn tại ném lỗi BackendError(UNKNOWN_FLIGHT)."""
    with pytest.raises(BackendError) as exc_info:
        t_book(db, customer, flight_id="VN999-20261020", passenger="Nguyen Van A")

    assert exc_info.value.code == "UNKNOWN_FLIGHT"
    assert "Không có chuyến" in str(exc_info.value)


def test_book_sold_out(db: MockAirline, customer):
    """15. test_book_sold_out - Đặt chuyến bay đã hết chỗ (seats_left = 0) ném lỗi BackendError(SOLD_OUT)."""
    flight_id = "VJ150-20261020"
    db.flights[flight_id]["seats_left"] = 0

    with pytest.raises(BackendError) as exc_info:
        t_book(db, customer, flight_id=flight_id, passenger="Nguyen Van A")

    assert exc_info.value.code == "SOLD_OUT"
    assert "hết chỗ" in str(exc_info.value)


def test_cancel_not_found(db: MockAirline, customer):
    """16. test_cancel_not_found - Huỷ booking không tồn tại ném lỗi BackendError(NOT_FOUND)."""
    with pytest.raises(BackendError) as exc_info:
        t_cancel(db, customer, booking_id="BK9999")

    assert exc_info.value.code == "NOT_FOUND"
    assert "Không có booking" in str(exc_info.value)


def test_cancel_already_cancelled(db: MockAirline, customer):
    """17. test_cancel_already_cancelled - Huỷ một booking đã huỷ trước đó ném lỗi BackendError(ALREADY_CANCELLED)."""
    # Đặt vé và huỷ lần đầu
    book_res = t_book(db, customer, flight_id="VN210-20261020", passenger="Nguyen Van A")
    booking_id = book_res["booking_id"]
    t_cancel(db, customer, booking_id=booking_id)

    # Huỷ lại lần 2
    with pytest.raises(BackendError) as exc_info:
        t_cancel(db, customer, booking_id=booking_id)

    assert exc_info.value.code == "ALREADY_CANCELLED"
    assert "huỷ trước đó" in str(exc_info.value)


def test_search_timeout(db: MockAirline, customer):
    """18. test_search_timeout - Khi transient['search_flights'] = 1, t_search ném ngoại lệ Timeout."""
    db.transient["search_flights"] = 1

    with pytest.raises(Timeout):
        t_search(db, customer, origin="SGN", destination="HAN", depart_date="2026-10-20")

    # Kiểm tra số lần timeout đã được trừ đi 1 (giảm về 0)
    assert db.transient["search_flights"] == 0

    # Lần gọi tiếp theo sau khi hết timeout phải thành công
    results = t_search(db, customer, origin="SGN", destination="HAN", depart_date="2026-10-20")
    assert len(results) == 3


def test_search_args_bad_origin():
    """19. test_search_args_bad_origin - SearchArgs từ chối mã origin sai định dạng hoặc quá ngắn như 'sg'."""
    with pytest.raises(ValidationError):
        SearchArgs(origin="sg", destination="HAN", depart_date="2026-10-20")

    with pytest.raises(ValidationError):
        SearchArgs(origin="sgn", destination="HAN", depart_date="2026-10-20")

    with pytest.raises(ValidationError):
        SearchArgs(origin="SGNN", destination="HAN", depart_date="2026-10-20")


def test_search_args_bad_date_past():
    """20. test_search_args_bad_date_past - SearchArgs từ chối ngày bay ở quá khứ (trước ngày TODAY)."""
    # Ngày hôm qua so với TODAY
    past_date = "2026-10-05"
    with pytest.raises(ValidationError) as exc_info:
        SearchArgs(origin="SGN", destination="HAN", depart_date=past_date)

    assert "quá khứ" in str(exc_info.value)


def test_search_args_bad_date_format():
    """21. test_search_args_bad_date_format - SearchArgs từ chối ngày sai định dạng ISO YYYY-MM-DD như 'not-a-date'."""
    with pytest.raises(ValidationError):
        SearchArgs(origin="SGN", destination="HAN", depart_date="not-a-date")

    with pytest.raises(ValidationError):
        SearchArgs(origin="SGN", destination="HAN", depart_date="2026/10/20")


def test_book_args_bad_flight_id():
    """22. test_book_args_bad_flight_id - BookArgs từ chối flight_id không đúng định dạng chuẩn như 'INVALID'."""
    with pytest.raises(ValidationError):
        BookArgs(flight_id="INVALID", passenger="Nguyen Van A")

    with pytest.raises(ValidationError):
        BookArgs(flight_id="vj150-20261020", passenger="Nguyen Van A")  # Chữ thường

    with pytest.raises(ValidationError):
        BookArgs(flight_id="VJ15-20261020", passenger="Nguyen Van A")   # Thiếu số hiệu chuyến bay


def test_book_args_empty_passenger():
    """23. test_book_args_empty_passenger - BookArgs từ chối tên hành khách có độ dài < 2 ký tự hoặc rỗng."""
    with pytest.raises(ValidationError):
        BookArgs(flight_id="VJ150-20261020", passenger="")

    with pytest.raises(ValidationError):
        BookArgs(flight_id="VJ150-20261020", passenger="A")


def test_cancel_args_bad_format():
    """24. test_cancel_args_bad_format - CancelArgs từ chối booking_id sai định dạng BK + 4 số như 'BOOKING1'."""
    with pytest.raises(ValidationError):
        CancelArgs(booking_id="BOOKING1")

    with pytest.raises(ValidationError):
        CancelArgs(booking_id="BK123")  # Chỉ 3 số

    with pytest.raises(ValidationError):
        CancelArgs(booking_id="bk0101")  # Chữ thường


def test_fill_after_search_race_condition(db: MockAirline, customer):
    """25. test_fill_after_search_race_condition - Mô phỏng race condition: vé hết chỗ ngay sau khi search."""
    flight_id = "VJ150-20261020"
    db.fill_after_search.add(flight_id)
    assert db.flights[flight_id]["seats_left"] > 0

    # Khi search, chuyến bay vẫn xuất hiện trong kết quả search ban đầu
    results = t_search(db, customer, origin="SGN", destination="HAN", depart_date="2026-10-20")
    matching = [f for f in results if f["flight_id"] == flight_id]
    assert len(matching) == 1

    # Nhưng ngay sau đó trong DB, seats_left bị đổi thành 0 và bị xoá khỏi fill_after_search
    assert db.flights[flight_id]["seats_left"] == 0
    assert flight_id not in db.fill_after_search

    # Lập tức đặt vé chuyến này sẽ nhận lỗi SOLD_OUT
    with pytest.raises(BackendError) as exc_info:
        t_book(db, customer, flight_id=flight_id, passenger="Nguyen Van A")
    assert exc_info.value.code == "SOLD_OUT"


# ==============================================================================
# REGRESSION TESTS (26 - 29)
# ==============================================================================

def test_tool_names_match_keys():
    """26. test_tool_names_match_keys - Khóa trong từ điển TOOLS phải khớp chính xác với ToolSpec.name."""
    for key, spec in TOOLS.items():
        assert key == spec.name


def test_tools_have_required_fields():
    """27. test_tools_have_required_fields - Mọi ToolSpec đều có đủ name, desc, args_model, fn, scope hợp lệ."""
    for name, spec in TOOLS.items():
        assert isinstance(spec.name, str) and len(spec.name) > 0
        assert isinstance(spec.desc, str) and len(spec.desc) > 0
        assert issubclass(spec.args_model, BaseModel)
        assert callable(spec.fn)
        assert isinstance(spec.scope, str) and len(spec.scope) > 0


def test_multiple_bookings_get_unique_ids(db: MockAirline, customer):
    """28. test_multiple_bookings_get_unique_ids - Đặt vé nhiều lần đảm bảo các booking_id sinh ra luôn phân biệt."""
    # Thử qua hàm new_booking_id
    generated_ids = [db.new_booking_id() for _ in range(5)]
    assert len(generated_ids) == len(set(generated_ids))

    # Thử qua hàm t_book thực tế
    b1 = t_book(db, customer, flight_id="VN210-20261020", passenger="Passenger 1")
    b2 = t_book(db, customer, flight_id="VN210-20261020", passenger="Passenger 2")
    b3 = t_book(db, customer, flight_id="QH320-20261020", passenger="Passenger 3")

    actual_bids = [b1["booking_id"], b2["booking_id"], b3["booking_id"]]
    assert len(actual_bids) == len(set(actual_bids))
    assert all(bid.startswith("BK") and len(bid) == 6 for bid in actual_bids)


def test_search_does_not_modify_db(db: MockAirline, customer):
    """29. test_search_does_not_modify_db - Hàm t_search ở trạng thái thông thường chỉ đọc dữ liệu, không làm thay đổi DB."""
    flights_before = copy.deepcopy(db.flights)
    bookings_before = copy.deepcopy(db.bookings)
    transient_before = copy.deepcopy(db.transient)
    fill_before = copy.deepcopy(db.fill_after_search)

    results = t_search(db, customer, origin="SGN", destination="HAN", depart_date="2026-10-20")
    assert len(results) > 0

    assert db.flights == flights_before
    assert db.bookings == bookings_before
    assert db.transient == transient_before
    assert db.fill_after_search == fill_before
