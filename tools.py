"""
Chỉ chứa:
  - MockAirline: CSDL giả lập trong bộ nhớ, có thể cài sẵn "sự cố" (timeout, hết chỗ, ...)
  - 3 hàm tool: search_flights, book_flight, cancel_booking
  - Schema tham số (Pydantic), bảng đăng ký TOOLS
  - langchain_tools(): bọc tool thành LangChain StructuredTool (tên + mô tả + schema là
    những gì model nhìn thấy và gọi được qua bind_tools)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from typing import Callable

from pydantic import BaseModel, Field, field_validator

from config import TODAY  # ngày hệ thống lấy từ config.yaml


class Timeout(Exception):
    """Lỗi tạm thời (mạng/timeout) - có thể thử lại."""


class BackendError(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(msg)
        self.code = code


class MockAirline:
    """CSDL giả lập trong bộ nhớ, có thể cài sẵn 'sự cố' để test agent."""

    def __init__(self):
        self.flights: dict[str, dict] = {}
        self.bookings: dict[str, dict] = {}
        self.transient: dict[str, int] = {}          # tool -> số lần timeout sắp tới
        self.fill_after_search: set[str] = set()     # chuyến hết chỗ NGAY sau khi được search
        self._n = 100

    def add_flight(self, fid, origin, dest, dt, price, seats):
        self.flights[fid] = dict(flight_id=fid, origin=origin, destination=dest,
                                 depart_date=dt, price=price, seats_left=seats)

    def add_booking(self, bid, owner, fid, status="confirmed"):
        f = self.flights[fid]
        self.bookings[bid] = dict(id=bid, owner=owner, flight_id=fid, price=f["price"], status=status)
        if status == "confirmed":
            f["seats_left"] -= 1

    def new_booking_id(self) -> str:
        self._n += 1
        return f"BK{self._n:04d}"


# Hàm tool thật sự (chỉ chạy SAU KHI harness cho phép)
def search_flights(db: MockAirline, user, origin: str, destination: str, depart_date: str) -> list[dict]:
    """Tìm các chuyến bay theo tuyến và ngày khởi hành. Trả về danh sách chuyến, mỗi chuyến gồm flight_id, price (VND) và seats_left (số ghế còn). Phải gọi tool này trước khi đặt vé để lấy flight_id hợp lệ."""
    if db.transient.get("search_flights", 0) > 0:
        db.transient["search_flights"] -= 1
        raise Timeout()
    res = [dict(f) for f in db.flights.values()
           if (f["origin"], f["destination"], f["depart_date"]) == (origin, destination, depart_date)]
    for f in res:  # mô phỏng race condition: vé bị người khác mua mất sau khi mình search
        if f["flight_id"] in db.fill_after_search:
            db.flights[f["flight_id"]]["seats_left"] = 0
            db.fill_after_search.discard(f["flight_id"])
    return res


def book_flight(db: MockAirline, user, flight_id: str, passenger: str) -> dict:
    """Đặt vé cho MỘT chuyến bay đã xuất hiện trong kết quả search_flights. Trả về booking_id. Lỗi SOLD_OUT nghĩa là chuyến vừa hết chỗ: hãy search lại và chọn chuyến khác."""
    f = db.flights.get(flight_id)
    if f is None:
        raise BackendError("UNKNOWN_FLIGHT", f"Không có chuyến {flight_id}")
    if f["seats_left"] <= 0:
        raise BackendError("SOLD_OUT", f"Chuyến {flight_id} đã hết chỗ")
    f["seats_left"] -= 1
    bid = db.new_booking_id()
    db.bookings[bid] = dict(id=bid, owner=user.user_id, flight_id=flight_id,
                            price=f["price"], status="confirmed", passenger=passenger)
    return dict(booking_id=bid, flight_id=flight_id, price=f["price"])


def cancel_booking(db: MockAirline, user, booking_id: str) -> dict:
    """Huỷ một booking đã có theo booking_id. Chỉ huỷ được booking của chính khách đang được phục vụ."""
    b = db.bookings.get(booking_id)
    if b is None:
        raise BackendError("NOT_FOUND", f"Không có booking {booking_id}")
    if b["status"] == "cancelled":
        raise BackendError("ALREADY_CANCELLED", "Booking đã huỷ trước đó")
    b["status"] = "cancelled"
    db.flights[b["flight_id"]]["seats_left"] += 1
    return dict(booking_id=booking_id, status="cancelled")


# Schema tham số (lớp ràng buộc dữ liệu)
class SearchArgs(BaseModel):
    origin: str = Field(pattern=r"^[A-Z]{3}$", description="Mã sân bay IATA 3 chữ in hoa của nơi đi. Ví dụ: SGN")
    destination: str = Field(pattern=r"^[A-Z]{3}$", description="Mã sân bay IATA 3 chữ in hoa của nơi đến. Ví dụ: HAN")
    depart_date: str = Field(description="Ngày khởi hành dạng YYYY-MM-DD, không ở quá khứ. Ví dụ: 2026-10-20")

    @field_validator("depart_date")
    @classmethod
    def _iso_future(cls, v):
        d = Date.fromisoformat(v)  # sai định dạng -> ValueError -> ValidationError
        if d < TODAY:
            raise ValueError("ngày bay ở quá khứ")
        return v


class BookArgs(BaseModel):
    flight_id: str = Field(pattern=r"^[A-Z]{2}\d{3,4}-\d{8}$", description="Mã chuyến bay, CHỈ lấy từ kết quả search_flights. Ví dụ: VJ150-20261020")
    passenger: str = Field(min_length=2, description="Họ tên đầy đủ của hành khách. Ví dụ: Nguyen Van A")


class CancelArgs(BaseModel):
    booking_id: str = Field(pattern=r"^BK\d{4}$", description="Mã booking cần huỷ, dạng BK + 4 chữ số. Ví dụ: BK0101")


@dataclass
class ToolSpec:
    name: str
    args_schema: type[BaseModel]
    func: Callable
    scope: str


TOOLS: dict[str, ToolSpec] = {s.name: s for s in [
    ToolSpec("search_flights", SearchArgs, search_flights, "flights:read"),
    ToolSpec("book_flight", BookArgs, book_flight, "flights:book"),
    ToolSpec("cancel_booking", CancelArgs, cancel_booking, "bookings:cancel"),
]}


# Tool "điều khiển": chỉ dành cho ReAct, KHÔNG chạy qua backend/harness
class FinishArgs(BaseModel):
    summary: str = Field(description="Tóm tắt ngắn những gì đã làm cho khách")


class HandoffArgs(BaseModel):
    reason: str = Field(description="Lý do cần chuyển cho nhân viên con người")


CONTROL_TOOLS: dict[str, tuple[str, type[BaseModel]]] = {
    "finish": ("Gọi khi mục tiêu của khách ĐÃ hoàn thành. Hệ thống sẽ kiểm tra lại bằng code.", FinishArgs),
    "handoff_to_human": ("Gọi khi không thể tiếp tục an toàn (thiếu quyền, cần người duyệt, "
                         "không có chuyến bay phù hợp) để chuyển cho nhân viên.", HandoffArgs),
}


def langchain_tools(executor: Callable[[str, dict], str] | None = None, include_control: bool = False):
    """
    Trả về danh sách LangChain StructuredTool để dùng với model.bind_tools().
    Model chỉ nhìn thấy 3 thứ từ mỗi tool:
      1. name         — tên tool (để gọi)
      2. description  — mô tả (lấy từ docstring của hàm)
      3. args_schema  — schema tham số (lấy từ Pydantic model, gồm tên + type hint + Field description)

    executor=None : tool chỉ để KHAI BÁO cho model; model chỉ đề xuất tool call.
    executor=f    : tool gọi được (tool.invoke(args)); mỗi lần gọi chạy f(tên_tool, args) -> chuỗi kết quả.
                    Harness.langchain_tools() dùng cách này để mọi lần gọi đều qua 4 lớp harness.
    """
    from langchain_core.tools import StructuredTool

    def make(name: str, desc: str, schema: type[BaseModel], runnable: bool):
        def func(**kwargs):
            """Placeholder — mô tả thật nằm ở description của StructuredTool."""
            if runnable and executor is not None:
                return executor(name, kwargs)
            raise RuntimeError(f"Tool '{name}' chỉ được chạy qua Harness.run_tool()")

        return StructuredTool.from_function(
            func=func,
            name=name,
            description=desc,        # model nhìn thấy mô tả này
            args_schema=schema,       # model nhìn thấy tên + type + description của từng tham số
            handle_validation_error=True,
        )

    tools = [make(s.name, s.func.__doc__ or "", s.args_schema, True) for s in TOOLS.values()]
    if include_control:
        tools += [make(n, d, m, False) for n, (d, m) in CONTROL_TOOLS.items()]
    return tools