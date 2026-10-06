"""
Mọi tool call của agent PHẢI đi qua Harness.run_tool(). Bốn lớp:
  Lớp 1  Ràng buộc dữ liệu   : schema Pydantic + flight_id phải từng xuất hiện trong kết quả search
  Lớp 2  Kiểm quyền           : scope của người dùng, quyền sở hữu booking
  Lớp 3  Bàn giao             : sai quyền / vượt hạn mức -> dừng agent, chuyển cho người
  Lớp 4  Tiêu chí hoàn thành  : goal_met() kiểm trạng thái DB bằng code, không tin lời agent
Ngoài ra: ngân sách số lần gọi, tự thử lại 1 lần khi timeout, audit log.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Optional

from pydantic import BaseModel, ValidationError

from config import MAX_TOOL_CALLS, PRICE_CAP
from tools import TOOLS, BackendError, MockAirline, Timeout, ToolSpec, langchain_tools


@dataclass
class UserCtx:
    user_id: str
    name: str
    scopes: set[str]


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None
    code: str = "OK"
    handoff: str | None = None   # != None  =>  agent PHẢI dừng và bàn giao cho người


@dataclass
class Obs:
    tool: str
    args: dict
    res: ToolResult


class Intent(BaseModel):
    """Mục tiêu có cấu trúc, trích từ câu của khách."""
    kind: Literal["book_cheapest", "cancel", "rebook"]
    origin: Optional[str] = None
    destination: Optional[str] = None
    depart_date: Optional[str] = None
    booking_id: Optional[str] = None
    passenger: str = ""


class Harness:
    def __init__(self, db: MockAirline, user: UserCtx, enforce: bool = True):
        self.db, self.user, self.enforce = db, user, enforce
        self.seen: dict[str, dict] = {}   # chuyến đã xuất hiện trong kết quả search của phiên này
        self.calls = self.rejected = self.blocked = 0
        self.audit: list[dict] = []

    # Thực thi 1 tool call qua 4 lớp
    def run_tool(self, name: str, args: dict) -> ToolResult:
        self.calls += 1
        if self.calls > MAX_TOOL_CALLS:
            return self._log(name, args, ToolResult(False, error="Hết ngân sách gọi tool", code="BUDGET"))
        spec = TOOLS.get(name)
        if spec is None:
            return self._log(name, args, ToolResult(False, error=f"Tool '{name}' không tồn tại", code="BAD_TOOL"))

        kwargs = args
        if self.enforce:
            # Lớp 1a: schema (kiểu, định dạng, ngày hợp lệ)
            try:
                kwargs = spec.args_schema.model_validate(args).model_dump()
            except ValidationError as e:
                self.rejected += 1
                msg = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
                return self._log(name, args, ToolResult(False, error=msg, code="BAD_ARGS"))
            # Lớp 1b: ràng buộc ngữ nghĩa - flight_id phải đến từ kết quả search (chống bịa mã)
            if name == "book_flight" and kwargs["flight_id"] not in self.seen:
                self.rejected += 1
                return self._log(name, args, ToolResult(
                    False, error="flight_id chưa từng xuất hiện trong kết quả search", code="UNKNOWN_FLIGHT"))
            # Lớp 2 + 3: kiểm quyền, nếu sai quyền -> bàn giao
            denied = self._authorize(spec, kwargs)
            if denied:
                self.blocked += 1
                return self._log(name, args, denied)

        # Thực thi (có tự thử lại 1 lần với lỗi tạm thời)
        last = ToolResult(False, error="TIMEOUT", code="TIMEOUT")
        for _ in range(2):
            try:
                data = spec.func(self.db, self.user, **kwargs)
            except Timeout:
                continue
            except BackendError as e:
                return self._log(name, args, ToolResult(False, error=str(e), code=e.code))
            except TypeError as e:  # chỉ xảy ra khi TẮT harness: tham số sai khoá/thiếu
                return self._log(name, args, ToolResult(False, error=str(e), code="BAD_ARGS"))
            if name == "search_flights":
                self.seen.update({f["flight_id"]: f for f in data})
            return self._log(name, args, ToolResult(True, data=data))
        return self._log(name, args, last)

    def langchain_tools(self):
        """Tool LangChain GỌI ĐƯỢC (tool.invoke), mỗi lần gọi đều đi qua run_tool() = qua 4 lớp harness."""
        def run(name: str, kwargs: dict) -> str:
            r = self.run_tool(name, kwargs)
            return json.dumps(r.data, ensure_ascii=False) if r.ok else f"{r.code}: {r.handoff or r.error}"
        return langchain_tools(executor=run)

    def _authorize(self, spec: ToolSpec, kw: dict) -> ToolResult | None:
        if spec.scope not in self.user.scopes:
            return ToolResult(False, code="NO_PERMISSION", error="Không đủ quyền",
                              handoff=f"Người dùng thiếu quyền '{spec.scope}' cho {spec.name}")
        if spec.name == "cancel_booking":
            b = self.db.bookings.get(kw["booking_id"])
            if b and b["owner"] != self.user.user_id:
                return ToolResult(False, code="NOT_OWNER", error="Booking không thuộc người dùng",
                                  handoff=f"Booking {kw['booking_id']} không thuộc người dùng này")
        if spec.name == "book_flight":
            price = self.seen[kw["flight_id"]]["price"]
            if price > PRICE_CAP:
                return ToolResult(False, code="NEEDS_APPROVAL", error="Vượt hạn mức tự động",
                                  handoff=f"Giá {price:,}đ vượt hạn mức {PRICE_CAP:,}đ, cần nhân viên duyệt")
        return None

    def _log(self, name, args, res: ToolResult) -> ToolResult:
        self.audit.append(dict(tool=name, args=args, code=res.code,
                               detail=res.handoff or res.error or "ok"))
        return res

    # Lớp 4: tiêu chí hoàn thành kiểm bằng code (KHÔNG tin lời agent)
    def goal_met(self, it: Intent) -> tuple[bool, str]:
        uid, db = self.user.user_id, self.db
        if it.kind in ("cancel", "rebook"):
            b = db.bookings.get(it.booking_id)
            if not b or b["owner"] != uid or b["status"] != "cancelled":
                return False, "booking chưa được huỷ"
        if it.kind in ("book_cheapest", "rebook"):
            route = {fid: f for fid, f in db.flights.items() if
                     (f["origin"], f["destination"], f["depart_date"]) == (it.origin, it.destination, it.depart_date)}
            mine = [b for b in db.bookings.values() if b["owner"] == uid and b["status"] == "confirmed"
                    and b["flight_id"] in route and b["id"] != it.booking_id]
            if not mine:
                return False, "chưa có booking mới đúng tuyến/ngày"
            booked = {b["flight_id"] for b in mine}
            best = min(f["price"] for fid, f in route.items() if f["seats_left"] > 0 or fid in booked)
            if min(b["price"] for b in mine) > best:
                return False, "booking không phải chuyến rẻ nhất còn chỗ"
        return True, ""

    def check_done(self, it: Intent) -> tuple[bool, str]:
        return self.goal_met(it) if self.enforce else (True, "")