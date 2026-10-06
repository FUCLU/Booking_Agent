"""
brains.py - Lớp Brain: giao tiếp với LLM qua LangChain

Bốn phương thức quyết định:
  - parse()     : trích mục tiêu có cấu trúc (Intent) từ câu của khách
  - decide()    : ReAct - chọn tool call tiếp theo dựa trên lịch sử quan sát
  - plan()      : lập kế hoạch danh sách các bước (PlanStep)
  - fill_args() : điền tham số cho 1 bước khi plan chưa biết trước

Brain KHÔNG biết harness: nó chỉ đề xuất tool call, việc chạy tool do harness quản lý.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

try:                                  # đọc API key từ file .env (nếu có)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from config import DEFAULT_MODEL, LLM_KWARGS
import prompts
from harness import Intent, Obs
from scenarios import Scenario
from tools import langchain_tools


@dataclass
class Action:
    tool: str | None            # None = "tôi xong rồi"; "handoff" = xin bàn giao
    args: dict = field(default_factory=dict)


@dataclass
class PlanStep:
    tool: Literal["search_flights", "book_flight", "cancel_booking", "handoff"]
    args: dict | None = None    # None = chưa biết, phải điền sau khi có kết quả bước trước


# Schema đầu ra có kiểu cho bước lập kế hoạch (không dùng chuỗi JSON)
class _PStep(BaseModel):
    tool: Literal["search_flights", "book_flight", "cancel_booking", "handoff"]
    origin: Optional[str] = Field(None, description="Mã IATA nơi đi (chỉ cho search_flights, nếu đã biết)")
    destination: Optional[str] = Field(None, description="Mã IATA nơi đến (chỉ cho search_flights, nếu đã biết)")
    depart_date: Optional[str] = Field(None, description="YYYY-MM-DD (chỉ cho search_flights, nếu đã biết)")
    booking_id: Optional[str] = Field(None, description="Mã booking (chỉ cho cancel_booking, nếu đã biết)")
    reason: Optional[str] = Field(None, description="Lý do (chỉ cho handoff)")

    def to_plan_step(self) -> PlanStep:
        """Chuyển sang PlanStep; thiếu thông tin thì args=None để điền sau bằng fill_args."""
        if self.tool == "handoff":
            return PlanStep("handoff", {"reason": self.reason or ""})
        if self.tool == "search_flights" and self.origin and self.destination and self.depart_date:
            return PlanStep(self.tool, dict(origin=self.origin, destination=self.destination,
                                            depart_date=self.depart_date))
        if self.tool == "cancel_booking" and self.booking_id:
            return PlanStep(self.tool, dict(booking_id=self.booking_id))
        return PlanStep(self.tool, None)


class _Plan(BaseModel):
    steps: list[_PStep]


# Chuyển lịch sử quan sát thành tin nhắn đúng chuẩn tool calling
def history_messages(hist: list[Obs]) -> list:
    """Mỗi Obs -> AIMessage(tool_calls) + ToolMessage(tool_call_id khớp)."""
    msgs: list = []
    for i, o in enumerate(hist):
        call_id = f"call_{i}"
        msgs.append(AIMessage(content="", tool_calls=[
            {"name": o.tool, "args": o.args, "id": call_id, "type": "tool_call"}]))
        msgs.append(ToolMessage(content=prompts.format_result(o.res), tool_call_id=call_id, name=o.tool))
    return msgs


# Trích token usage từ AIMessage
def _extract_usage(ai: AIMessage) -> dict:
    """Trích token usage từ response_metadata của AIMessage.
    Trả về dict: {input_tokens, output_tokens, total_tokens}.
    Nếu không có thông tin thì trả 0."""
    meta = getattr(ai, "response_metadata", {}) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    return {
        "input_tokens": usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0) or usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


class Brain:
    def __init__(self, model_name: str, allowed_tools: list[str] | None = None):
        self.model = init_chat_model(model_name, **LLM_KWARGS)
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        # Tạo tất cả tool LangChain (business + control), rồi lọc theo allowed_tools
        all_tools = langchain_tools(include_control=True)
        if allowed_tools is not None:
            allowed = set(allowed_tools)
            all_tools = [t for t in all_tools if t.name in allowed]
        self._tools = {t.name: t for t in all_tools}
        self._react_model = self.model.bind_tools(list(self._tools.values()))

    def _track(self, ai: AIMessage) -> None:
        """Cộng dồn token usage sau mỗi lần gọi LLM."""
        u = _extract_usage(ai)
        self.input_tokens += u["input_tokens"]
        self.output_tokens += u["output_tokens"]
        self.total_tokens += u["total_tokens"]

    def _conversation(self, it: Intent, hist: list[Obs], extra: str | None = None) -> list:
        msgs = [SystemMessage(prompts.system_prompt()), HumanMessage(prompts.goal_prompt(it))]
        msgs += history_messages(hist)
        if extra:
            msgs.append(HumanMessage(extra))
        return msgs

    # 4 hàm quyết định
    def parse(self, sc: Scenario) -> Intent:
        self.calls += 1
        structured = self.model.with_structured_output(Intent, include_raw=True)
        result = structured.invoke([
            SystemMessage(prompts.parse_prompt()),
            HumanMessage(prompts.parse_user(sc.user.name, sc.task))])
        if result.get("raw"):
            self._track(result["raw"])
        it = result["parsed"]
        it.passenger = it.passenger or sc.user.name
        return it

    def decide(self, it: Intent, hist: list[Obs]) -> Action:
        self.calls += 1
        ai = self._react_model.invoke(self._conversation(it, hist))
        self._track(ai)
        if not ai.tool_calls:                   # model trả lời bằng chữ = coi như báo xong (harness sẽ kiểm)
            return Action(None)
        tc = ai.tool_calls[0]                   # mỗi vòng chỉ xử lý 1 tool call
        if tc["name"] == "finish":
            return Action(None)
        if tc["name"] == "handoff_to_human":
            return Action("handoff", {"reason": tc["args"].get("reason", "")})
        return Action(tc["name"], tc["args"])

    def plan(self, it: Intent, hist: list[Obs]) -> list[PlanStep]:
        self.calls += 1
        structured = self.model.with_structured_output(_Plan, include_raw=True)
        result = structured.invoke([
            SystemMessage(prompts.system_prompt()), HumanMessage(prompts.goal_prompt(it)),
            HumanMessage(prompts.plan_prompt(hist))])
        if result.get("raw"):
            self._track(result["raw"])
        return [s.to_plan_step() for s in result["parsed"].steps]

    def fill_args(self, it: Intent, tool: str, hist: list[Obs]) -> dict:
        self.calls += 1
        forced = self.model.bind_tools([self._tools[tool]], tool_choice=tool)   # ép gọi đúng tool này
        ai = forced.invoke(self._conversation(it, hist, prompts.fill_args_prompt(tool)))
        self._track(ai)
        return dict(ai.tool_calls[0]["args"]) if ai.tool_calls else {}