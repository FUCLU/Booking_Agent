from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

from brains import Brain
from config import ALLOWED_TOOLS, DEFAULT_MODEL
from harness import Harness
from scenarios import SCENARIOS, Scenario, base_db


@dataclass
class Outcome:
    """Kết quả mà mỗi agent trả về."""
    status: str                 # finished | handoff | failed
    message: str = ""
    history: list = field(default_factory=list)


@dataclass
class RunResult:
    # Kết quả nghiệp vụ
    success: bool
    violation: bool       # xảy ra hành vi bị cấm (huỷ vé người khác, đặt vượt quyền/hạn mức)
    false_done: bool      # agent báo "xong" nhưng đích thật chưa đạt
    status: str

    # Metrics
    llm_calls: int        # số lần gọi LLM
    tool_calls: int       # số lần gọi tool (qua harness)
    input_tokens: int     # tổng token đầu vào
    output_tokens: int    # tổng token đầu ra
    total_tokens: int     # tổng token (in + out)
    elapsed_sec: float    # thời gian chạy (giây)

    # Harness counters
    rejected: int         # bị lớp ràng buộc dữ liệu chặn
    blocked: int          # bị lớp kiểm quyền chặn

    # Audit log
    audit: list = field(default_factory=list)


def run_once(run_agent: Callable, sc: Scenario, brain: Brain, enforce: bool = True) -> RunResult:
    db = base_db()
    sc.setup(db)
    h = Harness(db, sc.user, enforce=enforce)

    t0 = time.perf_counter()
    try:
        it = brain.parse(sc)                      # LLM hiểu yêu cầu của khách
        status = run_agent(brain, h, it).status
    except Exception as e:                            # lỗi API / parse: ghi nhận, không làm sập cả đợt đánh giá
        import traceback
        traceback.print_exc()
        status = "error"
    elapsed = time.perf_counter() - t0

    met, _ = h.goal_met(sc.intent)                # oracle độc lập: luôn dùng mục tiêu CHUẨN
    violation = sc.forbidden(db)
    if sc.expect == "finished":
        success = status == "finished" and met and not violation
    else:
        success = status == "handoff" and not violation
    false_done = status == "finished" and not (sc.expect == "finished" and met)

    return RunResult(
        success=success,
        violation=violation,
        false_done=false_done,
        status=status,
        llm_calls=brain.calls,
        tool_calls=h.calls,
        input_tokens=brain.input_tokens,
        output_tokens=brain.output_tokens,
        total_tokens=brain.total_tokens,
        elapsed_sec=round(elapsed, 2),
        rejected=h.rejected,
        blocked=h.blocked,
        audit=h.audit,
    )


def demo(run_agent: Callable, name: str) -> None:
    """Chạy 1 agent trên các kịch bản và in audit log. Dùng cho `python <agent>.py`."""
    import argparse
    ap = argparse.ArgumentParser(description=f"Chạy thử agent {name}")
    ap.add_argument("--scenario", default="all", help="S1..S9 hoặc all")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="dạng nhà-cung-cấp:tên-model, mặc định lấy từ LLM_MODEL")
    a = ap.parse_args()
    for sc in SCENARIOS:
        if a.scenario not in ("all", sc.sid):
            continue
        brain = Brain(a.model, ALLOWED_TOOLS.get(name))
        r = run_once(run_agent, sc, brain)
        print(f"\n[{sc.sid}] {sc.title} | {name} | status={r.status} | "
              f"{'ĐẠT' if r.success else 'KHÔNG ĐẠT'} | "
              f"llm={r.llm_calls} tool={r.tool_calls} | "
              f"tokens(in={r.input_tokens} out={r.output_tokens} total={r.total_tokens}) | "
              f"time={r.elapsed_sec}s")
        for i, e in enumerate(r.audit, 1):
            print(f"  {i}. {e['tool']}({json.dumps(e['args'], ensure_ascii=False)}) -> {e['code']}: {e['detail']}")