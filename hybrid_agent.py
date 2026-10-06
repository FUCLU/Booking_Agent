"""
hybrid_agent.py - Mẫu Lai (Plan + ReAct cục bộ + Replan)

PLAN -> EXECUTE (mỗi bước có retry cục bộ bằng fill_args) -> nếu lỗi không sửa được: REPLAN.
Cân bằng giữa kế hoạch tổng thể và khả năng phục hồi lỗi.
"""
from __future__ import annotations

from brains import Brain
from config import HYBRID_MAX_REPLANS, HYBRID_MAX_RETRIES, HYBRID_RETRYABLE
from harness import Harness, Intent, Obs
from runner import Outcome, demo


def run(brain: Brain, h: Harness, it: Intent) -> Outcome:
    hist: list[Obs] = []
    replans = 0
    plan = brain.plan(it, hist)                        # PLAN

    while True:
        need_replan = False
        for st in plan:                                # EXECUTE từng bước
            if st.tool == "handoff":
                return Outcome("handoff", (st.args or {}).get("reason", ""), hist)

            retries, use_plan_args = 0, st.args is not None
            while True:                                # vòng ReAct cục bộ cho 1 bước
                args = st.args if use_plan_args else brain.fill_args(it, st.tool, hist)
                res = h.run_tool(st.tool, args)
                hist.append(Obs(st.tool, args, res))
                if res.handoff:
                    return Outcome("handoff", res.handoff, hist)
                if res.code == "BUDGET":
                    return Outcome("failed", res.error or "", hist)
                if res.ok:
                    break
                if res.code in HYBRID_RETRYABLE and retries < HYBRID_MAX_RETRIES:
                    retries, use_plan_args = retries + 1, False   # sửa tham số dựa trên lỗi vừa thấy
                    continue
                need_replan = True                     # lỗi không sửa cục bộ được (vd SOLD_OUT)
                break

            if not need_replan and st.tool == "search_flights" and not res.data:
                need_replan = True                     # search rỗng: kế hoạch cũ không còn đúng
            if need_replan:
                break

        if not need_replan:                            # VERIFY bằng code
            ok, _ = h.check_done(it)
            if ok:
                return Outcome("finished", "", hist)
        if replans >= HYBRID_MAX_REPLANS:
            return Outcome("failed", "vượt số lần replan", hist)
        replans += 1
        plan = brain.plan(it, hist)                    # REPLAN dựa trên những gì đã quan sát


if __name__ == "__main__":
    demo(run, "hybrid_agent")