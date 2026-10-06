from __future__ import annotations

from brains import Brain
from config import REACT_MAX_NOT_DONE
from harness import Harness, Intent, Obs, ToolResult
from runner import Outcome, demo


def run(brain: Brain, h: Harness, it: Intent) -> Outcome:
    hist: list[Obs] = []
    while True:
        a = brain.decide(it, hist)                     # THINK: LLM chọn bước kế tiếp

        if a.tool == "handoff":                        # LLM tự xin bàn giao
            return Outcome("handoff", a.args.get("reason", ""), hist)

        if a.tool is None:                             # LLM nói "xong" -> harness kiểm bằng code
            ok, why = h.check_done(it)
            if ok:
                return Outcome("finished", "", hist)
            hist.append(Obs("finish", {}, ToolResult(False, code="NOT_DONE",
                                                     error=f"COMPLETION_CHECK_FAILED: {why}")))
            if sum(o.res.code == "NOT_DONE" for o in hist) >= REACT_MAX_NOT_DONE:
                return Outcome("failed", why, hist)
            continue

        res = h.run_tool(a.tool, a.args)               # ACT (qua 4 lớp harness)
        hist.append(Obs(a.tool, a.args, res))          # OBSERVE
        if res.handoff:
            return Outcome("handoff", res.handoff, hist)
        if res.code == "BUDGET":
            return Outcome("failed", res.error or "", hist)


if __name__ == "__main__":
    demo(run, "react_agent")