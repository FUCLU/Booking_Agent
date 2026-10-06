from __future__ import annotations

from brains import Brain
from harness import Harness, Intent, Obs
from runner import Outcome, demo


def run(brain: Brain, h: Harness, it: Intent) -> Outcome:
    hist: list[Obs] = []
    plan = brain.plan(it, [])                          # PLAN: 1 lần duy nhất

    for st in plan:                                    # EXECUTE: chạy tuần tự
        if st.tool == "handoff":
            return Outcome("handoff", (st.args or {}).get("reason", ""), hist)
        args = st.args if st.args is not None else brain.fill_args(it, st.tool, hist)
        res = h.run_tool(st.tool, args)
        hist.append(Obs(st.tool, args, res))
        if res.handoff:
            return Outcome("handoff", res.handoff, hist)
        if not res.ok:                                 # bước lỗi -> cả kế hoạch hỏng
            return Outcome("failed", f"{st.tool}: {res.code} - {res.error}", hist)

    ok, why = h.check_done(it)                         # VERIFY bằng code
    return Outcome("finished" if ok else "failed", why, hist)


if __name__ == "__main__":
    demo(run, "plan_execute_agent")