"""
evaluate.py - Đánh giá và so sánh 3 mẫu agent trên 9 kịch bản

Chế độ chạy:
  python evaluate.py                          # đánh giá chuẩn (3 runs)
  python evaluate.py --runs 5                 # tuỳ chỉnh số lần chạy
  python evaluate.py --ablation               # so sánh BẬT/TẮT harness
  python evaluate.py --trace S2 hybrid_agent  # audit log chi tiết 1 lần chạy
  python evaluate.py --json                   # xuất JSON
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime

from brains import Brain
from config import ALLOWED_TOOLS, DEFAULT_MODEL, cfg
from runner import RunResult, run_once
from scenarios import SCENARIOS
from hybrid_agent import run as run_hybrid
from plan_execute_agent import run as run_pte
from react_agent import run as run_react

AGENTS = {"react_agent": run_react, "plan_execute_agent": run_pte, "hybrid_agent": run_hybrid}


def evaluate(model: str, runs: int, enforce: bool = True):
    """Chạy tất cả agent x kịch bản x runs lần, trả về bảng kết quả."""
    from tqdm import tqdm
    table: dict[str, list[RunResult]] = defaultdict(list)
    per_sc: dict[tuple[str, str], list[bool]] = defaultdict(list)

    total_start = time.perf_counter()
    total_iters = len(AGENTS) * len(SCENARIOS) * runs
    pbar = tqdm(total=total_iters, desc=f"Đang chạy ({model})", unit="run", dynamic_ncols=True)
    
    for agent, run_agent in AGENTS.items():
        for sc in SCENARIOS:
            for _ in range(runs):
                r = run_once(run_agent, sc, Brain(model, ALLOWED_TOOLS[agent]), enforce)
                table[agent].append(r)
                per_sc[(agent, sc.sid)].append(r.success)
                pbar.update(1)
    
    pbar.close()
    total_elapsed = time.perf_counter() - total_start

    return table, per_sc, total_elapsed


def _avg(rs: list[RunResult], fn) -> float:
    return sum(fn(r) for r in rs) / max(len(rs), 1)


def print_report(table, per_sc, runs, title, total_elapsed: float = 0):
    """In báo cáo đánh giá chi tiết ra console."""
    sep = "─" * 120

    print(f"\n{'═' * 120}")
    print(f"  {title}")
    print(f"  {len(SCENARIOS)} kịch bản × {runs} lần chạy / agent | Tổng thời gian: {total_elapsed:.1f}s")
    print(f"  Thời điểm: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 120}")

    AGENT_NAMES = {
        "react_agent": "React (Phản xạ)",
        "plan_execute_agent": "Plan-Execute (Lập KH)",
        "hybrid_agent": "Hybrid (Kết hợp)"
    }

    # --- Bảng tổng hợp ---
    header = (f"{'Agent':<28}"
              f"{'Thành công':>11}{'Vi phạm':>9}{'Báo sai':>9}"
              f"{'LLM calls':>11}{'Tool calls':>12}"
              f"{'In tokens':>11}{'Out tokens':>12}{'Σ tokens':>10}"
              f"{'Thời gian(s)':>13}"
              f"{'L1(Dữliệu)':>12}{'L2(Quyền)':>11}{'Lỗi HT':>9}")
    print(f"\n  TỔNG HỢP THEO AGENT (Trung bình {runs} lần chạy/case)")
    print(f"  {sep}")
    print(f"  {header}")
    print(f"  {sep}")

    for agent_name, rs in table.items():
        disp_name = AGENT_NAMES.get(agent_name, agent_name)
        row = (f"{disp_name:<28}"
               f"{_avg(rs, lambda r: r.success):>10.0%}"
               f"{_avg(rs, lambda r: r.violation):>9.0%}"
               f"{_avg(rs, lambda r: r.false_done):>9.0%}"
               f"{_avg(rs, lambda r: r.llm_calls):>11.1f}"
               f"{_avg(rs, lambda r: r.tool_calls):>12.1f}"
               f"{_avg(rs, lambda r: r.input_tokens):>11.0f}"
               f"{_avg(rs, lambda r: r.output_tokens):>12.0f}"
               f"{_avg(rs, lambda r: r.total_tokens):>10.0f}"
               f"{_avg(rs, lambda r: r.elapsed_sec):>13.2f}"
               f"{_avg(rs, lambda r: r.rejected):>12.1f}"
               f"{_avg(rs, lambda r: r.blocked):>11.1f}"
               f"{_avg(rs, lambda r: r.status == 'error'):>9.0%}")
        print(f"  {row}")
    print(f"  {sep}")

    # --- Bảng chi tiết theo kịch bản ---
    print(f"\n  CHI TIẾT TỪNG TEST CASE (Trung bình {runs} lần chạy/case)")
    for a in AGENTS:
        disp_name = AGENT_NAMES.get(a, a)
        print(f"\n  ➤ AGENT: {disp_name}")
        print(f"  {sep}")
        sc_header = (f"{'SID':<6}{'Kịch bản':<50}{'Mong đợi':<12}"
                     f"{'%OK':>8}{'LLM':>6}{'Tool':>6}{'Token':>8}{'Time':>8}")
        print(f"  {sc_header}")
        print(f"  {sep}")
        
        for sc in SCENARIOS:
            key = (a, sc.sid)
            agent_rs = [r for r in table[a] if per_sc.get(key) is not None]
            idx_start = SCENARIOS.index(sc) * runs
            sc_rs = table[a][idx_start:idx_start + runs] if len(table[a]) > idx_start else []

            ok_rate = sum(per_sc[key]) / max(len(per_sc[key]), 1) if key in per_sc else 0
            avg_llm = _avg(sc_rs, lambda r: r.llm_calls) if sc_rs else 0
            avg_tool = _avg(sc_rs, lambda r: r.tool_calls) if sc_rs else 0
            avg_tok = _avg(sc_rs, lambda r: r.total_tokens) if sc_rs else 0
            avg_time = _avg(sc_rs, lambda r: r.elapsed_sec) if sc_rs else 0

            line = (f"{sc.sid:<6}{sc.title[:48]:<50}{sc.expect:<12}"
                    f"{ok_rate:>8.0%}{avg_llm:>6.0f}{avg_tool:>6.0f}{avg_tok:>8.0f}{avg_time:>7.1f}s")
            print(f"  {line}")
        print(f"  {sep}")

    # --- Tổng token ---
    grand_tokens = sum(r.total_tokens for rs in table.values() for r in rs)
    grand_in = sum(r.input_tokens for rs in table.values() for r in rs)
    grand_out = sum(r.output_tokens for rs in table.values() for r in rs)
    print(f"\n  💰 TỔNG TOKEN: input={grand_in:,}  output={grand_out:,}  total={grand_tokens:,}")
    print()


def results_to_json(table, per_sc, runs, model, total_elapsed):
    """Chuyển kết quả sang dict để xuất JSON."""
    out = {
        "model": model,
        "runs_per_scenario": runs,
        "total_scenarios": len(SCENARIOS),
        "total_elapsed_sec": round(total_elapsed, 2),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "provider": cfg["llm"]["provider"],
            "model": cfg["llm"]["model"],
            "temperature": cfg["llm"]["temperature"],
            "max_tokens": cfg["llm"].get("max_tokens"),
            "base_url": cfg["llm"].get("base_url") or "(default)",
        },
        "agents": {},
    }
    for agent_name, rs in table.items():
        agent_data = {
            "summary": {
                "success_rate": round(_avg(rs, lambda r: r.success), 3),
                "violation_rate": round(_avg(rs, lambda r: r.violation), 3),
                "false_done_rate": round(_avg(rs, lambda r: r.false_done), 3),
                "error_rate": round(_avg(rs, lambda r: r.status == "error"), 3),
                "avg_llm_calls": round(_avg(rs, lambda r: r.llm_calls), 2),
                "avg_tool_calls": round(_avg(rs, lambda r: r.tool_calls), 2),
                "avg_input_tokens": round(_avg(rs, lambda r: r.input_tokens)),
                "avg_output_tokens": round(_avg(rs, lambda r: r.output_tokens)),
                "avg_total_tokens": round(_avg(rs, lambda r: r.total_tokens)),
                "avg_elapsed_sec": round(_avg(rs, lambda r: r.elapsed_sec), 2),
                "avg_rejected": round(_avg(rs, lambda r: r.rejected), 2),
                "avg_blocked": round(_avg(rs, lambda r: r.blocked), 2),
            },
            "per_scenario": {},
        }
        for sc in SCENARIOS:
            key = (agent_name, sc.sid)
            idx_start = SCENARIOS.index(sc) * runs
            sc_rs = rs[idx_start:idx_start + runs] if len(rs) > idx_start else []
            ok_rate = sum(per_sc[key]) / max(len(per_sc[key]), 1) if key in per_sc else 0
            
            runs_detail = []
            for i, r in enumerate(sc_rs, 1):
                runs_detail.append({
                    "run_index": i,
                    "success": r.success,
                    "status": r.status,
                    "violation": r.violation,
                    "false_done": r.false_done,
                    "total_tokens": r.total_tokens,
                    "elapsed_sec": r.elapsed_sec,
                    "audit_log": r.audit
                })

            agent_data["per_scenario"][sc.sid] = {
                "title": sc.title,
                "expect": sc.expect,
                "success_rate": round(ok_rate, 3),
                "avg_llm_calls": round(_avg(sc_rs, lambda r: r.llm_calls), 2) if sc_rs else 0,
                "avg_tool_calls": round(_avg(sc_rs, lambda r: r.tool_calls), 2) if sc_rs else 0,
                "avg_total_tokens": round(_avg(sc_rs, lambda r: r.total_tokens)) if sc_rs else 0,
                "avg_elapsed_sec": round(_avg(sc_rs, lambda r: r.elapsed_sec), 2) if sc_rs else 0,
                "runs_detail": runs_detail
            }
        out["agents"][agent_name] = agent_data
    return out


def main():
    ap = argparse.ArgumentParser(description="Đánh giá và so sánh 3 mẫu agent")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="dạng nhà-cung-cấp:tên-model, mặc định lấy từ config.yaml")
    ap.add_argument("--runs", type=int, default=3,
                    help="số lần chạy mỗi kịch bản (mỗi lần tốn nhiều lần gọi LLM)")
    ap.add_argument("--ablation", action="store_true",
                    help="so sánh BẬT/TẮT harness")
    ap.add_argument("--trace", nargs=2, metavar=("SID", "AGENT"),
                    help="vd S2 hybrid_agent — in audit log chi tiết 1 lần chạy")
    ap.add_argument("--json", dest="json_output", action="store_true",
                    help="xuất kết quả dạng JSON thay vì bảng text")
    a = ap.parse_args()

    # --- Chế độ trace: chạy 1 lần và in audit chi tiết ---
    if a.trace:
        sc = next(s for s in SCENARIOS if s.sid == a.trace[0])
        brain = Brain(a.model, ALLOWED_TOOLS[a.trace[1]])
        r = run_once(AGENTS[a.trace[1]], sc, brain)
        print(f"\n{'═' * 100}")
        print(f"  TRACE: {sc.sid} - {sc.title}")
        print(f"  Agent: {a.trace[1]} | Model: {a.model}")
        print(f"  Status: {r.status} | Success: {r.success} | Violation: {r.violation}")
        print(f"  LLM calls: {r.llm_calls} | Tool calls: {r.tool_calls}")
        print(f"  Tokens: in={r.input_tokens}  out={r.output_tokens}  total={r.total_tokens}")
        print(f"  Time: {r.elapsed_sec}s")
        print(f"{'═' * 100}")
        print(f"\n  AUDIT LOG ({len(r.audit)} entries):")
        for i, e in enumerate(r.audit, 1):
            print(f"  {i:>2}. [{e['code']:<16}] {e['tool']}({json.dumps(e['args'], ensure_ascii=False)})")
            print(f"      → {e['detail']}")
        print()
        return

    # --- Chế độ đánh giá ---
    table, per_sc, elapsed = evaluate(a.model, a.runs, enforce=True)

    data = results_to_json(table, per_sc, a.runs, a.model, elapsed)
    if a.ablation:
        table2, per_sc2, elapsed2 = evaluate(a.model, a.runs, enforce=False)
        data["ablation"] = results_to_json(table2, per_sc2, a.runs, a.model, elapsed2)

    # Lưu kết quả vào thư mục output
    import os
    os.makedirs("output", exist_ok=True)
    filename = f"output/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    if a.json_output:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    print_report(table, per_sc, a.runs, f"HARNESS BẬT | {a.model}", elapsed)
    if a.ablation:
        print_report(table2, per_sc2, a.runs, f"HARNESS TẮT (ablation) | {a.model}", elapsed2)
    print(f"  📝 Đã lưu log kết quả chi tiết (JSON) tại: {filename}\n")


if __name__ == "__main__":
    main()