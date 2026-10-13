"""全量 500 题并发 runner:题级线程池,支持基线与 QC 全部系统,断点续跑。

用法: python3 run_full500.py <system> [workers=8]
  system ∈ {full, fifo, decay, qc, qc_no_qc, qc_view_only, qc_no_lifecycle}
结果逐题追加写 results/full500_<system>.jsonl。
断点续跑:只有带 judge 的成功行计入已完成;错误行下轮重试(会话缓存使重试廉价)。
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset, token_f1, llm_judge, CostLedger

SYSTEM = sys.argv[1]
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
TAG = sys.argv[3] if len(sys.argv) > 3 else ""  # 口径变更时用新 tag,避免与旧结果混写
OUT = f"/root/agent_memory_iclr/results/full500_{SYSTEM}{TAG}.jsonl"
_lock = threading.Lock()

if SYSTEM.startswith("qc"):
    import run_qc
    run_qc.ABLATE = {"qc": None, "qc_no_qc": "no_qc", "qc_view_only": "view_only",
                     "qc_no_lifecycle": "no_lifecycle"}[SYSTEM]
    def answer(q, ledger):
        ans, label, mode = run_qc.answer_question(q, ledger)
        return ans, {"qlabel": label, "view_mode": mode}
else:
    import run_baseline
    def answer(q, ledger):
        ans, mem_units = run_baseline.answer_question(q, SYSTEM, ledger)
        return ans, {"mem_units": mem_units}


def process(q, ledger):
    ans, extra = answer(q, ledger)
    f1 = token_f1(ans, q["answer"])
    correct = llm_judge(q["question"], q["question_type"], q["answer"], ans, ledger=ledger)
    return {"qid": q["question_id"], "type": q["question_type"],
            "judge": bool(correct), "f1": round(f1, 3), **extra, "pred": ans[:4000]}


def main():
    data = load_dataset("s_cleaned")
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if "judge" in r:  # 错误行不算完成,续跑时重试
                    done.add(r["qid"])
    todo = [q for q in data if q["question_id"] not in done]
    print(f"{SYSTEM}: total {len(data)}, done {len(done)}, todo {len(todo)}", flush=True)
    ledger = CostLedger()
    n_ok = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(process, q, ledger): q for q in todo}
        for fut in as_completed(futs):
            q = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"qid": q["question_id"], "type": q["question_type"], "error": str(e)[:200]}
            with _lock:
                with open(OUT, "a") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_ok += 1
            if n_ok % 25 == 0:
                print(f"progress {n_ok}/{len(todo)}", flush=True)
    ledger.save(f"/root/agent_memory_iclr/results/full500_{SYSTEM}{TAG}_ledger_{int(time.time())}.json")
    # 汇总:同 qid 取最后一次成功行
    latest = {}
    with open(OUT) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "judge" in r:
                latest[r["qid"]] = r
            elif r.get("qid") not in latest:
                latest[r["qid"]] = r
    rows = list(latest.values())
    ok = [r for r in rows if "judge" in r]
    errs = [r for r in rows if "judge" not in r]
    by = {}
    for r in ok:
        s = by.setdefault(r["type"], [0, 0])
        s[0] += 1
        s[1] += r["judge"]
    print(f"=== {SYSTEM} full500 ===")
    for t, (n, c) in sorted(by.items()):
        print(f"{t:28s} n={n:3d} acc={c/n:.3f}")
    print(f"TOTAL n={len(ok)} acc={sum(r['judge'] for r in ok)/max(1,len(ok)):.3f}")
    if errs:
        print(f"!!! WARNING: {len(errs)} unresolved error rows (n<{len(data)}): {[r['qid'] for r in errs][:10]}")
        print("!!! 重跑本命令可续跑这些题;主表 n 必须为 500。")
    print(f"FULL500 {SYSTEM} DONE", flush=True)


if __name__ == "__main__":
    main()
