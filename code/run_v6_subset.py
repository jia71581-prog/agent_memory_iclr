"""门控子集 runner:只跑某个 v6 flag 门内命中的题,与 v5 在同题上做配对比较。

依据:v6_gates.py 已逐字节验证门外题的上下文与 v5 完全相同,故门外结果可直接复用 v5,
只有门内题需要在线重跑。成本因此从 500 题降到门控命中数(AGG 175 / SSP 29 / TR 179)。

用法: QC_V5=1 QC_V6_AGG=1 python3 run_v6_subset.py QC_V6_AGG <workers=8>
结果写 results/v6_<FLAG>.jsonl(断点续跑:已有 judge 行的 qid 跳过)。
跑完自动打印与 v5 的配对 McNemar。
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset, token_f1, llm_judge, CostLedger

R = "/root/agent_memory_iclr/results"
FLAG = sys.argv[1]
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
OUT = f"{R}/v6_{FLAG}.jsonl"
_lock = threading.Lock()

import run_qc

if not (run_qc.V6_AGG or run_qc.V6SSP or run_qc.V6_TR):
    sys.exit(f"错误:{FLAG} 未生效,请通过环境变量开启(当前 V5={run_qc.V5})")


def process(q, ledger):
    ans, label, mode = run_qc.answer_question(q, ledger)
    f1 = token_f1(ans, q["answer"])
    correct = llm_judge(q["question"], q["question_type"], q["answer"], ans, ledger=ledger)
    return {"qid": q["question_id"], "type": q["question_type"], "judge": bool(correct),
            "f1": round(f1, 3), "qlabel": label, "view_mode": mode, "pred": ans[:4000]}


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def main():
    gates = json.load(open(f"{R}/v6_gates.json"))
    key = FLAG if FLAG in gates else FLAG + "2"
    fired = set(gates[key]["fired"])
    data = {q["question_id"]: q for q in load_dataset("s_cleaned")}
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "judge" in r:
                done.add(r["qid"])
    todo = [data[x] for x in sorted(fired) if x not in done]
    print(f"{FLAG}: 门控命中 {len(fired)}, 已完成 {len(done)}, 待跑 {len(todo)}", flush=True)
    ledger = CostLedger()
    n = 0
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
            n += 1
            if n % 25 == 0:
                print(f"progress {n}/{len(todo)}", flush=True)
    ledger.save(f"{R}/v6_{FLAG}_ledger_{int(time.time())}.json")

    # 与 v5 配对比较(门内子集)
    v6 = {}
    for line in open(OUT):
        r = json.loads(line)
        if "judge" in r:
            v6[r["qid"]] = r
    v5 = {}
    for line in open(f"{R}/full500_qc_v5.jsonl"):
        r = json.loads(line)
        if "judge" in r:
            v5[r["qid"]] = r
    common_q = sorted(set(v6) & set(v5) & fired)
    b = sum(1 for x in common_q if v6[x]["judge"] and not v5[x]["judge"])
    c = sum(1 for x in common_q if not v6[x]["judge"] and v5[x]["judge"])
    a5 = sum(v5[x]["judge"] for x in common_q) / max(1, len(common_q))
    a6 = sum(v6[x]["judge"] for x in common_q) / max(1, len(common_q))
    print(f"\n=== {FLAG} 门内子集 n={len(common_q)} ===")
    print(f"v5 {a5:.4f} -> v6 {a6:.4f}  delta {a6-a5:+.4f}  b={b} c={c}  McNemar p={mcnemar(b,c):.4f}")
    # 折算到全 500
    tot5 = sum(v5[x]["judge"] for x in v5) / len(v5)
    proj = (sum(v5[x]["judge"] for x in v5) - sum(v5[x]["judge"] for x in common_q)
            + sum(v6[x]["judge"] for x in common_q)) / len(v5)
    print(f"折算全 500: v5 {tot5:.4f} -> {proj:.4f} ({proj-tot5:+.4f})")
    by = {}
    for x in common_q:
        s = by.setdefault(v5[x]["type"], [0, 0, 0])
        s[0] += 1
        s[1] += v5[x]["judge"]
        s[2] += v6[x]["judge"]
    print(f"{'type':26s} {'n':>4s} {'v5':>7s} {'v6':>7s}")
    for t, (nn, s5, s6) in sorted(by.items()):
        print(f"{t:26s} {nn:4d} {s5/nn:7.3f} {s6/nn:7.3f}")
    errs = [1 for line in open(OUT) if '"error"' in line]
    if errs:
        print(f"!!! 有错误行,重跑本命令可续跑")
    print(f"V6 SUBSET {FLAG} DONE", flush=True)


if __name__ == "__main__":
    main()
