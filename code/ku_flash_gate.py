"""KU 门(flash 答题口径):同 10 题 knowledge-update,qc 臂与 full 基线臂同口径对照。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset, token_f1, llm_judge, CostLedger, ANSWER_MODEL
from run_baseline import stratified_sample, answer_question as baseline_answer
from run_qc import answer_question as qc_answer

data = load_dataset("s_cleaned")
sample = [q for q in stratified_sample(data, 60) if q["question_type"] == "knowledge-update"]
ledger = CostLedger()
out = {"answer_model": ANSWER_MODEL, "arms": {}}
for arm in ("qc", "full"):
    results = []
    for i, q in enumerate(sample):
        try:
            if arm == "qc":
                ans, _, _ = qc_answer(q, ledger)
            else:
                ans, _ = baseline_answer(q, "full", ledger)
            f1 = token_f1(ans, q["answer"])
            correct = llm_judge(q["question"], q["question_type"], q["answer"], ans, ledger=ledger)
        except Exception as e:
            print(f"[{arm} {i+1}] ERROR: {e}", flush=True)
            continue
        results.append({"qid": q["question_id"], "judge": bool(correct), "f1": round(f1, 3),
                        "pred": ans[:200]})
        print(f"[{arm} {i+1}/{len(sample)}] judge={'Y' if correct else 'N'} gold={str(q['answer'])[:40]!r} pred={ans[:60]!r}", flush=True)
    acc = sum(r["judge"] for r in results) / max(1, len(results))
    out["arms"][arm] = {"acc": acc, "results": results}
    print(f"== {arm} KU acc = {acc:.2f} ==", flush=True)

out["cost"] = ledger.summary()
with open("/root/agent_memory_iclr/results/ku_flash_gate.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("KU FLASH GATE DONE:", {a: round(v["acc"], 2) for a, v in out["arms"].items()})
