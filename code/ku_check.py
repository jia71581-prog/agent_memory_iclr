"""KU 子类快速验证:只跑 60 题样本中的 knowledge-update 10 题(v3 修复:抽取召回+原文优先)。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset, token_f1, llm_judge, CostLedger
from run_baseline import stratified_sample
from run_qc import answer_question

data = load_dataset("s_cleaned")
sample = [q for q in stratified_sample(data, 60) if q["question_type"] == "knowledge-update"]
ledger = CostLedger()
results = []
for i, q in enumerate(sample):
    try:
        ans, label, mode = answer_question(q, ledger)
        f1 = token_f1(ans, q["answer"])
        correct = llm_judge(q["question"], q["question_type"], q["answer"], ans, ledger=ledger)
    except Exception as e:
        print(f"[{i+1}] ERROR: {e}", flush=True)
        continue
    results.append({"qid": q["question_id"], "judge": bool(correct), "f1": round(f1, 3),
                    "gold": str(q["answer"])[:60], "pred": ans[:150]})
    print(f"[{i+1}/{len(sample)}] judge={'Y' if correct else 'N'} gold={str(q['answer'])[:40]!r} pred={ans[:60]!r}", flush=True)

acc = sum(r["judge"] for r in results) / max(1, len(results))
print(f"\nKU v3 acc = {acc:.2f} (v2 was 0.70, full baseline 0.90)")
with open("/root/agent_memory_iclr/results/ku_v3_check.json", "w") as f:
    json.dump({"acc": acc, "results": results, "cost": ledger.summary()}, f, ensure_ascii=False, indent=1)
print("KU CHECK DONE")
