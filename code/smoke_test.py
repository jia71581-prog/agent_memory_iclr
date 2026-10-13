"""冒烟测试:oracle 版(仅含答案会话)上跑通 读答案会话→回答→双指标判分→成本记账 全链路。
用法: python3 smoke_test.py [N题,默认10]
"""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset, chat, token_f1, llm_judge, CostLedger

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10

data = load_dataset("oracle")
# 按题型分层取样,保证覆盖 knowledge-update / temporal-reasoning / abstention
by_type = {}
for q in data:
    by_type.setdefault(q["question_type"], []).append(q)
sample = []
while len(sample) < N:
    for t, qs in by_type.items():
        if qs and len(sample) < N:
            sample.append(qs.pop(0))

ledger = CostLedger()
results = []
for i, q in enumerate(sample):
    ctx_parts = []
    for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        turns = "\n".join(f"[{t['role']}] {t['content']}" for t in sess)
        ctx_parts.append(f"=== Session ({date}) ===\n{turns}")
    ctx = "\n\n".join(ctx_parts)
    ans = chat([
        {"role": "system", "content": "Answer the question based on the conversation history. Be concise. If the history contains no relevant information, say you don't know."},
        {"role": "user", "content": f"Conversation history:\n{ctx}\n\nQuestion (asked on {q['question_date']}): {q['question']}"},
    ], model="deepseek-v4-pro", max_tokens=2048, ledger=ledger, op="answer")
    f1 = token_f1(ans, q["answer"])
    correct = llm_judge(q["question"], q["question_type"], q["answer"], ans, ledger=ledger)
    results.append({"qid": q["question_id"], "type": q["question_type"],
                    "gold": str(q["answer"])[:80], "pred": ans[:150],
                    "f1": round(f1, 3), "judge": correct})
    print(f"[{i+1}/{N}] {q['question_type'][:20]:20s} judge={'Y' if correct else 'N'} f1={f1:.2f}")

acc = sum(r["judge"] for r in results) / len(results)
mf1 = sum(r["f1"] for r in results) / len(results)
print(f"\njudge 准确率: {acc:.2f} | 平均 F1: {mf1:.3f}")
print("成本:", json.dumps(ledger.summary(), indent=1))
os.makedirs("/root/agent_memory_iclr/results", exist_ok=True)
with open("/root/agent_memory_iclr/results/smoke_test.json", "w") as f:
    json.dump({"results": results, "cost": ledger.summary()}, f, ensure_ascii=False, indent=1)
ledger.save("/root/agent_memory_iclr/results/smoke_test_ledger.json")
