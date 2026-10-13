import json, re
R = "/root/agent_memory_iclr/results"
rows = json.load(open(f"{R}/_adv6_rows3.json"))
QC = {}
for l in open(f"{R}/full500_qc.jsonl"):
    d = json.loads(l)
    if "judge" in d:
        QC[d["qid"]] = d
idx = {r["qid"]: r for r in rows}
WANT = ["88432d0a", "gpt4_2f8be40d", "gpt4_7fce9456", "gpt4_59c863d7", "6d550036",
        "e3038f8c", "6cb6f249", "67e0d0f2", "4f54b7c9", "3a704032", "7024f17c",
        "gpt4_731e37d7", "d851d5ba", "80ec1f4f_abs"]
for q in WANT:
    r = idx[q]
    print(f"--- {q}  type={r['type']} judge={QC[q]['judge']} f1={QC[q].get('f1')} n_ev={r['n_ev']} "
          f"union={r['union_dv']} maxg={r['max_dv']} goldsesshit={r['gold_sess_hit']} kind={r['kind']} shape={r['shape']}")
    print(f"    Q: {r['question']}")
    print(f"    GOLD: {r['gold']}")
    print(f"    PRED: {(QC[q].get('pred') or '')[:420]}")
