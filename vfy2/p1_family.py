"""Independent recomputation 1: family detection counts on TR n=133."""
import pickle, sys, json
from collections import Counter
sys.path.insert(0, "/root/agent_memory_iclr/code")
from v5_temporal import detect_family, resolve_question_anchors, parse_date

tr = pickle.load(open("/root/agent_memory_iclr/vfy2/tr.pkl", "rb"))
fams = Counter()
rows = []
for q in tr:
    fam, unit = detect_family(q["question"])
    fams[fam] += 1
    rows.append({"qid": q["question_id"], "fam": fam, "unit": unit,
                 "question": q["question"], "answer": q["answer"],
                 "qdate": q["question_date"],
                 "n_ans_sess": len(q.get("answer_session_ids", []))})
print(dict(fams))
al = [r for r in rows if r["fam"] == "anchor_lookup"]
print("anchor_lookup n =", len(al))
for r in al:
    print(" ", r["qid"], "| nans", r["n_ans_sess"], "|", r["question"][:95])
    print("      GOLD:", str(r["answer"])[:110])
json.dump(rows, open("/root/agent_memory_iclr/vfy2/fam_rows.json","w"), ensure_ascii=False, indent=1)
