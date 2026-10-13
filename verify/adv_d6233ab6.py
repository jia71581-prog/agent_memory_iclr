"""ADVERSARIAL recompute #3: end-to-end trace of d6233ab6 and the raw-turn channel."""
import json, os, hashlib, pickle, sys, re
sys.path.insert(0, "/root/agent_memory_iclr/code")
import lifecycle as LC
from run_baseline import bm25_rank

data = {q["question_id"]: q for q in json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))}
views = pickle.load(open("/root/agent_memory_iclr/verify/_views.pkl", "rb"))
qc = {r["qid"]: r for r in (json.loads(l) for l in open("/root/agent_memory_iclr/results/full500_qc.jsonl"))}

QID = "d6233ab6"
q = data[QID]
V = views[QID]
print("question:", q["question"])
print("question_date:", q["question_date"])
print("gold rubric:", q["answer"][:900])
print("view mode:", V["mode"], "| full chars", len(V["full"]), "| capped chars", len(V["cap"]),
      "| lines", len(V["full"].splitlines()), "->", len(V["cap"].splitlines()))

pat = re.compile(r"advanced placement|debate|high school", re.I)
print("\n--- lines matching 'advanced placement|debate|high school' in FULL view ---")
for l in V["full"].splitlines():
    if pat.search(l):
        print("  FULL:", l[:220], "   [IN CAP]" if l in V["cap"].splitlines() else "   [DROPPED BY CAP]")

# production raw-turn channel (run_qc.py: BM25 top-20 turns over all units)
units = []
for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
    for t in sess:
        if t.get("content"):
            units.append((date, t["role"], t["content"]))
order = sorted(bm25_rank(q["question"], [u[2] for u in units])[:20])
raw_txt = "\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in order)
print(f"\nraw top-20 chars {len(raw_txt)}")
for key in ["advanced placement", "economics", "debate", "high school"]:
    print(f"  '{key}': full_view={key in V['full'].lower()} capped_view={key in V['cap'].lower()} "
          f"raw20={key in raw_txt.lower()}")

# where does the gold evidence turn live, and is it retrievable at all?
sid2 = dict(zip(q["haystack_session_ids"], q["haystack_sessions"]))
ev = [t["content"] for sid in q["answer_session_ids"] for t in sid2.get(sid, []) if t.get("has_answer") and t.get("content")]
print("\ngold evidence turns:", len(ev))
for e in ev:
    print("  EV:", e[:400].replace("\n", " "))
    print("     in raw20:", e in raw_txt, "| bm25 rank of that turn:",
          [i for i, u in enumerate(units) if u[2] == e])
rank_order = bm25_rank(q["question"], [u[2] for u in units])
pos = {u: i for i, u in enumerate(rank_order)}
for e in ev:
    idxs = [i for i, u in enumerate(units) if u[2] == e]
    print("     bm25 position(s):", [pos[i] for i in idxs], "of", len(units))

print("\n--- prediction (qc arm) ---")
print(qc[QID]["judge"], qc[QID]["f1"])
print(qc[QID]["pred"][:1500])
