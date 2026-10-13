"""OFFLINE prep pass 2: qid -> answer_session_ids + haystack_session_ids (-> gold sidx list),
plus the oracle gold-session texts. No API."""
import json
import pickle

S = "/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"
O = "/root/agent_memory_iclr/data/longmemeval_oracle.json"
OUT = "/root/agent_memory_iclr/results/_v5_gold.pkl"

oracle = {q["question_id"]: q for q in json.load(open(O))}
data = json.load(open(S))
out = {}
for q in data:
    hs = q.get("haystack_session_ids", [])
    ans = q.get("answer_session_ids", [])
    gold_sidx = [i for i, sid in enumerate(hs) if sid in set(ans)]
    orc = oracle.get(q["question_id"], {})
    gold_txt = []
    for sid, sess in zip(orc.get("haystack_session_ids", []), orc.get("haystack_sessions", [])):
        if sid in set(ans):
            gold_txt.append((sid, "\n".join(f"[{t['role']}] {t.get('content','')}" for t in sess)))
    out[q["question_id"]] = {"answer_session_ids": ans, "gold_sidx": gold_sidx,
                             "n_hay": len(hs), "gold_sessions": gold_txt}
with open(OUT, "wb") as f:
    pickle.dump(out, f)
print("saved", OUT, len(out), "median gold sessions",
      sorted(len(v["gold_sidx"]) for v in out.values())[len(out) // 2])
