"""Independent: dump the TR questions (full haystack) to a pickle for repeated offline use."""
import json, pickle, sys
sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset

data = load_dataset("s_cleaned")
print("total", len(data))
from collections import Counter
print(Counter(q["question_type"] for q in data))
tr = [q for q in data if q["question_type"] == "temporal-reasoning"]
print("TR", len(tr))
with open("/root/agent_memory_iclr/vfy2/tr.pkl", "wb") as f:
    pickle.dump(tr, f)
# also dump the oracle answer sessions text for the TR qids
orc = {q["question_id"]: q for q in load_dataset("oracle")}
out = {}
for q in tr:
    o = orc.get(q["question_id"], {})
    out[q["question_id"]] = {
        "answer_session_ids": q.get("answer_session_ids", []),
        "hay_ids": q.get("haystack_session_ids", []),
        "oracle_sessions": [(sid, sess) for sid, sess in zip(o.get("haystack_session_ids", []),
                                                             o.get("haystack_sessions", []))
                            if sid in set(q.get("answer_session_ids", []))],
    }
with open("/root/agent_memory_iclr/vfy2/tr_gold.pkl", "wb") as f:
    pickle.dump(out, f)
print("saved", len(out))
