"""Per-evidence-turn retrieval flags for every error qid, both arms. No API calls."""
import json, os, sys
sys.path.insert(0, "/root/agent_memory_iclr/code")
sys.path.insert(0, "/root/agent_memory_iclr/results")
import _recon_ctx as RC
from run_baseline import bm25_rank

RES = "/root/agent_memory_iclr/results"
qc = {r["qid"]: r for r in (json.loads(l) for l in open(f"{RES}/full500_qc.jsonl"))}
fu = {r["qid"]: r for r in (json.loads(l) for l in open(f"{RES}/full500_full.jsonl"))}
need = sorted({k for k, r in qc.items() if not r["judge"]} | {k for k, r in fu.items() if not r["judge"]})
data = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
byq = {q["question_id"]: q for q in data}
out = {}
for qid in need:
    q = byq[qid]
    units = RC.units_of(q)
    docs = [u[2] for u in units]
    r = bm25_rank(q["question"], docs)
    top20 = {docs[i] for i in r[:20]}
    top30 = {docs[i] for i in r[:30]}
    ev = RC.evidence_turns(q)
    out[qid] = {
        "n_ev": len(ev),
        "ev_hit20": [e in top20 for e in ev],
        "ev_hit30": [e in top30 for e in ev],
        "ev_head": [" ".join(e.split())[:160] for e in ev],
    }
json.dump(out, open(f"{RES}/_ev_hits.json", "w"), ensure_ascii=False, indent=1)
print("saved", len(out))
