"""Rebuild the two view renderings offline from the session cache (no API) and measure
how often the qc-mode view text actually differs from the no_qc-mode view text."""
import json, os, sys, hashlib
sys.path.insert(0,"/root/agent_memory_iclr/code")
import common
_calls={"n":0}
def _no_api(*a,**k):
    _calls["n"]+=1
    raise RuntimeError("CACHE MISS -> would call API")
common.chat=_no_api
import lifecycle
lifecycle.chat=_no_api
from lifecycle import build_lifecycle, render_view, assemble_chains, _group_by_session
from common import load_dataset
from query_classifier import classify_query

R="/root/agent_memory_iclr/results/"
def load(t): return {r["qid"]: r for r in (json.loads(l) for l in open(R+f"full500_{t}.jsonl"))}
qc,nq=load("qc"),load("qc_no_qc")
data=load_dataset("s_cleaned")
byid={q["question_id"]:q for q in data}
print("dataset n:",len(data),"covers all qids:",all(q in byid for q in qc))

CAP=20000
def cap(txt):
    if len(txt)<=CAP: return txt
    kept,tot=[],0
    for line in txt.splitlines():
        if tot+len(line)+1>CAP: break
        kept.append(line); tot+=len(line)+1
    return "\n".join(kept)

rows=[]
miss=0
for i,(qid,r) in enumerate(qc.items()):
    q=byid[qid]
    try:
        store=build_lifecycle(q,None)
    except RuntimeError as e:
        miss+=1; continue
    cur=cap(render_view(store,"current")); hist=cap(render_view(store,"historical"))
    mode_qc=r["view_mode"]; mode_nq=nq[qid]["view_mode"]
    v_qc=cur if mode_qc=="current" else hist
    v_nq=cur if mode_nq=="current" else hist
    ngroups=[len(_group_by_session(c)) for c in store.values()]
    n_multi=sum(1 for g in ngroups if g>1)
    n_conf=sum(1 for c in store.values() if any(x["relation"]=="contradiction" for x in c))
    rows.append({"qid":qid,"type":r["type"],"qlabel":r["qlabel"],
                 "mode_qc":mode_qc,"mode_nq":mode_nq,
                 "same_view":v_qc==v_nq,"len_qc":len(v_qc),"len_nq":len(v_nq),
                 "n_chains":len(store),"n_multi_group_chains":n_multi,"n_conflict_chains":n_conf,
                 "len_cur":len(cur),"len_hist":len(hist),
                 "judge_qc":r["judge"],"judge_nq":nq[qid]["judge"]})
    if (i+1)%100==0: print("  ...",i+1,flush=True)
print("cache misses (questions skipped):",miss,"api call attempts:",_calls["n"])
json.dump(rows,open("/root/agent_memory_iclr/audit/view_diff.json","w"))
print("rebuilt rows:",len(rows))
