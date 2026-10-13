"""Adversarial verify: does 'evidence-session-complete' (flat_all) imply the answer-bearing
content is actually in the QC arm's context? Build the real qc context (view + raw20) and
measure per-gold-evidence-turn representation. No API."""
import json, re, sys, os
from collections import Counter
sys.path.insert(0,"/root/agent_memory_iclr/code"); sys.path.insert(0,"/root/agent_memory_iclr/results")
import _recon_ctx as RC
from common import load_dataset

R="/root/agent_memory_iclr/results"
V={r['qid']:r for r in json.load(open(f"{R}/_v5_offline_val.json"))}
ROWS=json.load(open(f"{R}/_v5_shape_rows.json"))
NA=[r for r in ROWS if not r['is_abs']]
ERR=[r for r in NA if not r['judge']]
comp=[r for r in ERR if V[r['qid']]['flat_all']]
inc=[r for r in ERR if not V[r['qid']]['flat_all']]
data={q['question_id']:q for q in load_dataset("s_cleaned")}

STOP=set("the a an and or of to in on at for with is are was were be been i my me you your it this that as by from do did does have has had will would can could not no if then than so about into over out up down we they he she them his her our their there here what which who when where how why all any some more most other new also just only very much many few per".split())
def toks(s):
    return [t for t in re.sub(r"[^a-z0-9 ]"," ",s.lower()).split() if len(t)>3 and t not in STOP]

out={}
for r in comp+inc:
    qid=r['qid']; q=data[qid]
    label,mode,view,raw_idx=RC.qc_context(q)
    units=RC.units_of(q)
    raw_txt=" ".join(units[i][2] for i in raw_idx)
    ctx=view+"\n"+raw_txt
    ctxn=re.sub(r"[^a-z0-9 ]"," ",ctx.lower())
    ctxset=set(ctxn.split())
    got_raw={units[i][2] for i in raw_idx}
    ev=RC.evidence_turns(q)
    per=[]
    for e in ev:
        t=toks(e)
        rec=(sum(1 for x in set(t) if x in ctxset)/len(set(t))) if t else None
        per.append({"in_raw": e in got_raw, "tok_recall_ctx": rec, "ntok": len(set(t)),
                    "head":" ".join(e.split())[:110]})
    out[qid]={"shape":r['shape'],"flat_all":V[qid]['flat_all'],"cov_all":V[qid]['cov_all'],
              "n_ev":len(ev),"n_ev_in_raw":sum(p['in_raw'] for p in per),
              "view_chars":len(view),"raw_chars":len(raw_txt),"mode":mode,
              "gold":str(q['answer']),"question":q['question'],"per":per}
json.dump(out,open(f"{R}/_vrf/v1_turnlevel.json","w"),ensure_ascii=False,indent=1)
print("saved",len(out),"cache_miss",RC.CACHE_MISS[0])
