"""Does the 20k alphabetical truncation destroy answers? gold containment uncapped vs capped."""
import json, re, sys
sys.path.insert(0,"/root/agent_memory_iclr/code")
import common; common.chat=lambda *a,**k:(_ for _ in ()).throw(RuntimeError("no api"))
import lifecycle; lifecycle.chat=common.chat
from lifecycle import build_lifecycle, render_view
from common import load_dataset
R="/root/agent_memory_iclr/results/"
def load(t): return {r["qid"]: r for r in (json.loads(l) for l in open(R+f"full500_{t}.jsonl"))}
qc,vo=load("qc"),load("qc_view_only")
byid={q["question_id"]:q for q in load_dataset("s_cleaned")}
CAP=20000
def cap(txt):
    if len(txt)<=CAP: return txt
    kept,tot=[],0
    for line in txt.splitlines():
        if tot+len(line)+1>CAP: break
        kept.append(line); tot+=len(line)+1
    return "\n".join(kept)
def nrm(s): return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9 ]"," ",str(s).lower())).strip()
stat={"in_full":0,"in_capped":0,"lost_by_cap":0,"n":0}
lost=[]
for qid,r in qc.items():
    q=byid[qid]; store=build_lifecycle(q,None)
    mode=r["view_mode"]
    full_v=nrm(render_view(store,mode)); cap_v=nrm(cap(render_view(store,mode)))
    g=nrm(q["answer"])
    if not g: continue
    stat["n"]+=1
    a=g in full_v; b=g in cap_v
    stat["in_full"]+=a; stat["in_capped"]+=b
    if a and not b:
        stat["lost_by_cap"]+=1; lost.append((qid,r["type"],vo[qid]["judge"],r["judge"]))
n=stat["n"]
print(f"questions with non-empty gold: {n}")
print(f"gold string present in UNCAPPED view : {stat['in_full']} ({stat['in_full']/n:.3f})")
print(f"gold string present in CAPPED   view : {stat['in_capped']} ({stat['in_capped']/n:.3f})")
print(f"answers DESTROYED by the 20k alphabetical truncation: {stat['lost_by_cap']} ({stat['lost_by_cap']/n:.3f})")
print(f"  of those, view_only still judged correct: {sum(1 for x in lost if x[2])}/{len(lost)}; qc correct: {sum(1 for x in lost if x[3])}/{len(lost)}")
from collections import Counter
print("  lost-by-cap type mix:",dict(Counter(x[1] for x in lost)))
