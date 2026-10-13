"""Sharpest mechanism test: on routed-different questions, does the mode choice change whether the
GOLD ANSWER is present in the view at all? Accuracy on that narrow subset."""
import json, re, sys, math
sys.path.insert(0,"/root/agent_memory_iclr/code")
import common; common.chat=lambda *a,**k:(_ for _ in ()).throw(RuntimeError("no api"))
import lifecycle; lifecycle.chat=common.chat
from lifecycle import build_lifecycle, render_view
from common import load_dataset
R="/root/agent_memory_iclr/results/"
def load(t): return {r["qid"]: r for r in (json.loads(l) for l in open(R+f"full500_{t}.jsonl"))}
qc,nq,vo=load("qc"),load("qc_no_qc"),load("qc_view_only")
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
recs=[]
for qid,r in qc.items():
    q=byid[qid]; store=build_lifecycle(q,None)
    cur=nrm(cap(render_view(store,"current"))); hist=nrm(cap(render_view(store,"historical")))
    g=nrm(q["answer"])
    gi_cur = bool(g) and g in cur
    gi_hist= bool(g) and g in hist
    recs.append({"qid":qid,"type":r["type"],"qlabel":r["qlabel"],"mode_qc":r["view_mode"],
                 "gold_in_cur":gi_cur,"gold_in_hist":gi_hist,
                 "gold_in_view_qc": gi_cur if r["view_mode"]=="current" else gi_hist,
                 "gold_in_view_nq": gi_hist,
                 "judge_qc":r["judge"],"judge_nq":nq[qid]["judge"],"judge_vo":vo[qid]["judge"]})
json.dump(recs,open("/root/agent_memory_iclr/audit/gold_in_view.json","w"))
n=len(recs)
print("gold-string containment in the capped view (substring, normalized):")
print("  in current view :",sum(r['gold_in_cur'] for r in recs),f"/{n}")
print("  in historical   :",sum(r['gold_in_hist'] for r in recs),f"/{n}")
routed=[r for r in recs if r['mode_qc']!='historical']
disc=[r for r in routed if r['gold_in_view_qc']!=r['gold_in_view_nq']]
print(f"\nrouted-different n={len(routed)}; gold-presence CHANGED by the mode choice: {len(disc)}")
for r in disc:
    print(f"   qid={r['qid']} type={r['type']} qlabel={r['qlabel']} gold_in_cur={r['gold_in_cur']} gold_in_hist={r['gold_in_hist']} judge qc={r['judge_qc']} nq={r['judge_nq']}")
if disc:
    a=sum(r['judge_qc'] for r in disc)/len(disc); b=sum(r['judge_nq'] for r in disc)/len(disc)
    print(f"  accuracy on that subset: qc={a:.3f} no_qc={b:.3f} delta={a-b:+.3f}")
print("\n== view completeness: gold in view vs view_only correctness ==")
for key,lab in [("gold_in_view_qc","qc-mode view")]:
    hit=[r for r in recs if r[key]]; nohit=[r for r in recs if not r[key]]
    print(f"  gold string present in {lab}: {len(hit)}/{n} = {len(hit)/n:.3f}")
    print(f"    view_only acc | gold present  = {sum(r['judge_vo'] for r in hit)/max(1,len(hit)):.3f} (n={len(hit)})")
    print(f"    view_only acc | gold absent   = {sum(r['judge_vo'] for r in nohit)/max(1,len(nohit)):.3f} (n={len(nohit)})")
    print(f"    qc acc        | gold present  = {sum(r['judge_qc'] for r in hit)/max(1,len(hit)):.3f}")
    print(f"    qc acc        | gold absent   = {sum(r['judge_qc'] for r in nohit)/max(1,len(nohit)):.3f}")
