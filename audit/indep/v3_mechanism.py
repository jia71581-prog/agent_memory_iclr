import json, re, sys, math, time
sys.path.insert(0,"/root/agent_memory_iclr/code")
import common
common.chat=lambda *a,**k:(_ for _ in ()).throw(RuntimeError("NO API CALLS ALLOWED"))
import lifecycle; lifecycle.chat=common.chat
from lifecycle import build_lifecycle, render_view
from common import load_dataset
R="/root/agent_memory_iclr/results/"
def load(t): return {r["qid"]: r for r in (json.loads(l) for l in open(R+f"full500_{t}.jsonl"))}
qc,vo=load("qc"),load("qc_view_only")
t0=time.time(); data=load_dataset("s_cleaned"); byid={q["question_id"]:q for q in data}
print("dataset loaded",len(data),"in %.1fs"%(time.time()-t0))
CAP=20000
def cap(txt):
    if len(txt)<=CAP: return txt
    kept,tot=[],0
    for line in txt.splitlines():
        if tot+len(line)+1>CAP: break
        kept.append(line); tot+=len(line)+1
    return "\n".join(kept)
def nrm(s): return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9 ]"," ",str(s).lower())).strip()
def toks(s): return set(nrm(s).split())
STOP=set("the a an of in on at to for and or is are was were i my me you your it that this with".split())
out=[]
for i,(qid,r) in enumerate(qc.items()):
    q=byid[qid]; store=build_lifecycle(q,None)
    mode=r["view_mode"]
    view=nrm(cap(render_view(store,mode)))
    g=q["answer"]; gt=toks(g)-STOP
    vt=set(view.split())
    cov=len(gt&vt)/len(gt) if gt else float('nan')
    out.append({"qid":qid,"type":r["type"],"abs":qid.endswith("_abs"),
                "gold_sub":bool(nrm(g)) and nrm(g) in view,
                "gold_tokcov":cov,"nfacts":len(store) if hasattr(store,'__len__') else -1,
                "viewlen":len(view),"judge_vo":bool(vo[qid]["judge"]),"judge_qc":bool(r["judge"])})
    if (i+1)%100==0: print("  ..",i+1,flush=True)
json.dump(out,open("/root/agent_memory_iclr/audit/indep/gold_in_view_indep.json","w"))
NON=[r for r in out if not r["abs"]]
print("\n== gold-answer presence in the qc-mode view (independent recompute, non-abstention only) ==")
print(f"{'type':28s} {'n':>4s} {'gold_substr':>12s} {'tokcov>=0.8':>12s} {'mean tokcov':>12s} {'vo acc':>8s} {'qc acc':>8s}")
for t in sorted({r['type'] for r in NON}):
    ids=[r for r in NON if r['type']==t]
    gs=sum(r['gold_sub'] for r in ids)/len(ids)
    hc=sum(1 for r in ids if r['gold_tokcov']>=0.8)/len(ids)
    mc=sum(r['gold_tokcov'] for r in ids)/len(ids)
    print(f"{t:28s} {len(ids):4d} {gs:12.3f} {hc:12.3f} {mc:12.3f} {sum(r['judge_vo'] for r in ids)/len(ids):8.3f} {sum(r['judge_qc'] for r in ids)/len(ids):8.3f}")
ids=NON
print(f"{'ALL-nonabs':28s} {len(ids):4d} {sum(r['gold_sub'] for r in ids)/len(ids):12.3f} {sum(1 for r in ids if r['gold_tokcov']>=0.8)/len(ids):12.3f} {sum(r['gold_tokcov'] for r in ids)/len(ids):12.3f} {sum(r['judge_vo'] for r in ids)/len(ids):8.3f} {sum(r['judge_qc'] for r in ids)/len(ids):8.3f}")
print("\n== is view_only accuracy tracking view CONTENT or answerer skill? ==")
hit=[r for r in NON if r['gold_tokcov']>=0.8]; miss=[r for r in NON if r['gold_tokcov']<0.8]
print(f"  gold-token-covered  n={len(hit)} vo acc={sum(r['judge_vo'] for r in hit)/len(hit):.3f} qc acc={sum(r['judge_qc'] for r in hit)/len(hit):.3f}")
print(f"  gold-token-missing  n={len(miss)} vo acc={sum(r['judge_vo'] for r in miss)/len(miss):.3f} qc acc={sum(r['judge_qc'] for r in miss)/len(miss):.3f}")
