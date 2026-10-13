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
STOP=set("the a an of in on at to for and or is are was were i my me you your it that this with".split())
def cov(g,v):
    gt=set(nrm(g).split())-STOP
    return len(gt&set(v.split()))/len(gt) if gt else float('nan')
rows=[]
for qid,r in qc.items():
    q=byid[qid]; store=build_lifecycle(q,None); mode=r["view_mode"]
    full=render_view(store,mode); cp=cap(full)
    nl_full=full.count("\n")+1; nl_cap=cp.count("\n")+1
    user_full=sum(1 for L in full.splitlines() if re.search(r"\buser \|",L))
    user_cap=sum(1 for L in cp.splitlines() if re.search(r"\buser \|",L))
    rows.append({"qid":qid,"type":r["type"],"abs":qid.endswith("_abs"),
      "lines_full":nl_full,"lines_cap":nl_cap,"kept":nl_cap/nl_full,
      "user_full":user_full,"user_cap":user_cap,
      "cov_full":cov(q["answer"],nrm(full)),"cov_cap":cov(q["answer"],nrm(cp)),
      "vo":bool(vo[qid]["judge"])})
NON=[r for r in rows if not r["abs"]]
print("== how binding is the 20000-char view cap? ==")
print(f"{'type':28s} {'n':>4s} {'lines_full':>11s} {'lines_kept':>11s} {'frac_kept':>10s} {'user_lines_full':>16s} {'user_kept':>10s}")
for t in sorted({r['type'] for r in NON}):
    ids=[r for r in NON if r['type']==t]
    print(f"{t:28s} {len(ids):4d} {sum(r['lines_full'] for r in ids)/len(ids):11.1f} {sum(r['lines_cap'] for r in ids)/len(ids):11.1f} {sum(r['kept'] for r in ids)/len(ids):10.3f} {sum(r['user_full'] for r in ids)/len(ids):16.1f} {sum(r['user_cap'] for r in ids)/len(ids):10.1f}")
print("\n== is missing gold content a CAP artifact or an EXTRACTION artifact? (non-abstention) ==")
print(f"{'type':28s} {'meancov_uncapped':>17s} {'meancov_capped':>15s} {'lost_to_cap':>12s} {'cov>=.8 uncap':>14s} {'cov>=.8 cap':>12s}")
for t in sorted({r['type'] for r in NON}):
    ids=[r for r in NON if r['type']==t]
    cf=sum(r['cov_full'] for r in ids)/len(ids); cc=sum(r['cov_cap'] for r in ids)/len(ids)
    hf=sum(1 for r in ids if r['cov_full']>=0.8)/len(ids); hc=sum(1 for r in ids if r['cov_cap']>=0.8)/len(ids)
    print(f"{t:28s} {cf:17.3f} {cc:15.3f} {cf-cc:12.3f} {hf:14.3f} {hc:12.3f}")
