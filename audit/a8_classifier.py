import json, sys, math, re
from collections import Counter, defaultdict
sys.path.insert(0,"/root/agent_memory_iclr/code")
from query_classifier import classify_query
from common import load_dataset
R="/root/agent_memory_iclr/results/"
def load(t): return {r["qid"]: r for r in (json.loads(l) for l in open(R+f"full500_{t}.jsonl"))}
qc,nq,vo,full,nolc=load("qc"),load("qc_no_qc"),load("qc_view_only"),load("full"),load("qc_no_lifecycle")
data=load_dataset("s_cleaned")
lab={}; fired_all=Counter()
for q in data:
    l,f=classify_query(q["question"]); lab[q["question_id"]]=l
    for p in f: fired_all[p]+=1
print("recomputed classifier matches stored qlabel:",all(lab[q]==qc[q]['qlabel'] for q in qc))
print("\n== routing distribution over 500 ==")
c=Counter(lab.values())
for k in ["historical","current","neutral"]:
    print(f"  {k:11s} {c[k]:3d}  {c[k]/500:.3f}")
print(f"  -> mode assigned: current {c['current']+c['neutral']} ({(c['current']+c['neutral'])/500:.3f}), historical {c['historical']} ({c['historical']/500:.3f})")
print("\n== rules that fired (count over 500 questions) ==")
for p,n in fired_all.most_common(): print(f"  {n:4d}  {p}")
print("\n== qlabel x question_type ==")
types=sorted({qc[q]['type'] for q in qc})
print(f"{'type':30s} {'n':>4s} {'hist':>5s} {'curr':>5s} {'neut':>5s}   {'%hist':>6s}")
for t in types:
    ids=[q for q in qc if qc[q]['type']==t]
    cc=Counter(lab[q] for q in ids)
    print(f"{t:30s} {len(ids):4d} {cc['historical']:5d} {cc['current']:5d} {cc['neutral']:5d}   {cc['historical']/len(ids):6.3f}")
print("\n== qc-arm accuracy by routed label ==")
def wilson(k,n,z=1.96):
    if n==0: return (0,0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (c-h,c+h)
for k in ["historical","current","neutral"]:
    ids=[q for q in qc if lab[q]==k]
    if not ids: continue
    for arm,name in [(qc,"qc"),(nq,"no_qc"),(vo,"view_only"),(full,"full"),(nolc,"no_lifecycle")]:
        a=sum(arm[q]['judge'] for q in ids); acc=a/len(ids)
        lo,hi=wilson(a,len(ids))
        print(f"  {k:11s} n={len(ids):3d} {name:13s} acc={acc:.4f} ({a:3d}) 95%CI[{lo:.3f},{hi:.3f}]")
    print()
print("== per-label f1 (qc arm) ==")
for k in ["historical","current","neutral"]:
    ids=[q for q in qc if lab[q]==k]
    print(f"  {k:11s} n={len(ids):3d} f1={sum(qc[q]['f1'] for q in ids)/len(ids):.4f}")
print("\n== is routing aligned with need? knowledge-update questions routed historical ==")
ku=[q for q in qc if qc[q]['type']=='knowledge-update']
print(f"  knowledge-update n={len(ku)}: routed historical={sum(1 for q in ku if lab[q]=='historical')}, current={sum(1 for q in ku if lab[q]=='current')}, neutral={sum(1 for q in ku if lab[q]=='neutral')}")
print("  qc acc on KU routed historical: %.3f (n=%d)"%(
  sum(qc[q]['judge'] for q in ku if lab[q]=='historical')/max(1,sum(1 for q in ku if lab[q]=='historical')),
  sum(1 for q in ku if lab[q]=='historical')))
print("  qc acc on KU routed current/neutral: %.3f (n=%d)"%(
  sum(qc[q]['judge'] for q in ku if lab[q]!='historical')/max(1,sum(1 for q in ku if lab[q]!='historical')),
  sum(1 for q in ku if lab[q]!='historical')))
print("\n== bound on max possible effect of query conditioning as implemented ==")
print(f"  vs always-historical (no_qc): differs on {sum(1 for q in qc if lab[q]!='historical')} / 500 questions")
print(f"  vs always-current (NOT RUN):  would differ on {sum(1 for q in qc if lab[q]=='historical')} / 500 questions = {sum(1 for q in qc if lab[q]=='historical')/500:.3f}")
