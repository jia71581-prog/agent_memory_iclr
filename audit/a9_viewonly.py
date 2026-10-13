import json, math, re
from collections import Counter
R="/root/agent_memory_iclr/results/"
def load(t): return {r["qid"]: r for r in (json.loads(l) for l in open(R+f"full500_{t}.jsonl"))}
qc,nq,vo,nolc,full=load("qc"),load("qc_no_qc"),load("qc_view_only"),load("qc_no_lifecycle"),load("full")
qids=list(qc)
print("abstention-style qids (suffix _abs):",sum(1 for q in qids if str(q).endswith("_abs")))
print("qid prefixes:",Counter(('gpt4_' if str(q).startswith('gpt4_') else 'plain') for q in qids))
def mcn(a,b,ids):
    n01=sum(1 for q in ids if not a[q]['judge'] and b[q]['judge']); n10=sum(1 for q in ids if a[q]['judge'] and not b[q]['judge'])
    n=n01+n10; p=1.0 if n==0 else min(1.0,sum(math.comb(n,i) for i in range(min(n01,n10)+1))/2**n*2)
    return n01,n10,p
def wilson(k,n,z=1.96):
    if n==0: return (0,0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (c-h,c+h)
print("\n== view_only per question_type (view completeness) ==")
print(f"{'type':30s} {'n':>4s} {'view_only':>10s} {'95%CI':>16s} {'qc':>7s} {'nolc':>7s} {'full':>7s} {'vo_f1':>7s}")
for t in sorted({qc[q]['type'] for q in qids}):
    ids=[q for q in qids if qc[q]['type']==t]
    k=sum(vo[q]['judge'] for q in ids); lo,hi=wilson(k,len(ids))
    print(f"{t:30s} {len(ids):4d} {k/len(ids):10.3f} [{lo:.3f},{hi:.3f}] {sum(qc[q]['judge'] for q in ids)/len(ids):7.3f} "
          f"{sum(nolc[q]['judge'] for q in ids)/len(ids):7.3f} {sum(full[q]['judge'] for q in ids)/len(ids):7.3f} "
          f"{sum(vo[q]['f1'] for q in ids)/len(ids):7.3f}")
k=sum(vo[q]['judge'] for q in qids); lo,hi=wilson(k,500)
print(f"{'TOTAL':30s} {500:4d} {k/500:10.3f} [{lo:.3f},{hi:.3f}]")

print("\n== attribution: where does qc's gain over qc_no_lifecycle come from? ==")
n01,n10,p=mcn(nolc,qc,qids)
print(f"  qc={sum(qc[q]['judge'] for q in qids)/500:.4f} no_lifecycle={sum(nolc[q]['judge'] for q in qids)/500:.4f} delta={+ (sum(qc[q]['judge'] for q in qids)-sum(nolc[q]['judge'] for q in qids))/500:+.4f}")
print(f"  qc-right/nolc-wrong={n01}  qc-wrong/nolc-right={n10}  exact p={p:.4g}")
gain=[q for q in qids if not nolc[q]['judge'] and qc[q]['judge']]
loss=[q for q in qids if nolc[q]['judge'] and not qc[q]['judge']]
print(f"  of the {len(gain)} gained questions, view_only also correct: {sum(vo[q]['judge'] for q in gain)} ({sum(vo[q]['judge'] for q in gain)/len(gain):.3f})")
print(f"  of the {len(loss)} lost questions,   view_only correct: {sum(vo[q]['judge'] for q in loss)}")
print("  gained-question type mix:",dict(Counter(qc[q]['type'] for q in gain)))
print("  lost-question type mix:  ",dict(Counter(qc[q]['type'] for q in loss)))

print("\n== 2x2: view_only x no_lifecycle -> qc ==")
cells=Counter((vo[q]['judge'],nolc[q]['judge'],qc[q]['judge']) for q in qids)
print(f"{'vo':>5s} {'nolc':>5s} -> n, qc acc within cell")
for v in (True,False):
    for nn in (True,False):
        ids=[q for q in qids if vo[q]['judge']==v and nolc[q]['judge']==nn]
        if not ids: continue
        print(f"{str(v):>5s} {str(nn):>5s} -> n={len(ids):3d}  qc acc={sum(qc[q]['judge'] for q in ids)/len(ids):.3f}")
u=[q for q in qids if not vo[q]['judge'] and not nolc[q]['judge']]
print(f"  neither view nor raw-turns alone gets it: n={len(u)}, qc solves {sum(qc[q]['judge'] for q in u)} ({sum(qc[q]['judge'] for q in u)/len(u):.3f}) -> true synergy")
print(f"  union coverage (vo OR nolc correct): {sum(1 for q in qids if vo[q]['judge'] or nolc[q]['judge'])}/500 = {sum(1 for q in qids if vo[q]['judge'] or nolc[q]['judge'])/500:.3f}; qc achieves {sum(qc[q]['judge'] for q in qids)/500:.3f}")

print("\n== view_only abstention/IDK rate (does it just say it doesn't know?) ==")
idk=re.compile(r"don't know|do not know|no (relevant )?information|not (mentioned|provided|available|found)|cannot determine|unable to",re.I)
for name,arm in [("view_only",vo),("qc",qc),("no_lifecycle",nolc)]:
    n=sum(1 for q in qids if idk.search(arm[q]['pred']))
    nj=sum(1 for q in qids if idk.search(arm[q]['pred']) and arm[q]['judge'])
    print(f"  {name:13s} IDK-phrase preds={n:3d} ({n/500:.3f}); of those judged CORRECT={nj}")
