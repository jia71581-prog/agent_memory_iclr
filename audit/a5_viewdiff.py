import json, math
from collections import Counter
rows=json.load(open("/root/agent_memory_iclr/audit/view_diff.json"))
R="/root/agent_memory_iclr/results/"
def load(t): return {r["qid"]: r for r in (json.loads(l) for l in open(R+f"full500_{t}.jsonl"))}
qc,nq=load("qc"),load("qc_no_qc")
def mcn(ids):
    n01=sum(1 for q in ids if not nq[q]['judge'] and qc[q]['judge'])
    n10=sum(1 for q in ids if nq[q]['judge'] and not qc[q]['judge'])
    n=n01+n10
    p=1.0 if n==0 else min(1.0,sum(math.comb(n,i) for i in range(min(n01,n10)+1))/2**n*2)
    return n01,n10,p

routed=[r for r in rows if r["mode_qc"]!=r["mode_nq"]]
print("routed-different n:",len(routed))
print("  of those, view TEXT identical anyway:",sum(1 for r in routed if r["same_view"]))
print("  view TEXT actually differs         :",sum(1 for r in routed if not r["same_view"]))
print("\nchain stats over all 500 questions:")
print("  mean chains/question       :",sum(r['n_chains'] for r in rows)/len(rows))
print("  mean multi-session chains  :",sum(r['n_multi_group_chains'] for r in rows)/len(rows))
print("  questions with 0 multi-session chains:",sum(1 for r in rows if r['n_multi_group_chains']==0))
print("  mean conflict-tagged chains:",sum(r['n_conflict_chains'] for r in rows)/len(rows))
print("  questions with >=1 conflict chain    :",sum(1 for r in rows if r['n_conflict_chains']>0))
print("  mean view chars current/historical   :",
      sum(r['len_cur'] for r in rows)/len(rows), sum(r['len_hist'] for r in rows)/len(rows))
print("  questions where 20k cap binds (current):",sum(1 for r in rows if r['len_cur']>=19000),
      " (historical):",sum(1 for r in rows if r['len_hist']>=19000))

EFF=[r["qid"] for r in routed if not r["same_view"]]
NOEFF=[r["qid"] for r in routed if r["same_view"]]
same_mode=[r["qid"] for r in rows if r["mode_qc"]==r["mode_nq"]]
print("\n== accuracy on the EFFECTIVE-DIFFERENT subset (mode differs AND view text differs) ==")
for name,ids in [("EFFECTIVE (view text differs)",EFF),
                 ("mode differs but view identical",NOEFF),
                 ("mode identical (both historical)",same_mode)]:
    if not ids: print(f"  [{name}] n=0"); continue
    a=sum(qc[q]['judge'] for q in ids)/len(ids); b=sum(nq[q]['judge'] for q in ids)/len(ids)
    n01,n10,p=mcn(ids)
    jf=sum(1 for q in ids if qc[q]['judge']!=nq[q]['judge'])
    print(f"  [{name}] n={len(ids)}  qc={a:.4f} no_qc={b:.4f} delta={a-b:+.4f}  flips {n01}/{n10} (rate {jf/len(ids):.3f}) exact p={p:.4f}")

print("\n== EFFECTIVE subset per type ==")
for t in sorted({qc[q]['type'] for q in EFF}):
    sub=[q for q in EFF if qc[q]['type']==t]
    a=sum(qc[q]['judge'] for q in sub)/len(sub); b=sum(nq[q]['judge'] for q in sub)/len(sub)
    n01,n10,_=mcn(sub)
    print(f"  {t:30s} n={len(sub):3d} qc={a:.3f} no_qc={b:.3f} d={a-b:+.3f} flips {n01}/{n10}")

# how big is the textual difference where it exists
d=[(r['len_hist']-r['len_cur']) for r in routed if not r['same_view']]
print("\nview length delta (historical - current) on EFFECTIVE subset: mean %.0f chars, median %.0f, min %d max %d"
      %(sum(d)/len(d), sorted(d)[len(d)//2], min(d), max(d)))
