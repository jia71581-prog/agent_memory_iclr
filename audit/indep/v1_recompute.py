import json, math, re
from collections import Counter
from itertools import combinations
R="/root/agent_memory_iclr/results/"
def load(t):
    d={}
    for l in open(R+f"full500_{t}.jsonl"):
        r=json.loads(l)
        assert r["qid"] not in d, ("dup qid",t,r["qid"])
        d[r["qid"]]=r
    return d
arms={t:load(t) for t in ["qc","qc_no_qc","qc_view_only","qc_no_lifecycle","full","fifo","decay"]}
vo=arms["qc_view_only"]; qc=arms["qc"]; nq=arms["qc_no_qc"]; nolc=arms["qc_no_lifecycle"]
# ground truth metadata from the ORIGINAL dataset, not from result rows
ds=json.load(open("/root/agent_memory_iclr/data/longmemeval_oracle.json"))
meta={x["question_id"]:x for x in ds}
print("== file integrity ==")
for t,a in arms.items(): print(f"  {t:18s} rows={len(a)}")
qids=sorted(vo)
print("  qid sets identical across arms:", all(set(a)==set(vo) for a in arms.values()))
print("  qids all present in oracle dataset:", all(q in meta for q in qids), " n_qids:",len(qids))
# type label agreement between result rows and dataset
mism=[q for q in qids if vo[q]["type"]!=meta[q]["question_type"]]
print("  type-label mismatches vs dataset:",len(mism))
print("  judge field dtype:",Counter(type(vo[q]['judge']).__name__ for q in qids))

def wilson(k,n,z=1.96):
    if n==0: return (float('nan'),)*2
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (c-h,c+h)
def clopper(k,n,alpha=0.05):
    # exact binomial CI via beta quantiles (scipy-free bisection on binom cdf)
    from math import comb
    def cdf(p,k,n): return sum(comb(n,i)*p**i*(1-p)**(n-i) for i in range(k+1))
    def bis(f,lo,hi):
        for _ in range(200):
            m=(lo+hi)/2
            if f(m)>0: lo=m
            else: hi=m
        return (lo+hi)/2
    lo=0.0 if k==0 else bis(lambda p: cdf(p,k-1,n)-(1-alpha/2),0,1)
    hi=1.0 if k==n else bis(lambda p: cdf(p,k,n)-alpha/2,0,1)
    return lo,hi

ABS=[q for q in qids if str(q).endswith("_abs")]
NON=[q for q in qids if not str(q).endswith("_abs")]
print(f"\n== abstention set (qid suffix _abs, official LongMemEval convention) ==")
print(f"  n_abs={len(ABS)}  n_non={len(NON)}")
print("  abs type mix:",dict(Counter(meta[q]['question_type'] for q in ABS)))
print("  gold answers of abs questions (first 5):",[meta[q]['answer'][:60] for q in ABS[:5]])

print("\n== HEADLINE / SPLIT ==")
for name in ["qc_view_only","qc","qc_no_qc","qc_no_lifecycle","full"]:
    a=arms[name]
    k=sum(bool(a[q]['judge']) for q in qids); ka=sum(bool(a[q]['judge']) for q in ABS); kn=sum(bool(a[q]['judge']) for q in NON)
    lo,hi=wilson(k,500)
    print(f"  {name:16s} TOTAL {k:3d}/500={k/500:.4f} [{lo:.3f},{hi:.3f}]   abs {ka:2d}/{len(ABS)}={ka/len(ABS):.4f}   non-abs {kn:3d}/{len(NON)}={kn/len(NON):.4f}   diff={ka/len(ABS)-kn/len(NON):+.4f}")

print("\n== view_only per question_type (ALL 500, as stated) ==")
rows=[]
for t in sorted({meta[q]['question_type'] for q in qids}):
    ids=[q for q in qids if meta[q]['question_type']==t]
    k=sum(bool(vo[q]['judge']) for q in ids); lo,hi=wilson(k,len(ids)); clo,chi=clopper(k,len(ids))
    rows.append((t,len(ids),k,k/len(ids),lo,hi,clo,chi))
    print(f"  {t:28s} n={len(ids):3d} k={k:3d} acc={k/len(ids):.3f} Wilson[{lo:.3f},{hi:.3f}] Exact[{clo:.3f},{chi:.3f}]")
print("\n== view_only per question_type, ABSTENTION REMOVED (true view completeness) ==")
rows2=[]
for t in sorted({meta[q]['question_type'] for q in qids}):
    ids=[q for q in NON if meta[q]['question_type']==t]
    nab=[q for q in ABS if meta[q]['question_type']==t]
    k=sum(bool(vo[q]['judge']) for q in ids); lo,hi=wilson(k,len(ids))
    ka=sum(bool(vo[q]['judge']) for q in nab)
    rows2.append((t,len(ids),k,k/len(ids)))
    print(f"  {t:28s} n={len(ids):3d} k={k:3d} acc={k/len(ids):.3f} Wilson[{lo:.3f},{hi:.3f}]   (dropped {len(nab)} abs, vo correct on {ka})")
