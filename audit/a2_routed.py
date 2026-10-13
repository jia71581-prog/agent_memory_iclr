import json, math
from collections import Counter
R="/root/agent_memory_iclr/results/"
def load(t): return {r["qid"]: r for r in (json.loads(l) for l in open(R+f"full500_{t}.jsonl"))}
qc,nq,vo=load("qc"),load("qc_no_qc"),load("qc_view_only")
qids=list(qc)
print("== view_mode / qlabel consistency ==")
print("qc  (qlabel,view_mode):",Counter((qc[q]['qlabel'],qc[q]['view_mode']) for q in qids))
print("nq  (qlabel,view_mode):",Counter((nq[q]['qlabel'],nq[q]['view_mode']) for q in qids))
print("vo  (qlabel,view_mode):",Counter((vo[q]['qlabel'],vo[q]['view_mode']) for q in qids))
print("qlabel identical across arms:",all(qc[q]['qlabel']==nq[q]['qlabel']==vo[q]['qlabel'] for q in qids))

diff=[q for q in qids if qc[q]['view_mode']!=nq[q]['view_mode']]
same=[q for q in qids if qc[q]['view_mode']==nq[q]['view_mode']]
print(f"\nrouted-differently (qc=current, nq=historical): {len(diff)}")
print(f"identical-mode (both historical):               {len(same)}")
print("diff set all (qc=current & nq=historical):",all(qc[q]['view_mode']=='current' and nq[q]['view_mode']=='historical' for q in diff))
print("diff set qlabel dist:",Counter(qc[q]['qlabel'] for q in diff))
print("same set qlabel dist:",Counter(qc[q]['qlabel'] for q in same))

def mcnemar(a,b,ids):
    n01=sum(1 for q in ids if (not a[q]["judge"]) and b[q]["judge"])
    n10=sum(1 for q in ids if a[q]["judge"] and (not b[q]["judge"]))
    n=n01+n10
    if n==0: return n01,n10,1.0
    k=min(n01,n10)
    return n01,n10,min(1.0,sum(math.comb(n,i) for i in range(k+1))/2**n*2)

for name,ids in [("ROUTED-DIFFERENT",diff),("IDENTICAL-MODE",same)]:
    a=sum(qc[q]['judge'] for q in ids)/len(ids); b=sum(nq[q]['judge'] for q in ids)/len(ids)
    n01,n10,p=mcnemar(nq,qc,ids)
    fa=sum(qc[q]['f1'] for q in ids)/len(ids); fb=sum(nq[q]['f1'] for q in ids)/len(ids)
    print(f"\n[{name}] n={len(ids)}")
    print(f"  qc acc={a:.4f} ({sum(qc[q]['judge'] for q in ids)})  no_qc acc={b:.4f} ({sum(nq[q]['judge'] for q in ids)})  delta={a-b:+.4f}")
    print(f"  f1 qc={fa:.4f} no_qc={fb:.4f} delta={fa-fb:+.4f}")
    print(f"  judge-flips: qc-right/nq-wrong={n01}  qc-wrong/nq-right={n10}  exact p={p:.4f}")
    print("  per type:")
    for t in sorted({qc[q]['type'] for q in ids}):
        sub=[q for q in ids if qc[q]['type']==t]
        aa=sum(qc[q]['judge'] for q in sub)/len(sub); bb=sum(nq[q]['judge'] for q in sub)/len(sub)
        n1,n2,pp=mcnemar(nq,qc,sub)
        print(f"    {t:30s} n={len(sub):3d} qc={aa:.3f} no_qc={bb:.3f} d={aa-bb:+.3f} flips {n1}/{n2}")
