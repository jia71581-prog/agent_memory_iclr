import json, math
from collections import Counter, defaultdict
R="/root/agent_memory_iclr/results/"
def load(t):
    return {r["qid"]: r for r in (json.loads(l) for l in open(R+f"full500_{t}.jsonl"))}
arms={t:load(t) for t in ["full","decay","fifo","qc_no_lifecycle","qc","qc_no_qc","qc_view_only"]}
qids=list(arms["qc"].keys())
print("n qids:",len(qids),"all arms same qid set:",all(set(a)==set(qids) for a in arms.values()))
print("\n== overall judge acc ==")
for t,a in arms.items():
    acc=sum(r["judge"] for r in a.values())/len(a)
    f1=sum(r["f1"] for r in a.values())/len(a)
    print(f"{t:18s} acc={acc:.4f} ({sum(r['judge'] for r in a.values())}/{len(a)}) f1={f1:.4f}")

def mcnemar(a,b,ids):
    # a,b dicts qid->row ; returns b01(a wrong,b right), b10, exact two-sided p
    n01=sum(1 for q in ids if (not a[q]["judge"]) and b[q]["judge"])
    n10=sum(1 for q in ids if a[q]["judge"] and (not b[q]["judge"]))
    n=n01+n10
    if n==0: return n01,n10,1.0
    k=min(n01,n10)
    p=sum(math.comb(n,i) for i in range(0,k+1))/2**n*2
    return n01,n10,min(1.0,p)

print("\n== qc vs qc_no_qc, overall (paired McNemar exact) ==")
qc,nq=arms["qc"],arms["qc_no_qc"]
acc_qc=sum(r["judge"] for r in qc.values())/500; acc_nq=sum(r["judge"] for r in nq.values())/500
n01,n10,p=mcnemar(nq,qc,qids)   # nq baseline, qc treatment: n01 = nq wrong & qc right
print(f"qc={acc_qc:.4f} no_qc={acc_nq:.4f} delta={acc_qc-acc_nq:+.4f}")
print(f"qc-right/nq-wrong={n01}  qc-wrong/nq-right={n10}  discordant={n01+n10}  exact p={p:.4f}")

print("\n== per question_type ==")
types=sorted({r['type'] for r in qc.values()})
hdr=f"{'type':30s} {'n':>4s} {'qc':>7s} {'no_qc':>7s} {'delta':>7s} {'n01':>4s} {'n10':>4s} {'p':>7s}"
print(hdr)
for t in types:
    ids=[q for q in qids if qc[q]['type']==t]
    a=sum(qc[q]['judge'] for q in ids)/len(ids); b=sum(nq[q]['judge'] for q in ids)/len(ids)
    n01,n10,p=mcnemar(nq,qc,ids)
    print(f"{t:30s} {len(ids):4d} {a:7.3f} {b:7.3f} {a-b:+7.3f} {n01:4d} {n10:4d} {p:7.3f}")
