"""Controls: (1) chance baseline for 'NN shares shape'; (2) answerable-vs-answerable NN
Jaccard distribution (is 0.67 special?); (3) arm-level abstention/answerable tradeoff."""
import json, re, sys, statistics, collections, glob, os, random
sys.path.insert(0,"/root/agent_memory_iclr/code")
from v5_shape import classify_shape
R="/root/agent_memory_iclr/results"
ROWS=json.load(open(f"{R}/_v5_shape_rows.json"))
AB=[r for r in ROWS if r["is_abs"]]; NA=[r for r in ROWS if not r["is_abs"]]
W=re.compile(r"[a-z0-9']+")
STOP={"i","my","the","a","an","do","did","was","is","in","of","to","have","how"}
tok=lambda s:{t for t in W.findall(s.lower()) if t not in STOP}
TA={r["qid"]:tok(r["q"]) for r in ROWS}

# (1) chance baseline: random answerable partner
p=collections.Counter(r["shape"] for r in NA); n=len(NA)
chance=sum(v/n * (p[s]/n) for s,v in p.items())
print(f"[J] chance same-shape with a RANDOM answerable question = {chance:.3f}")
# empirical: random partner, 2000 draws
rng=random.Random(0); hits=[]
for _ in range(2000):
    hits.append(sum(a["shape"]==rng.choice(NA)["shape"] for a in AB))
print(f"[J] random-partner same-shape count: mean={statistics.mean(hits):.1f}/30 "
      f"p95={sorted(hits)[int(.95*2000)]}/30  observed=26/30")
print(f"[J] one-sided p(random >= 26) = {sum(h>=26 for h in hits)/2000:.4f}")

# (2) answerable NN control
def nnj(q, pool):
    ta=TA[q["qid"]]; b=None; bj=-1
    for o in pool:
        if o["qid"]==q["qid"]: continue
        to=TA[o["qid"]]; j=len(ta&to)/max(1,len(ta|to))
        if j>bj: bj,b=j,o
    return b,bj
absj=[nnj(a,NA)[1] for a in AB]
ansn=[nnj(a,NA) for a in NA]
ansj=[j for _,j in ansn]
print(f"\n[K] abs  NN-Jaccard: median={statistics.median(absj):.2f} mean={statistics.mean(absj):.2f} "
      f">=0.5:{sum(j>=.5 for j in absj)}/30")
print(f"[K] ans  NN-Jaccard: median={statistics.median(ansj):.2f} mean={statistics.mean(ansj):.2f} "
      f">=0.5:{sum(j>=.5 for j in ansj)}/470 = {sum(j>=.5 for j in ansj)/470:.3f}")
# Mann-Whitney style: fraction of answerable with NN-J >= abs median
print(f"[K] answerable questions with NN-J >= abs median({statistics.median(absj):.2f}): "
      f"{sum(j>=statistics.median(absj) for j in ansj)}/470 = {sum(j>=statistics.median(absj) for j in ansj)/470:.3f}")
same_ans=sum(a["shape"]==b["shape"] for (b,_),a in zip(ansn,NA))
print(f"[K] answerable NN same-shape control = {same_ans}/470 = {same_ans/470:.3f}")
hi=[(b,j,a) for (b,j),a in zip(ansn,NA) if j>=0.5]
print(f"[K] among answerable with NN-J>=0.5 (n={len(hi)}): same-shape "
      f"{sum(b['shape']==a['shape'] for b,_,a in hi)}/{len(hi)} = "
      f"{sum(b['shape']==a['shape'] for b,_,a in hi)/len(hi):.3f}")

# (3) arms
print()
pts=[]
for f in sorted(glob.glob(f"{R}/full500_*.jsonl")):
    rows=[json.loads(l) for l in open(f)]
    ab=[r for r in rows if r["qid"].endswith("_abs")]; an=[r for r in rows if not r["qid"].endswith("_abs")]
    pts.append((os.path.basename(f), sum(r['judge'] for r in an)/len(an), sum(r['judge'] for r in ab)/len(ab)))
xs=[p[1] for p in pts]; ys=[p[2] for p in pts]
mx,my=statistics.mean(xs),statistics.mean(ys)
r=sum((a-mx)*(b-my) for a,b in zip(xs,ys))/((sum((a-mx)**2 for a in xs)*sum((b-my)**2 for b in ys))**.5)
print(f"[L] across 8 arms: pearson r(answerable acc, abstention acc) = {r:.3f}")
for n_,x,y in sorted(pts,key=lambda t:t[1]): print(f"    {n_:30s} ans={x:.3f} abs={y:.3f}")
