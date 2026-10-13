import json,sys,math,random
sys.path.insert(0,'/root/agent_memory_iclr/code')
from v5_guards import is_decline
R='/root/agent_memory_iclr/results/'
ARMS=['qc','qc_no_qc','qc_no_lifecycle','full','decay','qc_view_only','fifo']
D={a:{r['qid']:r for r in map(json.loads,open(R+'full500_%s.jsonl'%a))} for a in ARMS}
QIDS=list(D['qc']); AN=[q for q in QIDS if not q.endswith('_abs')]; AB=[q for q in QIDS if q.endswith('_abs')]
def wilson(k,n,z=1.96):
    if n==0: return (0,0)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); m=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return ((c-m)/d,(c+m)/d)
print("=== abstention accuracy, 95% Wilson CI (n=30) ===")
for a in ARMS:
    k=sum(D[a][q]['judge'] for q in AB); lo,hi=wilson(k,30)
    print(f"  {a:16s} {k:2d}/30 = {k/30:.3f}  [{lo:.3f},{hi:.3f}]")
print("\n=== bootstrap over questions: P(qc has lowest answerable decline rate) ===")
random.seed(11); wins={a:0 for a in ARMS}; B=4000
dec={a:[is_decline(D[a][q]['pred']) for q in AN] for a in ARMS}
n=len(AN)
for _ in range(B):
    idx=[random.randrange(n) for _ in range(n)]
    tot={a:sum(dec[a][i] for i in idx) for a in ARMS}
    m=min(tot.values())
    for a in ARMS:
        if tot[a]==m: wins[a]+=1
for a in ARMS: print(f"  {a:16s} argmin share {wins[a]/B:.3f}")
print("\n=== per-question-type answerable decline rates ===")
types=sorted({D['qc'][q]['type'] for q in AN})
print(f"  {'type':26s} {'n':>4s} " + " ".join(f"{a[:9]:>10s}" for a in ARMS))
for t in types:
    sub=[q for q in AN if D['qc'][q]['type']==t]
    print(f"  {t:26s} {len(sub):4d} " + " ".join(
        f"{sum(1 for q in sub if is_decline(D[a][q]['pred']))/len(sub):10.3f}" for a in ARMS))
print("\n=== pooled / answerable / abstention accuracy summary ===")
for a in ARMS:
    al=[D[a][q]['judge'] for q in QIDS]; an=[D[a][q]['judge'] for q in AN]; ab=[D[a][q]['judge'] for q in AB]
    print(f"  {a:16s} pooled500 {sum(al)/500:.3f}  answerable470 {sum(an)/470:.3f}  abs30 {sum(ab)/30:.3f}")
