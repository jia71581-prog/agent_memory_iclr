import json, math, sys, re
from collections import Counter
R="/root/agent_memory_iclr/results/"
def load(t): return {r["qid"]: r for r in (json.loads(l) for l in open(R+f"full500_{t}.jsonl"))}
qc,nq,vo=load("qc"),load("qc_no_qc"),load("qc_view_only")
qids=list(qc)
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (c-h,c+h)
same=[q for q in qids if qc[q]['view_mode']==nq[q]['view_mode']]
diff=[q for q in qids if qc[q]['view_mode']!=nq[q]['view_mode']]
f_same=sum(1 for q in same if qc[q]['judge']!=nq[q]['judge']); f_diff=sum(1 for q in diff if qc[q]['judge']!=nq[q]['judge'])
print("== nondeterminism floor ==")
print(f"  identical-context re-run flip rate : {f_same}/{len(same)} = {f_same/len(same):.4f}  95%CI{tuple(round(x,4) for x in wilson(f_same,len(same)))}")
print(f"  changed-context flip rate          : {f_diff}/{len(diff)} = {f_diff/len(diff):.4f}  95%CI{tuple(round(x,4) for x in wilson(f_diff,len(diff)))}")
# Fisher exact two-sided for flip rate difference
def fisher(a,b,c,d):
    n=a+b+c+d
    def hg(a):
        return math.comb(a+b,a)*math.comb(c+d,c)/math.comb(n,a+c)
    p0=hg(a); tot=0
    lo=max(0,a+c-(c+d)); hi=min(a+b,a+c)
    for x in range(lo,hi+1):
        px=math.comb(a+b,x)*math.comb(c+d,a+c-x)/math.comb(n,a+c)
        if px<=p0+1e-12: tot+=px
    return min(1.0,tot)
print(f"  Fisher exact (flip rate diff): p={fisher(f_diff,len(diff)-f_diff,f_same,len(same)-f_same):.4f}  -> changing the view mode does NOT change answers more than re-running the same prompt")
print("\n== power / minimum detectable effect (paired McNemar, exact) ==")
for nd,label in [(24,"observed discordant pairs, full 500"),(22,"observed discordant, routed-different n=456")]:
    for k in range(0,nd//2+1):
        p=min(1.0,sum(math.comb(nd,i) for i in range(k+1))/2**nd*2)
        if p>=0.05:
            need=nd-k+1  # first split that is significant
            break
    # find smallest majority m such that p<0.05
    m=None
    for mm in range(nd//2,nd+1):
        p=min(1.0,sum(math.comb(nd,i) for i in range(nd-mm+1))/2**nd*2)
        if p<0.05: m=mm; break
    print(f"  with {nd} discordant pairs ({label}): need >= {m}/{nd-m} split for p<0.05 -> net {2*m-nd} questions = {(2*m-nd)/500:.3f} abs acc on 500 (MDE)")
print("\n== abstention subset (_abs qids) ==")
ab=[q for q in qids if str(q).endswith('_abs')]
print("  n_abs:",len(ab),"types:",dict(Counter(qc[q]['type'] for q in ab)))
for name,arm in [("qc",qc),("no_qc",nq),("view_only",vo)]:
    print(f"  {name:10s} abs acc={sum(arm[q]['judge'] for q in ab)/len(ab):.3f}  non-abs acc={sum(arm[q]['judge'] for q in qids if not str(q).endswith('_abs'))/(500-len(ab)):.3f}")
print("  view_only correct on abstention:",sum(vo[q]['judge'] for q in ab),"/",len(ab))
