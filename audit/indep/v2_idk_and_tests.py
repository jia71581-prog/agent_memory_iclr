import json, math, re
from collections import Counter
R="/root/agent_memory_iclr/results/"
def load(t): return {r["qid"]: r for r in (json.loads(l) for l in open(R+f"full500_{t}.jsonl"))}
arms={t:load(t) for t in ["qc","qc_no_qc","qc_view_only","qc_no_lifecycle","full"]}
vo=arms["qc_view_only"]; qc=arms["qc"]; nolc=arms["qc_no_lifecycle"]
ds=json.load(open("/root/agent_memory_iclr/data/longmemeval_oracle.json"))
meta={x["question_id"]:x for x in ds}
qids=sorted(vo); ABS=[q for q in qids if q.endswith("_abs")]; NON=[q for q in qids if not q.endswith("_abs")]

# --- three independent IDK definitions (robustness of the 18.6% figure) ---
defs={
 "stated_regex": re.compile(r"don't know|do not know|no (relevant )?information|not (mentioned|provided|available|found)|cannot determine|unable to",re.I),
 "narrow": re.compile(r"\b(i )?(don't|do not|dont) know\b|no information (in|about|available)|there is no (information|record|mention)",re.I),
 "broad": re.compile(r"don'?t know|do not know|no (relevant |such )?(information|record|mention|data)|not (mentioned|provided|available|found|specified|stated|recorded)|cannot (determine|find|tell|answer)|unable to|isn'?t (any )?(information|mention)|no mention",re.I),
}
print("== IDK-phrase rate per arm, 3 definitions ==")
for name,a in arms.items():
    out=[]
    for dn,rx in defs.items():
        n=sum(1 for q in qids if rx.search(a[q]['pred']))
        out.append(f"{dn}={n:3d}({n/500:.3f})")
    print(f"  {name:16s} "+"  ".join(out))

rx=defs["stated_regex"]
print("\n== view_only refusal precision (stated regex) ==")
ref=[q for q in qids if rx.search(vo[q]['pred'])]
print(f"  refusals={len(ref)}  of those _abs={sum(1 for q in ref if q.endswith('_abs'))}  judged correct={sum(bool(vo[q]['judge']) for q in ref)}")
print(f"  precision(refusal correct)={sum(bool(vo[q]['judge']) for q in ref)/len(ref):.3f}")
print(f"  abs recall by refusal: {sum(1 for q in ABS if rx.search(vo[q]['pred']))}/{len(ABS)}")
print(f"  abs correct WITHOUT refusal phrase: {sum(1 for q in ABS if bool(vo[q]['judge']) and not rx.search(vo[q]['pred']))}")
print(f"  non-abs refusals (false refusals): {sum(1 for q in NON if rx.search(vo[q]['pred']))}  of which judged correct: {sum(1 for q in NON if rx.search(vo[q]['pred']) and vo[q]['judge'])}")
for name,a in [("qc",qc),("no_lifecycle",nolc)]:
    r2=[q for q in qids if rx.search(a[q]['pred'])]
    print(f"  {name:12s} refusals={len(r2)} abs-recall={sum(1 for q in ABS if rx.search(a[q]['pred']))}/{len(ABS)} abs acc={sum(bool(a[q]['judge']) for q in ABS)}/{len(ABS)}")

print("\n== per-type IDK rate for view_only (mechanism check) ==")
for t in sorted({meta[q]['question_type'] for q in qids}):
    ids=[q for q in NON if meta[q]['question_type']==t]
    n=sum(1 for q in ids if rx.search(vo[q]['pred']))
    print(f"  {t:28s} non-abs n={len(ids):3d} IDK={n:3d} ({n/len(ids):.3f})  vo acc={sum(bool(vo[q]['judge']) for q in ids)/len(ids):.3f}  qc acc={sum(bool(qc[q]['judge']) for q in ids)/len(ids):.3f}")

# --- omnibus heterogeneity test across the 6 subtypes (guards the fishing concern) ---
def chi2_p(x2,df):
    # regularized upper incomplete gamma via series/continued fraction
    a=df/2.0; xx=x2/2.0
    def gser(a,x):
        ap=a; s=1.0/a; d=s
        for n in range(1,500):
            ap+=1; d*=x/ap; s+=d
            if abs(d)<abs(s)*1e-14: break
        return s*math.exp(-x+a*math.log(x)-math.lgamma(a))
    def gcf(a,x):
        tiny=1e-300; b=x+1-a; c=1/tiny; d=1/b; h=d
        for i in range(1,500):
            an=-i*(i-a); b+=2; d=an*d+b
            if abs(d)<tiny: d=tiny
            c=b+an/c
            if abs(c)<tiny: c=tiny
            d=1/d; de=d*c; h*=de
            if abs(de-1)<1e-14: break
        return math.exp(-x+a*math.log(x)-math.lgamma(a))*h
    return gcf(a,xx) if xx>=a+1 else 1-gser(a,xx)
def chi2_table(counts):
    K=sum(k for k,n in counts); N=sum(n for k,n in counts); p=K/N; x2=0.0
    for k,n in counts:
        for obs,exp in [(k,n*p),(n-k,n*(1-p))]:
            x2+=(obs-exp)**2/exp
    return x2,len(counts)-1
for label,idset in [("all 500",qids),("non-abstention 470",NON)]:
    cts=[]
    for t in sorted({meta[q]['question_type'] for q in qids}):
        ids=[q for q in idset if meta[q]['question_type']==t]
        cts.append((sum(bool(vo[q]['judge']) for q in ids),len(ids)))
    x2,df=chi2_table(cts)
    print(f"\n== omnibus heterogeneity across 6 subtypes ({label}): chi2={x2:.2f} df={df} p={chi2_p(x2,df):.3e} ==")
    print("   cells:",cts)

# --- pairwise subtype comparisons with Bonferroni/Holm (fishing-expedition guard) ---
def fisher2(a,b,c,d):
    from math import comb
    n=a+b+c+d
    def px(x): return comb(a+b,x)*comb(c+d,a+c-x)/comb(n,a+c)
    p0=px(a); lo=max(0,a+c-(c+d)); hi=min(a+b,a+c)
    return min(1.0,sum(px(x) for x in range(lo,hi+1) if px(x)<=p0*(1+1e-9)))
types=sorted({meta[q]['question_type'] for q in qids})
res=[]
for i in range(len(types)):
    for j in range(i+1,len(types)):
        A=[q for q in NON if meta[q]['question_type']==types[i]]
        B=[q for q in NON if meta[q]['question_type']==types[j]]
        ka=sum(bool(vo[q]['judge']) for q in A); kb=sum(bool(vo[q]['judge']) for q in B)
        p=fisher2(ka,len(A)-ka,kb,len(B)-kb)
        res.append((p,types[i],types[j],f"{ka}/{len(A)}={ka/len(A):.3f}",f"{kb}/{len(B)}={kb/len(B):.3f}"))
res.sort()
m=len(res); print(f"\n== all {m} pairwise subtype Fisher tests (non-abstention), Holm-corrected ==")
for r,(p,t1,t2,s1,s2) in enumerate(res):
    holm=min(1.0,p*(m-r)); print(f"  {t1:26s} vs {t2:26s} {s1:>14s} {s2:>14s} p={p:.2e} holm={holm:.2e} {'SIG' if holm<0.05 else 'ns'}")

# --- abs vs non-abs test for each arm (is view_only's abstention edge special?) ---
print("\n== abs vs non-abs within each arm, Fisher exact (unpaired, different questions) ==")
for name,a in arms.items():
    ka=sum(bool(a[q]['judge']) for q in ABS); kn=sum(bool(a[q]['judge']) for q in NON)
    p=fisher2(ka,len(ABS)-ka,kn,len(NON)-kn)
    print(f"  {name:16s} abs {ka}/{len(ABS)}={ka/len(ABS):.3f} vs non {kn}/{len(NON)}={kn/len(NON):.3f}  p={p:.3e}")
