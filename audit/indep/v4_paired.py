import json, math
from collections import Counter
R="/root/agent_memory_iclr/results/"
def load(t): return {r["qid"]: r for r in (json.loads(l) for l in open(R+f"full500_{t}.jsonl"))}
arms={t:load(t) for t in ["qc","qc_no_qc","qc_view_only","qc_no_lifecycle","full"]}
ds={x["question_id"]:x for x in json.load(open("/root/agent_memory_iclr/data/longmemeval_oracle.json"))}
qids=sorted(arms["qc"]); ABS=[q for q in qids if q.endswith("_abs")]
pairs=[(q,q[:-4]) for q in ABS if q[:-4] in arms["qc"]]
print("paired abs/base-twin pairs available:",len(pairs),"of",len(ABS))
def mcnemar(pairs_bool):
    n01=sum(1 for a,b in pairs_bool if (not a) and b); n10=sum(1 for a,b in pairs_bool if a and not b)
    n=n01+n10
    p=1.0 if n==0 else min(1.0, 2*sum(math.comb(n,i) for i in range(min(n01,n10)+1))/2**n)
    return n01,n10,p
print("\n== PAIRED: abstention question vs its own answerable twin (same haystack) ==")
for name,a in arms.items():
    pb=[(bool(a[x]["judge"]),bool(a[y]["judge"])) for x,y in pairs]
    ka=sum(1 for u,v in pb if u); kb=sum(1 for u,v in pb if v)
    n01,n10,p=mcnemar(pb)
    print(f"  {name:16s} abs {ka:2d}/{len(pb)}={ka/len(pb):.3f}  twin {kb:2d}/{len(pb)}={kb/len(pb):.3f}  abs-only-right={n10} twin-only-right={n01} exact p={p:.4f}")
print("\n== same-question paired comparison ON THE ABSTENTION SUBSET: view_only vs qc / no_lifecycle ==")
vo=arms["qc_view_only"]
for name in ["qc","qc_no_qc","qc_no_lifecycle","full"]:
    a=arms[name]
    pb=[(bool(vo[q]["judge"]),bool(a[q]["judge"])) for q in ABS]
    n01,n10,p=mcnemar(pb)
    print(f"  abs subset (n=30): view_only {sum(1 for u,v in pb if u)} vs {name} {sum(1 for u,v in pb if v)}   vo-only={n10} other-only={n01} p={p:.4f}")
NON=[q for q in qids if not q.endswith("_abs")]
for name in ["qc","qc_no_lifecycle"]:
    a=arms[name]
    pb=[(bool(vo[q]["judge"]),bool(a[q]["judge"])) for q in NON]
    n01,n10,p=mcnemar(pb)
    print(f"  non-abs (n=470): view_only {sum(1 for u,v in pb if u)} vs {name} {sum(1 for u,v in pb if v)}   vo-only={n10} other-only={n01} p={p:.3e}")

print("\n== interaction: is the abs-vs-nonabs gap SPECIFIC to view_only? (2x2x2, Breslow-Day-style) ==")
def orat(a,b,c,d): return ((a+0.5)*(d+0.5))/((b+0.5)*(c+0.5))
for name,a in arms.items():
    ka=sum(bool(a[q]["judge"]) for q in ABS); kn=sum(bool(a[q]["judge"]) for q in NON)
    print(f"  {name:16s} OR(abs correct vs nonabs correct) = {orat(ka,30-ka,kn,470-kn):7.2f}   logOR={math.log(orat(ka,30-ka,kn,470-kn)):+.2f}")
# z-test on difference of log-ORs view_only vs qc
def lor_se(a,b,c,d): return math.sqrt(1/(a+.5)+1/(b+.5)+1/(c+.5)+1/(d+.5))
def cell(a): 
    ka=sum(bool(a[q]["judge"]) for q in ABS); kn=sum(bool(a[q]["judge"]) for q in NON); return ka,30-ka,kn,470-kn
for other in ["qc","qc_no_qc","qc_no_lifecycle","full"]:
    A=cell(vo); B=cell(arms[other])
    d=math.log(orat(*A))-math.log(orat(*B)); se=math.sqrt(lor_se(*A)**2+lor_se(*B)**2)
    z=d/se; p=math.erfc(abs(z)/math.sqrt(2))
    print(f"  view_only vs {other:16s} interaction logOR diff={d:+.2f} se={se:.2f} z={z:+.2f} p={p:.2e}")

print("\n== decomposition of the 0.500 headline ==")
k=sum(bool(vo[q]["judge"]) for q in qids); ka=sum(bool(vo[q]["judge"]) for q in ABS); kn=sum(bool(vo[q]["judge"]) for q in NON)
print(f"  0.500 = ({ka} abs-correct + {kn} nonabs-correct)/500;  abs contributes {ka/500:.3f} of the 0.500 headline ({ka/k:.1%} of all its correct answers)")
print(f"  counterfactual: if view_only scored its non-abs rate on the abs items, headline = {(kn/len(NON)*30+kn)/500:.4f}")
print(f"  reported non-abs completeness = {kn/len(NON):.4f}; overstatement of headline = {k/500-kn/len(NON):+.4f}")
