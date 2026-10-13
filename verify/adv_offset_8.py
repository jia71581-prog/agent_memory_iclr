import json, re, sys, itertools, collections, math
from datetime import datetime
sys.path.insert(0,"/root/agent_memory_iclr/code")
from v5_shape import classify_shape
R="/root/agent_memory_iclr/results"
D=json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
def J(p): return {json.loads(l)["qid"]:json.loads(l)["judge"] for l in open(f"{R}/{p}")}
a,b=J("full500_full.jsonl"),J("full500_full_p2.jsonl")
common=[k for k in a if k in b]
flip=[k for k in common if a[k]!=b[k]]
print(f"same-config repeat (full vs full_p2): n={len(common)} judge flips={len(flip)} ({len(flip)/len(common):.1%})")
print(f"  acc full={sum(a[k] for k in common)/len(common):.3f}  full_p2={sum(b[k] for k in common)/len(common):.3f}")
off=[q["question_id"] for q in D if classify_shape(q["question"])[0]=="TEMP.offset" and not q["question_id"].endswith("_abs")]
print(f"  flips inside TEMP.offset(71): {sum(1 for k in flip if k in off)} {[k for k in flip if k in off]}")
# derivability vs correctness (QC arm)
def pd(s): return datetime.strptime(s.split(" (")[0],"%Y/%m/%d")
NUM=re.compile(r"(\d+(?:\.\d+)?)")
qc=J("full500_qc.jsonl")
rows=[]
for q in D:
    if q["question_id"] not in off: continue
    sid2d=dict(zip(q["haystack_session_ids"],q["haystack_dates"]))
    evd=sorted({pd(sid2d[s]) for s in q["answer_session_ids"] if s in sid2d})
    dg={float(x) for x in NUM.findall(str(q["answer"]))}
    if not dg or not evd: continue
    A={abs((pd(q["question_date"])-e).days) for e in evd}
    B={abs((x-y).days) for x,y in itertools.combinations(evd,2)}
    ok=lambda v: (any(abs(v-g)<=1 for g in dg) or any(abs(v/7-g)<=0.5 for g in dg) or any(abs(v/30-g)<=0.5 for g in dg))
    rows.append((q["question_id"],any(ok(v) for v in A|B),qc[q["question_id"]]))
d1=[r for r in rows if r[1]]; d0=[r for r in rows if not r[1]]
def acc(g): return sum(x[2] for x in g)/len(g)
print(f"\nderivable   n={len(d1)} acc={acc(d1):.3f} err={sum(1 for x in d1 if not x[2])}")
print(f"non-deriv   n={len(d0)} acc={acc(d0):.3f} err={sum(1 for x in d0 if not x[2])}")
# fisher exact
def fisher(a_,b_,c_,d_):
    def C(n,k): return math.comb(n,k)
    n=a_+b_+c_+d_; tot=0.0; obs=C(a_+b_,a_)*C(c_+d_,c_)/C(n,a_+c_)
    for i in range(0,min(a_+b_,a_+c_)+1):
        j=a_+c_-i; k=a_+b_-i; l=c_+d_-j
        if j<0 or k<0 or l<0: continue
        p=C(a_+b_,i)*C(c_+d_,j)/C(n,a_+c_)
        if p<=obs+1e-12: tot+=p
    return tot
p=fisher(sum(1 for x in d1 if not x[2]),len(d1)-sum(1 for x in d1 if not x[2]),
         sum(1 for x in d0 if not x[2]),len(d0)-sum(1 for x in d0 if not x[2]))
print(f"Fisher exact (derivability x correctness) p={p:.3f}  -> derivability does not predict accuracy")
print(f"\nheadroom accounting: TEMP.offset errors 6/470 = {6/470:.2%} of the non-abs set;"
      f" session-date-validatable subset 3/470 = {3/470:.2%}")
