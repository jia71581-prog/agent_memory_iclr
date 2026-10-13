"""Audit judge labels on the 65 eligible TEMP.offset questions (numeric golds make a
rule-based check possible), and test tolerance/denominator sensitivity of 61.5%."""
import json, re, sys, itertools, collections
from datetime import datetime
sys.path.insert(0,"/root/agent_memory_iclr/code")
from v5_shape import classify_shape
D=json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
R="/root/agent_memory_iclr/results"
QC={json.loads(l)["qid"]:json.loads(l) for l in open(f"{R}/full500_qc.jsonl")}
def pd(s): return datetime.strptime(s.split(" (")[0],"%Y/%m/%d")
NUM=re.compile(r"(\d+(?:\.\d+)?)")
NUMW={"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
      "eleven":11,"twelve":12,"fifteen":15,"twenty":20}
GU=re.compile(r"(\d+(?:\.\d+)?|"+"|".join(NUMW)+r")\s*(?:and a half\s*)?(second|minute|hour|day|night|week|month|year)s?",re.I)
QUNIT=re.compile(r"\b(seconds?|minutes?|hours?|days?|nights?|weeks?|months?|years?)\b",re.I)
rows=[]
for q in D:
    if classify_shape(q["question"])[0]!="TEMP.offset" or q["question_id"].endswith("_abs"): continue
    sid2d=dict(zip(q["haystack_session_ids"],q["haystack_dates"]))
    evd=sorted({pd(sid2d[s]) for s in q["answer_session_ids"] if s in sid2d})
    gold=str(q["answer"]); digits={float(x) for x in NUM.findall(gold)}
    gu=[(x.group(1).lower(),x.group(2).lower()) for x in GU.finditer(gold)]
    qu=[u.lower().rstrip("s") for u in QUNIT.findall(q["question"].lower())]
    gunits=sorted({u for _,u in gu}) or sorted(set(qu))
    rows.append(dict(qid=q["question_id"],q=q["question"],gold=gold,digits=digits,gunits=gunits,
                     evd=evd,qd=pd(q["question_date"]),judge=QC[q["question_id"]]["judge"],
                     pred=QC[q["question_id"]]["pred"],f1=QC[q["question_id"]]["f1"]))
E=[r for r in rows if r["digits"] and r["evd"]]
def loose(v,g): return any(abs(v-x)<=1 for x in g) or any(abs(v/7-x)<=0.5 for x in g) or any(abs(v/30-x)<=0.5 for x in g)
def deriv(r,tol_day=1):
    A={abs((r["qd"]-e).days) for e in r["evd"]}
    B={abs((a-b).days) for a,b in itertools.combinations(r["evd"],2)}
    def ok(v): return (any(abs(v-x)<=tol_day for x in r["digits"]) or
                       any(abs(v/7-x)<=0.5 for x in r["digits"]) or
                       any(abs(v/30-x)<=0.5 for x in r["digits"]))
    return any(ok(v) for v in A), any(ok(v) for v in B)

print("=== judge-label audit on the 65 eligible (rule check: gold digit present in pred) ===")
dis=[]
for r in E:
    pn={float(x) for x in NUM.findall(r["pred"])}
    hit=any(any(abs(p-g)<=1e-9 for p in pn) for g in r["digits"])
    if hit!=r["judge"]: dis.append((r,hit,pn))
print(f"judge=True {sum(r['judge'] for r in E)}  judge=False {sum(not r['judge'] for r in E)}")
print(f"disagreements between judge and 'gold number appears verbatim in pred': {len(dis)}")
for r,hit,pn in dis:
    print(f"  {r['qid'][:14]:15s} judge={r['judge']} numhit={hit} gold={r['gold'][:28]:28s} prednums={sorted(pn)[:6]}")
    print(f"      pred: {r['pred'][:220]}")
print("\n=== the 6 QC-arm errors in full ===")
for r in E:
    if r["judge"]: continue
    a,b=deriv(r)
    print(f"  {r['qid'][:14]:15s} deriv(qd-ev)={a} deriv(pair)={b} gold={r['gold'][:40]}")
    print(f"      Q: {r['q'][:120]}")
    print(f"      ev dates: {[d.strftime('%Y/%m/%d') for d in r['evd']]}  qdate={r['qd'].strftime('%Y/%m/%d')}")
    print(f"      pred: {r['pred'][:240]}")
print("\n=== tolerance / denominator sensitivity ===")
for tol in (0,1,2,3):
    n=len(E); k=sum(1 for r in E if any(deriv(r,tol))); 
    errd=sum(1 for r in E if not r["judge"] and any(deriv(r,tol)))
    print(f"  day-tolerance +-{tol}: either={k}/{n} = {k/n:.1%}  derivable-errors={errd}")
HOUR=[r for r in E if r["gunits"]==["hour"] or (r["gunits"] and all(u in("hour","minute","second") for u in r["gunits"]))]
print(f"  hour/minute-unit golds among 65: {len(HOUR)} {[r['qid'] for r in HOUR]}")
sub=[r for r in E if r not in HOUR]
k=sum(1 for r in sub if any(deriv(r))); print(f"  excluding hour-unit golds: either={k}/{len(sub)} = {k/len(sub):.1%} (2/3 = 66.7%)")
for tol in (1,2):
    k=sum(1 for r in sub if any(deriv(r,tol))); print(f"    with day-tol +-{tol}: {k}/{len(sub)} = {k/len(sub):.1%}")
print("\n=== mechanism check: span structure of the 25 non-derivable ===")
nd=[r for r in E if not any(deriv(r))]
print(f"  non-derivable n={len(nd)}")
print(f"  of these, only ONE distinct ev-session date (no session pair exists): {sum(1 for r in nd if len(r['evd'])==1)}")
print(f"  of these, max session span = 0 days: {sum(1 for r in nd if len(set(r['evd']))<=1)}")
spans=[max((abs((a-b).days) for a,b in itertools.combinations(r['evd'],2)), default=0) for r in nd]
print(f"  span days distribution: {collections.Counter(spans)}")
print(f"  gold unit distribution: {collections.Counter(tuple(r['gunits']) for r in nd)}")
json.dump([r["qid"] for r in nd],open("/root/agent_memory_iclr/verify/off_nonderiv.json","w"))
