"""Hunt for judge false positives among derivable+judged-correct TEMP.offset rows:
if any judged-True row is in fact wrong AND session-date derivable, the '<=3' bound breaks."""
import json, re, sys, itertools
from datetime import datetime
sys.path.insert(0,"/root/agent_memory_iclr/code")
from v5_shape import classify_shape
D=json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
R="/root/agent_memory_iclr/results"
QC={json.loads(l)["qid"]:json.loads(l) for l in open(f"{R}/full500_qc.jsonl")}
def pd(s): return datetime.strptime(s.split(" (")[0],"%Y/%m/%d")
NUMW={"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
      "eleven":11,"twelve":12,"fifteen":15,"twenty":20,"a":1,"an":1,"half":0.5}
PAIR=re.compile(r"(\d+(?:\.\d+)?|"+"|".join(k for k in NUMW if len(k)>1)+r")\s*(?:\+\s*\d+\s*)?(?:and a half\s*)?(second|minute|hour|day|night|week|month|year)s?",re.I)
NUM=re.compile(r"(\d+(?:\.\d+)?)")
QUNIT=re.compile(r"\b(seconds?|minutes?|hours?|days?|nights?|weeks?|months?|years?)\b",re.I)
def pairs(s):
    out=set()
    for m in PAIR.finditer(s):
        v=m.group(1).lower()
        v=float(v) if re.match(r"^\d",v) else float(NUMW[v])
        out.add((v,m.group(2).lower().rstrip("s")))
    return out
rows=[]
for q in D:
    if classify_shape(q["question"])[0]!="TEMP.offset" or q["question_id"].endswith("_abs"): continue
    sid2d=dict(zip(q["haystack_session_ids"],q["haystack_dates"]))
    evd=sorted({pd(sid2d[s]) for s in q["answer_session_ids"] if s in sid2d})
    gold=str(q["answer"]); dg={float(x) for x in NUM.findall(gold)}
    gp=pairs(gold); qu=[u.lower().rstrip("s") for u in QUNIT.findall(q["question"].lower())]
    if not gp and dg and qu: gp={(v,qu[0]) for v in dg}
    r=QC[q["question_id"]]
    rows.append(dict(qid=q["question_id"],q=q["question"],gold=gold,dg=dg,gp=gp,evd=evd,
                     qd=pd(q["question_date"]),judge=r["judge"],pred=r["pred"]))
E=[r for r in rows if r["dg"] and r["evd"]]
def deriv(r):
    A={abs((r["qd"]-e).days) for e in r["evd"]}
    B={abs((a-b).days) for a,b in itertools.combinations(r["evd"],2)}
    def ok(v): return (any(abs(v-x)<=1 for x in r["dg"]) or any(abs(v/7-x)<=0.5 for x in r["dg"])
                       or any(abs(v/30-x)<=0.5 for x in r["dg"]))
    return any(ok(v) for v in A|B)
sus=[]
for r in E:
    if not r["judge"]: continue
    pp=pairs(r["pred"])
    # gold value+unit stated anywhere in pred?
    ok=any(any(abs(pv-gv)<=0.05 and pu==gu for pv,pu in pp) for gv,gu in r["gp"])
    if not ok: sus.append((r,pp))
print(f"eligible={len(E)} judged-correct={sum(r['judge'] for r in E)} derivable={sum(deriv(r) for r in E)}")
print(f"judged-correct rows whose pred never states gold value+unit: {len(sus)}")
for r,pp in sus:
    print(f"\n  {r['qid'][:14]:15s} derivable={deriv(r)} gold={r['gold'][:60]}")
    print(f"    goldpairs={sorted(r['gp'])} predpairs={sorted(pp)[:8]}")
    print(f"    Q: {r['q'][:120]}")
    print(f"    pred: {r['pred'][:300].replace(chr(10),' ')}")
