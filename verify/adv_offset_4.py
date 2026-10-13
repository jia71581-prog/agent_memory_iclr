"""Independent recomputation of the TEMP.offset session-date derivability claim.
Variants of the derivability oracle, from the design agent's rule to strictly more
generous ones (all pairs, all sessions) and stricter unit-aware ones."""
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
WORDNUM=re.compile(r"\b("+"|".join(NUMW)+r")\b",re.I)
QUNIT=re.compile(r"\b(seconds?|minutes?|hours?|days?|nights?|weeks?|months?|years?)\b",re.I)
GU=re.compile(r"(\d+(?:\.\d+)?|"+"|".join(NUMW)+r")\s*(?:and a half\s*)?(second|minute|hour|day|night|week|month|year)s?",re.I)

def months_between(a,b):
    return (b.year-a.year)*12+(b.month-a.month)+(1 if b.day>=a.day else 0)-1 if False else (b.year-a.year)*12+(b.month-a.month)

rows=[]
for q in D:
    if classify_shape(q["question"])[0]!="TEMP.offset": continue
    if q["question_id"].endswith("_abs"): continue
    sid2d=dict(zip(q["haystack_session_ids"],q["haystack_dates"]))
    evd=sorted({pd(sid2d[s]) for s in q["answer_session_ids"] if s in sid2d})
    gold=str(q["answer"])
    digits={float(x) for x in NUM.findall(gold)}
    words={float(NUMW[w.lower()]) for w in WORDNUM.findall(gold)}
    gu=[(x.group(1).lower(),x.group(2).lower()) for x in GU.finditer(gold)]
    qunits=[u.lower().rstrip("s") for u in QUNIT.findall(q["question"].lower())]
    # gold unit: from gold string if present else from question
    gunits=sorted({u for _,u in gu}) or sorted(set(qunits))
    rows.append(dict(qid=q["question_id"],q=q["question"],gold=gold,digits=digits,words=words,
                     gu=gu,gunits=gunits,qunits=qunits,evd=evd,qd=pd(q["question_date"]),
                     judge=QC[q["question_id"]]["judge"],n_ev=len(q["answer_session_ids"]),
                     n_evdate=len(evd)))
print("non-abs TEMP.offset:",len(rows))
print("eligible (digit in gold and >=1 ev date):",sum(1 for r in rows if r["digits"] and r["evd"]))
print("distinct ev-session dates distribution:",collections.Counter(r["n_evdate"] for r in rows))
print("ev-session count distribution:",collections.Counter(r["n_ev"] for r in rows))

def loose_ok(v,golds):
    return any(abs(v-g)<=1 for g in golds) or any(abs(v/7-g)<=0.5 for g in golds) or any(abs(v/30-g)<=0.5 for g in golds)

def strict_ok(v,goldpairs,fallback_units,golds):
    """unit-aware: gold (value,unit) must match the day-count v read in that unit
    with human conventions: day -> v or v+1 ; week -> v/7 exact/floor/round ;
    month -> round(v/30.44) or floor ; year -> round(v/365.25)."""
    pairs=list(goldpairs)
    if not pairs:
        pairs=[(g,u) for g in golds for u in (fallback_units or ["day"])]
    for g,u in pairs:
        g=float(NUMW[g]) if isinstance(g,str) and not re.match(r"^\d",str(g)) else float(g)
        u=u.rstrip("s")
        if u in ("day","night"): cands={v,v+1}
        elif u=="week": cands={v/7.0,float(v//7),float(round(v/7.0))}
        elif u=="month": cands={round(v/30.44,2),float(v//30),float(round(v/30.44))}
        elif u=="year": cands={round(v/365.25,2),float(round(v/365.25))}
        else: cands=set()
        if any(abs(c-g)<=0.05 for c in cands): return True
    return False

def cand_sets(r,mode):
    qd,ev=r["qd"],r["evd"]
    A=set(); B=set()
    if mode=="agent":                 # first/last only, as in v5_offline_report
        if ev:
            A={abs((qd-ev[-1]).days),abs((qd-ev[0]).days)}
            if len(ev)>=2: B={abs((ev[-1]-ev[0]).days)}
    elif mode=="allpairs":            # every ev date vs qdate, every ev pair
        A={abs((qd-e).days) for e in ev}
        B={abs((a-b).days) for a,b in itertools.combinations(ev,2)}
    return A,B

def run(mode,matcher,include_words,label):
    elig=[r for r in rows if (r["digits"] or (include_words and r["words"])) and r["evd"]]
    hitA=hitB=hitAny=0; det=[]
    for r in elig:
        golds=set(r["digits"])|(r["words"] if include_words else set())
        gp=[(v,u) for v,u in r["gu"]]
        A,B=cand_sets(r,mode)
        if matcher=="loose":
            oa=any(loose_ok(v,golds) for v in A); ob=any(loose_ok(v,golds) for v in B)
        else:
            oa=any(strict_ok(v,gp,r["gunits"],golds) for v in A)
            ob=any(strict_ok(v,gp,r["gunits"],golds) for v in B)
        hitA+=oa; hitB+=ob; hitAny+=(oa or ob)
        det.append((r,oa,ob))
    n=len(elig)
    err=[d for d in det if not d[0]["judge"]]
    errder=[d for d in err if d[1] or d[2]]
    print(f"{label:46s} n={n:3d} qd-ev={hitA:3d} evpair={hitB:3d} either={hitAny:3d} ({hitAny/n:5.1%})"
          f"  errors={len(err)} derivable-errors={len(errder)} {[d[0]['qid'] for d in errder]}")
    return det

print()
d_agent=run("agent","loose",False,"V1 agent rule (first/last, unit-blind loose)")
run("allpairs","loose",False,"V2 all pairs+all qd-ev, unit-blind loose")
run("agent","loose",True,"V3 agent rule + word-number golds")
run("allpairs","loose",True,"V4 all pairs + word-number golds")
d_strict=run("allpairs","strict",True,"V5 all pairs + words, unit-aware strict")
run("agent","strict",True,"V6 agent cands, unit-aware strict")
json.dump([[r["qid"],oa,ob] for r,oa,ob in d_agent],open("/root/agent_memory_iclr/verify/off_v1.json","w"))
