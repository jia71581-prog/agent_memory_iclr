"""Adversarial recomputation of the hyponym-table claim. OFFLINE."""
import json, pickle, re, sys, os, statistics as st
from collections import Counter
sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains, render_view
import v5_chains as V

D = {x["qid"]: x for x in pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl", "rb"))}
G = pickle.load(open("/root/agent_memory_iclr/results/_v5_gold.pkl", "rb"))
QC = {}
for line in open("/root/agent_memory_iclr/results/full500_qc.jsonl"):
    r = json.loads(line)
    if "judge" in r:
        QC[r["qid"]] = r

WORDNUM = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
           "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,
           "sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,"twenty":20}
GOLD_UNIT = re.compile(r"\b(dollars?|\$|hours?|minutes?|miles?|pounds?|lbs?|kg|pages?|"
                       r"calories|percent|%|days?|weeks?|months?|years?|nights?|dozen|"
                       r"degrees?|stars?|followers?|points?|steps?|reps?|km|liters?|"
                       r"cups?|ounces?|oz|gallons?)\b")
ARITH_Q = re.compile(r"\bhow many (days?|weeks?|months?|years?|hours?|minutes?|nights?)\b|"
                     r"\b(ago|older|younger|since I|before my|before the|left to|remaining)\b")

def gold_number(ans):
    s = str(ans).lower().replace(",", "")
    m = re.search(r"-?\d+(\.\d+)?", s)
    if m: return float(m.group(0))
    for w, v in WORDNUM.items():
        if re.search(rf"\b{w}\b", s): return float(v)
    return None

def is_enum_count(q, gold, gnum):
    ql = q.lower()
    if not re.search(r"\bhow many\b|\bnumber of\b", ql): return False
    if gnum is None or gnum != int(gnum) or gnum > 12 or gnum < 1: return False
    if GOLD_UNIT.search(str(gold).lower()): return False
    if re.search(r"\bhow much\b", ql) or ARITH_Q.search(ql): return False
    return True

HAND = ["jewelry", "doctor", "sibling", "relative", "vehicle"]
FULL = dict(V.HYPONYMS)
NOHAND = {k: v for k, v in FULL.items() if k not in HAND}

def arm(store, question, hyp_table, use_hyp=True):
    V.HYPONYMS = hyp_table
    kept, _ = V.aggregation_printed_groups(store, question, use_hyp)
    V.HYPONYMS = FULL
    printed = [[it[1] for it in g["items"]] for g in kept]
    allitems = [c for gi in printed for c in gi]
    return kept, printed, allitems

rows = []
for qid, x in D.items():
    is_agg, _ = V.is_aggregation_question(x["question"])
    if not is_agg: continue
    gnum = gold_number(x["answer"])
    if not is_enum_count(x["question"], x["answer"], gnum): continue
    store4 = assemble_chains(x["facts"])
    store5 = V.merge_chains(store4)
    gold_sidx = set(G[qid]["gold_sidx"])
    rec = {"qid": qid, "type": x["type"], "q": x["question"], "gold": x["answer"],
           "gnum": gnum, "ngs": len(gold_sidx), "judge": QC.get(qid, {}).get("judge"),
           "pred": (QC.get(qid, {}).get("pred") or "")[:200]}
    for tag, store, tbl, uh in [("v5", store5, FULL, True),
                                ("nohyp", store5, FULL, False),
                                ("nohand", store5, NOHAND, True),
                                ("nomerge", store4, FULL, True)]:
        kept, printed, allitems = arm(store, x["question"], tbl, uh)
        cov = {c.get("sidx") for c in allitems} & gold_sidx
        rec[tag] = {"items": len(allitems), "groups": len(printed),
                    "best": max((len(gi) for gi in printed), default=0),
                    "gold_hit": len(cov),
                    "terms": [g["term"] for g in kept],
                    "sizes": [len(gi) for gi in printed]}
    rows.append(rec)

json.dump(rows, open("/root/agent_memory_iclr/verify/adv_hyp_rows.json", "w"), indent=1)

def d1(r, t): return r[t]["items"] >= r["gnum"] and r[t]["gold_hit"] == r["ngs"]
def d2(r, t): return r[t]["best"] >= r["gnum"] and r[t]["gold_hit"] == r["ngs"]

n = len(rows); tgs = sum(r["ngs"] for r in rows)
print(f"enum-count subset n={n}  total gold sessions={tgs}")
print(f"{'arm':10s} {'D1':>12s} {'D2':>12s} {'recall':>16s} {'items med':>10s}")
for t in ["v5", "nohyp", "nohand", "nomerge"]:
    a = sum(d1(r, t) for r in rows); b = sum(d2(r, t) for r in rows)
    c = sum(r[t]["gold_hit"] for r in rows)
    print(f"{t:10s} {a/n:8.3f}({a:2d}) {b/n:8.3f}({b:2d}) {c/tgs:10.3f}({c:3d}/{tgs}) "
          f"{st.median([r[t]['items'] for r in rows]):10.1f}")

print("\n--- decompose failure cause (v5 arm) ---")
for t in ["v5", "nohyp"]:
    cnt_fail = sum(1 for r in rows if r[t]["items"] < r["gnum"])
    cov_fail = sum(1 for r in rows if r[t]["gold_hit"] != r["ngs"])
    both = sum(1 for r in rows if r[t]["items"] < r["gnum"] and r[t]["gold_hit"] != r["ngs"])
    print(f"{t}: count-part fails {cnt_fail}, coverage-part fails {cov_fail}, both {both}")

print("\n--- questions that FLIP between v5 and nohyp ---")
for r in rows:
    f1 = d1(r, "v5") != d1(r, "nohyp"); f2 = d2(r, "v5") != d2(r, "nohyp")
    if f1 or f2:
        print(f"{r['qid']:15s} gold={r['gnum']:.0f} ngs={r['ngs']} "
              f"D1 {d1(r,'nohyp')}->{d1(r,'v5')} D2 {d2(r,'nohyp')}->{d2(r,'v5')} "
              f"judge={r['judge']}")
        print(f"    items {r['nohyp']['items']}->{r['v5']['items']} best "
              f"{r['nohyp']['best']}->{r['v5']['best']} gh {r['nohyp']['gold_hit']}->"
              f"{r['v5']['gold_hit']}/{r['ngs']}")
        print(f"    terms {r['nohyp']['terms']} -> {r['v5']['terms']}")
        print(f"    Q: {r['q'][:110]}")
