"""Adversarial independent recomputation of D1 on the enumeration-count subset.
Re-derives the enum subset and D1 from the raw pickles, NOT from _v5_agg2.json.
"""
import json, pickle, re, sys, os, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifecycle import assemble_chains
from v5_chains import merge_chains, aggregation_printed_groups, is_aggregation_question

D = {x["qid"]: x for x in pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl", "rb"))}
G = pickle.load(open("/root/agent_memory_iclr/results/_v5_gold.pkl", "rb"))

WORDNUM = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,
           "nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,
           "fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,"twenty":20}
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

out = []
for qid, x in D.items():
    is_agg, _ = is_aggregation_question(x["question"])
    if not is_agg: continue
    gnum = gold_number(x["answer"])
    if not is_enum_count(x["question"], x["answer"], gnum): continue
    store4 = assemble_chains(x["facts"])
    store5 = merge_chains(store4)
    gold_sidx = set(G[qid]["gold_sidx"]) if not isinstance(G[qid]["gold_sidx"], str) else set(eval(G[qid]["gold_sidx"]))
    rec = {"qid": qid, "gold": x["answer"], "gold_num": gnum, "q": x["question"],
           "n_gold_sess": len(gold_sidx)}
    for tag, store, hyp in [("v5", store5, True), ("nohyp", store5, False), ("nomerge", store4, True)]:
        kept, _ = aggregation_printed_groups(store, x["question"], hyp)
        printed = [[it[1] for it in g["items"]] for g in kept]
        allitems = [c for gi in printed for c in gi]
        cov = {c.get("sidx") for c in allitems} & gold_sidx
        rec[f"{tag}_items"] = len(allitems)
        rec[f"{tag}_best"] = max((len(gi) for gi in printed), default=0)
        rec[f"{tag}_gold_hit"] = len(cov)
    out.append(rec)

json.dump(out, open("/root/agent_memory_iclr/results/_adv_d1.json", "w"), indent=1)
print("enum subset size:", len(out))
def d1(r, t="v5"): return r[f"{t}_items"] >= r["gold_num"] and r[f"{t}_gold_hit"] == r["n_gold_sess"]
def d2(r, t="v5"): return r[f"{t}_best"] >= r["gold_num"] and r[f"{t}_gold_hit"] == r["n_gold_sess"]
for t in ["v5", "nohyp", "nomerge"]:
    print(f"{t:10s} D1={sum(d1(r,t) for r in out)/len(out):.3f}  D2={sum(d2(r,t) for r in out)/len(out):.3f}"
          f"  items_med={st.median([r[f'{t}_items'] for r in out]):.1f}")
