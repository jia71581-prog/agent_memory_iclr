"""Measurement 2 (v2): aggregation questions with an explicit ENUMERATION-COUNT subset.
OFFLINE - no API. Criteria:
  D1 loose : union of printed groups >= gold count AND every gold evidence session
             contributes at least one printed item
  D2 strict: ONE printed group alone has >= gold count items AND full gold-session coverage
Contrast arms: no hyponym table / no chain merging / v4 current view (same matched chains).
"""
import json
import pickle
import re
import statistics as st
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifecycle import assemble_chains, render_view
from v5_chains import (merge_chains, render_aggregation_view, aggregation_printed_groups,
                       is_aggregation_question, aggregation_kind)

D = {x["qid"]: x for x in pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl", "rb"))}
G = pickle.load(open("/root/agent_memory_iclr/results/_v5_gold.pkl", "rb"))
QC = {}
for line in open("/root/agent_memory_iclr/results/full500_qc.jsonl"):
    r = json.loads(line)
    if "judge" in r:
        QC[r["qid"]] = r

WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
           "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
           "nineteen": 19, "twenty": 20}
# magnitude units in the GOLD string -> the answer is a measured amount, not an item count
GOLD_UNIT = re.compile(r"\b(dollars?|\$|hours?|minutes?|miles?|pounds?|lbs?|kg|pages?|"
                       r"calories|percent|%|days?|weeks?|months?|years?|nights?|dozen|"
                       r"degrees?|stars?|followers?|points?|steps?|reps?|km|liters?|"
                       r"cups?|ounces?|oz|gallons?)\b")
# date/interval arithmetic, not enumeration of items
ARITH_Q = re.compile(r"\bhow many (days?|weeks?|months?|years?|hours?|minutes?|nights?)\b|"
                     r"\b(ago|older|younger|since I|before my|before the|left to|remaining)\b")


def gold_number(ans):
    s = str(ans).lower().replace(",", "")
    m = re.search(r"-?\d+(\.\d+)?", s)
    if m:
        return float(m.group(0))
    for w, v in WORDNUM.items():
        if re.search(rf"\b{w}\b", s):
            return float(v)
    return None


def is_enum_count(q, gold, gnum):
    """ENUMERATION-COUNT question: 'how many <things>' whose gold is a small integer and
    whose gold string is not a money/time/length magnitude. These are the questions whose
    answer must be DERIVED by counting a complete item set."""
    ql = q.lower()
    if not re.search(r"\bhow many\b|\bnumber of\b", ql):
        return False
    if gnum is None or gnum != int(gnum) or gnum > 12 or gnum < 1:
        return False
    if GOLD_UNIT.search(str(gold).lower()):
        return False
    if re.search(r"\bhow much\b", ql) or ARITH_Q.search(ql):
        return False
    return True


rows = []
for qid, x in D.items():
    is_agg, fired = is_aggregation_question(x["question"])
    if not is_agg:
        continue
    store4 = assemble_chains(x["facts"])
    store5 = merge_chains(store4)
    gnum = gold_number(x["answer"])
    gold_sidx = set(G[qid]["gold_sidx"])
    arms = {}
    for tag, store, hyp in [("v5", store5, True), ("nohyp", store5, False),
                            ("nomerge", store4, True)]:
        kept, _ = aggregation_printed_groups(store, x["question"], hyp)
        printed = [[it[1] for it in g["items"]] for g in kept]
        strong = [[it[1] for it in g["items"] if it[3] >= 2] for g in kept]
        allitems = [c for gi in printed for c in gi]
        cov = {c.get("sidx") for c in allitems} & gold_sidx
        arms[tag] = {"items": len(allitems), "groups": len(printed),
                     "best_group": max((len(gi) for gi in printed), default=0),
                     "best_strong": max((len(gi) for gi in strong), default=0),
                     "strong": sum(len(gi) for gi in strong),
                     "gold_hit": len(cov),
                     "gold_hit_best": max((len({c.get("sidx") for c in gi} & gold_sidx)
                                           for gi in printed), default=0)}
    # v4 contrast: distinct values the current-mode view exposes for the same matched chains
    groups5, _ = aggregation_printed_groups(store5, x["question"])
    v4_shown = 0
    for g in groups5:
        for k in g["chain_keys"]:
            chain = store5[k]
            if chain[0]["relation"] == "accumulative":
                v4_shown += len({c["value"].lower() for c in chain})
            else:
                last = (chain[-1]["date"], chain[-1].get("sidx"))
                v4_shown += len({c["value"].lower() for c in chain
                                 if (c["date"], c.get("sidx")) == last})
    # (i) does the FULL v4 current view expose at least one fact from every gold session?
    #     (ii) what fraction of gold-session facts does current-mode rendering drop?
    rendered_sidx, gold_facts, gold_rendered = set(), 0, 0
    for chain in store5.values():
        if chain[0]["relation"] == "accumulative":
            vis = chain
        else:
            last = (chain[-1]["date"], chain[-1].get("sidx"))
            vis = [c for c in chain if (c["date"], c.get("sidx")) == last]
        for c in vis:
            rendered_sidx.add(c.get("sidx"))
        for c in chain:
            if c.get("sidx") in gold_sidx:
                gold_facts += 1
                if c in vis:
                    gold_rendered += 1
    txt = render_aggregation_view(store5, x["question"])
    rows.append({
        "qid": qid, "type": x["type"], "question": x["question"], "gold": x["answer"],
        "gold_num": gnum, "enum": is_enum_count(x["question"], x["answer"], gnum),
        "kind": aggregation_kind(x["question"]),
        "judge": QC.get(qid, {}).get("judge"), "pred": (QC.get(qid, {}).get("pred") or "")[:400],
        "n_gold_sess": len(gold_sidx), "v4_current_shown": v4_shown,
        "agg_chars": len(txt), "view_chars": len(render_view(store5, "current")),
        "view_gold_hit": len(rendered_sidx & gold_sidx),
        "gold_facts": gold_facts, "gold_facts_rendered": gold_rendered,
        "snippet": txt, **{f"{t}_{k}": v for t, d in arms.items() for k, v in d.items()},
    })

with open("/root/agent_memory_iclr/results/_v5_agg2.json", "w") as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)

enum = [r for r in rows if r["enum"]]
print(f"aggregation trigger fires on {len(rows)}/500 questions "
      f"(count-kind {sum(1 for r in rows if r['kind']=='count')}, "
      f"amount-kind {sum(1 for r in rows if r['kind']=='amount')})")
print(f"ENUMERATION-COUNT subset: {len(enum)} questions; by type "
      f"{dict(Counter(r['type'] for r in enum))}")
print(f"v4 judge on the enum subset: {sum(1 for r in enum if r['judge'])}/"
      f"{sum(1 for r in enum if r['judge'] is not None)}")


def d1(r, t="v5"):
    return r[f"{t}_items"] >= r["gold_num"] and r[f"{t}_gold_hit"] == r["n_gold_sess"]


def d2(r, t="v5"):
    return r[f"{t}_best_group"] >= r["gold_num"] and r[f"{t}_gold_hit"] == r["n_gold_sess"]


print("\n=== enumeration-count derivability ===")
print(f"{'arm':34s} {'D1 union':>10s} {'D2 best group':>14s} {'gold-sess recall':>17s} "
      f"{'items med':>10s}")
tot_gs = sum(r["n_gold_sess"] for r in enum)
for t, lab in [("v5", "v5 merge + hyponyms (default)"), ("nohyp", "v5 without hyponym table"),
               ("nomerge", "v5 on UNMERGED v4 store")]:
    print(f"{lab:34s} {sum(d1(r,t) for r in enum)/len(enum):10.3f} "
          f"{sum(d2(r,t) for r in enum)/len(enum):14.3f} "
          f"{sum(r[f'{t}_gold_hit'] for r in enum)/tot_gs:17.3f} "
          f"{st.median([r[f'{t}_items'] for r in enum]):10.1f}")
print(f"{'v4 current view, same chains':34s} {sum(r['v4_current_shown'] >= r['gold_num'] for r in enum)/len(enum):10.3f} "
      f"{'n/a':>14s} {'n/a':>17s} {st.median([r['v4_current_shown'] for r in enum]):10.1f}")
print(f"gold count median {st.median([r['gold_num'] for r in enum])}, "
      f"mean {st.mean([r['gold_num'] for r in enum]):.1f}")
print(f"\nblock size: median {st.median([r['agg_chars'] for r in rows]):.0f} chars, "
      f"p90 {sorted(r['agg_chars'] for r in rows)[int(0.9*len(rows))]}, "
      f"max {max(r['agg_chars'] for r in rows)}; "
      f"full current view median {st.median([r['view_chars'] for r in rows]):.0f} chars "
      f"({st.median([r['agg_chars'] for r in rows])/st.median([r['view_chars'] for r in rows]):.3%} of it)")

print("\n=== concentration / structural loss (all 227 triggered questions) ===")
tg = sum(r["n_gold_sess"] for r in rows)
print(f"gold evidence sessions represented: full v4 current view "
      f"{sum(r['view_gold_hit'] for r in rows)/tg:.3f} at {st.median([r['view_chars'] for r in rows]):.0f} chars"
      f"  vs  v5 aggregation block {sum(r['v5_gold_hit'] for r in rows)/tg:.3f} at "
      f"{st.median([r['agg_chars'] for r in rows]):.0f} chars")
gf, gr = sum(r["gold_facts"] for r in rows), sum(r["gold_facts_rendered"] for r in rows)
print(f"facts extracted from gold sessions: {gf}; rendered by 'current' mode: {gr} "
      f"({gr/gf:.3f}) -> supersession truncation hides {1-gr/gf:.1%} of gold-session facts")

print("\n=== per-question, enumeration-count subset ===")
print(f"{'qid':16s} {'type':6s} {'gold':>4s} {'un':>3s} {'best':>4s} {'gs':>5s} {'D1':>5s} "
      f"{'D2':>5s} {'v4j':>4s}  question")
for r in sorted(enum, key=lambda r: (r["type"], r["qid"])):
    print(f"{r['qid']:16s} {r['type'][:6]:6s} {r['gold_num']:4.0f} {r['v5_items']:3d} "
          f"{r['v5_best_group']:4d} {r['v5_gold_hit']}/{r['n_gold_sess']:<3d} "
          f"{str(d1(r))[:5]:>5s} {str(d2(r))[:5]:>5s} {str(r['judge'])[:4]:>4s}  {r['question'][:66]}")
