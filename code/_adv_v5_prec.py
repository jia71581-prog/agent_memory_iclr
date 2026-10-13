"""Adversarial control 7: PRECISION of the printed item list, which no reported metric
measures.  On an enumeration-count question the answer is a count of items drawn from the
gold (answer) sessions, so lines sourced from non-gold sessions are distractors that push
the model to over-count.  Compare arms on precision as well as recall. OFFLINE."""
import statistics as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifecycle import assemble_chains
import v5_chains as V
from _adv_v5_bugs import D, G, apply_arm, gold_number, is_enum_count


def run(bugs, dedupe, label):
    apply_arm(bugs, dedupe)
    P = []
    tot_items = tot_gold_items = 0
    ratio = []
    for qid, x in D.items():
        gnum = gold_number(x["answer"])
        if not is_enum_count(x["question"], x["answer"], gnum):
            continue
        gs = set(G[qid]["gold_sidx"])
        store = V.merge_chains(assemble_chains(x["facts"]))
        kept, _ = V.aggregation_printed_groups(store, x["question"])
        items = [it[1] for g in kept for it in g["items"]]
        if not items:
            continue
        ng = sum(1 for c in items if c.get("sidx") in gs)
        tot_items += len(items)
        tot_gold_items += ng
        P.append(ng / len(items))
        ratio.append(len(items) / gnum)
    apply_arm(set())
    print(f"{label:34s} lines from gold sessions {tot_gold_items}/{tot_items} = "
          f"{tot_gold_items/tot_items:.3f} (macro {st.mean(P):.3f}) | printed/gold-count "
          f"median {st.median(ratio):.1f}x mean {st.mean(ratio):.1f}x")


print("=== (g) precision of the printed list on the 60 enumeration-count questions ===")
run(set(), "exact", "SHIPPED (all 4 fixes)")
run(set(), 3, "shipped, dedupe slack 3")
run(set(), 6, "shipped, dedupe slack 6")
run({"i"}, "exact", "unbounded subset dedupe (bug i)")
run({"i", "ii", "iii_nodbl", "iv"}, "exact", "ALL FOUR BUGS")
