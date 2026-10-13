"""Adversarial control 2: (a) per-question monotonicity of the dedupe sweep (is the
'recall drops with slack' finding a tautology?), (b) matched-budget control: give the
all-bugs arm a bigger print budget until its char count matches the shipped arm.
OFFLINE."""
import json
import pickle
import statistics as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _adv_v5_bugs import (D, G, measure, summarize, line, HDR, gold_number, is_enum_count,
                          apply_arm)
import v5_chains as V
from lifecycle import assemble_chains

SUB = None

# ---------- (a) per-question monotonicity of the printed set under the dedupe sweep
arms = {}
for tag, bugs, ded in [("exact", set(), "exact"), ("slack3", set(), 3),
                       ("slack6", set(), 6), ("unbound", {"i"}, "exact")]:
    arms[tag] = {r["qid"]: r for r in measure(bugs, ded) if r["enum"]}

qids = sorted(arms["exact"])
viol_items, viol_gold, strict_items = 0, 0, 0
for q in qids:
    seq = [arms[t][q]["items"] for t in ("exact", "slack3", "slack6", "unbound")]
    gseq = [arms[t][q]["gold_hit"] for t in ("exact", "slack3", "slack6", "unbound")]
    if any(seq[i] < seq[i + 1] for i in range(3)):
        viol_items += 1
    if any(gseq[i] < gseq[i + 1] for i in range(3)):
        viol_gold += 1
    if seq[0] > seq[-1]:
        strict_items += 1
print("=== (a) dedupe sweep: is it a tautology? (60 enum questions) ===")
print(f"printed-item count non-increasing exact >= slack3 >= slack6 >= unbounded: "
      f"{60-viol_items}/60 questions ({viol_items} violations)")
print(f"gold-session hits   non-increasing along the same order: "
      f"{60-viol_gold}/60 questions ({viol_gold} violations)")
print(f"questions where exact-only strictly prints more items than unbounded: {strict_items}/60")
print("total printed items:", {t: sum(r["items"] for r in a.values()) for t, a in arms.items()})
print("total gold-sess hits:", {t: sum(r["gold_hit"] for r in a.values()) for t, a in arms.items()})
print("total agg block chars:", {t: sum(r["agg_chars"] for r in a.values()) for t, a in arms.items()})

# ---------- (b) matched-budget control
print("\n=== (b) matched-budget control: all-four-bugs arm given a wider print budget ===")
print(HDR)
print(line(summarize(measure(set(), "exact"), "SHIPPED (all 4 fixes) rel_keep .35")))
ALL = {"i", "ii", "iii", "iv"}


def measure_budget(bugs, rel_keep, max_groups, max_items, label):
    apply_arm(bugs, "exact")
    rows = []
    for qid, x in D.items():
        if not V.is_aggregation_question(x["question"])[0]:
            continue
        store5 = V.merge_chains(assemble_chains(x["facts"]))
        gnum = gold_number(x["answer"])
        gs = set(G[qid]["gold_sidx"])
        kept, _ = V.aggregation_printed_groups(store5, x["question"], rel_keep=rel_keep,
                                              max_groups=max_groups,
                                              max_items_per_group=max_items)
        printed = [[it[1] for it in g["items"]] for g in kept]
        allitems = [c for gi in printed for c in gi]
        txt = V.render_aggregation_view(store5, x["question"], rel_keep=rel_keep,
                                        max_groups=max_groups, max_items_per_group=max_items)
        rows.append({"qid": qid, "type": x["type"], "question": x["question"],
                     "gold_num": gnum, "n_gold_sess": len(gs),
                     "enum": is_enum_count(x["question"], x["answer"], gnum),
                     "items": len(allitems), "groups": len(printed),
                     "best_group": max((len(gi) for gi in printed), default=0),
                     "gold_hit": len({c.get("sidx") for c in allitems} & gs),
                     "agg_chars": len(txt), "judge": None})
    apply_arm(set())
    return summarize(rows, label)


for rk, mg, mi in [(0.35, 8, 40), (0.20, 10, 60), (0.12, 14, 60), (0.06, 20, 80),
                   (0.02, 30, 120)]:
    print(line(measure_budget(ALL, rk, mg, mi, f"ALL-4-BUGS rel_keep {rk} groups {mg}")))
