"""Adversarial control 8: can the three reported sweep values 0.886 / 0.867 / 0.861 be
reproduced from ANY reachable dedupe variant?  Tries symmetric and one-directional subset
tests, under both the shipped and the pre-fix stemmer.  Also unit-tests the DISCRIMINATIVE
precision guard on the two examples the module's own docstring cites. OFFLINE."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_chains as V
from _adv_v5_bugs import measure, summarize, line, HDR, apply_arm, _ORIG


def one_dir(k):
    """dup if the NEW key is contained in an already-kept key (with slack k; None=unbounded)."""
    def f(a, b):
        if a == b:
            return True
        if not (a <= b):
            return False
        return True if k is None else (len(b) - len(a) <= k)
    return f


print("=== (h) can the reported sweep values 0.886 / 0.867 / 0.861 be reproduced? ===")
print(HDR)
for stem_lab, bugs in [("FIXED stem/SYN", set()), ("PRE-FIX stem/SYN", {"ii", "iii_nodbl", "iv"})]:
    for ded_lab, mk in [("exact", None), ("sym slack3", 3), ("sym slack6", 6),
                        ("dir slack3", "d3"), ("dir slack6", "d6"), ("dir unbounded", "du")]:
        if mk is None:
            arm_bugs, ded, patch = bugs, "exact", None
        elif isinstance(mk, int):
            arm_bugs, ded, patch = bugs, mk, None
        else:
            arm_bugs, ded = bugs, "exact"
            patch = one_dir({"d3": 3, "d6": 6, "du": None}[mk])
        rows = None
        if patch is None:
            rows = measure(arm_bugs, ded)
        else:
            apply_arm(arm_bugs, "exact")
            V._same_item = patch
            saved = V._same_item
            rows = []
            import pickle
            from lifecycle import assemble_chains
            from _adv_v5_bugs import D, G, gold_number, is_enum_count
            for qid, x in D.items():
                if not V.is_aggregation_question(x["question"])[0]:
                    continue
                V._same_item = patch
                store5 = V.merge_chains(assemble_chains(x["facts"]))
                gnum = gold_number(x["answer"])
                gs = set(G[qid]["gold_sidx"])
                kept, _ = V.aggregation_printed_groups(store5, x["question"])
                printed = [[it[1] for it in g["items"]] for g in kept]
                allitems = [c for gi in printed for c in gi]
                rows.append({"qid": qid, "type": x["type"], "question": x["question"],
                             "gold_num": gnum, "n_gold_sess": len(gs),
                             "enum": is_enum_count(x["question"], x["answer"], gnum),
                             "items": len(allitems), "groups": len(printed),
                             "best_group": max((len(gi) for gi in printed), default=0),
                             "gold_hit": len({c.get("sidx") for c in allitems} & gs),
                             "agg_chars": 0, "judge": None})
            apply_arm(set())
        print(line(summarize(rows, f"{stem_lab} / {ded_lab}")))

print("\n=== (i) DISCRIMINATIVE guard: the two examples the docstring claims it blocks ===")
apply_arm(set())
for a, b in [("original price range", "revised price range"),
             ("intended wake time", "actual wake time"),
             ("planned budget", "revised budget"),
             ("estimated cost", "cost"), ("expected delivery", "delivery"),
             ("updated price", "price"), ("intended bedtime", "bedtime"),
             ("original price", "price"), ("last payment", "payment")]:
    sa, sb = V._canon_tokens(a), V._canon_tokens(b)
    r = V._mergeable(sa, sb, None, 0.85, V._trailing_prep(a), V._trailing_prep(b))
    print(f"  {a:24s} <=> {b:22s} sigs {str(sa):28s} {str(sb):22s} -> "
          f"{'MERGED by ' + r if r else 'blocked'}")
