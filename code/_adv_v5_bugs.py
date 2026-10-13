"""ADVERSARIAL recomputation of the four-bug claim in v5_chains.py. OFFLINE, no API.

Reverts each claimed bug independently and measures D1 / D2 / gold-session recall /
printed-item count / rendered block chars on the 60-question ENUMERATION-COUNT subset,
using the exact metric code of _v5_meas_agg2.py.
"""
import json
import pickle
import re
import statistics as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifecycle import assemble_chains, render_view
import v5_chains as V

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
GOLD_UNIT = re.compile(r"\b(dollars?|\$|hours?|minutes?|miles?|pounds?|lbs?|kg|pages?|"
                       r"calories|percent|%|days?|weeks?|months?|years?|nights?|dozen|"
                       r"degrees?|stars?|followers?|points?|steps?|reps?|km|liters?|"
                       r"cups?|ounces?|oz|gallons?)\b")
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


# ----------------------------------------------------------------- bug reverts
_ORIG = {"_stem": V._stem, "_restore_e": V._restore_e, "_same_item": V._same_item,
         "_stem_syn_table": V._stem_syn_table}


def stem_old_order(t):
    """BUG (ii): plural stripping AFTER the -ing/-ed strip."""
    if len(t) > 4 and t.endswith("ing") and t[:-3][-1] not in "aeiou":
        t = V._restore_e(t[:-3])
    elif len(t) > 3 and t.endswith("ed") and t[:-2][-1] not in "aeiou":
        t = V._restore_e(t[:-2])
    if len(t) > 4 and t.endswith("ies"):
        t = t[:-3] + "y"
    elif len(t) > 4 and t.endswith(("sses", "shes", "ches", "xes", "zes")):
        t = t[:-2]
    elif len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        t = t[:-1]
    if len(t) > 4 and t.endswith("e") and V._measure(t[:-1]) > 1:
        t = t[:-1]
    return t


def restore_e_none(u):
    """BUG (iii): no -e restoration at all."""
    return u


def restore_e_nodbl(u):
    """BUG (iii) variant: keep the double-consonant reduction, drop the +e branches."""
    if len(u) > 2 and u[-1] == u[-2] and u[-1] not in "aeiouls":
        return u[:-1]
    return u


def syn_table_unstemmed(tbl):
    """BUG (iv): table keyed on unstemmed words only."""
    return dict(tbl)


def make_same_item(mode):
    """mode: 'exact' (shipped), 'unbounded' (BUG i), or an int slack."""
    if mode == "exact":
        return lambda a, b: a == b
    if mode == "unbounded":
        def f(a, b):
            return a == b or a <= b or b <= a
        return f
    k = int(mode)

    def f(a, b):
        if a == b:
            return True
        small, big = (a, b) if len(a) <= len(b) else (b, a)
        return small <= big and len(big) - len(small) <= k
    return f


def apply_arm(bugs, dedupe="exact"):
    V._stem = stem_old_order if "ii" in bugs else _ORIG["_stem"]
    if "iii" in bugs:
        V._restore_e = restore_e_none
    elif "iii_nodbl" in bugs:
        V._restore_e = restore_e_nodbl
    else:
        V._restore_e = _ORIG["_restore_e"]
    V._stem_syn_table = syn_table_unstemmed if "iv" in bugs else _ORIG["_stem_syn_table"]
    V._same_item = make_same_item("unbounded" if "i" in bugs else dedupe)
    V._SYN_STEMMED = None          # reset the memoised synonym table


def measure(bugs, dedupe="exact", max_items=40, subset=None, keep_rows=False):
    apply_arm(bugs, dedupe)
    rows = []
    for qid, x in D.items():
        if subset is not None and qid not in subset:
            continue
        is_agg, _ = V.is_aggregation_question(x["question"])
        if not is_agg:
            continue
        store4 = assemble_chains(x["facts"])
        store5 = V.merge_chains(store4)
        gnum = gold_number(x["answer"])
        gold_sidx = set(G[qid]["gold_sidx"])
        kept, _m = V.aggregation_printed_groups(store5, x["question"],
                                                max_items_per_group=max_items)
        printed = [[it[1] for it in g["items"]] for g in kept]
        allitems = [c for gi in printed for c in gi]
        cov = {c.get("sidx") for c in allitems} & gold_sidx
        txt = V.render_aggregation_view(store5, x["question"], max_items_per_group=max_items)
        rows.append({"qid": qid, "type": x["type"], "question": x["question"],
                     "gold_num": gnum, "n_gold_sess": len(gold_sidx),
                     "enum": is_enum_count(x["question"], x["answer"], gnum),
                     "items": len(allitems), "groups": len(printed),
                     "best_group": max((len(gi) for gi in printed), default=0),
                     "gold_hit": len(cov), "agg_chars": len(txt),
                     "judge": QC.get(qid, {}).get("judge")})
    apply_arm(set())               # restore
    return rows


def summarize(rows, label):
    enum = [r for r in rows if r["enum"]]
    tot_gs = sum(r["n_gold_sess"] for r in enum)
    d1 = sum(r["items"] >= r["gold_num"] and r["gold_hit"] == r["n_gold_sess"] for r in enum)
    d2 = sum(r["best_group"] >= r["gold_num"] and r["gold_hit"] == r["n_gold_sess"]
             for r in enum)
    gh = sum(r["gold_hit"] for r in enum)
    return {"label": label, "n_enum": len(enum), "D1": d1 / len(enum), "D1_n": d1,
            "D2": d2 / len(enum), "recall": gh / tot_gs, "gh": gh, "tot_gs": tot_gs,
            "items_med": st.median([r["items"] for r in enum]),
            "items_sum": sum(r["items"] for r in enum),
            "chars_med": st.median([r["agg_chars"] for r in enum]),
            "chars_sum": sum(r["agg_chars"] for r in enum),
            "all_recall": sum(r["gold_hit"] for r in rows) / sum(r["n_gold_sess"] for r in rows),
            "all_chars_med": st.median([r["agg_chars"] for r in rows]),
            "n_all": len(rows)}


HDR = (f"{'arm':44s} {'nQ':>3s} {'D1':>6s} {'D1n':>6s} {'D2':>6s} {'recall':>7s} {'gh/tot':>9s} "
       f"{'itMed':>6s} {'itSum':>6s} {'chMed':>6s} {'chSum':>7s}")


def line(s):
    return (f"{s['label']:44s} {s['n_enum']:3d} {s['D1']:6.3f} {s['D1_n']:6d} {s['D2']:6.3f} "
            f"{s['recall']:7.3f} {str(s['gh'])+'/'+str(s['tot_gs']):>9s} "
            f"{s['items_med']:6.1f} {s['items_sum']:6d} {s['chars_med']:6.0f} {s['chars_sum']:7d}")


if __name__ == "__main__":
    arms = [
        (set(), "exact", "SHIPPED (all 4 fixes)"),
        ({"i"}, "exact", "revert (i) unbounded subset dedupe"),
        ({"ii"}, "exact", "revert (ii) plural-after-ing"),
        ({"iii"}, "exact", "revert (iii) no -e restore (identity)"),
        ({"iii_nodbl"}, "exact", "revert (iii) no -e restore (keep dbl)"),
        ({"iv"}, "exact", "revert (iv) unstemmed SYN table"),
        ({"i", "ii", "iii", "iv"}, "exact", "revert ALL FOUR"),
        ({"i", "ii", "iii_nodbl", "iv"}, "exact", "revert ALL FOUR (iii keep dbl)"),
        (set(), 3, "shipped code, dedupe slack 3"),
        (set(), 6, "shipped code, dedupe slack 6"),
        ({"ii", "iii", "iv"}, "exact", "old stemmer+SYN, dedupe exact"),
        ({"ii", "iii", "iv"}, 3, "old stemmer+SYN, dedupe slack 3"),
        ({"ii", "iii", "iv"}, 6, "old stemmer+SYN, dedupe slack 6"),
        ({"ii", "iii", "iv"}, "unbounded", "old stemmer+SYN, dedupe unbounded"),
    ]
    out = []
    print(HDR)
    for bugs, ded, lab in arms:
        s = summarize(measure(bugs, ded), lab)
        out.append(s)
        print(line(s), flush=True)
    with open("/root/agent_memory_iclr/results/_adv_v5_bugs.json", "w") as f:
        json.dump(out, f, indent=1)
