"""Adversarial regression probe: the scaffold header says "trust these numbers".
Measure how often the pre-computed block asserts something that CONTRADICTS the gold,
split by whether the current pipeline already answers that question correctly.

A wrong assertion on a currently-correct question is a regression channel the
per-case claim never accounts for. Zero API.
"""
import json
import os
import re
import statistics as st
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from v5_temporal import (parse_date, days_between, months_between, detect_family, extract_events,
                         precompute_arithmetic, render_timeline, question_alternatives,
                         match_item_date, _delta_str)
from v5_temporal_measure import parse_gold_values, delta_in_unit, gold_match, norm
from v5_temporal_val import build_store_cached

R = "/root/agent_memory_iclr/results"


def unit_numbers(lines, unit):
    """Numbers the block prints IN THE ASKED UNIT (unit-aware, unlike a bare number scan)."""
    u = (unit or "day").rstrip("s")
    vals = set()
    pat = re.compile(r"(\d+(?:\.\d+)?)\s*(?:full\s+|calendar\s+)?" + u + r"s?\b", re.I)
    for ln in lines:
        for m in pat.finditer(ln):
            vals.add(float(m.group(1)))
    return vals


def stated_order(q, events):
    """For a 2-alternative ordering question, what order does the scaffold's
    chronological block imply? -> ('A_first'|'B_first'|'tie'|'undecidable', dates)"""
    alts = question_alternatives(q["question"])
    if len(alts) != 2:
        return "n_alts!=2", None
    d1 = match_item_date(alts[0], events)
    d2 = match_item_date(alts[1], events)
    if d1 is None or d2 is None:
        return "undecidable", (d1, d2)
    if d1 < d2:
        return "first_named_earlier", (d1, d2)
    if d2 < d1:
        return "second_named_earlier", (d1, d2)
    return "tie", (d1, d2)


def gold_picks(q, alts):
    """Which named alternative does the gold string pick? 0, 1, or None."""
    g = norm(q["answer"])
    sc = []
    for a in alts:
        toks = [t for t in norm(a).split() if len(t) > 2]
        sc.append(sum(t in g for t in toks) / max(1, len(toks)))
    if sc[0] > sc[1] and sc[0] >= 0.5:
        return 0
    if sc[1] > sc[0] and sc[1] >= 0.5:
        return 1
    return None


def main():
    data = [q for q in load_dataset("s_cleaned") if q["question_type"] == "temporal-reasoning"]
    qc = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{R}/full500_qc.jsonl")}

    delta_rows, ord_rows, sizes = [], [], []
    for q in data:
        qid = q["question_id"]
        store, miss = build_store_cached(q)
        family, unit = detect_family(q["question"])
        events = extract_events(q, store, q["question"])
        lines = precompute_arithmetic(q["question"], q["question_date"], events, family, unit,
                                     q_for_anchor=q)
        judge = qc.get(qid, {}).get("judge")
        sizes.append(len(render_timeline(q, store, q["question"])))

        if family in ("delta_between", "delta_since"):
            gv = parse_gold_values(q["answer"], unit)
            if gv:
                printed = unit_numbers(lines, unit)
                u = (unit or "day").rstrip("s")
                gold_here = {v for v, gu in gv if gu == u}
                hit = any(abs(c - v) <= 0.05 for c in printed for v in gold_here)
                delta_rows.append({"qid": qid, "unit": u, "judge": judge, "hit": hit,
                                   "n_printed": len(printed), "gold": sorted(gold_here),
                                   "printed": sorted(printed)})

        if family == "ordering":
            alts = question_alternatives(q["question"])
            if len(alts) == 2:
                st_, dts = stated_order(q, events)
                gp = gold_picks(q, alts)
                if gp is not None and st_ in ("first_named_earlier", "second_named_earlier", "tie"):
                    # question asks 'first' vs 'most recently/latest'
                    asks_first = bool(re.search(r"\bfirst\b|\bearliest\b", q["question"], re.I))
                    if st_ == "tie":
                        verdict = "tie_no_answer"
                    else:
                        scaffold_pick = 0 if st_ == "first_named_earlier" else 1
                        if not asks_first:
                            scaffold_pick = 1 - scaffold_pick
                        verdict = "agrees" if scaffold_pick == gp else "CONTRADICTS"
                    ord_rows.append({"qid": qid, "judge": judge, "verdict": verdict,
                                     "alts": alts, "gold": q["answer"][:40], "dates": str(dts)})
                else:
                    ord_rows.append({"qid": qid, "judge": judge, "verdict": "undecidable/" + st_,
                                     "alts": alts, "gold": q["answer"][:40], "dates": str(dts)})

    json.dump({"delta": delta_rows, "ordering": ord_rows, "sizes": sizes},
              open(f"{R}/_ver3_regress.json", "w"))

    print("=== DELTA families, numeric gold: is gold printed IN THE ASKED UNIT? ===")
    for j in (True, False):
        s = [r for r in delta_rows if r["judge"] is j]
        if s:
            print(f"  currently {'CORRECT' if j else 'WRONG  '} n={len(s):3d}  "
                  f"gold printed {sum(r['hit'] for r in s)}/{len(s)} = {sum(r['hit'] for r in s)/len(s):.3f}")
    allr = delta_rows
    print(f"  overall n={len(allr)} gold printed {sum(r['hit'] for r in allr)}/{len(allr)} "
          f"= {sum(r['hit'] for r in allr)/len(allr):.3f}")
    print(f"  numbers printed in asked unit: median {st.median([r['n_printed'] for r in allr])}, "
          f"mean {st.mean([r['n_printed'] for r in allr]):.1f}, max {max(r['n_printed'] for r in allr)}")
    print("  GOLD NOT PRINTED but block still asserts numbers (currently-correct questions):")
    for r in allr:
        if r["judge"] and not r["hit"] and r["n_printed"] > 0:
            print(f"    {r['qid']:16s} unit={r['unit']:6s} gold={r['gold']} printed={r['printed'][:9]}")

    print("\n=== ORDERING (2 named alternatives): does the chronological block agree with gold? ===")
    print(" ", Counter(r["verdict"] for r in ord_rows))
    for j in (True, False):
        s = [r for r in ord_rows if r["judge"] is j]
        c = Counter(r["verdict"] for r in s)
        dec = c["agrees"] + c["CONTRADICTS"]
        print(f"  currently {'CORRECT' if j else 'WRONG  '} n={len(s):3d} decidable={dec} "
              f"agrees={c['agrees']} CONTRADICTS={c['CONTRADICTS']} "
              f"tie={c['tie_no_answer']} " + (f"(accuracy {c['agrees']/dec:.3f})" if dec else ""))
    print("  CONTRADICTS on currently-CORRECT questions (regression candidates):")
    for r in ord_rows:
        if r["judge"] and r["verdict"] == "CONTRADICTS":
            print(f"    {r['qid']:16s} alts={r['alts']} gold={r['gold']!r} dates={r['dates']}")

    print(f"\n=== SIZE over all TR questions: median {st.median(sizes):.0f} mean {st.mean(sizes):.0f} "
          f"p90 {sorted(sizes)[int(.9*len(sizes))]:.0f} max {max(sizes)} chars ===")


if __name__ == "__main__":
    main()
