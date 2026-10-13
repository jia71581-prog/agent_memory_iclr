"""Independent adversarial recompute: is 'gold number literally printed in the arithmetic
block' a discriminative signal, or is it satisfied by construction?

Zero API. Nothing trusted from disk artifacts; everything recomputed from the module.
"""
import json
import os
import re
import statistics as st
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from v5_temporal import (parse_date, days_between, months_between, detect_family,
                         extract_events, precompute_arithmetic, render_timeline)
from v5_temporal_measure import parse_gold_values, delta_in_unit, gold_match
from v5_temporal_val import build_store_cached

R = "/root/agent_memory_iclr/results"

# numbers as literally rendered in the arithmetic block
_NUM = re.compile(r"(?<![\d/])(\d+(?:\.\d+)?)(?![\d/])")


def literal_numbers(lines):
    """Every number token the arithmetic block prints, excluding the date strings
    themselves (yyyy/mm/dd is stripped first)."""
    vals = set()
    for ln in lines:
        s = re.sub(r"\d{4}/\d{1,2}/\d{1,2}", " ", ln)
        s = re.sub(r"\d{1,2}/\d{1,2}=", " ", s)
        for m in _NUM.finditer(s):
            vals.add(float(m.group(1)))
    return vals


def main():
    data = [q for q in load_dataset("s_cleaned") if q["question_type"] == "temporal-reasoning"]
    qcrows = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{R}/full500_qc.jsonl")}

    recs = []
    for q in data:
        qid = q["question_id"]
        store, miss = build_store_cached(q)
        family, unit = detect_family(q["question"])
        events = extract_events(q, store, q["question"])
        lines = precompute_arithmetic(q["question"], q["question_date"], events, family, unit,
                                     q_for_anchor=q)
        gold_vals = parse_gold_values(q["answer"], unit)
        lit = literal_numbers(lines)
        tl = render_timeline(q, store, q["question"])
        recs.append({
            "qid": qid, "family": family, "unit": unit,
            "gold": str(q["answer"]), "gold_vals": gold_vals,
            "n_literal": len(lit), "literal": sorted(lit),
            "judge": qcrows.get(qid, {}).get("judge"),
            "cache_miss": miss, "tl_chars": len(tl), "n_lines": len(lines),
        })

    with open(f"{R}/_ver3_null.json", "w") as f:
        json.dump(recs, f)

    # ---- restrict to numeric-gold delta families (the pool the claim scores) ----
    pool = [r for r in recs if r["family"] in ("delta_between", "delta_since") and r["gold_vals"]]
    print(f"delta-family numeric-gold pool n={len(pool)}")

    def hit(r, gvals):
        u = (r["unit"] or "day").rstrip("s")
        for v, gu in gvals:
            if gu != u:
                continue
            for c in r["literal"]:
                if abs(c - v) <= 0.05:
                    return True
        return False

    true_hits = [hit(r, r["gold_vals"]) for r in pool]
    print(f"TRUE  gold literally printed: {sum(true_hits)}/{len(pool)} = {sum(true_hits)/len(pool):.3f}")

    # permutation null: swap in another question's gold (same asked unit)
    null_rates = []
    per_q = []
    for r in pool:
        others = [o for o in pool if o["qid"] != r["qid"]
                  and (o["unit"] or "day").rstrip("s") == (r["unit"] or "day").rstrip("s")]
        if not others:
            continue
        h = [hit(r, o["gold_vals"]) for o in others]
        per_q.append(sum(h) / len(h))
        null_rates.extend(h)
    print(f"NULL  other question's gold printed: {sum(null_rates)}/{len(null_rates)} "
          f"= {sum(null_rates)/len(null_rates):.3f}  (mean per-question {st.mean(per_q):.3f})")

    print(f"\ndistinct literal numbers per question: median {st.median([r['n_literal'] for r in pool])}, "
          f"mean {st.mean([r['n_literal'] for r in pool]):.1f}, max {max(r['n_literal'] for r in pool)}")

    # small-integer saturation: how much of the plausible answer space is printed?
    for r in pool[:0]:
        pass
    covered = []
    for r in pool:
        u = (r["unit"] or "day").rstrip("s")
        span = {"day": 60, "week": 20, "month": 12, "year": 5}.get(u, 60)
        ints = {float(k) for k in range(1, span + 1)}
        covered.append(len(ints & set(r["literal"])) / len(ints))
    print(f"fraction of plausible small-integer answer space literally printed: "
          f"median {st.median(covered):.3f} mean {st.mean(covered):.3f}")

    print("\n--- per-case: the 13 claimed evidence-present errors + 3 bonus ---")
    ids = ["b46e15ed", "6e984301", "gpt4_65aabe59", "gpt4_483dd43c", "gpt4_1e4a8aec", "eac54add",
           "gpt4_4929293b", "370a8ff4", "71017277", "gpt4_f420262d", "gpt4_f420262c",
           "gpt4_7abb270c", "d01c6aa8", "9a707b81", "2c63a862", "8c18457d"]
    by = {r["qid"]: r for r in recs}
    for i in ids:
        r = by[i]
        print(f"{i:18s} fam={r['family']:14s} unit={str(r['unit']):6s} nlit={r['n_literal']:3d} "
              f"gold={r['gold'][:28]:30s} hit={hit(r, r['gold_vals']) if r['gold_vals'] else 'n/a'} "
              f"miss_sess={r['cache_miss']}")


if __name__ == "__main__":
    main()
