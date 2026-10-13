"""Adversarial control 4: (a) same-knob comparison of buggy vs fixed at several print
budgets, (b) precision of the lines that fix (i) adds back, (c) overcount exposure
(printed items vs gold count), (d) association of D1 with the actual v4 judge. OFFLINE."""
import json
import statistics as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _adv_v5_bugs import D, G, measure, summarize, line, HDR
from _adv_v5_ctl import measure_budget

ALL = {"i", "ii", "iii", "iv"}
print("=== (a) SAME-KNOB comparison: does the fix add anything once the budget is wide? ===")
print(HDR)
for rk, mg, mi in [(0.35, 8, 40), (0.20, 10, 60), (0.12, 14, 60), (0.06, 20, 80)]:
    a = measure_budget(set(), rk, mg, mi, f"FIXED    rel_keep {rk} groups {mg}")
    b = measure_budget(ALL, rk, mg, mi, f"ALL-4-BUGS rel_keep {rk} groups {mg}")
    print(line(a))
    print(line(b))
    print(f"{'  -> delta attributable to the 4 fixes':44s} {'':3s} "
          f"{a['D1']-b['D1']:+6.3f} {a['D1_n']-b['D1_n']:+6d} {a['D2']-b['D2']:+6.3f} "
          f"{a['recall']-b['recall']:+7.3f} {'':>9s} {a['items_med']-b['items_med']:+6.1f} "
          f"{a['items_sum']-b['items_sum']:+6d} {a['chars_med']-b['chars_med']:+6.0f} "
          f"{a['chars_sum']-b['chars_sum']:+7d}")

print("\n=== (b) precision of the lines that fix (i) restores ===")
fixed = {r["qid"]: r for r in measure(set(), "exact") if r["enum"]}
unb = {r["qid"]: r for r in measure({"i"}, "exact") if r["enum"]}
di = sum(fixed[q]["items"] - unb[q]["items"] for q in fixed)
dg = sum(fixed[q]["gold_hit"] - unb[q]["gold_hit"] for q in fixed)
dc = sum(fixed[q]["agg_chars"] - unb[q]["agg_chars"] for q in fixed)
print(f"exact-only vs unbounded-subset dedupe on the 60 enum questions: "
      f"+{di} printed lines, +{dg} gold-session hits, +{dc} chars "
      f"({dc/sum(unb[q]['agg_chars'] for q in unb):+.1%})")
print(f"-> {dg}/{di} = {dg/di:.1%} of the restored lines carry a gold session "
      f"that was not already covered; the other {di-dg} are redundant for the metric")

print("\n=== (c) overcount exposure on the enum subset (gold answer IS the item count) ===")
for tag, arm in [("fixed (exact dedupe)", fixed), ("buggy (unbounded dedupe)", unb)]:
    rat = [arm[q]["items"] / fixed[q]["gold_num"] for q in arm]
    print(f"{tag:26s} printed items / gold count: median {st.median(rat):.1f}x, "
          f"mean {st.mean(rat):.1f}x, max {max(rat):.0f}x; "
          f"questions where printed > 3x gold: {sum(1 for r in rat if r > 3)}/60")

print("\n=== (d) does D1 track the real (v4) judge on these 60 questions? ===")
rows = [r for r in measure(set(), "exact") if r["enum"]]
tab = {}
for r in rows:
    d1 = r["items"] >= r["gold_num"] and r["gold_hit"] == r["n_gold_sess"]
    tab[(d1, r["judge"])] = tab.get((d1, r["judge"]), 0) + 1
print("  (D1, v4 judge) -> n :", {f"D1={k[0]},judge={k[1]}": v for k, v in sorted(tab.items(), key=str)})
jt = [r for r in rows if r["judge"] is not None]
d1t = [r for r in jt if r["items"] >= r["gold_num"] and r["gold_hit"] == r["n_gold_sess"]]
d1f = [r for r in jt if not (r["items"] >= r["gold_num"] and r["gold_hit"] == r["n_gold_sess"])]
print(f"  v4 answer accuracy | D1 True : {sum(1 for r in d1t if r['judge'])}/{len(d1t)} = "
      f"{sum(1 for r in d1t if r['judge'])/max(1,len(d1t)):.3f}")
print(f"  v4 answer accuracy | D1 False: {sum(1 for r in d1f if r['judge'])}/{len(d1f)} = "
      f"{sum(1 for r in d1f if r['judge'])/max(1,len(d1f)):.3f}")
allb = {r["qid"]: r for r in measure(ALL, "exact") if r["enum"]}


def d1of(r):
    return r["items"] >= r["gold_num"] and r["gold_hit"] == r["n_gold_sess"]


flips = [q for q in fixed if d1of(fixed[q]) and not d1of(allb[q])]
unflips = [q for q in fixed if not d1of(fixed[q]) and d1of(allb[q])]
print(f"\n  questions whose D1 flips False->True from all-4-bugs to fixed: {len(flips)}")
for q in flips:
    print(f"    {q:16s} gold={fixed[q]['gold_num']:.0f} items {allb[q]['items']:3d}->"
          f"{fixed[q]['items']:3d} goldsess {allb[q]['gold_hit']}/{fixed[q]['n_gold_sess']}->"
          f"{fixed[q]['gold_hit']}/{fixed[q]['n_gold_sess']} v4judge={fixed[q]['judge']}"
          f"  {fixed[q]['question'][:58]}")
print(f"  questions whose D1 flips True->False (regressions): {len(unflips)} {unflips}")
