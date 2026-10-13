"""Stage 4: does derivability predict answer correctness / headroom? + mechanism check."""
import json
import re
import statistics as st
import sys
from collections import Counter
from datetime import date

sys.path.insert(0, "/root/agent_memory_iclr/code")

R = "/root/agent_memory_iclr/results"
rows = json.load(open(f"{R}/_adv_v5t_tr.json"))
by = {r["qid"]: r for r in rows}
DV = [r for r in rows if r["family"] in ("delta_since", "delta_between") and r["gold_vals"]]
DALL = [r for r in rows if r["family"] in ("delta_since", "delta_between")]

runs = {}
for tag in ("qc", "full", "full_p2", "fifo", "decay", "qc_no_lifecycle", "qc_no_qc",
            "qc_view_only"):
    p = f"{R}/full500_{tag}.jsonl"
    try:
        runs[tag] = {json.loads(l)["qid"]: json.loads(l) for l in open(p)}
    except FileNotFoundError:
        pass
print("runs loaded:", {k: len(v) for k, v in runs.items()})


def j(x):
    return str(x).strip().lower() in ("true", "1", "yes")


print()
print("=" * 100)
print("TEST 8  does derivTL / literal predict actual answer correctness of the existing systems?")
print("        (existing runs have no temporal scaffold: this tests whether the proxy tracks")
print("         which delta questions are actually hard)")
print("=" * 100)
for tag, rr in runs.items():
    tab = {}
    for r in DALL:
        row = rr.get(r["qid"])
        if row is None:
            continue
        k = (bool(r["gold_derivable_from_timeline"]), j(row["judge"]))
        tab[k] = tab.get(k, 0) + 1
    n1 = tab.get((True, True), 0) + tab.get((True, False), 0)
    n0 = tab.get((False, True), 0) + tab.get((False, False), 0)
    a1 = tab.get((True, True), 0) / n1 if n1 else float("nan")
    a0 = tab.get((False, True), 0) / n0 if n0 else float("nan")
    print(f"  {tag:16s} acc|derivTL=1 = {a1:.3f} (n={n1})   acc|derivTL=0 = {a0:.3f} (n={n0})")

print()
print("=" * 100)
print("TEST 9  headroom placement: on the delta questions the best current system gets WRONG,")
print("        does the scaffold actually supply the arithmetic?")
print("=" * 100)
best = max(runs, key=lambda t: sum(j(v["judge"]) for v in runs[t].values()))
print("best run by overall judge accuracy:", best,
      round(sum(j(v["judge"]) for v in runs[best].values()) / len(runs[best]), 4))
for tag in ("qc", "full_p2", best):
    rr = runs[tag]
    wrong = [r for r in DALL if r["qid"] in rr and not j(rr[r["qid"]]["judge"])]
    right = [r for r in DALL if r["qid"] in rr and j(rr[r["qid"]]["judge"])]
    if not wrong:
        continue
    print(f"  {tag}: delta questions wrong={len(wrong)} right={len(right)}")
    print(f"      derivTL among WRONG = {sum(bool(r['gold_derivable_from_timeline']) for r in wrong)}/{len(wrong)}"
          f" = {sum(bool(r['gold_derivable_from_timeline']) for r in wrong)/len(wrong):.3f}")
    print(f"      derivTL among RIGHT = {sum(bool(r['gold_derivable_from_timeline']) for r in right)}/{len(right)}"
          f" = {sum(bool(r['gold_derivable_from_timeline']) for r in right)/len(right):.3f}")
    print(f"      literal among WRONG = {sum(bool(r['gold_literal_in_arith']) for r in wrong)}/{len(wrong)}")
    print("      wrong qids:", [r["qid"][:12] + ("+" if r["gold_derivable_from_timeline"] else "-")
                                for r in wrong])

print()
print("=" * 100)
print("TEST 10  mechanism check: is the delta_between rescue really 'two events inside one session'?")
print("=" * 100)
DB = [r for r in DV if r["family"] == "delta_between"]
one_sess = [r for r in DB if len(r["pool_ans"]) <= 1]
print(f"delta_between numeric n={len(DB)}; with <=1 distinct answer-session date: {len(one_sess)}")
print(f"  of those, derivEvid=1: {sum(bool(r['gold_derivable_from_evidence']) for r in one_sess)}")
print(f"  of those, derivTL=1  : {sum(bool(r['gold_derivable_from_timeline']) for r in one_sess)}")
resc = [r for r in DB if r["gold_derivable_from_evidence"] and not r["gold_derivable_from_ans_dates"]]
print(f"  rescued (derivAns=0 -> derivEvid=1): {len(resc)}")
print(f"    of the rescued, needed the low-confidence 'med' pool: "
      f"{sum(1 for r in resc if str(r['evidence_derivation']).startswith('high+med'))}")
print("    rescued qids:", [r["qid"][:12] for r in resc])
DS = [r for r in DV if r["family"] == "delta_since"]
print(f"delta_since n={len(DS)}: derivAns=1 already {sum(bool(r['gold_derivable_from_ans_dates']) for r in DS)}"
      f"  -> in-text resolution adds "
      f"{sum(bool(r['gold_derivable_from_evidence']) and not bool(r['gold_derivable_from_ans_dates']) for r in DS)}")
miss = [r for r in DS if not r["gold_derivable_from_timeline"]]
print("delta_since misses:", [(r["qid"], r["gold"][:30], r["question"][:70]) for r in miss])

print()
print("=" * 100)
print("TEST 11  pool sizes / decoy density / candidate-count reconciliation")
print("=" * 100)
print("  |pool_ans| median", st.median([len(r["pool_ans"]) for r in DV]),
      " |pool_ev_all| median", st.median([len(r["pool_ev_all"]) for r in DV]),
      "max", max(len(r["pool_ev_all"]) for r in DV))
print("  |pool_tl| median", st.median([len(r["pool_tl"]) for r in DV]),
      "max", max(len(r["pool_tl"]) for r in DV),
      " n_tl_pairs median", st.median([r["n_tl_pairs"] for r in DV]),
      "max", max(r["n_tl_pairs"] for r in DV))
succ = [r for r in DV if r["gold_derivable_from_timeline"]]
print("  among TL successes: firing pairs / total pairs =",
      round(st.mean([r["n_tl_hit_pairs"] / max(1, r["n_tl_pairs"]) for r in succ]), 3),
      " median firing pairs", st.median([r["n_tl_hit_pairs"] for r in succ]))
nz = [r["n_arith_candidates"] for r in DV if r["n_arith_candidates"]]
print("  n_arith_candidates: median(all)", st.median([r["n_arith_candidates"] for r in DV]),
      " median(nonzero)", st.median(nz), " mean(nonzero)", round(st.mean(nz), 2),
      " median(lit=1 only)",
      st.median([r["n_arith_candidates"] for r in DV if r["gold_literal_in_arith"]]))
print("  arith block line counts: median",
      st.median([r["arith_text"].count(chr(10)) + 1 for r in DV]))
print("  chars: delta families median", st.median([r["chars"] for r in DV]),
      "mean", round(st.mean([r["chars"] for r in DV]), 1))
