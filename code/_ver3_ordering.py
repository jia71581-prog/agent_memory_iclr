"""Recompute the missing-item-note precision over the full TR ordering family, plus family counts."""
import json
import os
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
from collections import Counter
from v5_temporal import (extract_events, detect_family, question_alternatives, match_item_date,
                         precompute_arithmetic, render_timeline)
from v5_temporal_val import build_store_cached, data_by_id

R = "/root/agent_memory_iclr/results"
d = data_by_id()
TR = [q for q in d.values() if q["question_type"] == "temporal-reasoning"]
TR.sort(key=lambda q: q["question_id"])
print("TR n =", len(TR))

fams = Counter()
rows = []
for i, q in enumerate(TR):
    fam, unit = detect_family(q["question"])
    fams[fam] += 1
    if fam != "ordering":
        continue
    store, miss = build_store_cached(q)
    ev = extract_events(q, store, q["question"])
    alts = question_alternatives(q["question"])
    md = {a: match_item_date(a, ev) for a in alts}
    missing = [a for a, v in md.items() if v is None]
    rows.append(dict(qid=q["question_id"], is_abs=q["question_id"].endswith("_abs"),
                     q=q["question"], gold=str(q["answer"])[:120],
                     alts=alts, missing=missing, fires=bool(alts and missing),
                     all_missing=bool(alts) and len(missing) == len(alts),
                     n_events=len(ev), miss_sess=miss))
    if (i + 1) % 40 == 0:
        print(f"  ...{i+1}/{len(TR)}", flush=True)

print("\nfamily counts:", dict(fams), "sum", sum(fams.values()))
O = rows
nabs = [r for r in O if r["is_abs"]]
nans = [r for r in O if not r["is_abs"]]
print(f"\nordering family n={len(O)}  abstention={len(nabs)}  answerable={len(nans)}")
tp = [r for r in nabs if r["fires"]]
fn = [r for r in nabs if not r["fires"]]
fp = [r for r in nans if r["fires"]]
print(f"note fires on abstention (TP): {len(tp)}/{len(nabs)}   -> {[r['qid'] for r in tp]}")
print(f"note misses abstention (FN): {len(fn)}   -> {[r['qid'] for r in fn]}")
print(f"note false-fires on answerable (FP): {len(fp)}/{len(nans)} -> {[r['qid'] for r in fp]}")
den = len(tp) + len(fp)
print(f"precision = {len(tp)}/{den} = {len(tp)/den:.4f}" if den else "no fires")
print(f"recall on abstention = {len(tp)}/{len(nabs)} = {len(tp)/max(1,len(nabs)):.3f}")

print("\n--- alts extraction coverage over ordering family ---")
print("questions with >=2 alts extracted:", sum(1 for r in O if r["alts"]), "/", len(O))

print("\n--- TP detail (does it flag ONLY the truly-absent item?) ---")
for r in tp:
    print(f"  {r['qid']}: alts={r['alts']}")
    print(f"      missing={r['missing']}  all_missing={r['all_missing']}")
    print(f"      gold={r['gold']}")

print("\n--- FP detail ---")
for r in fp:
    print(f"  {r['qid']}: {r['q'][:130]}")
    print(f"      alts={r['alts']}  missing={r['missing']}  n_events={r['n_events']}")
    print(f"      gold={r['gold']}")

json.dump(O, open(f"{R}/_ver3_ordering.json", "w"), indent=1, default=str)
print("\nsaved _ver3_ordering.json")
