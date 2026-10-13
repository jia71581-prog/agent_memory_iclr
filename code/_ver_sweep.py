"""Precisely: does predicate_sweep really have 'exactly zero effect / bit-identical metrics'?
Compare per-question timelines and verdicts, sweep OFF vs ON, in the shipped config."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_temporal as V
from common import load_dataset
from v5_temporal_measure import analyse
from v5_temporal_val import build_store_cached

data = [q for q in load_dataset("s_cleaned") if q["question_type"] == "temporal-reasoning"]
stores = {q["question_id"]: build_store_cached(q)[0] for q in data}
V.USE_STEM, V.USE_QFORM_STOP = True, False
V.W_CUE = V.W_ANCHOR = V.W_INTENT = 1.0
res = {}
for sw in (False, True):
    V.USE_SWEEP = sw
    res[sw] = {}
    for q in data:
        rec, tl = analyse(q, stores[q["question_id"]])
        res[sw][q["question_id"]] = (rec["family"], rec["gold_derivable_from_timeline"],
                                     str(rec.get("ordering_item_correct")), rec["chars"],
                                     rec["n_events"], tl)
diff_tl = [k for k in res[False] if res[False][k][5] != res[True][k][5]]
diff_chars = [k for k in res[False] if res[False][k][3] != res[True][k][3]]
diff_verdict = [k for k in res[False] if res[False][k][:3] != res[True][k][:3]]
print(f"questions with a DIFFERENT rendered timeline under sweep ON: {len(diff_tl)}/{len(res[False])}")
print(f"  of which char count differs: {len(diff_chars)}")
print(f"  verdict (family/dTL/ordering) differs on: {len(diff_verdict)} -> {diff_verdict}")
for k in diff_chars[:4]:
    print(f"  {k}: chars {res[False][k][3]} -> {res[True][k][3]}, "
          f"n_events {res[False][k][4]} -> {res[True][k][4]}, fam {res[False][k][0]}")
V.USE_SWEEP = False
