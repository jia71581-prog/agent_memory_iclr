"""Independent adversarial recompute of the v5_temporal abstention claim (offline, zero API)."""
import json
import os
import re
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
import v5_temporal as T
from v5_temporal import (render_timeline, extract_events, detect_family, precompute_arithmetic,
                         question_alternatives, match_item_date, parse_date)
from v5_temporal_val import build_store_cached, data_by_id

R = "/root/agent_memory_iclr/results"
d = data_by_id()
ABS = [q for q in d.values()
       if q["question_type"] == "temporal-reasoning" and q["question_id"].endswith("_abs")]
ABS.sort(key=lambda q: q["question_id"])

out = {}
for q in ABS:
    store, miss = build_store_cached(q)
    fam, unit = detect_family(q["question"])
    events = extract_events(q, store, q["question"])
    arith = precompute_arithmetic(q["question"], q["question_date"], events, fam, unit, q_for_anchor=q)
    tl = render_timeline(q, store, q["question"])
    alts = question_alternatives(q["question"])
    missing = [a for a in alts if match_item_date(a, events) is None]
    out[q["question_id"]] = dict(fam=fam, unit=unit, miss=miss, n_events=len(events),
                                 chars=len(tl), alts=alts, missing=missing,
                                 n_arith=len(arith))
    print("=" * 110)
    print(q["question_id"], "| family:", fam, "| unit:", unit, "| cache-miss:", miss,
          "| n_events:", len(events), "| chars:", len(tl))
    print("Q:", q["question"])
    print("GOLD:", q["answer"])
    print("alts:", alts, "| missing:", missing)
    print("-" * 40, "SCAFFOLD", "-" * 40)
    print(tl)
    print()

json.dump(out, open(f"{R}/_ver3_abs.json", "w"), indent=1, default=str)
