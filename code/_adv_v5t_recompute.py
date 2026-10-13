"""Adversarial independent recompute of the v5_temporal delta-family claim (zero API).

Stage 1: reproduce reported numbers with the CURRENT module (artifacts predate the
module's last edit, so nothing is trusted from disk).
Stage 2: null tests for metric permissiveness + wrong-pair diagnostics.
"""
import json
import os
import random
import re
import statistics as st
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from v5_temporal import (parse_date, day0, days_between, months_between, detect_family,
                         extract_events, precompute_arithmetic, render_timeline, _sentences,
                         resolve_expressions)
from v5_temporal_measure import (parse_gold_values, delta_in_unit, gold_match, evidence_dates,
                                 analyse)
from v5_temporal_val import build_store_cached

R = "/root/agent_memory_iclr/results"


def fires(pool, vals, unit, q_dt, allow_qdate=True):
    """Does some pair in pool (or pool-vs-question-date) reproduce one of vals?
    Returns (bool, list_of_matching_descriptors)."""
    hits = []
    P = sorted(set(pool))
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            if gold_match(vals, delta_in_unit(P[i], P[j], unit), unit):
                hits.append((P[i], P[j]))
    if allow_qdate:
        for d in P:
            if gold_match(vals, delta_in_unit(d, q_dt, unit), unit):
                hits.append((d, q_dt))
    return bool(hits), hits


def n_pairs(pool, q_dt):
    P = sorted(set(pool))
    return len(P) * (len(P) - 1) // 2 + len(P)


def main():
    data = [q for q in load_dataset("s_cleaned") if q["question_type"] == "temporal-reasoning"]
    rows = []
    for i, q in enumerate(data):
        store, miss = build_store_cached(q)
        rec, tl = analyse(q, store)
        rec["type"] = q["question_type"]
        rec["cache_miss"] = miss
        # --- extra adversarial fields
        qd = q["question_date"]
        q_dt = parse_date(qd)
        family, unit = detect_family(q["question"])
        events = extract_events(q, store, q["question"])
        arith = precompute_arithmetic(q["question"], qd, events, family, unit, q_for_anchor=q)
        ev_hi, ev_med = evidence_dates(q)
        s2d = dict(zip(q["haystack_session_ids"], q["haystack_dates"]))
        ans_dates = sorted({day0(parse_date(s2d[s])) for s in (q.get("answer_session_ids") or [])
                            if s in s2d and parse_date(s2d[s])})
        tl_pool = sorted({e.ev_dt for e in events})
        rec["pool_ans"] = [d.isoformat() for d in ans_dates]
        rec["pool_ev_hi"] = [d.isoformat() for d in sorted(set(ev_hi))]
        rec["pool_ev_all"] = [d.isoformat() for d in sorted(set(ev_hi) | set(ev_med))]
        rec["pool_tl"] = [d.isoformat() for d in tl_pool]
        rec["q_dt"] = q_dt.isoformat()
        rec["arith_text"] = "\n".join(arith)
        rec["question"] = q["question"]
        rec["gold_full"] = str(q["answer"])
        rec["n_tl_pairs"] = n_pairs(tl_pool, q_dt)
        vals = rec["gold_vals"]
        vals = [(float(v), u) for v, u in vals]
        if vals:
            ok_tl, hits_tl = fires(tl_pool, vals, unit, q_dt)
            ok_ev, hits_ev = fires(sorted(set(ev_hi) | set(ev_med)), vals, unit, q_dt)
            rec["tl_hit_pairs"] = [[a.isoformat(), b.isoformat()] for a, b in hits_tl]
            rec["n_tl_hit_pairs"] = len(hits_tl)
            rec["n_ev_hit_pairs"] = len(hits_ev)
            rec["chk_tl"] = ok_tl
            rec["chk_ev"] = ok_ev
        rows.append(rec)
        if (i + 1) % 40 == 0:
            print(f"  ...{i+1}/{len(data)}", flush=True)
    with open(f"{R}/_adv_v5t_tr.json", "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1, default=str)
    print("families", Counter(r["family"] for r in rows))
    for fam in ("delta_since", "delta_between"):
        A = [r for r in rows if r["family"] == fam]
        V = [r for r in A if r["gold_vals"]]
        def rate(key, S):
            return round(sum(bool(r[key]) for r in S) / max(1, len(S)), 4)
        print(f"\n{fam}: n_all={len(A)} n_numeric_gold={len(V)}")
        for den, S in (("den=all", A), ("den=numeric", V)):
            print(f"  {den}: cov={round(st.mean(r['dates_present'] for r in S),4)} "
                  f"derivAns={rate('gold_derivable_from_ans_dates',S)} "
                  f"derivEvid={rate('gold_derivable_from_evidence',S)} "
                  f"derivTL={rate('gold_derivable_from_timeline',S)} "
                  f"lit={rate('gold_literal_in_arith',S)}")
        print("  med n_arith_candidates(numeric)", st.median([r["n_arith_candidates"] for r in V]))
    both = [r for r in rows if r["family"] in ("delta_since", "delta_between") and r["gold_vals"]]
    print(f"\nBOTH numeric n={len(both)} "
          f"derivEvid={round(sum(r['gold_derivable_from_evidence'] for r in both)/len(both),4)} "
          f"derivTL={round(sum(r['gold_derivable_from_timeline'] for r in both)/len(both),4)} "
          f"lit={round(sum(r['gold_literal_in_arith'] for r in both)/len(both),4)} "
          f"medcand={st.median([r['n_arith_candidates'] for r in both])}")


if __name__ == "__main__":
    main()
