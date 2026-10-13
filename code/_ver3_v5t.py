"""Independent adversarial recompute of v5_temporal claims + sham/null controls.

Null controls answer: does 'gold is derivable / literally present' carry ANY signal,
or is it an artifact of enumerating a large candidate set of small integers?
"""
import json, os, random, statistics as st, sys
from collections import Counter
sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
import v5_temporal as V
from v5_temporal_measure import (analyse, parse_gold_values, delta_in_unit, gold_match)
from v5_temporal_val import build_store_cached

R = "/root/agent_memory_iclr/results"
data = [q for q in load_dataset("s_cleaned") if q["question_type"] == "temporal-reasoning"]
print("TR questions:", len(data), flush=True)

stores = {}
for q in data:
    stores[q["question_id"]] = build_store_cached(q)[0]
print("stores built", flush=True)

rows = []
for q in data:
    r = analyse(q, stores[q["question_id"]])
    r = r[0] if isinstance(r, tuple) else r
    rows.append(r)
print("analysed", len(rows), flush=True)
json.dump(rows, open(f"{R}/_ver3_v5t_rows.json", "w"))

def f(g, k):
    return sum(bool(x.get(k)) for x in g) / max(1, len(g))

print("\n=== INDEPENDENT RECOMPUTE (per family) ===")
for fam in ["delta_between", "delta_since", "ordering", "anchor_lookup", "count_before", "other"]:
    g = [r for r in rows if r["family"] == fam]
    if not g:
        continue
    nc = [r["n_arith_candidates"] for r in g if r["n_arith_candidates"]]
    print(f"{fam:14s} n={len(g):3d} datesPres={st.mean([r['dates_present'] for r in g]):.3f} "
          f"derivAns={f(g,'gold_derivable_from_ans_dates'):.3f} "
          f"derivTL={f(g,'gold_derivable_from_timeline'):.3f} "
          f"literal={f(g,'gold_literal_in_arith'):.3f} "
          f"medCand={(st.median(nc) if nc else 0):.1f}")

dg = [r for r in rows if r["family"] in ("delta_between", "delta_since") and r["gold_vals"]]
print(f"\ndelta n={len(dg)}: derivEvid={f(dg,'gold_derivable_from_evidence'):.3f} "
      f"derivTL={f(dg,'gold_derivable_from_timeline'):.3f} literal={f(dg,'gold_literal_in_arith'):.3f}")

# ---------------------------------------------------------------- null controls
qmap = {q["question_id"]: q for q in data}
rmap = {r["qid"]: r for r in rows}

def hay_dates(q, exclude_evidence=True):
    ans = set(q.get("answer_session_ids") or [])
    out = []
    for sid, ds in zip(q["haystack_session_ids"], q["haystack_dates"]):
        if exclude_evidence and sid in ans:
            continue
        d = V.parse_date(ds)
        if d:
            out.append(V.day0(d))
    return sorted(set(out))

def n_timeline_dates(q, store):
    ev = V.extract_events(q, store, q["question"])
    return len({e.ev_dt for e in ev})

SEEDS = 24
print("\n=== NULL CONTROL 1: sham timeline dates (same count, drawn from NON-evidence sessions) ===")
per_q = {}
ntl_list = []
for r in dg:
    q = qmap[r["qid"]]
    store = stores[r["qid"]]
    k = n_timeline_dates(q, store)
    ntl_list.append(k)
    pool = hay_dates(q)
    q_dt = V.parse_date(q["question_date"])
    vals = r["gold_vals"]
    vals = [(v, u) for v, u in vals]
    unit = r["unit"]
    hits = 0
    for s in range(SEEDS):
        rnd = random.Random(hash((r["qid"], s)) & 0xffffffff)
        sham = sorted(rnd.sample(pool, min(k, len(pool))))
        ok = False
        for i in range(len(sham)):
            for j in range(i + 1, len(sham)):
                if gold_match(vals, delta_in_unit(sham[i], sham[j], unit), unit):
                    ok = True
                    break
            if ok:
                break
        if not ok:
            for d in sham:
                if gold_match(vals, delta_in_unit(d, q_dt, unit), unit):
                    ok = True
                    break
        hits += ok
    per_q[r["qid"]] = hits / SEEDS
print(f"n distinct timeline dates: median={st.median(ntl_list)} mean={st.mean(ntl_list):.1f} "
      f"max={max(ntl_list)}")
print(f"real derivTL = {f(dg,'gold_derivable_from_timeline'):.3f}")
print(f"SHAM derivTL = {st.mean(per_q.values()):.3f}  (mean over {SEEDS} seeds/question)")

print("\n=== NULL CONTROL 2: literal-in-arith with sham dates ===")
def arith_numbers(dates, q_dt, unit, family, max_pairs=15, n_pair=6):
    """Reproduce the numeric candidate set the arith block would emit."""
    import re
    dd = sorted(dates)[:n_pair] if family == "delta_between" else sorted(dates)[:10]
    got = set()
    u = (unit or "day").rstrip("s")
    def emit(d1, d2):
        nd = abs(V.days_between(d1, d2))
        if u == "week":
            got.update({round(nd / 7.0, 1), float(nd // 7)})
        elif u == "month":
            got.update({float(abs(V.months_between(min(d1, d2), max(d1, d2)))),
                        round(nd / 30.44, 1)})
        else:
            got.add(float(nd))
    if family == "delta_between":
        n = 0
        for i in range(len(dd)):
            for j in range(i + 1, len(dd)):
                emit(dd[i], dd[j]); n += 1
                if n >= max_pairs: break
            if n >= max_pairs: break
    else:
        for d in dd:
            emit(d, q_dt)
    return got

sham_lit, real_lit = {}, {}
for r in dg:
    q = qmap[r["qid"]]
    store = stores[r["qid"]]
    ev = V.extract_events(q, store, q["question"])
    q_dt = V.parse_date(q["question_date"])
    unit, fam, vals = r["unit"], r["family"], r["gold_vals"]
    real_dates = V.top_dates_by_relevance(ev, 6 if fam == "delta_between" else 10)
    real_lit[r["qid"]] = gold_match(vals, arith_numbers(real_dates, q_dt, unit, fam), unit)
    pool = hay_dates(q)
    k = len(real_dates)
    hits = 0
    for s in range(SEEDS):
        rnd = random.Random(hash((r["qid"], "L", s)) & 0xffffffff)
        sham = rnd.sample(pool, min(k, len(pool)))
        hits += gold_match(vals, arith_numbers(sham, q_dt, unit, fam), unit)
    sham_lit[r["qid"]] = hits / SEEDS
print(f"real literal (my reimpl of block)= {sum(real_lit.values())/len(real_lit):.3f}")
print(f"SHAM literal                     = {st.mean(sham_lit.values()):.3f}")
print(f"module-reported literal          = {f(dg,'gold_literal_in_arith'):.3f}")

json.dump({"sham_derivTL": per_q, "sham_lit": sham_lit,
           "real_lit": {k: bool(v) for k, v in real_lit.items()}},
          open(f"{R}/_ver3_v5t_null.json", "w"), indent=1)
