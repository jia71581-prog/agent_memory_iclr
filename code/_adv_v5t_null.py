"""Stage 2: validity attacks on the deriv* metric (offline)."""
import json
import random
import re
import statistics as st
import sys
from collections import Counter
from datetime import date

sys.path.insert(0, "/root/agent_memory_iclr/code")
from v5_temporal_measure import delta_in_unit, gold_match

R = "/root/agent_memory_iclr/results"
rows = json.load(open(f"{R}/_adv_v5t_tr.json"))
D = [r for r in rows if r["family"] in ("delta_since", "delta_between")]
DV = [r for r in D if r["gold_vals"]]


def pd(s):
    s = str(s).split("T")[0].split(" ")[0]
    y, m, dd = [int(x) for x in s.split("-")]
    return date(y, m, dd)


def fires(pool, vals, unit, q_dt):
    P = sorted({pd(x) if isinstance(x, str) else x for x in pool})
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            if gold_match(vals, delta_in_unit(P[i], P[j], unit), unit):
                return True
    for d in P:
        if gold_match(vals, delta_in_unit(d, q_dt, unit), unit):
            return True
    return False


def hitpairs(pool, vals, unit, q_dt):
    P = sorted({pd(x) if isinstance(x, str) else x for x in pool})
    out = []
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            if gold_match(vals, delta_in_unit(P[i], P[j], unit), unit):
                out.append((P[i], P[j]))
    for d in P:
        if gold_match(vals, delta_in_unit(d, q_dt, unit), unit):
            out.append((d, q_dt))
    return out


GRID = {"day": list(range(1, 61)), "week": list(range(1, 16)),
        "month": list(range(1, 13)), "year": list(range(1, 6))}

print("=" * 100)
print("TEST 1  metric permissiveness: fraction of PLAUSIBLE gold values the same pool 'derives'")
print("        (structure-preserving slide: keeps the gold's multi-reading pattern & unit)")
print("=" * 100)
for name, poolkey in (("timeline", "pool_tl"), ("evidence(hi+med)", "pool_ev_all"),
                      ("ans-session", "pool_ans")):
    obs, nulls, nullsu = [], [], []
    for r in DV:
        unit = r["unit"]
        vals = [(float(v), u) for v, u in r["gold_vals"]]
        q_dt = pd(r["q_dt"])
        u = (unit or "day").rstrip("s")
        usable = [(v, uu) for v, uu in vals if uu == u]
        if not usable:                      # unit mismatch: metric structurally cannot fire
            obs.append(0.0)
            nulls.append(0.0)
            nullsu.append(0.0)
            continue
        base = min(v for v, _ in usable)
        offs = sorted({v - base for v, _ in usable})
        grid = GRID.get(u, list(range(1, 61)))
        hit = 0
        for g in grid:
            cand = [(g + o, u) for o in offs if g + o > 0]
            if fires(r[poolkey], cand, unit, q_dt):
                hit += 1
        obs.append(1.0 if fires(r[poolkey], vals, unit, q_dt) else 0.0)
        nulls.append(hit / len(grid))
        nullsu.append(hit / len(grid))
    print(f"{name:18s} observed={st.mean(obs):.3f}   "
          f"null(random plausible value in same unit)={st.mean(nulls):.3f}   "
          f"n={len(obs)}")

print()
print("=" * 100)
print("TEST 1b  empirical permutation null: give each question another question's gold magnitude")
print("=" * 100)
random.seed(0)
byunit = {}
for r in DV:
    u = (r["unit"] or "day").rstrip("s")
    for v, uu in r["gold_vals"]:
        if uu == u:
            byunit.setdefault(u, []).append(float(v))
for name, poolkey in (("timeline", "pool_tl"), ("evidence(hi+med)", "pool_ev_all")):
    tot = []
    for _ in range(200):
        hits = 0
        for r in DV:
            unit = r["unit"]
            u = (unit or "day").rstrip("s")
            vals = [(float(v), uu) for v, uu in r["gold_vals"] if uu == u]
            if not vals or u not in byunit:
                continue
            base = min(v for v, _ in vals)
            offs = sorted({v - base for v, _ in vals})
            g = random.choice(byunit[u])
            cand = [(g + o, u) for o in offs if g + o > 0]
            hits += fires(r[poolkey], cand, unit, q_dt=pd(r["q_dt"]))
        tot.append(hits / len(DV))
    print(f"{name:18s} permutation-null mean={st.mean(tot):.3f} sd={st.pstdev(tot):.3f}")

print()
print("=" * 100)
print("TEST 2  wrong-pair diagnostic: is the firing pair the pair that actually supports gold?")
print("=" * 100)
tlsucc = [r for r in DV if r["gold_derivable_from_timeline"]]
rows_out = []
for r in tlsucc:
    unit, q_dt = r["unit"], pd(r["q_dt"])
    vals = [(float(v), u) for v, u in r["gold_vals"]]
    hp = hitpairs(r["pool_tl"], vals, unit, q_dt)
    ans = {pd(x) for x in r["pool_ans"]}
    ev = {pd(x) for x in r["pool_ev_all"]}
    # a pair "supports gold" if both endpoints are answer-session dates or evidence-derived
    # dates (question date always allowed)
    def ok(p):
        return all((x in ans or x in ev or x == q_dt) for x in p)
    good = [p for p in hp if ok(p)]
    rows_out.append((r["qid"], len(hp), len(good), r["n_tl_pairs"], r["unit"], r["gold"][:32]))
print(f"TL successes n={len(tlsucc)}")
print(f"  median #firing pairs = {st.median([x[1] for x in rows_out])}, "
      f"mean = {st.mean([x[1] for x in rows_out]):.2f}, max = {max(x[1] for x in rows_out)}")
print(f"  #successes where NO firing pair is evidence-grounded (pure decoy coincidence): "
      f"{sum(1 for x in rows_out if x[2]==0)}")
print(f"  #successes with >=2 firing pairs (model still has to pick): "
      f"{sum(1 for x in rows_out if x[1]>=2)}")
print(f"  median #candidate pairs available in timeline = {st.median([x[3] for x in rows_out])}")
for x in sorted(rows_out, key=lambda y: -y[1])[:12]:
    print("   ", x)

print()
print("=" * 100)
print("TEST 3  unit=None / unit-mismatch structural zeros (are these 'evidence-side ceiling'?)")
print("=" * 100)
bad = [r for r in DV if not any(u == (r["unit"] or "day").rstrip("s") for _, u in r["gold_vals"])]
print(f"questions whose gold unit can NEVER match the metric's unit: {len(bad)} of {len(DV)}")
for r in bad:
    print(f"  {r['qid'][:12]} unit={r['unit']} gold={r['gold'][:40]!r} "
          f"derivEvid={int(r['gold_derivable_from_evidence'])} lit={int(r['gold_literal_in_arith'])}")
    print(f"      Q: {r['question'][:150]}")
