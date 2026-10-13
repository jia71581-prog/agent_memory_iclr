"""Stage 3: unit-oracle ceiling, literal-block null, per-family nulls, arith inspection."""
import json
import random
import re
import statistics as st
import sys
from collections import Counter
from datetime import date

sys.path.insert(0, "/root/agent_memory_iclr/code")
from v5_temporal_measure import delta_in_unit, gold_match, parse_gold_values

R = "/root/agent_memory_iclr/results"
rows = json.load(open(f"{R}/_adv_v5t_tr.json"))
DV = [r for r in rows if r["family"] in ("delta_since", "delta_between") and r["gold_vals"]]


def pd(s):
    s = str(s).split("T")[0].split(" ")[0]
    y, m, dd = [int(x) for x in s.split("-")]
    return date(y, m, dd)


def fires(pool, vals, unit, q_dt):
    P = sorted({pd(x) for x in pool})
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            if gold_match(vals, delta_in_unit(P[i], P[j], unit), unit):
                return True
    for d in P:
        if gold_match(vals, delta_in_unit(d, q_dt, unit), unit):
            return True
    return False


GRID = {"day": list(range(1, 61)), "week": list(range(1, 16)),
        "month": list(range(1, 13)), "year": list(range(1, 6))}

print("=" * 100)
print("TEST 4  ceiling recomputed with the UNIT TAKEN FROM GOLD (removes the unit-detector zeros)")
print("=" * 100)
for fam in ("delta_since", "delta_between", "both"):
    S = [r for r in DV if fam == "both" or r["family"] == fam]
    a = b = c = d = 0
    flips = []
    for r in S:
        q_dt = pd(r["q_dt"])
        # unit-oracle: try every unit present in the gold string
        units = sorted({u for _, u in r["gold_vals"]})
        okE = any(fires(r["pool_ev_all"], [(float(v), u) for v, u in r["gold_vals"] if u == u2],
                        u2 + "s", q_dt) for u2 in units)
        okT = any(fires(r["pool_tl"], [(float(v), u) for v, u in r["gold_vals"] if u == u2],
                        u2 + "s", q_dt) for u2 in units)
        a += r["gold_derivable_from_evidence"]
        b += okE
        c += r["gold_derivable_from_timeline"]
        d += okT
        if okE and not r["gold_derivable_from_evidence"]:
            flips.append(r["qid"])
    print(f"{fam:14s} n={len(S)}  derivEvid as-reported={a/len(S):.3f} -> unit-oracle={b/len(S):.3f} | "
          f"derivTL as-reported={c/len(S):.3f} -> unit-oracle={d/len(S):.3f}")
    if fam == "both":
        print("   evidence-level failures rescued purely by fixing the unit:", flips)

print()
print("=" * 100)
print("TEST 5  per-family null (grid) and the delta_since triviality check")
print("=" * 100)
for fam in ("delta_since", "delta_between"):
    S = [r for r in DV if r["family"] == fam]
    for name, key in (("timeline", "pool_tl"), ("evidence", "pool_ev_all"), ("ans", "pool_ans")):
        obs, nul = [], []
        for r in S:
            u = (r["unit"] or "day").rstrip("s")
            vals = [(float(v), uu) for v, uu in r["gold_vals"] if uu == u]
            q_dt = pd(r["q_dt"])
            if not vals:
                obs.append(0.0)
                nul.append(0.0)
                continue
            base = min(v for v, _ in vals)
            offs = sorted({v - base for v, _ in vals})
            grid = GRID.get(u, list(range(1, 61)))
            h = sum(fires(r[key], [(g + o, u) for o in offs if g + o > 0], r["unit"], q_dt)
                    for g in grid)
            obs.append(1.0 if fires(r[key], vals, r["unit"], q_dt) else 0.0)
            nul.append(h / len(grid))
        print(f"  {fam:14s} {name:9s} observed={st.mean(obs):.3f} null={st.mean(nul):.3f} n={len(S)}")
    # how many delta_since need only ONE date (question-date subtraction)?
    if fam == "delta_since":
        one = 0
        for r in S:
            q_dt = pd(r["q_dt"])
            vals = [(float(v), u) for v, u in r["gold_vals"]]
            singles = [d for d in {pd(x) for x in r["pool_tl"]}
                       if gold_match(vals, delta_in_unit(d, q_dt, r["unit"]), r["unit"])]
            one += bool(singles)
        print(f"  delta_since: solvable by ONE date minus question date: {one}/{len(S)}")

print()
print("=" * 100)
print("TEST 6  literal-in-arith: candidate counts, null rate, and lit-vs-derivable inconsistency")
print("=" * 100)
cands, allnums, litnull, obs = [], [], [], []
incons = []
for r in DV:
    u = (r["unit"] or "day").rstrip("s")
    atxt = r["arith_text"]
    got = {float(m.group(1)) for m in
           re.finditer(r"(\d+(?:\.\d+)?)\s*(?:full\s+)?" + u + r"s?\b", atxt)} if r["unit"] else set()
    allnum = {float(x) for x in re.findall(r"\b(\d+(?:\.\d+)?)\b", atxt)}
    cands.append(len(got))
    allnums.append(len(allnum))
    vals = [(float(v), uu) for v, uu in r["gold_vals"]]
    obs.append(1.0 if gold_match(vals, got, r["unit"]) else 0.0)
    grid = GRID.get(u, list(range(1, 61)))
    vv = [(v, uu) for v, uu in vals if uu == u]
    if vv and r["unit"]:
        base = min(v for v, _ in vv)
        offs = sorted({v - base for v, _ in vv})
        litnull.append(sum(gold_match([(g + o, u) for o in offs if g + o > 0], got, r["unit"])
                           for g in grid) / len(grid))
    else:
        litnull.append(0.0)
    if r["gold_literal_in_arith"] and not r["gold_derivable_from_timeline"]:
        incons.append((r["qid"], r["unit"], r["gold"][:34], len(got)))
print(f"n={len(DV)}  literal observed={st.mean(obs):.3f}  null(random plausible gold)={st.mean(litnull):.3f}")
print(f"  distinct unit-tagged candidate numbers in block: median={st.median(cands)} "
      f"mean={st.mean(cands):.2f} p90={sorted(cands)[int(.9*len(cands))]} max={max(cands)}")
print(f"  ALL distinct numbers printed in block:          median={st.median(allnums)} "
      f"mean={st.mean(allnums):.2f} max={max(allnums)}")
print(f"  lit=1 while timeline cannot derive gold (coincidence hits): {len(incons)} -> {incons}")

print()
print("=" * 100)
print("TEST 7  what the arithmetic block looks like when the unit detector returns None")
print("=" * 100)
for r in DV:
    if r["unit"] is None:
        print("-" * 90)
        print(r["qid"], "| gold:", r["gold"])
        print("Q:", r["question"][:140])
        print(r["arith_text"][:700] or "<EMPTY ARITHMETIC BLOCK>")
        break
none_units = [r for r in DV if r["unit"] is None]
print(f"\nunit=None questions: {len(none_units)}; of these, empty arith block: "
      f"{sum(1 for r in none_units if not r['arith_text'].strip())}")
print("chars of timeline (all 133 TR):", st.median([r["chars"] for r in rows]),
      "mean", round(st.mean([r["chars"] for r in rows]), 1),
      "max", max(r["chars"] for r in rows))
