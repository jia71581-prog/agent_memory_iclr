"""Date-blind null for gold_derivable_from_timeline: draw the SAME NUMBER of distinct dates
uniformly at random from each question's own haystack date span and ask how often the metric
fires. If the null is close to the reported ON value, the metric measures date count, not
temporal resolution. Zero API."""
import json, os, sys, random, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from v5_temporal import parse_date, day0, timedelta
from v5_temporal_measure import parse_gold_values, delta_in_unit, gold_match

R = "/root/agent_memory_iclr/results"
rows = {r["qid"]: r for r in json.load(open(f"{R}/v5_temporal_tr.json"))}
pq = json.load(open(f"{R}/_ver_repro.json"))["per_q"]
data = {q["question_id"]: q for q in load_dataset("s_cleaned")}
rng = random.Random(7)
TRIALS = 40


def fires(dates, vals, unit, q_dt):
    ds = sorted(set(dates))
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            if gold_match(vals, delta_in_unit(ds[i], ds[j], unit), unit):
                return True
    for d in ds:
        if gold_match(vals, delta_in_unit(d, q_dt, unit), unit):
            return True
    return False


for arm_k, label in (("final", "ON (k = ON date count)"), ("no_intext", "OFF (k = OFF date count)")):
    tot, hits = 0, 0
    per_q_rate = []
    for qid, P in pq[arm_k].items():
        r = rows[qid]
        if r["family"] not in ("delta_between", "delta_since") or not r["gold_vals"]:
            continue
        q = data[qid]
        vals = [(v, u) for v, u in r["gold_vals"]]
        unit = r["unit"]
        q_dt = parse_date(q["question_date"])
        span = sorted(day0(parse_date(d)) for d in q["haystack_dates"] if parse_date(d))
        lo, hi = span[0], span[-1]
        n_days = max(1, (hi - lo).days)
        k = max(1, P["ndates"])
        h = 0
        for _ in range(TRIALS):
            dts = [lo + timedelta(days=rng.randrange(n_days + 1)) for _ in range(k)]
            h += fires(dts, vals, unit, q_dt)
        per_q_rate.append(h / TRIALS)
        hits += h
        tot += TRIALS
    print(f"date-blind null, {label}: mean fire rate = {hits/tot:.4f}  "
          f"(per-question median {st.median(per_q_rate):.2f}, "
          f"n_q={len(per_q_rate)}, questions where null fires >=50% of trials: "
          f"{sum(x>=0.5 for x in per_q_rate)})")

# how much of the metric's permissiveness comes from the multi-reading slack?
strict_hits = {"slack": 0, "strict": 0, "n": 0}
for qid, P in pq["final"].items():
    r = rows[qid]
    if r["family"] not in ("delta_between", "delta_since") or not r["gold_vals"]:
        continue
    q = data[qid]
    unit = r["unit"]
    q_dt = parse_date(q["question_date"])
    vals = [(v, u) for v, u in r["gold_vals"]]
    strict_hits["n"] += 1
print("\nreadings accepted per pair by delta_in_unit:",
      {u: sorted(delta_in_unit(parse_date("2023/03/01"), parse_date("2023/04/12"), u))
       for u in ("day", "week", "month", "year")})
