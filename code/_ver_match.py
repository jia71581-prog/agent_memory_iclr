"""Exactly count-matched random control for gold_derivable_from_timeline.

For each delta-family question: take the OFF-arm timeline date set, then add exactly as many
extra dates as in-text resolution added, each drawn as (a random OFF timeline date) - (a random
real in-text offset). Same date count as the ON arm, zero real semantics. 30 trials/question.
"""
import json, os, sys, random, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_temporal as V
from common import load_dataset
from v5_temporal_measure import parse_gold_values, delta_in_unit, gold_match
from v5_temporal_val import build_store_cached

R = "/root/agent_memory_iclr/results"
rows = {r["qid"]: r for r in json.load(open(f"{R}/v5_temporal_tr.json"))}
data = [q for q in load_dataset("s_cleaned") if q["question_type"] == "temporal-reasoning"]
stores = {q["question_id"]: build_store_cached(q)[0] for q in data}
_orig = V.best_event_date


def no_intext(text, session_dt):
    d = V.parse_date(session_dt)
    return (V.day0(d), "") if d else (None, "")


# real offsets
shifts = []
for q in data:
    for date_s, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        for t in sess:
            if t.get("role") != "user":
                continue
            for s in V._sentences(t.get("content") or ""):
                ev, _ = _orig(s, date_s)
                sd = V.parse_date(date_s)
                if ev is not None and sd is not None and V.day0(sd) != ev:
                    shifts.append((V.day0(sd) - ev).days)
print("n real nonzero offsets", len(shifts), "median", st.median(shifts))


def fires(ds, vals, unit, q_dt):
    ds = sorted(set(ds))
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            if gold_match(vals, delta_in_unit(ds[i], ds[j], unit), unit):
                return True
    return any(gold_match(vals, delta_in_unit(d, q_dt, unit), unit) for d in ds)


V.USE_STEM, V.USE_QFORM_STOP, V.USE_SWEEP = True, False, False
sets = {}
for name, fn in (("ON", _orig), ("OFF", no_intext)):
    V.best_event_date = fn
    for q in data:
        r = rows[q["question_id"]]
        if r["family"] not in ("delta_between", "delta_since") or not r["gold_vals"]:
            continue
        ev = V.extract_events(q, stores[q["question_id"]], q["question"])
        sets.setdefault(q["question_id"], {})[name] = sorted({e.ev_dt for e in ev})
V.best_event_date = _orig

rng = random.Random(99)
TR = 30
on_h, off_h, ctl = 0, 0, []
n = 0
for qid, S in sets.items():
    r = rows[qid]
    q = next(x for x in data if x["question_id"] == qid)
    vals = [(v, u) for v, u in r["gold_vals"]]
    unit, q_dt = r["unit"], V.parse_date(q["question_date"])
    k_add = max(0, len(S["ON"]) - len(S["OFF"]))
    on_h += fires(S["ON"], vals, unit, q_dt)
    off_h += fires(S["OFF"], vals, unit, q_dt)
    h = 0
    for _ in range(TR):
        ds = list(S["OFF"])
        for _ in range(k_add):
            base = rng.choice(S["OFF"]) if S["OFF"] else q_dt
            ds.append(base - V.timedelta(days=rng.choice(shifts)))
        h += fires(ds, vals, unit, q_dt)
    ctl.append(h / TR)
    n += 1
print(f"n={n}  ON={on_h/n:.4f}  OFF={off_h/n:.4f}  "
      f"count-matched random control={st.mean(ctl):.4f} (sd over questions {st.pstdev(ctl):.3f})")
gain = on_h / n - off_h / n
print(f"gain ON-OFF = {gain:+.4f}; control gain = {st.mean(ctl)-off_h/n:+.4f} "
      f"= {100*(st.mean(ctl)-off_h/n)/max(1e-9,gain):.0f}% of it")
print("mean added dates per question:",
      st.mean([len(S["ON"]) - len(S["OFF"]) for S in sets.values()]))
# Poisson-binomial null: is ON=48/60 outside what the count-matched control predicts?
import math
mu = sum(ctl)
var = sum(p * (1 - p) for p in ctl)
z = (on_h - mu) / math.sqrt(max(1e-9, var))
print(f"count-matched null: expected hits {mu:.1f} +/- {math.sqrt(var):.1f}; observed {on_h}; z={z:.2f}")
print(f"one-sided normal p ~ {0.5*math.erfc(z/math.sqrt(2)):.3f}")
json.dump({"ctl": ctl, "on": on_h, "off": off_h}, open(f"{R}/_ver_match.json", "w"))
