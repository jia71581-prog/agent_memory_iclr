"""Adversarial recompute of v5_temporal component ablation. Zero API.

Arms:
  final            shipped config (stem on, intext on)
  no_stem          stem off, intext on
  no_intext        intext off (session date only)
  no_stem_nointext both off
  jitter_K         intext off, but each event date randomly shifted with prob p and
                   magnitude drawn from the REAL in-text shift distribution (chance control)
  shuffle_K        intext on, but the resolved shifts are permuted across events
Per-question booleans are dumped for paired tests.
"""
import json, os, sys, random, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_temporal as V
from common import load_dataset
from v5_temporal_measure import analyse
from v5_temporal_val import build_store_cached

OUT = "/root/agent_memory_iclr/results/_ver_repro.json"
data = [q for q in load_dataset("s_cleaned") if q["question_type"] == "temporal-reasoning"]
stores, miss_tot = {}, 0
for q in data:
    s, m = build_store_cached(q)
    stores[q["question_id"]] = s
    miss_tot += m
print("stores built", len(stores), "cache-miss sessions:", miss_tot, flush=True)

_orig = V.best_event_date

# ---- measure the real in-text shift distribution (days between session date and event date)
shifts = []
for q in data:
    for date_s, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        for t in sess:
            if t.get("role") != "user":
                continue
            for s in V._sentences(t.get("content") or ""):
                ev, prov = _orig(s, date_s)
                sd = V.parse_date(date_s)
                if ev is None or sd is None:
                    continue
                shifts.append((V.day0(sd) - ev).days)
nz = [x for x in shifts if x != 0]
p_shift = len(nz) / max(1, len(shifts))
print(f"real in-text shifts: n_sent={len(shifts)} p_nonzero={p_shift:.4f} "
      f"median_shift={st.median(nz):.1f} mean={st.mean(nz):.1f}", flush=True)


def no_intext(text, session_dt):
    d = V.parse_date(session_dt)
    return (V.day0(d), "") if d else (None, "")


def make_jitter(seed):
    rng = random.Random(seed)

    def f(text, session_dt):
        d = V.parse_date(session_dt)
        if d is None:
            return None, ""
        if rng.random() < p_shift:
            return V.day0(d) - V.timedelta(days=rng.choice(nz)), "jitter"
        return V.day0(d), ""
    return f


def make_shuffle(seed):
    """Keep the real decision of WHICH sentences carry an in-text date, but replace the
    resolved offset with a random real offset (destroys the semantics, keeps the statistics)."""
    rng = random.Random(seed)

    def f(text, session_dt):
        d = V.parse_date(session_dt)
        if d is None:
            return None, ""
        ev, prov = _orig(text, session_dt)
        if ev is None:
            return None, ""
        if V.day0(d) == ev:
            return ev, prov
        return V.day0(d) - V.timedelta(days=rng.choice(nz)), prov
    return f


def metrics(rows):
    dg = [r for r in rows if r["family"] in ("delta_between", "delta_since") and r["gold_vals"]]
    o = [r for r in rows if r["family"] == "ordering"]
    oi = [r for r in o if r.get("ordering_item_correct") is not None]
    al = [r for r in rows if r["family"] == "anchor_lookup"]
    ndates = [len({d for d in r["_tl_dates"]}) for r in rows]
    return {
        "n_delta": len(dg), "n_ord": len(o), "n_anchor": len(al),
        "deltaTL": sum(r["gold_derivable_from_timeline"] for r in dg) / len(dg),
        "deltaLiteral": sum(r["gold_literal_in_arith"] for r in dg) / len(dg),
        "deltaEvid": sum(r["gold_derivable_from_evidence"] for r in dg) / len(dg),
        "ord_computable": len(oi) / len(o),
        "ord_correct_of_all": sum(bool(r.get("ordering_item_correct")) for r in o) / len(o),
        "ord_wrong_of_all": sum(r.get("ordering_item_correct") is False for r in o) / len(o),
        "same_day_split": sum(bool(r.get("same_day_split")) for r in o),
        "anchor_hit": sum(bool(r.get("anchor_hit")) for r in al) / len(al),
        "datesPres": st.mean([r["dates_present"] for r in rows]),
        "chars_median": st.median([r["chars"] for r in rows]),
        "chars_mean": st.mean([r["chars"] for r in rows]),
        "tl_dates_mean": st.mean(ndates),
        "tl_dates_median": st.median(ndates),
        "n_events_mean": st.mean([r["n_events"] for r in rows]),
    }


ARMS = [("final", True, True, _orig),
        ("no_stem", False, True, _orig),
        ("no_intext", True, False, no_intext),
        ("no_stem_nointext", False, False, no_intext)]
for k in range(3):
    ARMS.append((f"jitter_{k}", True, False, make_jitter(1000 + k)))
for k in range(3):
    ARMS.append((f"shuffle_{k}", True, True, make_shuffle(2000 + k)))

out, per_q = {}, {}
for name, stem, _it, fn in ARMS:
    V.USE_STEM = stem
    V.USE_QFORM_STOP = False
    V.USE_SWEEP = False
    V.best_event_date = fn
    rows = []
    for q in data:
        rec, tl = analyse(q, stores[q["question_id"]])
        ev = V.extract_events(q, stores[q["question_id"]], q["question"])
        rec["_tl_dates"] = sorted({e.ev_dt.strftime("%Y%m%d") for e in ev})
        rows.append(rec)
    out[name] = metrics(rows)
    per_q[name] = {r["qid"]: {"fam": r["family"], "dTL": r["gold_derivable_from_timeline"],
                              "dLit": r["gold_literal_in_arith"],
                              "dEv": r["gold_derivable_from_evidence"],
                              "ordc": r.get("ordering_item_correct"),
                              "sds": r.get("same_day_split"),
                              "ndates": len(r["_tl_dates"]), "chars": r["chars"],
                              "gold": r["gold"], "unit": r["unit"],
                              "tl_deriv": r["tl_derivation"],
                              "ans_dates": r["ans_dates"],
                              "dpres": r["dates_present"]} for r in rows}
    print(f"{name:18s} " + "  ".join(f"{k}={v:.4g}" for k, v in out[name].items()), flush=True)

V.best_event_date = _orig
json.dump({"metrics": out, "per_q": per_q, "p_shift": p_shift}, open(OUT, "w"), indent=1)
print("saved", OUT)
