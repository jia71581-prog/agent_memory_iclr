"""Is the ordering 'collapse' partly a tie-break artifact? With in-text resolution off, every
event in a session shares the session date, so many candidate items get IDENTICAL dates and the
verdict is decided by Python's stable sort = the order the question mentions the candidates.
Recompute the OFF arm's ordering score with an explicit random tie-break. Zero API."""
import json, os, sys, random, statistics as st
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_temporal as V
from common import load_dataset
import v5_temporal_measure as M
from v5_temporal_val import build_store_cached

data = [q for q in load_dataset("s_cleaned") if q["question_type"] == "temporal-reasoning"]
stores = {q["question_id"]: build_store_cached(q)[0] for q in data}
_orig = V.best_event_date


def no_intext(text, session_dt):
    d = V.parse_date(session_dt)
    return (V.day0(d), "") if d else (None, "")


def ordering_detail(q, store):
    question = q["question"]
    family, unit = M.detect_family(question)
    if family != "ordering":
        return None
    events = V.extract_events(q, store, question)
    gseq, cands = M.ordering_items(question, q["answer"])
    dates_i = {c: M._match_event(c, events) for c in cands}
    if len(cands) < 2 or any(v is None for v in dates_i.values()):
        return {"computable": False}
    vals = list(dates_i.values())
    tied = len(set(vals)) < len(vals)
    all_tied = len(set(vals)) == 1
    got = sorted(cands, key=lambda c: dates_i[c])
    if len(gseq) >= 3:
        verdict = [M.norm(x) for x in got] == [M.norm(x) for x in gseq]
    elif gseq:
        verdict = M.contains(got[0], gseq[0]) or M.contains(gseq[0], got[0])
    else:
        verdict = None
    return {"computable": True, "tied": tied, "all_tied": all_tied, "verdict": verdict,
            "n_items": len(cands), "gold_first": gseq[0] if gseq else None,
            "picked": got[0], "cands": cands}


for name, fn in (("ON (shipped)", _orig), ("OFF (no in-text)", no_intext)):
    V.best_event_date = fn
    V.USE_STEM, V.USE_QFORM_STOP, V.USE_SWEEP = True, False, False
    det = [(q["question_id"], ordering_detail(q, stores[q["question_id"]])) for q in data]
    det = [(k, d) for k, d in det if d is not None]
    comp = [(k, d) for k, d in det if d["computable"]]
    ok = [1 for _, d in comp if d["verdict"]]
    tied = [(k, d) for k, d in comp if d["tied"]]
    at = [(k, d) for k, d in comp if d["all_tied"]]
    print(f"\n{name}: ordering n={len(det)} computable={len(comp)} correct={sum(ok)} "
          f"(of all: {sum(ok)/len(det):.4f})")
    print(f"  computable questions whose candidate dates are TIED: {len(tied)} "
          f"(all-tied: {len(at)})")
    print(f"  verdict on tied ones: {Counter(str(d['verdict']) for _, d in tied)}")
    n_items = Counter(d["n_items"] for _, d in comp)
    print(f"  n_items distribution: {dict(sorted(n_items.items()))}")
    # random tie-break: any fully-tied binary question is a coin flip
    rng = random.Random(11)
    trials = []
    for _ in range(200):
        s = 0
        for _, d in comp:
            if d["all_tied"]:
                s += (rng.random() < (1.0 / max(2, d["n_items"]) if d["n_items"] > 2 else 0.5))
            elif d["tied"]:
                s += (d["verdict"] if rng.random() < 0.5 else rng.random() < 0.5)
            else:
                s += bool(d["verdict"])
        trials.append(s / len(det))
    print(f"  ordering correct-of-all with RANDOM tie-break: {st.mean(trials):.4f} "
          f"+/- {st.pstdev(trials):.4f}")
V.best_event_date = _orig
