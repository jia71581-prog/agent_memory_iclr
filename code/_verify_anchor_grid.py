"""Stability of the anchor effect across the provenance-hyperparameter grid, and under
extraction-noise perturbation of the fact store. Offline only, no API."""
import hashlib
import json
import math
import os
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from lifecycle import assemble_chains, _session_text, SESSION_CACHE_DIR
from query_classifier import classify_query
import v5_retrieval as V5

R = "/root/agent_memory_iclr/results"
DATA = None

GRID = []
for ts, ps, npv in [(4, 2, 6), (4, 3, 6), (6, 2, 6), (6, 3, 6), (8, 3, 6), (8, 2, 6),
                    (4, 2, 4), (4, 2, 8), (4, 3, 8), (10, 3, 10), (4, 2, 10)]:
    for aw in (0.0, 1.0):
        GRID.append((f"ts{ts}p{ps}n{npv}_a{int(aw)}",
                     dict(view="rank", n_bm25=20 - npv, n_prov=npv, session_cap=None,
                          top_sessions=ts, per_session=ps, anchor_weight=aw)))
GRID.append(("rankview", dict(view="rank", n_bm25=20, n_prov=0, session_cap=None)))
# fact-store perturbation (extraction noise): drop each fact w.p. 0.1, 3 seeds
DROPS = [(0.1, 1), (0.1, 2), (0.2, 3)]


def cached_facts(sess):
    text = _session_text(sess)
    if not text:
        return []
    key = hashlib.md5(text.encode()).hexdigest()[:16]
    p = os.path.join(SESSION_CACHE_DIR, f"{key}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def one(qi):
    q = DATA[qi]
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    raw_facts = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for e, a, v in cached_facts(sess):
            raw_facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
    units = V5.build_units(q)
    flags = []
    for si, sess in enumerate(q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                flags.append(bool(t.get("has_answer")))
    ev = {i for i, f in enumerate(flags) if f}
    bm = V5.bm25_scores(q["question"], [u[3] for u in units])
    out = {"qid": q["question_id"], "type": q["question_type"],
           "is_abs": q["question_id"].endswith("_abs"), "n_ev": len(ev), "v": {}}

    def run(store, tag, cfgs):
        for name, cfg in cfgs:
            _vt, _rt, dbg = V5.build_context(q, store, q["question"], config=cfg, mode=mode,
                                             units=units, bm_scores=bm)
            out["v"][tag + name] = len(set(dbg["turn_idx"]) & ev)

    run(assemble_chains(raw_facts), "", GRID)
    for frac, seed in DROPS:
        rnd = random.Random(hash((q["question_id"], seed)) & 0xFFFFFFFF)
        kept = [f for f in raw_facts if rnd.random() >= frac]
        run(assemble_chains(kept), f"d{int(frac*100)}s{seed}_",
            [(n, c) for n, c in GRID if n in ("ts4p2n6_a0", "ts4p2n6_a1", "rankview")])
    return out


def main():
    global DATA
    DATA = load_dataset("s_cleaned")
    with Pool(30) as p:
        rows = p.map(one, range(len(DATA)), chunksize=2)
    json.dump(rows, open(f"{R}/_verify_anchor_grid.json", "w"))
    print("saved", len(rows))


if __name__ == "__main__":
    main()
