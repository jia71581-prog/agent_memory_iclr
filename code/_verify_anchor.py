"""Adversarial recomputation of the value-anchoring claim. No API. Offline only."""
import hashlib
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from lifecycle import assemble_chains, _session_text, SESSION_CACHE_DIR
from query_classifier import classify_query
import v5_retrieval as V5

R = "/root/agent_memory_iclr/results"
DATA = None

# ------------- sham anchors: question-independent perturbations of the SAME magnitude
def _sham_rand(content, values, qt):
    h = hashlib.md5(content.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF * 1.5


def _sham_len(content, values, qt):
    return min(len(content) / 2000.0, 1.5)


def _sham_first(content, values, qt):
    # constant -> no reordering at all (equivalent to anchor_weight 0 but through the same path)
    return 0.0


SHAM = {"rand": _sham_rand, "len": _sham_len, "const": _sham_first}

# variants: name -> (config dict, sham key or None)
VAR = {
    "baseline":   (dict(view="alpha", n_bm25=20, n_prov=0, session_cap=None), None),
    "rankview":   (dict(view="rank", n_bm25=20, n_prov=0, session_cap=None), None),
    # p6 family (top_sessions=8, per_session=3 defaults)
    "p6":         (dict(view="rank", n_bm25=14, n_prov=6, session_cap=None), None),
    "p6_a0":      (dict(view="rank", n_bm25=14, n_prov=6, session_cap=None, anchor_weight=0.0), None),
    "p6_a2":      (dict(view="rank", n_bm25=14, n_prov=6, session_cap=None, anchor_weight=2.0), None),
    # the MISSING ablation: the deployed drop-in geometry (ts=4, ps=2) with the anchor OFF
    "ts4_a1":     (dict(view="rank", n_bm25=14, n_prov=6, session_cap=None,
                        top_sessions=4, per_session=2), None),
    "ts4_a0":     (dict(view="rank", n_bm25=14, n_prov=6, session_cap=None,
                        top_sessions=4, per_session=2, anchor_weight=0.0), None),
    "ts4_a2":     (dict(view="rank", n_bm25=14, n_prov=6, session_cap=None,
                        top_sessions=4, per_session=2, anchor_weight=2.0), None),
    "ts4_a05":    (dict(view="rank", n_bm25=14, n_prov=6, session_cap=None,
                        top_sessions=4, per_session=2, anchor_weight=0.5), None),
    # sham anchors at the same weight: question-independent tie-perturbations
    "p6_sham_rand":  (dict(view="rank", n_bm25=14, n_prov=6, session_cap=None), "rand"),
    "p6_sham_len":   (dict(view="rank", n_bm25=14, n_prov=6, session_cap=None), "len"),
    "ts4_sham_rand": (dict(view="rank", n_bm25=14, n_prov=6, session_cap=None,
                           top_sessions=4, per_session=2), "rand"),
    "ts4_sham_len":  (dict(view="rank", n_bm25=14, n_prov=6, session_cap=None,
                           top_sessions=4, per_session=2), "len"),
}


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
    all_facts = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        tri = cached_facts(sess)
        if tri is None:
            return None
        for e, a, v in tri:
            all_facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
    store = assemble_chains(all_facts)
    units = V5.build_units(q)
    flags = []
    for si, sess in enumerate(q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                flags.append(bool(t.get("has_answer")))
    ev_turns = {i for i, f in enumerate(flags) if f}
    ans_ids = set(q["answer_session_ids"])
    ev_sess = {i for i, sid in enumerate(q["haystack_session_ids"]) if sid in ans_ids}
    bm = V5.bm25_scores(q["question"], [u[3] for u in units])

    out = {"qid": q["question_id"], "type": q["question_type"],
           "is_abs": q["question_id"].endswith("_abs"),
           "n_ev_turns": len(ev_turns), "n_ev_sessions": len(ev_sess), "v": {}}
    real = V5._value_anchor
    for name, (cfg, sham) in VAR.items():
        try:
            if sham:
                V5._value_anchor = SHAM[sham]
            vt, rt, dbg = V5.build_context(q, store, q["question"], config=cfg, mode=mode,
                                           units=units, bm_scores=bm)
        finally:
            V5._value_anchor = real
        idx = dbg["turn_idx"]
        out["v"][name] = {
            "view_chars": len(vt), "raw_chars": len(rt), "n_turns": len(idx),
            "ev_turns_hit": len(set(idx) & ev_turns),
            "ev_sess_hit": len({units[i][0] for i in idx} & ev_sess),
            "n_turn_sessions": len(dbg["turn_sessions"]),
            "turn_idx": idx,
        }
    return out


def main():
    global DATA
    DATA = load_dataset("s_cleaned")
    with Pool(30) as p:
        rows = p.map(one, range(len(DATA)), chunksize=2)
    assert all(r is not None for r in rows), "cache miss"
    with open(f"{R}/_verify_anchor.json", "w") as f:
        json.dump(rows, f)
    print("saved", len(rows))


if __name__ == "__main__":
    main()
