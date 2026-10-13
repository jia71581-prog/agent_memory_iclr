"""INDEPENDENT verification of the 'provenance must stay a minority slice' negative claim.

Recomputes, from cached extractions only (no API):
  allEv  = every has_answer turn of the question is inside the raw turn block
  evRec  = micro recall of has_answer turns
  turns / sessions / chars actually delivered
for (a) the exact named configs in the claim and (b) clean single-variable dose sweeps that
hold top_sessions / per_session / count_mode / total turns fixed while varying n_prov.
"""
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

# ---- configs under test -------------------------------------------------------------
CFG = {}
# named arms quoted in the claim
CFG["baseline"] = dict(view="alpha", n_bm25=20, n_prov=0, session_cap=None)
CFG["p6_ts4"] = dict(view="rank", n_bm25=14, n_prov=6, session_cap=None, top_sessions=4, per_session=2)
CFG["p8"] = dict(view="rank", n_bm25=12, n_prov=8, session_cap=None)
CFG["p10"] = dict(view="rank", n_bm25=10, n_prov=10, session_cap=None)
CFG["prov_only"] = dict(view="rank", n_bm25=0, n_prov=20, session_cap=None)
CFG["v5_realloc"] = dict(view="rank", n_bm25=25, n_prov=6, session_cap=None,
                         top_sessions=4, per_session=2, view_cap=8000)
CFG["split8k"] = dict(view="rank", n_bm25=19, n_prov=12, session_cap=None, count_mode=True,
                      top_sessions=12, view_cap=8000)
CFG["split8k_bm"] = dict(view="rank", n_bm25=31, n_prov=0, session_cap=None, view_cap=8000)

# ---- clean dose sweep A: 20 turns, top_sessions=8/per_session=3 fixed (p-family default)
for p in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
    CFG[f"A20_p{p}"] = dict(view="rank", n_bm25=20 - p, n_prov=p, session_cap=None,
                            top_sessions=8, per_session=3)
# ---- clean dose sweep B: 20 turns, breadth scaled so prov CAN fill its slice (ps=2, ts=p/2)
for p in [2, 4, 6, 8, 10, 12, 14, 16, 20]:
    CFG[f"B20_p{p}"] = dict(view="rank", n_bm25=20 - p, n_prov=p, session_cap=None,
                            top_sessions=max(1, (p + 1) // 2), per_session=2)
# ---- clean dose sweep C: 31 turns, view_cap 8000, ps=2, ts=ceil(p/2)  (isolates n_prov from
#      the count_mode / top_sessions differences that separate v5_realloc from split8k)
for p in [0, 6, 12, 18, 24, 31]:
    CFG[f"C31_p{p}"] = dict(view="rank", n_bm25=31 - p, n_prov=p, session_cap=None,
                            top_sessions=max(1, (p + 1) // 2), per_session=2, view_cap=8000)
# ---- D: v5_realloc's exact params but n_prov raised with breadth held at ts4/ps2 (cap = 8)
for p in [2, 4, 6, 8]:
    CFG[f"D31_ts4_p{p}"] = dict(view="rank", n_bm25=31 - p, n_prov=p, session_cap=None,
                                top_sessions=4, per_session=2, view_cap=8000)
# ---- E: 31-turn 12-prov with count_mode ON vs OFF at identical breadth (ts12/ps3)
CFG["E31_p12_count"] = dict(view="rank", n_bm25=19, n_prov=12, session_cap=None, count_mode=True,
                            top_sessions=12, view_cap=8000)
CFG["E31_p12_nocount"] = dict(view="rank", n_bm25=19, n_prov=12, session_cap=None, count_mode=False,
                              top_sessions=12, view_cap=8000)
CFG["E31_p12_ts6ps2"] = dict(view="rank", n_bm25=19, n_prov=12, session_cap=None, count_mode=False,
                             top_sessions=6, per_session=2, view_cap=8000)
# ---- F: prov_only with the turn deficit removed (BM25 back-fill to the full 20 turns)
CFG["F_provonly_ts10ps2"] = dict(view="rank", n_bm25=0, n_prov=20, session_cap=None,
                                 top_sessions=10, per_session=2)
CFG["F_provonly_ts20ps1"] = dict(view="rank", n_bm25=0, n_prov=20, session_cap=None,
                                 top_sessions=20, per_session=1)
CFG["F_provonly_ts7ps3"] = dict(view="rank", n_bm25=0, n_prov=20, session_cap=None,
                                top_sessions=7, per_session=3)


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

    all_facts, missing = [], False
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        tri = cached_facts(sess)
        if tri is None:
            missing = True
            continue
        for e, a, v in tri:
            all_facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
    store = assemble_chains(all_facts)

    units = V5.build_units(q)
    flags = []
    for si, sess in enumerate(q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                flags.append(bool(t.get("has_answer")))
    assert len(flags) == len(units)
    ev_turns = {i for i, f in enumerate(flags) if f}
    ans_ids = set(q["answer_session_ids"])
    ev_sess = {i for i, sid in enumerate(q["haystack_session_ids"]) if sid in ans_ids}

    bm = V5.bm25_scores(q["question"], [u[3] for u in units])
    out = {"qid": q["question_id"], "type": q["question_type"], "mode": mode,
           "is_abs": q["question_id"].endswith("_abs"), "cache_missing": missing,
           "n_ev_turns": len(ev_turns), "n_ev_sessions": len(ev_sess),
           "n_units": len(units), "n_sessions": len(q["haystack_sessions"]),
           "is_count": bool(V5.COUNT_RE.search(q["question"])), "v": {}}
    for name, cfg in CFG.items():
        vt, rt, dbg = V5.build_context(q, store, q["question"], config=cfg, mode=mode,
                                       units=units, bm_scores=bm)
        idx = dbg["turn_idx"]
        out["v"][name] = {
            "n_turns": len(idx), "n_prov_actual": len(dbg["prov_idx"]),
            "n_turn_sessions": len(dbg["turn_sessions"]),
            "view_chars": len(vt), "raw_chars": len(rt),
            "ev_hit": len(set(idx) & ev_turns),
            "ev_sess_hit": len({units[i][0] for i in idx} & ev_sess),
            "allev": len(set(idx) & ev_turns) == len(ev_turns),
        }
    return out


def main():
    global DATA
    DATA = load_dataset("s_cleaned")
    with Pool(30) as p:
        rows = p.map(one, range(len(DATA)), chunksize=2)
    with open(f"{R}/_verify_prov_dose.json", "w") as f:
        json.dump(rows, f)
    print("saved", len(rows))


if __name__ == "__main__":
    main()
