"""Independent adversarial recomputation of the count_mode / focus_head ablation.

Written from scratch by the verifier: own has_answer alignment, own count-question
subsetting, own oracle aggregation measure. Imports only the module under test
(v5_retrieval) and the fact-store assembler (lifecycle.assemble_chains).
No API calls: all session extractions come from the md5 cache.
"""
import hashlib
import json
import os
import re
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from lifecycle import assemble_chains, _session_text, SESSION_CACHE_DIR
from query_classifier import classify_query
import v5_retrieval as V5

R = "/root/agent_memory_iclr/results"
DATA = None

# configs: the design agent's ablation pair, plus count_mode switched ON at the two
# RECOMMENDED operating points (which the agent's ablation never tested).
CFG = {
    "baseline":      dict(view="alpha", n_bm25=20, n_prov=0, session_cap=None),
    "v5":            dict(view="rank", n_bm25=12, n_prov=8, session_cap=None, count_mode=True),
    "v5_nocount":    dict(view="rank", n_bm25=12, n_prov=8, session_cap=None, count_mode=False),
    "v5_focus":      dict(view="rank", n_bm25=12, n_prov=8, session_cap=None, count_mode=True,
                          focus_head=10),
    "v5_nc_focus":   dict(view="rank", n_bm25=12, n_prov=8, session_cap=None, count_mode=False,
                          focus_head=10),
    "dropin":        dict(view="rank", n_bm25=14, n_prov=6, session_cap=None,
                          top_sessions=4, per_session=2),
    "dropin_count":  dict(view="rank", n_bm25=14, n_prov=6, session_cap=None,
                          top_sessions=4, per_session=2, count_mode=True),
    "realloc":       dict(view="rank", n_bm25=25, n_prov=6, session_cap=None,
                          top_sessions=4, per_session=2, view_cap=8000),
    "realloc_count": dict(view="rank", n_bm25=25, n_prov=6, session_cap=None,
                          top_sessions=4, per_session=2, view_cap=8000, count_mode=True),
    # count_mode with a session budget wide enough to actually hold the chain's provenance set
    "v5_count_ts14": dict(view="rank", n_bm25=12, n_prov=8, session_cap=None, count_mode=True,
                          top_sessions=14, per_session=2),
    "v5_nc_ts14":    dict(view="rank", n_bm25=12, n_prov=8, session_cap=None, count_mode=False,
                          top_sessions=14, per_session=2),
}

CNT = re.compile(r"how many|how much|total|number of", re.I)
NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def gold_num(ans):
    m = re.findall(r"\d+(?:\.\d+)?", str(ans).replace(",", ""))
    if m:
        return float(m[0])
    for w, v in NUMW.items():
        if re.search(rf"\b{w}\b", str(ans).lower()):
            return float(v)
    return None


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def contains(hay, needle):
    h, n = norm(hay), norm(needle)
    if not n:
        return False
    if n in h:
        return True
    toks = [t for t in n.split() if len(t) > 2 and t not in ("the", "and", "for", "was", "were",
                                                             "you", "your")]
    return bool(toks) and all(t in h for t in toks)


def cached(sess):
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

    facts, miss = [], 0
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        tri = cached(sess)
        if tri is None:
            miss += 1
            continue
        for e, a, v in tri:
            facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
    store = assemble_chains(facts)

    units = V5.build_units(q)
    # my own has_answer alignment, built independently of build_units but with the same
    # non-empty-content filter; asserted below to be index-aligned.
    flags, chk = [], []
    for si, sess in enumerate(q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                flags.append(bool(t.get("has_answer")))
                chk.append((si, t["role"], t["content"]))
    assert len(flags) == len(units)
    assert all(c[0] == u[0] and c[1] == u[2] and c[2] == u[3] for c, u in zip(chk, units))
    ev = {i for i, f in enumerate(flags) if f}
    ans_ids = set(q["answer_session_ids"])
    evs = {i for i, sid in enumerate(q["haystack_session_ids"]) if sid in ans_ids}
    # oracle: sessions that actually contain a has_answer turn
    ev_from_turns = {units[i][0] for i in ev}

    bm = V5.bm25_scores(q["question"], [u[3] for u in units])
    is_cnt = bool(CNT.search(q["question"]))
    gn = gold_num(q["answer"]) if is_cnt else None

    out = {"qid": q["question_id"], "type": q["question_type"], "abs": q["question_id"].endswith("_abs"),
           "miss": miss, "n_ev": len(ev), "n_evs": len(evs), "n_ev_turn_sess": len(ev_from_turns),
           "is_cnt": is_cnt, "gold_num": gn, "n_sess": len(q["haystack_sessions"]),
           "n_units": len(units), "v": {}}
    for name, cfg in CFG.items():
        vt, rt, dbg = V5.build_context(q, store, q["question"], config=cfg, mode=mode,
                                       units=units, bm_scores=bm)
        idx = dbg["turn_idx"]
        pos = norm(vt).find(norm(q["answer"]))
        out["v"][name] = {
            "ev_hit": len(set(idx) & ev), "all_ev": len(set(idx) & ev) == len(ev),
            "evs_hit": len({units[i][0] for i in idx} & evs),
            "ev_turn_sess_hit": len({units[i][0] for i in idx} & ev_from_turns),
            "all_ev_sess": len({units[i][0] for i in idx} & evs) == len(evs),
            "view_chars": len(vt), "raw_chars": len(rt), "tot": len(vt) + len(rt),
            "n_turns": len(idx), "n_turn_sess": len(dbg["turn_sessions"]),
            "in_view": contains(vt, q["answer"]), "in_raw": contains(rt, q["answer"]),
            "gold_pos": pos, "first2k": 0 <= pos < 2000,
            "cm": bool(dbg["count_mode"]), "prov_sess": len(dbg["prov_sessions"]),
        }
        out["v"][name]["in_ctx"] = out["v"][name]["in_view"] or out["v"][name]["in_raw"]
    return out


def main():
    global DATA
    DATA = load_dataset("s_cleaned")
    with Pool(30) as p:
        rows = p.map(one, range(len(DATA)), chunksize=2)
    with open(f"{R}/_verify_countmode.json", "w") as f:
        json.dump(rows, f)
    print("saved", len(rows))


if __name__ == "__main__":
    main()
