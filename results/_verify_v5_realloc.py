"""INDEPENDENT adversarial recomputation of the v5_realloc claim. No API.

Baseline context is rebuilt from run_qc.py's own logic (render_view alphabetical + 20k line
truncation + sorted(bm25_rank(...)[:20])) -- NOT from v5_retrieval's "baseline" config -- so a
mis-specified baseline in the candidate module would show up as a mismatch.
"""
import hashlib
import json
import math
import os
import re
import sys
from multiprocessing import Pool

sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from coverage_diag import contains, norm
from lifecycle import assemble_chains, render_view, _session_text, SESSION_CACHE_DIR
from query_classifier import classify_query
from run_baseline import bm25_rank
import v5_retrieval as V5

R = "/root/agent_memory_iclr/results"
DATA = None
VARIANTS = ["baseline", "rankview", "v5_drop_in", "v5_realloc", "final8k_bm", "final8k",
            "bm25_k30", "bm25_k50", "divcap2"]
MATCHED = {"v5_realloc_m": "v5_realloc", "final8k_bm_m": "final8k_bm"}


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
    gold = q["answer"]
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"

    all_facts, missing = [], False
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        tri = cached_facts(sess)
        if tri is None:
            missing = True
            continue
        for e, a, v in tri:
            all_facts.append({"entity": e, "attribute": a, "value": v,
                              "date": date, "sidx": si})
    store = assemble_chains(all_facts)

    # ---- independent units + has_answer flags (my own construction) ----
    units, flags = [], []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for t in sess:
            if t.get("content"):
                units.append((si, date, t["role"], t["content"]))
                flags.append(bool(t.get("has_answer")))
    ev_turns = {i for i, f in enumerate(flags) if f}
    ans_ids = set(q["answer_session_ids"])
    ev_sess = {i for i, sid in enumerate(q["haystack_session_ids"]) if sid in ans_ids}

    # ---- INDEPENDENT baseline exactly as run_qc.py builds it ----
    view_full = render_view(store, mode)
    if len(view_full) > 20000:
        kept, tot = [], 0
        for line in view_full.splitlines():
            if tot + len(line) + 1 > 20000:
                break
            kept.append(line)
            tot += len(line) + 1
        bview = "\n".join(kept)
    else:
        bview = view_full
    border = sorted(bm25_rank(q["question"], [u[3] for u in units])[:20])
    braw = "\n".join(f"[{units[i][1]}] [{units[i][2]}] {units[i][3]}" for i in border)

    bm = V5.bm25_scores(q["question"], [u[3] for u in units])

    out = {"qid": q["question_id"], "type": q["question_type"], "mode": mode,
           "is_abs": q["question_id"].endswith("_abs"), "cache_missing": missing,
           "n_ev_turns": len(ev_turns), "n_ev_sessions": len(ev_sess),
           "n_units": len(units), "n_sessions": len(q["haystack_sessions"]),
           "n_chains": len(store), "view_full_chars": len(view_full),
           "view_full_lines": len(view_full.splitlines()),
           "is_count": bool(V5.COUNT_RE.search(q["question"])),
           "gold_len": len(str(gold)), "v": {}}

    def row(vt, rt, idx):
        raw_c = " ".join(units[i][3] for i in idx)
        gpos = norm(vt).find(norm(gold)) if gold else -1
        r = {"view_chars": len(vt), "raw_chars": len(rt), "n_turns": len(idx),
             "n_turn_sessions": len({units[i][0] for i in idx}),
             "view_lines": len(vt.splitlines()),
             "in_view": contains(vt, gold), "in_raw": contains(rt, gold),
             "in_raw_c": contains(raw_c, gold), "gold_view_pos": gpos,
             "ev_hit": len(set(idx) & ev_turns),
             "ev_sess_hit": len({units[i][0] for i in idx} & ev_sess),
             "idx": sorted(idx)}
        r["allev"] = (r["ev_hit"] == len(ev_turns))
        r["in_ctx"] = r["in_view"] or r["in_raw"]
        r["in_ctx_c"] = r["in_view"] or r["in_raw_c"]
        return r

    out["v"]["BASE_INDEP"] = row(bview, braw, border)
    base_total = len(bview) + len(braw)

    for name in VARIANTS:
        vt, rt, dbg = V5.build_context(q, store, q["question"], config=name, mode=mode,
                                       units=units, bm_scores=bm)
        out["v"][name] = row(vt, rt, dbg["turn_idx"])
    for name, cfg in MATCHED.items():
        v0, _r0, _d0 = V5.build_context(q, store, q["question"], config=cfg, mode=mode,
                                        units=units, bm_scores=bm)
        rc = max(0, base_total - len(v0))
        vt, rt, dbg = V5.build_context(q, store, q["question"], config=cfg, mode=mode,
                                       units=units, bm_scores=bm, raw_char_cap=rc)
        out["v"][name] = row(vt, rt, dbg["turn_idx"])
    return out


def main():
    global DATA
    DATA = load_dataset("s_cleaned")
    with Pool(30) as p:
        rows = p.map(one, range(len(DATA)), chunksize=2)
    with open(f"{R}/_verify_v5.json", "w") as f:
        json.dump(rows, f)
    print("saved", len(rows))


if __name__ == "__main__":
    main()
