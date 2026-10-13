"""ADVERSARIAL independent recomputation of the T1 budget claim.

Reimplements the DEPLOYED baseline directly from run_qc.py (render_view + line-boundary
truncation at 20000, bm25_rank top-20, "[date] [role] content" join) WITHOUT going through
v5_retrieval.CONFIGS["baseline"], and independently re-derives has_answer alignment.
Then compares against v5_retrieval's own variants.
"""
import hashlib
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from lifecycle import assemble_chains, render_view, _session_text, SESSION_CACHE_DIR
from query_classifier import classify_query
from run_baseline import bm25_rank
import v5_retrieval as V5

DATA = None


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

    facts = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        tri = cached_facts(sess)
        if tri is None:
            return None
        for e, a, v in tri:
            facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
    store = assemble_chains(facts)

    # ---- independent units + has_answer, built EXACTLY as run_qc.py builds units ----
    units, flags = [], []
    for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                units.append((date, t["role"], t["content"]))
                flags.append(bool(t.get("has_answer")))
    ev = {i for i, f in enumerate(flags) if f}

    # ---- DEPLOYED baseline, transcribed from run_qc.answer_question ----
    view = render_view(store, mode)
    if len(view) > 20000:
        kept, tot = [], 0
        for line in view.splitlines():
            if tot + len(line) + 1 > 20000:
                break
            kept.append(line)
            tot += len(line) + 1
        view = "\n".join(kept)
    order = bm25_rank(q["question"], [u[2] for u in units])

    res = {"qid": q["question_id"], "n_ev": len(ev), "n_units": len(units), "v": {}}
    for k in (20, 30, 50):
        idx = sorted(order[:k])
        raw = "\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in idx)
        res["v"][f"deployed_k{k}"] = {
            "view_chars": len(view), "raw_chars": len(raw), "n_turns": len(idx),
            "allev": len(set(idx) & ev) == len(ev), "ev_hit": len(set(idx) & ev),
        }

    # ---- v5_retrieval variants, recomputed live (not read from the artifact) ----
    bm = V5.bm25_scores(q["question"], [u[2] for u in units])
    qq = {"haystack_dates": q["haystack_dates"], "haystack_sessions": q["haystack_sessions"]}
    u5 = V5.build_units(qq)
    assert [x[3] for x in u5] == [x[2] for x in units], "unit order divergence"
    base_total = None
    for name in ["baseline", "rankview", "bm25_k30", "bm25_k50", "v5_realloc", "final8k_bm",
                 "v5_realloc_m", "final8k_bm_m", "v5_drop_in"]:
        cfg = {"v5_realloc_m": "v5_realloc", "final8k_bm_m": "final8k_bm"}.get(name, name)
        rc = None
        if name.endswith("_m"):
            v0, _r, _d = V5.build_context(qq, store, q["question"], config=cfg, mode=mode,
                                          units=u5, bm_scores=bm)
            rc = max(0, base_total - len(v0))
        vt, rt, dbg = V5.build_context(qq, store, q["question"], config=cfg, mode=mode,
                                       units=u5, bm_scores=bm, raw_char_cap=rc)
        if name == "baseline":
            base_total = len(vt) + len(rt)
        idx = dbg["turn_idx"]
        res["v"][name] = {
            "view_chars": len(vt), "raw_chars": len(rt), "n_turns": len(idx),
            "allev": len(set(idx) & ev) == len(ev), "ev_hit": len(set(idx) & ev),
        }
    return res


def main():
    global DATA
    DATA = load_dataset("s_cleaned")
    with Pool(30) as p:
        rows = [r for r in p.map(one, range(len(DATA)), chunksize=2) if r]
    json.dump(rows, open("/root/agent_memory_iclr/results/_verify_budget.json", "w"))
    print("rows", len(rows))


if __name__ == "__main__":
    main()
