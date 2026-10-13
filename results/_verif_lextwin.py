"""Adversarial control: a LEXICAL TWIN of v5_drop_in that keeps the same shape
(6 turns round-robin over the top-4 sessions, <=2 each, then BM25 fill to 20)
but ranks sessions by BM25 mass instead of fact-store provenance.
If this matches v5_drop_in's allEv, the gain is 'session shape', not provenance.
Also records v5_drop_in / baseline turn indices to test where the gained evidence turns live.
"""
import hashlib, json, math, os, re, sys
from multiprocessing import Pool

sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains, _session_text, SESSION_CACHE_DIR
from query_classifier import classify_query
import v5_retrieval as V5

DATA = None
R = "/root/agent_memory_iclr/results"


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


def rr(order, by_sess, ranked, n, per_session):
    picked, slot = [], 0
    while len(picked) < n and slot < per_session:
        added = False
        for si in order:
            if len(picked) >= n:
                break
            lst = ranked.get(si, [])
            if slot < len(lst):
                picked.append(lst[slot])
                added = True
        if not added:
            break
        slot += 1
    return picked


def one(qi):
    q = DATA[qi]
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    facts = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        tri = cached_facts(sess)
        if tri is None:
            continue
        for e, a, v in tri:
            facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
    store = assemble_chains(facts)
    units = V5.build_units(q)
    flags = []
    for si, sess in enumerate(q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                flags.append(bool(t.get("has_answer")))
    ev = {i for i, f in enumerate(flags) if f}
    bm = V5.bm25_scores(q["question"], [u[3] for u in units])
    by_sess = {}
    for i, u in enumerate(units):
        by_sess.setdefault(u[0], []).append(i)

    out = {"qid": q["question_id"], "type": q["question_type"],
           "is_abs": q["question_id"].endswith("_abs"), "n_ev": len(ev), "v": {}}

    # --- reference variants through their own module
    for name in ["baseline", "v5_drop_in"]:
        _vt, _rt, dbg = V5.build_context(q, store, q["question"], config=name, mode=mode,
                                         units=units, bm_scores=bm)
        idx = dbg["turn_idx"]
        out["v"][name] = {"idx": idx, "sess": sorted({units[i][0] for i in idx}),
                          "ev_hit": len(set(idx) & ev), "allev": int(len(set(idx) & ev) == len(ev)),
                          "prov_idx": dbg["prov_idx"],
                          "prov_sessions": dbg["prov_sessions"][:4]}

    # --- LEXICAL TWIN: session ranking from BM25 mass, identical shape to v5_drop_in
    for tag, agg in [("lex_twin_max", max), ("lex_twin_sum", sum)]:
        sscore = {si: agg(bm[i] for i in idxs) for si, idxs in by_sess.items()}
        order = sorted(sscore, key=lambda si: (-sscore[si], si))[:4]
        ranked = {si: sorted(by_sess[si], key=lambda i: (-bm[i], i))[:2] for si in order}
        prov = rr(order, by_sess, ranked, 6, 2)
        gl = sorted(range(len(units)), key=lambda i: (-bm[i], i))
        picked, seen = [], set()
        for i in list(prov) + gl:
            if i in seen:
                continue
            seen.add(i)
            picked.append(i)
            if len(picked) >= 20:
                break
        out["v"][tag] = {"idx": sorted(picked), "sess": sorted({units[i][0] for i in picked}),
                         "ev_hit": len(set(picked) & ev),
                         "allev": int(len(set(picked) & ev) == len(ev))}

    # --- CONCENTRATION control: BM25 top-20 restricted to the top-4 BM25 sessions, then fill
    sscore = {si: max(bm[i] for i in idxs) for si, idxs in by_sess.items()}
    top4 = set(sorted(sscore, key=lambda si: (-sscore[si], si))[:4])
    gl = sorted(range(len(units)), key=lambda i: (-bm[i], i))
    picked = [i for i in gl if units[i][0] in top4][:20]
    picked += [i for i in gl if i not in picked][:20 - len(picked)]
    out["v"]["concentrate4"] = {"idx": sorted(picked), "sess": sorted({units[i][0] for i in picked}),
                                "ev_hit": len(set(picked) & ev),
                                "allev": int(len(set(picked) & ev) == len(ev))}
    out["ev"] = sorted(ev)
    out["ev_sess"] = sorted({units[i][0] for i in ev})
    return out


def main():
    global DATA
    DATA = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
    with Pool(30) as p:
        rows = p.map(one, range(len(DATA)), chunksize=2)
    json.dump(rows, open(f"{R}/_verif_lextwin.json", "w"))
    print("saved", len(rows))


if __name__ == "__main__":
    main()
