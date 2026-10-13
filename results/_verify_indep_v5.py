"""INDEPENDENT recomputation of the v5 retrieval claim. No API. Does not read prov_v5_offline.json.

Rebuilds, from the dataset + the session extraction cache only:
  - the deployed baseline context (alphabetical render_view capped at 20k + BM25 top-20 turns)
  - gold-string containment (published `contains`, plus a STRICT normalized-substring variant)
  - oracle evidence-turn / evidence-session hits
  - the A set (published definition: gold not in UNCAPPED view and not in BM25 top-20 content)
and then the v5 variants through v5_retrieval.build_context.
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
from lifecycle import assemble_chains, render_view, _session_text, SESSION_CACHE_DIR
from query_classifier import classify_query
import v5_retrieval as V5

R = "/root/agent_memory_iclr/results"
DATA = None

_w = re.compile(r"[a-z0-9']+")


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def contains(hay, needle):
    """published metric (coverage_diag.contains)"""
    h, n = norm(hay), norm(needle)
    if not n:
        return False
    if n in h:
        return True
    toks = [t for t in n.split() if len(t) > 2 and t not in ("the", "and", "for", "was", "were", "you", "your")]
    if not toks:
        return False
    return all(t in h for t in toks)


def contains_strict(hay, needle):
    """STRICT: normalized substring only (no bag-of-words disjunct)"""
    h, n = norm(hay), norm(needle)
    return bool(n) and n in h


def my_bm25_order(query, docs, k1=1.5, b=0.75):
    """Independent BM25 implementation (standard Okapi, same k1/b as the deployed ranker)."""
    qt = _w.findall(query.lower())
    dt = [_w.findall(d.lower()) for d in docs]
    N = len(dt)
    avgdl = sum(len(d) for d in dt) / max(1, N)
    df = {}
    for d in dt:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    sc = []
    for d in dt:
        tf = {}
        for w in d:
            tf[w] = tf.get(w, 0) + 1
        s = 0.0
        for w in qt:
            if w in tf:
                idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avgdl))
        sc.append(s)
    return sorted(range(N), key=lambda i: (-sc[i], i)), sc


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

    facts, missing = [], False
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        tri = cached_facts(sess)
        if tri is None:
            missing = True
            continue
        for e, a, v in tri:
            facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
    store = assemble_chains(facts)

    # units in haystack order, identical unit definition to the deployed runner
    units, flags = [], []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for t in sess:
            if t.get("content"):
                units.append((si, date, t["role"], t["content"]))
                flags.append(bool(t.get("has_answer")))
    ev_turns = {i for i, f in enumerate(flags) if f}
    ans_ids = set(q["answer_session_ids"])
    ev_sess = {i for i, sid in enumerate(q["haystack_session_ids"]) if sid in ans_ids}

    order, sc = my_bm25_order(q["question"], [u[3] for u in units])

    # ---- deployed baseline, rebuilt from scratch ----
    full_view = render_view(store, mode)
    kept, tot = [], 0
    for line in full_view.splitlines():
        if tot + len(line) + 1 > 20000:
            break
        kept.append(line)
        tot += len(line) + 1
    view_capped = "\n".join(kept)
    top20 = sorted(order[:20])
    raw20_c = " ".join(units[i][3] for i in top20)
    raw50_c = " ".join(units[i][3] for i in sorted(order[:50]))
    hay = " ".join(u[3] for u in units)

    row = {
        "qid": q["question_id"], "type": q["question_type"], "is_abs": q["question_id"].endswith("_abs"),
        "cache_missing": missing, "n_ev_turns": len(ev_turns), "n_ev_sess": len(ev_sess),
        "gold": str(gold)[:300], "gold_is_num": bool(re.fullmatch(r"\s*\d+(\.\d+)?\s*", str(gold))),
        # published A-set inputs (UNCAPPED view, as coverage_diag.py measured)
        "in_view_uncapped": contains(full_view, gold),
        "in_raw20": contains(raw20_c, gold),
        "in_raw50": contains(raw50_c, gold),
        "in_hay": contains(hay, gold),
        "in_hay_strict": contains_strict(hay, gold),
        "view_chars_uncapped": len(full_view),
        # deployed (capped) baseline
        "base_in_view": contains(view_capped, gold),
        "base_in_raw_c": contains(raw20_c, gold),
        "base_ev_hit": len(set(top20) & ev_turns),
        "base_ev_sess_hit": len({units[i][0] for i in top20} & ev_sess),
        "base_chars": len(view_capped) + len("\n".join(f"[{units[i][1]}] [{units[i][2]}] {units[i][3]}" for i in top20)),
        "v": {},
    }
    row["base_in_ctx"] = row["base_in_view"] or row["base_in_raw_c"]

    # cross-check: does v5's bm25 reproduce my independent top-20?
    v5_bm = V5.bm25_scores(q["question"], [u[3] for u in units])
    row["bm25_match"] = sorted(range(len(units)), key=lambda i: (-v5_bm[i], i))[:20] == order[:20]

    for name in ["baseline", "rankview", "v5_drop_in", "v5_realloc", "bm25_k50", "final8k_bm"]:
        vt, rt, dbg = V5.build_context(q, store, q["question"], config=name, mode=mode,
                                       units=units, bm_scores=v5_bm)
        idx = dbg["turn_idx"]
        raw_c = " ".join(units[i][3] for i in idx)
        row["v"][name] = {
            "in_view": contains(vt, gold), "in_raw_c": contains(raw_c, gold),
            "in_ctx_c": contains(vt, gold) or contains(raw_c, gold),
            "in_ctx_strict": contains_strict(vt, gold) or contains_strict(raw_c, gold),
            "ev_hit": len(set(idx) & ev_turns), "allEv": len(set(idx) & ev_turns) == len(ev_turns),
            "ev_sess_hit": len({units[i][0] for i in idx} & ev_sess),
            "chars": len(vt) + len(rt), "n_turns": len(idx),
        }
    return row


def main():
    global DATA
    DATA = load_dataset("s_cleaned")
    with Pool(30) as p:
        rows = p.map(one, range(len(DATA)), chunksize=2)
    with open(f"{R}/_verify_indep_v5.json", "w") as f:
        json.dump(rows, f)
    print("saved", len(rows))


if __name__ == "__main__":
    main()
