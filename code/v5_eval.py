"""Offline (no-API) evaluation of v5 provenance-guided retrieval vs the deployed baseline.

Per question and per variant we record, from already-cached data only:
  gold-string containment (coverage_diag.contains, so numbers are comparable to coverage_diag.json)
  oracle evidence recall  (has_answer turns / answer_session_ids from the dataset)
  aggregation-completeness proxy for counting questions
  context size (chars, est. tokens = chars/4)
Output: results/prov_v5_offline.json  (one compact row per question x variant)
"""
import hashlib
import json
import os
import re
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from coverage_diag import contains, norm
from lifecycle import assemble_chains, _session_text, SESSION_CACHE_DIR
from query_classifier import classify_query
import v5_retrieval as V5

R = "/root/agent_memory_iclr/results"
DATA = None
VARIANTS = ["baseline", "rankview", "divcap2", "prov_hybrid6", "prov_hybrid", "prov_only",
            "v5", "v5_divcap", "v5_deep", "v5_wide", "v5_noanchor", "v5_nocount", "v5_focus",
            "bm25_k30", "bm25_k50", "realloc", "realloc_bm", "realloc_bm_a", "realloc8k", "realloc8k_bm", "v5_final", "split4k", "split4k_bm", "split6k", "split6k_bm", "split8k", "split8k_bm", "split10k", "split10k_bm", "p4", "p6", "p8", "p10", "p6_ts4", "p6_ts6", "p6_a0", "p6_a2", "p6_c15", "p6_c80", "final12k", "final12k_bm", "final10k", "final10k_bm", "final8k", "final8k_bm", "v5_drop_in", "v5_realloc"]
COUNT_RE = V5.COUNT_RE
GENERIC = set("""how many much total number days times hours weeks months years items did have has
had was were do does i my me currently now still overall together altogether count""".split())


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


def gold_int(ans):
    m = re.findall(r"\d+(?:\.\d+)?", str(ans).replace(",", ""))
    if m:
        return float(m[0])
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
    for w, v in words.items():
        if re.search(rf"\b{w}\b", str(ans).lower()):
            return float(v)
    return None


def target_terms(question, units, min_len=4, df_frac=0.5):
    """Aggregation proxy, part 1: the question's content terms that could name the items being
    counted. Generic counting vocabulary is removed, and terms appearing in more than df_frac of
    the haystack SESSIONS are dropped as non-discriminative (they cannot individuate items)."""
    cand = [t for t in V5.qterms(question) if t not in GENERIC and len(t) >= min_len]
    if not cand:
        return []
    sess_tok = {}
    for si, _d, _r, c in units:
        sess_tok.setdefault(si, set()).update(V5.toks(c))
    ns = max(1, len(sess_tok))
    keep = [t for t in cand if sum(t in s for s in sess_tok.values()) / ns <= df_frac]
    return keep or cand


def agg_support(picked_idx, units, terms):
    """Aggregation proxy, part 2: how many DISTINCT dated evidence items the raw context exposes
    for the counted target = number of distinct provenance sessions among the selected turns that
    mention at least one target term. Honest limits: (a) it counts sessions, not items, so two
    items reported in one session count once and one item discussed twice in a session counts
    once; (b) term matching is lexical, so a paraphrased item is missed; (c) it says the context
    COULD support the count, never that the model would compute it."""
    if not terms:
        return 0
    hit = set()
    for i in picked_idx:
        c = set(V5.toks(units[i][3]))
        if any(t in c for t in terms):
            hit.add(units[i][0])
    return len(hit)


def chain_agg_support(store, scored, view_txt, mode, gold_num, topn=5, min_sessions=2):
    """Aggregation proxy, CHAIN version (stronger than the session-count proxy).

    An accumulative / multi-session chain literally enumerates the items to be counted
    ("user | plants acquired = a monstera (05/20); a fern (05/22); a cactus (05/24)").
    For the top-`topn` question-matching chains that span >= min_sessions sessions we count the
    distinct values, but ONLY if that chain's rendered line actually survived into the view -
    which is the whole point: the deployed 20k cap drops 34.5% of lines alphabetically.
    Returns (best_n_values, best_n_values_present_in_view).
    """
    best, best_in = 0, 0
    for k, s, n in scored[:topn]:
        if s <= 0 or n < min_sessions:
            continue
        chain = store[k]
        nv = len({norm(c["value"]) for c in chain})
        best = max(best, nv)
        line = V5._render_chain_line(chain, mode)
        if line and line in view_txt:
            best_in = max(best_in, nv)
    return best, best_in


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
            all_facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
    store = assemble_chains(all_facts)

    units = V5.build_units(q)
    # has_answer flags in haystack order (build_units skips empty-content turns identically)
    flags = []
    for si, sess in enumerate(q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                flags.append(bool(t.get("has_answer")))
    ev_turns = {i for i, f in enumerate(flags) if f}
    ans_ids = set(q["answer_session_ids"])
    ev_sess = {i for i, sid in enumerate(q["haystack_session_ids"]) if sid in ans_ids}

    bm = V5.bm25_scores(q["question"], [u[3] for u in units])
    is_count = bool(COUNT_RE.search(q["question"]))
    gnum = gold_int(gold) if is_count else None
    terms = target_terms(q["question"], units) if is_count else []

    base_raw_chars = None
    out = {"qid": q["question_id"], "type": q["question_type"], "mode": mode, "qlabel": label,
           "cache_missing": missing, "is_abs": q["question_id"].endswith("_abs"),
           "n_chains": len(store), "n_units": len(units), "n_sessions": len(q["haystack_sessions"]),
           "n_ev_turns": len(ev_turns), "n_ev_sessions": len(ev_sess),
           "is_count": is_count, "gold_num": gnum, "n_terms": len(terms), "v": {}}

    MATCH = {"v5_realloc_m": "v5_realloc", "v5_bm": "v5", "v5f_bm": "v5_final", "split4k_m": "split4k", "split4k_bm_m": "split4k_bm", "split6k_m": "split6k", "split6k_bm_m": "split6k_bm", "split8k_m": "split8k", "split8k_bm_m": "split8k_bm", "split10k_m": "split10k", "split10k_bm_m": "split10k_bm", "final12k_m": "final12k", "final12k_bm_m": "final12k_bm", "final10k_m": "final10k", "final10k_bm_m": "final10k_bm", "final8k_m": "final8k", "final8k_bm_m": "final8k_bm", "final12k_m": "final12k", "final12k_bm_m": "final12k_bm", "final10k_m": "final10k", "final10k_bm_m": "final10k_bm", "final8k_m": "final8k", "final8k_bm_m": "final8k_bm"}   # strictly char-matched to the baseline total
    base_total = None
    for name in VARIANTS + ["v5_bm", "v5f_bm"] + ["v5_realloc_m"] + [x+"_m" for x in ['final12k', 'final12k_bm', 'final10k', 'final10k_bm', 'final8k', 'final8k_bm'] + ['split4k', 'split4k_bm', 'split6k', 'split6k_bm', 'split8k', 'split8k_bm', 'split10k', 'split10k_bm']]:
        cfg = MATCH.get(name, name)
        if name in MATCH:
            # two-pass: render the view, then cap the raw block so total chars <= baseline total
            v0, _r0, d0 = V5.build_context(q, store, q["question"], config=cfg, mode=mode,
                                           units=units, bm_scores=bm)
            rc = max(0, base_total - len(v0))
        else:
            rc = None
        vt, rt, dbg = V5.build_context(q, store, q["question"], config=cfg, mode=mode,
                                       units=units, bm_scores=bm, raw_char_cap=rc)
        if name == "baseline":
            base_raw_chars, base_total = len(rt), len(vt) + len(rt)
        idx = dbg["turn_idx"]
        # content-only raw text = exactly what coverage_diag.py measured (no [date] [role] prefix),
        # so in_raw_c is strictly comparable to coverage_diag's in_raw20; in_raw is the decorated
        # text the answerer really sees (the date prefix can create spurious hits for date golds).
        raw_c = " ".join(units[i][3] for i in idx)
        cagg, cagg_in = chain_agg_support(store, dbg["scored"], vt, mode, gnum) if is_count else (0, 0)
        gpos = norm(vt).find(norm(gold)) if gold else -1
        row = {
            "view_chars": len(vt), "raw_chars": len(rt), "n_turns": len(idx),
            "n_turn_sessions": len(dbg["turn_sessions"]), "n_chains_kept": dbg["n_chains_kept"],
            "in_view": contains(vt, gold), "in_raw": contains(rt, gold),
            "in_raw_c": contains(raw_c, gold),
            "gold_view_pos": gpos, "gold_view_first2k": 0 <= gpos < 2000,
            "ev_turns_hit": len(set(idx) & ev_turns),
            "ev_sess_hit": len({units[i][0] for i in idx} & ev_sess),
            "agg": agg_support(idx, units, terms) if is_count else 0,
            "chain_agg": cagg, "chain_agg_in_view": cagg_in,
        }
        row["in_ctx"] = row["in_view"] or row["in_raw"]
        row["in_ctx_c"] = row["in_view"] or row["in_raw_c"]
        out["v"][name] = row
    return out


def main():
    global DATA
    DATA = load_dataset("s_cleaned")
    with Pool(30) as p:
        rows = p.map(one, range(len(DATA)), chunksize=2)
    with open(f"{R}/prov_v5_offline.json", "w") as f:
        json.dump(rows, f)
    print("saved", len(rows), "->", f"{R}/prov_v5_offline.json")


if __name__ == "__main__":
    main()
