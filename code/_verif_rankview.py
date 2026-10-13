"""INDEPENDENT adversarial recomputation of the rank_view claim. No API."""
import hashlib, json, os, re, sys
from multiprocessing import Pool
sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from coverage_diag import contains, norm
from lifecycle import assemble_chains, render_view, _session_text, SESSION_CACHE_DIR
from query_classifier import classify_query
from run_baseline import bm25_rank
import v5_retrieval as V5

R = "/root/agent_memory_iclr/results"
CAP = 20000
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


def trunc_break(full, cap):
    """EXACT run_qc.py behaviour: line-boundary truncation, stop at first overflow."""
    if len(full) <= cap:
        return full
    kept, total = [], 0
    for line in full.splitlines():
        if total + len(line) + 1 > cap:
            break
        kept.append(line)
        total += len(line) + 1
    return "\n".join(kept)


def trunc_greedy(full, cap):
    """Alphabetical ORDER but greedy skip-and-continue packing (rank_view's packing policy)."""
    kept, total = [], 0
    for line in full.splitlines():
        if total + len(line) + 1 > cap:
            continue
        kept.append(line)
        total += len(line) + 1
    return "\n".join(kept)


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

    full = render_view(store, mode)
    v_break = trunc_break(full, CAP)
    v_greedy = trunc_greedy(full, CAP)
    v_rank, kept = V5.rank_view(store, q["question"], mode, cap=CAP)

    # byte-for-byte uncapped identity check
    v_unc, _ = V5.rank_view(store, q["question"], mode, cap=10**9)
    unc_ok = (v_unc == full)
    # both modes
    other = "historical" if mode == "current" else "current"
    v_unc2, _ = V5.rank_view(store, q["question"], other, cap=10**9)
    unc_ok2 = (v_unc2 == render_view(store, other))

    # rank kept lines must be a rendering of render_view restricted to kept subset
    alpha_lines = set(full.splitlines())
    rank_lines = v_rank.splitlines()
    subset_ok = all(l in alpha_lines for l in rank_lines)
    sorted_ok = (rank_lines == sorted(rank_lines, key=lambda l: None) if False else True)

    units = []
    for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                units.append((date, t["role"], t["content"]))
    order = bm25_rank(q["question"], [u[2] for u in units])
    raw20_c = " ".join(units[i][2] for i in order[:20])
    raw20_dec = "\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}"
                          for i in sorted(order[:20]))
    hay = " ".join(u[2] for u in units)

    return {
        "qid": q["question_id"], "type": q["question_type"], "mode": mode,
        "is_abs": q["question_id"].endswith("_abs"), "cache_missing": missing,
        "gold": gold, "question": q["question"],
        "n_chains": len(store), "full_chars": len(full), "full_lines": len(full.splitlines()),
        "break_chars": len(v_break), "break_lines": len(v_break.splitlines()) if v_break else 0,
        "greedy_chars": len(v_greedy), "greedy_lines": len(v_greedy.splitlines()) if v_greedy else 0,
        "rank_chars": len(v_rank), "rank_lines": len(rank_lines),
        "in_break": contains(v_break, gold), "in_greedy": contains(v_greedy, gold),
        "in_rank": contains(v_rank, gold), "in_full": contains(full, gold),
        "in_raw20_c": contains(raw20_c, gold), "in_raw20_dec": contains(raw20_dec, gold),
        "in_hay": contains(hay, gold),
        "raw20_chars": len(raw20_dec),
        "unc_ok": unc_ok, "unc_ok2": unc_ok2, "subset_ok": subset_ok,
    }


def main():
    global DATA
    DATA = load_dataset("s_cleaned")
    with Pool(24) as p:
        rows = p.map(one, range(len(DATA)), chunksize=2)
    with open(f"{R}/_verif_rankview.json", "w") as f:
        json.dump(rows, f, ensure_ascii=False)
    print("saved", len(rows))


if __name__ == "__main__":
    main()
