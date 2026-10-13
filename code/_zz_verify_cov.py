"""Independent adversarial recomputation of the flat-vs-coverage complete-evidence-set claim.
No API. Recomputes from the raw dataset with the pipeline's own bm25_rank.
"""
import json
import os
import re
import sys
from collections import defaultdict, Counter
from multiprocessing import Pool

sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from run_baseline import bm25_rank
from v5_shape import classify_shape

R = "/root/agent_memory_iclr/results"
TOP_K = 20


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def contains(hay, needle):
    h, n = norm(hay), norm(needle)
    if not n:
        return False
    if n in h:
        return True
    toks = [t for t in n.split() if len(t) > 2 and t not in ("the", "and", "for", "was", "were", "you", "your")]
    if not toks:
        return False
    return all(t in h for t in toks)


def work(q):
    qid = q["question_id"]
    sid2idx = {s: i for i, s in enumerate(q["haystack_session_ids"])}
    ev = {sid2idx[s] for s in q["answer_session_ids"] if s in sid2idx}
    units = []  # (sidx, date, role, content)
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for t in sess:
            if t.get("content"):
                units.append((si, date, t["role"], t["content"]))
    docs = [u[3] for u in units]
    order = bm25_rank(q["question"], docs)

    flat = order[:TOP_K]
    flat_sess = {units[i][0] for i in flat}

    best_by_sess = defaultdict(list)
    for i in order:
        best_by_sess[units[i][0]].append(i)
    sess_rank = sorted(best_by_sess, key=lambda s: order.index(best_by_sess[s][0]))
    cov, si_ptr = [], 0
    while len(cov) < TOP_K and si_ptr < 2:
        for s in sess_rank:
            if len(cov) >= TOP_K:
                break
            if len(best_by_sess[s]) > si_ptr:
                cov.append(best_by_sess[s][si_ptr])
        si_ptr += 1
    cov_sess = {units[i][0] for i in cov}

    gold = str(q["answer"])
    flat_txt = " ".join(units[i][3] for i in flat)
    cov_txt = " ".join(units[i][3] for i in cov)
    hay_txt = " ".join(u[3] for u in units)
    # gold-bearing turns anywhere in the haystack (turn-level literal ceiling)
    gold_turns = {i for i, u in enumerate(units) if contains(u[3], gold)}
    # gold-bearing turns restricted to evidence sessions
    gold_turns_ev = {i for i in gold_turns if units[i][0] in ev}

    shape, fired = classify_shape(q["question"])
    return {
        "qid": qid, "type": q["question_type"], "shape": shape,
        "is_abs": qid.endswith("_abs"),
        "n_ev": len(ev), "n_sess": len(q["haystack_sessions"]), "n_units": len(units),
        "flat_all": ev <= flat_sess, "cov_all": ev <= cov_sess,
        "flat_hit": len(ev & flat_sess), "cov_hit": len(ev & cov_sess),
        "flat_subset_cov": flat_sess <= cov_sess,
        "n_flat_sess": len(flat_sess), "n_cov_sess": len(cov_sess),
        "flat_chars": sum(len(units[i][3]) for i in flat),
        "cov_chars": sum(len(units[i][3]) for i in cov),
        # depth inside evidence sessions
        "flat_ev_turns": sum(1 for i in flat if units[i][0] in ev),
        "cov_ev_turns": sum(1 for i in cov if units[i][0] in ev),
        # gold-string level
        "gold_in_flat": contains(flat_txt, gold), "gold_in_cov": contains(cov_txt, gold),
        "gold_in_hay": contains(hay_txt, gold),
        "n_gold_turns": len(gold_turns), "n_gold_turns_ev": len(gold_turns_ev),
        "goldturn_in_flat": bool(gold_turns & set(flat)),
        "goldturn_in_cov": bool(gold_turns & set(cov)),
        "goldturn_ev_in_flat": bool(gold_turns_ev & set(flat)),
        "goldturn_ev_in_cov": bool(gold_turns_ev & set(cov)),
        "gold": gold, "question": q["question"],
    }


if __name__ == "__main__":
    data = load_dataset("s_cleaned")
    with Pool(16) as p:
        rows = p.map(work, data, chunksize=4)
    json.dump(rows, open(f"{R}/_zz_verify_cov.json", "w"), ensure_ascii=False)
    print("saved", len(rows))
