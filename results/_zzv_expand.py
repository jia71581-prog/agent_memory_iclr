"""Independent adversarial recomputation: whole-session expansion vs turn-level BM25.

No API. Rebuilds every arm from the raw dataset with the pipeline's own bm25_rank.
Arms:
  flat_K      top-K BM25 turns (K = 20/30/40/60/80)   [K=20 == deployed run_qc raw slice]
  cov20       session-coverage round robin, 20 turns   [design agent's 'coverage top-20']
  sessN       top-N sessions by best-turn rank, FULL expansion (N = 1/2/3/6/10)
Metrics (per question, all computed on the identical unit list):
  evcomplete  every answer_session_id touched by the arm      <- design agent's metric
  hacomplete  every has_answer turn present in the arm        <- apples-to-apples
  hamicro     fraction of has_answer turns present
  goldin      gold string contained in the arm text
  chars       sum of selected turn content lengths
"""
import json
import os
import re
import sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from run_baseline import bm25_rank
from v5_shape import classify_shape

R = "/root/agent_memory_iclr/results"
FLAT_KS = [20, 30, 40, 60, 80]
SESS_NS = [1, 2, 3, 6, 10]


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
    units = []  # (sidx, role, content, has_answer)
    for si, sess in enumerate(q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                units.append((si, t["role"], t["content"], bool(t.get("has_answer"))))
    docs = [u[2] for u in units]
    order = bm25_rank(q["question"], docs)
    ha = {i for i, u in enumerate(units) if u[3]}
    ha_asst = {i for i in ha if units[i][1] == "assistant"}
    gold = str(q["answer"])
    turns_by_sess = defaultdict(list)
    for i, u in enumerate(units):
        turns_by_sess[u[0]].append(i)

    def metrics(sel):
        sel = list(sel)
        ss = {units[i][0] for i in sel}
        txt = " ".join(units[i][2] for i in sel)
        selset = set(sel)
        return {
            "chars": sum(len(units[i][2]) for i in sel),
            "nturns": len(sel),
            "nsess": len(ss),
            "evcomplete": ev <= ss,
            "evhit": len(ev & ss),
            "hacomplete": bool(ha) and ha <= selset,
            "hamicro": (len(ha & selset) / len(ha)) if ha else None,
            "haany": bool(ha & selset),
            "haasst_complete": bool(ha_asst) and ha_asst <= selset,
            "goldin": contains(txt, gold),
        }

    row = {"qid": qid, "type": q["question_type"], "is_abs": qid.endswith("_abs"),
           "shape": classify_shape(q["question"])[0],
           "n_ev": len(ev), "n_ha": len(ha), "n_sess": len(q["haystack_sessions"]),
           "n_units": len(units), "gold": gold,
           "ev_turns_total": sum(len(turns_by_sess[s]) for s in ev),
           "ev_chars_total": sum(len(units[i][2]) for s in ev for i in turns_by_sess[s]),
           "hay_chars": sum(len(u[2]) for u in units)}

    # flat arms
    for K in FLAT_KS:
        sel = order[:K]
        for k, v in metrics(sel).items():
            row[f"flat{K}_{k}"] = v
        if K == 20:
            row["flat20_ev_turns_in"] = sum(1 for i in sel if units[i][0] in ev)

    # coverage round robin (same construction as _zz_verify_cov.py)
    best_by_sess = defaultdict(list)
    for i in order:
        best_by_sess[units[i][0]].append(i)
    pos = {i: r for r, i in enumerate(order)}
    sess_rank = sorted(best_by_sess, key=lambda s: pos[best_by_sess[s][0]])
    cov, ptr = [], 0
    while len(cov) < 20 and ptr < 2:
        for s in sess_rank:
            if len(cov) >= 20:
                break
            if len(best_by_sess[s]) > ptr:
                cov.append(best_by_sess[s][ptr])
        ptr += 1
    for k, v in metrics(cov).items():
        row[f"cov20_{k}"] = v

    # session expansion
    for N in SESS_NS:
        sel = [i for s in sess_rank[:N] for i in turns_by_sess[s]]
        for k, v in metrics(sel).items():
            row[f"sess{N}_{k}"] = v

    # matched-budget flat arm: grow K until chars >= sess3 chars
    tgt = row["sess3_chars"]
    acc, K = 0, 0
    while K < len(order) and acc < tgt:
        acc += len(units[order[K]][2])
        K += 1
    for k, v in metrics(order[:K]).items():
        row[f"flatmatch3_{k}"] = v
    row["flatmatch3_K"] = K

    tgt6 = row["sess6_chars"]
    acc, K = 0, 0
    while K < len(order) and acc < tgt6:
        acc += len(units[order[K]][2])
        K += 1
    for k, v in metrics(order[:K]).items():
        row[f"flatmatch6_{k}"] = v
    row["flatmatch6_K"] = K
    return row


if __name__ == "__main__":
    data = load_dataset("s_cleaned")
    with Pool(16) as p:
        rows = p.map(work, data, chunksize=4)
    json.dump(rows, open(f"{R}/_zzv_expand.json", "w"), ensure_ascii=False)
    print("saved", len(rows))
