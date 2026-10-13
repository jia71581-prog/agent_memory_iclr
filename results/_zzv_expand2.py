"""Stress the session-expansion family: is the refuted alternative implemented at full strength?

Arms added:
  sbm25_N   top-N sessions ranked by BM25 over the CONCATENATED session text, full expansion
  win_K     top-K BM25 turns each expanded with +-1 neighbour turn (turn-neighbourhood)
  hyb_N     top-N sessions fully expanded + flat turns from elsewhere until the flat20 budget
Same metrics as _zzv_expand.py so the two files can be merged on qid.
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
    units, sess_units = [], defaultdict(list)
    for si, sess in enumerate(q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                units.append((si, t["role"], t["content"], bool(t.get("has_answer"))))
                sess_units[si].append(len(units) - 1)
    docs = [u[2] for u in units]
    order = bm25_rank(q["question"], docs)
    ha = {i for i, u in enumerate(units) if u[3]}
    gold = str(q["answer"])

    def metrics(sel):
        sel = list(dict.fromkeys(sel))
        ss = {units[i][0] for i in sel}
        selset = set(sel)
        return {"chars": sum(len(units[i][2]) for i in sel), "nturns": len(sel), "nsess": len(ss),
                "evcomplete": ev <= ss, "hacomplete": bool(ha) and ha <= selset,
                "hamicro": (len(ha & selset) / len(ha)) if ha else None,
                "goldin": contains(" ".join(units[i][2] for i in sel), gold)}

    row = {"qid": qid, "type": q["question_type"], "is_abs": qid.endswith("_abs"),
           "shape": classify_shape(q["question"])[0], "n_ev": len(ev)}

    # session-level BM25 ranking (whole-session retrieval done the standard way)
    sdocs = [" ".join(units[i][2] for i in sess_units[s]) for s in sorted(sess_units)]
    skeys = sorted(sess_units)
    sorder = bm25_rank(q["question"], sdocs)
    for N in (1, 2, 3, 6, 10):
        sel = [i for r in sorder[:N] for i in sess_units[skeys[r]]]
        for k, v in metrics(sel).items():
            row[f"sbm25{N}_{k}"] = v

    # turn-neighbourhood expansion: top-K turns plus +-1 neighbour inside the same session
    pos_in_sess = {}
    for s, idxs in sess_units.items():
        for j, i in enumerate(idxs):
            pos_in_sess[i] = (s, j)
    for K in (7, 10, 14, 20):
        sel = []
        for i in order[:K]:
            s, j = pos_in_sess[i]
            idxs = sess_units[s]
            for jj in (j - 1, j, j + 1):
                if 0 <= jj < len(idxs):
                    sel.append(idxs[jj])
        for k, v in metrics(sel).items():
            row[f"win{K}_{k}"] = v

    # hybrid: fully expand top-1 session, then fill with flat turns up to ~20.5k chars
    best_by_sess = defaultdict(list)
    for i in order:
        best_by_sess[units[i][0]].append(i)
    p = {i: r for r, i in enumerate(order)}
    sess_rank = sorted(best_by_sess, key=lambda s: p[best_by_sess[s][0]])
    for N in (1, 2):
        sel = [i for s in sess_rank[:N] for i in sess_units[s]]
        used = sum(len(units[i][2]) for i in sel)
        seen = set(sel)
        for i in order:
            if used >= 20500:
                break
            if i in seen:
                continue
            sel.append(i)
            seen.add(i)
            used += len(units[i][2])
        for k, v in metrics(sel).items():
            row[f"hyb{N}_{k}"] = v
    return row


if __name__ == "__main__":
    data = load_dataset("s_cleaned")
    with Pool(16) as p:
        rows = p.map(work, data, chunksize=4)
    json.dump(rows, open(f"{R}/_zzv_expand2.json", "w"), ensure_ascii=False)
    print("saved", len(rows))
