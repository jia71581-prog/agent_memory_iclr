"""Matched-budget control for the prov_only arm + BM25 allEv as a function of k.

prov_only delivers only 17.3 turns (it runs out of scored provenance sessions), so comparing it
to a 20-turn BM25 baseline is an UNMATCHED comparison. Here we compute, per question, the BM25
allEv at every k = 1..40 so a per-question turn-matched control can be built.
"""
import hashlib
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from lifecycle import assemble_chains, _session_text, SESSION_CACHE_DIR
from query_classifier import classify_query
import v5_retrieval as V5

R = "/root/agent_memory_iclr/results"
DATA = None
KMAX = 40


def one(qi):
    q = DATA[qi]
    units = V5.build_units(q)
    flags = []
    for sess in q["haystack_sessions"]:
        for t in sess:
            if t.get("content"):
                flags.append(bool(t.get("has_answer")))
    ev = {i for i, f in enumerate(flags) if f}
    bm = V5.bm25_scores(q["question"], [u[3] for u in units])
    order = sorted(range(len(units)), key=lambda i: (-bm[i], i))
    hits, sess_seen, allev_k, nsess_k, chars_k = 0, set(), [], [], []
    tot = 0
    for k in range(1, KMAX + 1):
        if k <= len(order):
            i = order[k - 1]
            if i in ev:
                hits += 1
            sess_seen.add(units[i][0])
            tot += len(f"[{units[i][1]}] [{units[i][2]}] {units[i][3]}") + 1
        allev_k.append(hits == len(ev))
        nsess_k.append(len(sess_seen))
        chars_k.append(tot)
    return {"qid": q["question_id"], "is_abs": q["question_id"].endswith("_abs"),
            "type": q["question_type"], "n_ev": len(ev), "n_units": len(units),
            "allev_k": allev_k, "nsess_k": nsess_k, "chars_k": chars_k}


def main():
    global DATA
    DATA = load_dataset("s_cleaned")
    with Pool(30) as p:
        rows = p.map(one, range(len(DATA)), chunksize=4)
    with open(f"{R}/_verify_bm25_k.json", "w") as f:
        json.dump(rows, f)
    print("saved", len(rows))


if __name__ == "__main__":
    main()
