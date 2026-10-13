"""Offline reconstruction of the exact context each arm saw (no API calls).

Retrieval (BM25) is deterministic pure python; the lifecycle view is rebuilt from the
on-disk extraction cache. This lets us separate "evidence never reached the context"
(retrieval miss) from "evidence was in context but the answer was still wrong"
(reasoning failure: stale-fact / conflict-confusion / aggregation).
"""
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
import lifecycle as LC
from run_baseline import bm25_rank
from query_classifier import classify_query

CACHE_MISS = [0]


def extract_cached(session_turns):
    text = LC._session_text(session_turns)
    if not text:
        return []
    key = hashlib.md5(text.encode()).hexdigest()[:16]
    p = os.path.join(LC.SESSION_CACHE_DIR, f"{key}.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    CACHE_MISS[0] += 1
    return []


def build_store(q):
    all_facts = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for ent, attr, val in extract_cached(sess):
            all_facts.append({"entity": ent, "attribute": attr, "value": val,
                              "date": date, "sidx": si})
    return LC.assemble_chains(all_facts)


def units_of(q):
    units = []
    for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                units.append((date, t["role"], t["content"]))
    return units


def qc_context(q, ablate=None):
    """Reproduce run_qc.answer_question's context exactly."""
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    if ablate == "no_qc":
        mode = "historical"
    view_txt = ""
    if ablate != "no_lifecycle":
        view_txt = LC.render_view(build_store(q), mode)
        if len(view_txt) > 20000:
            kept, total = [], 0
            for line in view_txt.splitlines():
                if total + len(line) + 1 > 20000:
                    break
                kept.append(line)
                total += len(line) + 1
            view_txt = "\n".join(kept)
    raw_idx = []
    if ablate != "view_only":
        units = units_of(q)
        raw_idx = sorted(bm25_rank(q["question"], [u[2] for u in units])[:20])
    return label, mode, view_txt, raw_idx


def full_context(q, top_k=30):
    """Reproduce run_baseline 'full' arm retrieval (BM25 top-30 over all turns)."""
    units = units_of(q)
    return sorted(bm25_rank(q["question"], [u[2] for u in units])[:top_k])


def evidence_turns(q):
    """The has_answer=True turns: the ground-truth supporting evidence."""
    sid2sess = dict(zip(q["haystack_session_ids"], q["haystack_sessions"]))
    out = []
    for sid in q["answer_session_ids"]:
        for t in sid2sess.get(sid, []):
            if t.get("has_answer") and t.get("content"):
                out.append(t["content"])
    return out


def coverage(q, kept_idx):
    """Fraction of gold evidence turns present in the retrieved raw context."""
    units = units_of(q)
    got = {units[i][2] for i in kept_idx}
    ev = evidence_turns(q)
    if not ev:
        return None, 0, 0
    hit = sum(1 for e in ev if e in got)
    return hit / len(ev), hit, len(ev)
