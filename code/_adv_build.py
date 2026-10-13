"""Adversarial verifier: offline store builder (cache-only, never calls the API)."""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifecycle import _session_text, assemble_chains, SESSION_CACHE_DIR

MISS = {"n": 0}


def cached_facts(session_turns):
    text = _session_text(session_turns)
    if not text:
        return []
    key = hashlib.md5(text.encode()).hexdigest()[:16]
    p = os.path.join(SESSION_CACHE_DIR, f"{key}.json")
    if not os.path.exists(p):
        MISS["n"] += 1
        return None
    with open(p) as f:
        return json.load(f)


def build_store_offline(q):
    """Returns (store, n_missing_sessions)."""
    all_facts, miss = [], 0
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        tr = cached_facts(sess)
        if tr is None:
            miss += 1
            continue
        for t in tr:
            if len(t) != 3:
                continue
            ent, attr, val = t
            all_facts.append({"entity": ent, "attribute": attr, "value": val,
                              "date": date, "sidx": si})
    return assemble_chains(all_facts), miss
