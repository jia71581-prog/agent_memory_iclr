"""v5_temporal 离线检查工具:从 lifecycle 缓存拼 store(零 API),打印/检查时间脚手架。"""
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from lifecycle import assemble_chains, _session_text, SESSION_CACHE_DIR


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


def build_store_cached(q):
    """只用缓存构建 lifecycle store。返回 (store, n_missing_sessions)。"""
    facts, missing = [], 0
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        tri = cached_facts(sess)
        if tri is None:
            missing += 1
            continue
        for e, a, v in tri:
            facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
    return assemble_chains(facts), missing


_DATA = None


def data_by_id():
    global _DATA
    if _DATA is None:
        _DATA = {q["question_id"]: q for q in load_dataset("s_cleaned")}
    return _DATA


if __name__ == "__main__":
    from v5_temporal import render_timeline
    d = data_by_id()
    for qid in sys.argv[1:]:
        q = d[qid]
        store, miss = build_store_cached(q)
        txt = render_timeline(q, store, q["question"])
        print("=" * 110)
        print(qid, "| qdate", q["question_date"], "| cache-miss sessions:", miss)
        print("Q:", q["question"])
        print("GOLD:", q["answer"])
        print(txt)
        print(f"[size] {len(txt)} chars, {len(txt.split())} whitespace tokens")
