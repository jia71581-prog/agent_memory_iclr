"""OFFLINE prep for v5 analysis: dump per-question facts (from the session cache) + metadata
into one pickle so downstream measurement scripts never re-load the 278MB dataset. No API."""
import hashlib
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from lifecycle import _session_text, SESSION_CACHE_DIR
from run_baseline import bm25_rank

OUT = "/root/agent_memory_iclr/results/_v5_facts.pkl"


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


def main():
    data = load_dataset("s_cleaned")
    out = []
    n_missing = 0
    for qi, q in enumerate(data):
        facts, missing = [], False
        for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
            tri = cached_facts(sess)
            if tri is None:
                missing = True
                continue
            for e, a, v in tri:
                facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
        n_missing += missing
        # BM25 top-20 raw turns (same as run_qc) so we can compare evidence sets offline
        units = []
        for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
            for t in sess:
                if t.get("content"):
                    units.append((date, t["role"], t["content"]))
        order = sorted(bm25_rank(q["question"], [u[2] for u in units])[:20])
        raw20 = [(units[i][0], units[i][1], units[i][2]) for i in order]
        out.append({
            "qid": q["question_id"], "type": q["question_type"],
            "question": q["question"], "answer": q["answer"],
            "question_date": q["question_date"],
            "facts": facts, "cache_missing": missing,
            "n_sessions": len(q["haystack_sessions"]),
            "raw20": raw20,
            "hay_chars": sum(len(u[2]) for u in units),
        })
        if (qi + 1) % 50 == 0:
            print(f"{qi+1}/{len(data)}", flush=True)
    with open(OUT, "wb") as f:
        pickle.dump(out, f)
    print("saved", OUT, "questions", len(out), "cache_missing_questions", n_missing)


if __name__ == "__main__":
    main()
