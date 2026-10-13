"""ADVERSARIAL VERIFIER #2, independent prep. Builds per-question facts straight from the
dataset + md5 session cache (no reuse of _v5_facts.pkl / _verify_facts.pkl). No API."""
import hashlib
import json
import os
import pickle
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from lifecycle import _session_text, SESSION_CACHE_DIR

OUT = "/root/agent_memory_iclr/results/_adv2_facts.pkl"


def main():
    data = load_dataset("s_cleaned")
    out = []
    miss_sess = 0
    tot_sess = 0
    for qi, q in enumerate(data):
        facts = []
        missing = False
        try:
            gold_ids = set(eval(q["answer_session_ids"])) if isinstance(q["answer_session_ids"], str) \
                else set(q["answer_session_ids"])
        except Exception:
            gold_ids = set()
        gold_sidx = [i for i, sid in enumerate(q["haystack_session_ids"]) if sid in gold_ids]
        for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
            text = _session_text(sess)
            tot_sess += 1
            if not text:
                continue
            key = hashlib.md5(text.encode()).hexdigest()[:16]
            p = os.path.join(SESSION_CACHE_DIR, f"{key}.json")
            if not os.path.exists(p):
                missing = True
                miss_sess += 1
                continue
            with open(p) as f:
                tri = json.load(f)
            for e, a, v in tri:
                facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
        out.append({"qid": q["question_id"], "type": q["question_type"],
                    "question": q["question"], "answer": q["answer"],
                    "question_date": q["question_date"], "facts": facts,
                    "cache_missing": missing, "gold_sidx": gold_sidx,
                    "n_sessions": len(q["haystack_sessions"]),
                    "gold_sess_texts": [q["haystack_sessions"][i] for i in gold_sidx]})
        if (qi + 1) % 100 == 0:
            print(f"{qi+1}/{len(data)}", flush=True)
    with open(OUT, "wb") as f:
        pickle.dump(out, f)
    print("saved", OUT, "q", len(out), "questions_with_missing_cache",
          sum(1 for x in out if x["cache_missing"]), "missing_sessions", miss_sess, "/", tot_sess)


if __name__ == "__main__":
    main()
