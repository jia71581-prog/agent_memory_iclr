"""ADVERSARIAL VERIFIER step 1: rebuild per-question facts from dataset + session cache,
INDEPENDENT of results/_v5_facts.pkl. Then cross-check against that pickle."""
import hashlib
import json
import os
import pickle
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")

DATA = "/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"
CACHE = "/root/agent_memory_iclr/results/lifecycle_cache/sessions_nt"
OUT = "/root/agent_memory_iclr/results/_verify_facts.pkl"


def session_text(turns):
    text = "\n".join(f"[{t['role']}] {t['content']}" for t in turns if t.get("content"))
    return text[:24000]


def main():
    with open(DATA) as f:
        data = json.load(f)
    print("questions", len(data), flush=True)
    out = []
    miss_sessions = 0
    tot_sessions = 0
    for qi, q in enumerate(data):
        facts = []
        missing = False
        gold_sidx = []
        for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
            tot_sessions += 1
            t = session_text(sess)
            if not t:
                continue
            key = hashlib.md5(t.encode()).hexdigest()[:16]
            p = os.path.join(CACHE, key + ".json")
            if not os.path.exists(p):
                missing = True
                miss_sessions += 1
                continue
            with open(p) as fh:
                tri = json.load(fh)
            for e, a, v in tri:
                facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
        sids = q.get("answer_session_ids") or []
        hids = [s for s in q.get("haystack_session_ids", [])]
        for si, sid in enumerate(hids):
            if sid in sids or any(str(sid).endswith(str(x)) or str(x).endswith(str(sid)) for x in sids):
                gold_sidx.append(si)
        out.append({"qid": q["question_id"], "type": q["question_type"],
                    "question": q["question"], "answer": q["answer"],
                    "facts": facts, "cache_missing": missing,
                    "n_sessions": len(q["haystack_sessions"]),
                    "gold_sidx": gold_sidx,
                    "answer_session_ids": sids})
        if (qi + 1) % 100 == 0:
            print(qi + 1, flush=True)
    with open(OUT, "wb") as f:
        pickle.dump(out, f)
    print("saved", OUT)
    print("sessions total", tot_sessions, "missing-cache sessions", miss_sessions)
    print("questions with any missing cache", sum(1 for x in out if x["cache_missing"]))
    print("total facts", sum(len(x["facts"]) for x in out))


if __name__ == "__main__":
    main()
