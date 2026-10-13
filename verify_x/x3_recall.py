"""Independent recomputation #3: RECALL.assistant single-session verbatim view risk + size."""
import json, os, re, sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from run_baseline import bm25_rank
from v5_shape import classify_shape

R = "/root/agent_memory_iclr/results"
OUT = "/root/agent_memory_iclr/verify_x"
TOP_K = 20
STOP = {"the", "and", "for", "was", "were", "you", "your", "with", "that", "this",
        "from", "have", "has", "had", "are", "not"}


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def contains(hay, needle):
    h, n = norm(hay), norm(needle)
    if not n:
        return False
    if n in h:
        return True
    toks = [t for t in n.split() if len(t) > 2 and t not in STOP]
    if not toks:
        return False
    return all(t in h for t in toks)


def work(q):
    shape, _ = classify_shape(q["question"])
    if shape != "RECALL.assistant":
        return None
    sid2idx = {s: i for i, s in enumerate(q["haystack_session_ids"])}
    ev = {sid2idx[s] for s in q["answer_session_ids"] if s in sid2idx}
    units, sidx = [], []
    for si, sess in enumerate(q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                units.append(t)
                sidx.append(si)
    order = bm25_rank(q["question"], [u["content"] for u in units])
    flat = order[:TOP_K]
    # session picking strategies
    top1_sess = sidx[order[0]]
    # session score = sum of its turn ranks in top-20 (votes)
    votes = defaultdict(int)
    for r, i in enumerate(flat):
        votes[sidx[i]] += TOP_K - r
    vote_sess = max(votes, key=lambda s: votes[s])
    # assistant-only session-level bm25
    sess_assist_txt = [" ".join(t.get("content", "") for t in s if t.get("role") == "assistant")
                       for s in q["haystack_sessions"]]
    sa_order = bm25_rank(q["question"], sess_assist_txt)
    sa_sess = sa_order[0]

    def sess_chars(si, role=None):
        return sum(len(t.get("content", "")) for t in q["haystack_sessions"][si]
                   if role is None or t.get("role") == role)

    gold = str(q["answer"])
    return {
        "qid": q["question_id"], "type": q["question_type"], "n_ev": len(ev),
        "gold": gold, "question": q["question"],
        "top1_ok": top1_sess in ev, "vote_ok": vote_sess in ev, "sa_ok": sa_sess in ev,
        "flat_chars": sum(len(units[i]["content"]) for i in flat),
        "one_sess_all_chars": sess_chars(vote_sess),
        "one_sess_assist_chars": sess_chars(vote_sess, "assistant"),
        "gold_sess_assist_chars": sum(sess_chars(i, "assistant") for i in ev),
        "gold_sess_all_chars": sum(sess_chars(i) for i in ev),
        "flat_gold": contains(" ".join(units[i]["content"] for i in flat), gold),
        "one_sess_gold": contains(" ".join(t.get("content", "") for t in q["haystack_sessions"][vote_sess]), gold),
        "one_sess_assist_gold": contains(" ".join(t.get("content", "") for t in q["haystack_sessions"][vote_sess]
                                                  if t.get("role") == "assistant"), gold),
        "n_sess": len(q["haystack_sessions"]),
    }


if __name__ == "__main__":
    data = load_dataset("s_cleaned")
    with Pool(16) as p:
        rows = [r for r in p.map(work, data, chunksize=2) if r]
    judge = {}
    for line in open(f"{R}/full500_qc.jsonl"):
        d = json.loads(line)
        if "judge" in d:
            judge[d["qid"]] = d["judge"]
    for r in rows:
        r["judge"] = judge.get(r["qid"])
    json.dump(rows, open(f"{OUT}/x3_recall.json", "w"), ensure_ascii=False)
    print("saved", len(rows))
