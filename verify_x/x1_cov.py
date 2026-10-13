"""Independent adversarial recomputation #1: session-coverage retriever claim.

Recomputes from raw dataset only (no API):
  - flat BM25 top-20 turns (exactly production run_qc.py)
  - session-coverage round-robin at the SAME 20-turn budget (v5_offline_val.py)
  - complete-evidence session recall (all answer_session_ids covered)
  - budget in chars AND bm25 tokens
  - regressions (flat_all True -> cov_all False)
  - overlap with the 62 non-abstention errors
"""
import json, os, re, sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from run_baseline import bm25_rank, tokenize
from v5_shape import classify_shape

R = "/root/agent_memory_iclr/results"
OUT = "/root/agent_memory_iclr/verify_x"
TOP_K = 20

judge = {}
for line in open(f"{R}/full500_qc.jsonl"):
    d = json.loads(line)
    if "judge" in d:
        judge[d["qid"]] = d["judge"]


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


STOP = {"the", "and", "for", "was", "were", "you", "your", "with", "that", "this",
        "from", "have", "has", "had", "are", "not"}


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


def cov_select(order, units, top_k=TOP_K, per_pass=2):
    best_by_sess = defaultdict(list)
    for i in order:
        best_by_sess[units[i][0]].append(i)
    pos = {i: r for r, i in enumerate(order)}
    sess_rank = sorted(best_by_sess, key=lambda s: pos[best_by_sess[s][0]])
    cov, ptr = [], 0
    while len(cov) < top_k and ptr < per_pass:
        for s in sess_rank:
            if len(cov) >= top_k:
                break
            if len(best_by_sess[s]) > ptr:
                cov.append(best_by_sess[s][ptr])
        ptr += 1
    return cov


def work(q):
    qid = q["question_id"]
    shape, _ = classify_shape(q["question"])
    sid2idx = {s: i for i, s in enumerate(q["haystack_session_ids"])}
    ev = {sid2idx[s] for s in q["answer_session_ids"] if s in sid2idx}
    units = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for t in sess:
            if t.get("content"):
                units.append((si, date, t["role"], t["content"], bool(t.get("has_answer"))))
    order = bm25_rank(q["question"], [u[3] for u in units])
    flat = order[:TOP_K]
    cov = cov_select(order, units)
    flat_sess = {units[i][0] for i in flat}
    cov_sess = {units[i][0] for i in cov}
    ev_turns = {i for i, u in enumerate(units) if u[0] in ev and u[4]}
    any_ha = len(ev_turns) > 0
    flat_txt = " ".join(units[i][3] for i in flat)
    cov_txt = " ".join(units[i][3] for i in cov)
    gold = str(q["answer"])
    return {
        "qid": qid, "shape": shape, "type": q["question_type"],
        "is_abs": qid.endswith("_abs"),
        "n_ev": len(ev), "n_sess": len(q["haystack_sessions"]), "n_units": len(units),
        "flat_all": ev <= flat_sess, "cov_all": ev <= cov_sess,
        "flat_any": bool(ev & flat_sess), "cov_any": bool(ev & cov_sess),
        "flat_hit": len(ev & flat_sess), "cov_hit": len(ev & cov_sess),
        "flat_chars": sum(len(units[i][3]) for i in flat),
        "cov_chars": sum(len(units[i][3]) for i in cov),
        "flat_toks": sum(len(tokenize(units[i][3])) for i in flat),
        "cov_toks": sum(len(tokenize(units[i][3])) for i in cov),
        "flat_nsess": len(flat_sess), "cov_nsess": len(cov_sess),
        "n_ha_turns": len(ev_turns), "any_ha": any_ha,
        "flat_ha_in": len(ev_turns & set(flat)), "cov_ha_in": len(ev_turns & set(cov)),
        "flat_ha_all": bool(any_ha and ev_turns <= set(flat)),
        "cov_ha_all": bool(any_ha and ev_turns <= set(cov)),
        "flat_gold": contains(flat_txt, gold), "cov_gold": contains(cov_txt, gold),
        "gold": gold, "question": q["question"], "qdate": q["question_date"],
        "ev_dates": [q["haystack_dates"][i] for i in sorted(ev)],
    }


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    data = load_dataset("s_cleaned")
    with Pool(16) as p:
        rows = p.map(work, data, chunksize=2)
    for r in rows:
        r["judge"] = judge.get(r["qid"])
    json.dump(rows, open(f"{OUT}/x1_cov.json", "w"), ensure_ascii=False)
    print("saved", len(rows))
