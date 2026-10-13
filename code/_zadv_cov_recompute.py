"""Independent adversarial recomputation of the flat-vs-coverage complete-evidence-set claim.

No API. Recomputes from raw dataset:
  - flat top-20 turn retrieval (exactly as production run_qc.py does)
  - session-coverage round-robin at the same 20-turn budget (as v5_offline_val.py describes)
  - session-level complete recall (all answer_session_ids covered)
  - TURN-level evidence recall inside the evidence sessions
  - gold-string containment in the retrieved raw turns (the answer-bearing test)
  - char/token budget of both arms
"""
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from run_baseline import bm25_rank, tokenize
from v5_shape import classify_shape

R = "/root/agent_memory_iclr/results"
TOP_K = 20

judge = {}
for line in open(f"{R}/full500_qc.jsonl"):
    r = json.loads(line)
    if "judge" in r:
        judge[r["qid"]] = r["judge"]


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def contains(hay, needle):
    h, n = norm(hay), norm(needle)
    if not n:
        return False
    if n in h:
        return True
    toks = [t for t in n.split() if len(t) > 2 and t not in
            ("the", "and", "for", "was", "were", "you", "your")]
    if not toks:
        return False
    return all(t in h for t in toks)


def cov_select(order, units, top_k=TOP_K, per_pass=2):
    """v5_offline_val.py's round-robin, reimplemented."""
    best_by_sess = defaultdict(list)
    for i in order:
        best_by_sess[units[i][0]].append(i)
    sess_rank = sorted(best_by_sess, key=lambda s: order.index(best_by_sess[s][0]))
    cov, ptr = [], 0
    while len(cov) < top_k and ptr < per_pass:
        for s in sess_rank:
            if len(cov) >= top_k:
                break
            if len(best_by_sess[s]) > ptr:
                cov.append(best_by_sess[s][ptr])
        ptr += 1
    return cov


data = load_dataset("s_cleaned")
out = []
for qi, q in enumerate(data):
    qid = q["question_id"]
    shape, _ = classify_shape(q["question"])
    sid2idx = {s: i for i, s in enumerate(q["haystack_session_ids"])}
    ev = {sid2idx[s] for s in q["answer_session_ids"] if s in sid2idx}
    units = []  # (sidx, date, role, content)
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for t in sess:
            if t.get("content"):
                units.append((si, date, t["role"], t["content"]))
    order = bm25_rank(q["question"], [u[3] for u in units])

    flat = order[:TOP_K]
    cov = cov_select(order, units)
    flat_sess = {units[i][0] for i in flat}
    cov_sess = {units[i][0] for i in cov}

    # turn-level: which turns inside evidence sessions actually got retrieved
    ev_turns = {i for i, u in enumerate(units) if u[0] in ev}
    # gold-string containment in the retrieved raw turn text only
    flat_txt = " ".join(units[i][3] for i in flat)
    cov_txt = " ".join(units[i][3] for i in cov)
    gold = str(q["answer"])

    out.append({
        "qid": qid, "shape": shape, "type": q["question_type"],
        "is_abs": qid.endswith("_abs"),
        "judge": judge.get(qid),
        "n_ev": len(ev), "n_sess": len(q["haystack_sessions"]), "n_units": len(units),
        "flat_all": ev <= flat_sess, "cov_all": ev <= cov_sess,
        "flat_any": bool(ev & flat_sess), "cov_any": bool(ev & cov_sess),
        "flat_hit": len(ev & flat_sess), "cov_hit": len(ev & cov_sess),
        "flat_sess_superset_ok": flat_sess <= cov_sess,
        "n_flat_turns": len(flat), "n_cov_turns": len(cov),
        "flat_chars": sum(len(units[i][3]) for i in flat),
        "cov_chars": sum(len(units[i][3]) for i in cov),
        "flat_toks": sum(len(tokenize(units[i][3])) for i in flat),
        "cov_toks": sum(len(tokenize(units[i][3])) for i in cov),
        "flat_ev_turns_in": len(ev_turns & set(flat)),
        "cov_ev_turns_in": len(ev_turns & set(cov)),
        "n_ev_turns": len(ev_turns),
        "flat_gold": contains(flat_txt, gold), "cov_gold": contains(cov_txt, gold),
        "gold": gold, "question": q["question"],
        "n_distinct_sess_flat": len(flat_sess), "n_distinct_sess_cov": len(cov_sess),
    })
    if (qi + 1) % 100 == 0:
        print("  ...", qi + 1, flush=True)

json.dump(out, open(f"{R}/_zadv_cov.json", "w"), ensure_ascii=False)
print("saved", len(out))
