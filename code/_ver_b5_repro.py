"""Independent recomputation of the B5 session-augmented role-quota retrieval claim."""
import json, os, sys, random, statistics as st
sys.path.insert(0, "/root/agent_memory_iclr/code")
from run_baseline import bm25_rank

DATA = "/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"
OUT = "/root/agent_memory_iclr/results/_verify/b5_repro.json"

def units_of(q):
    out = []
    for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                out.append((date, t["role"], t["content"]))
    return out

def session_docs(q, user_only=True):
    docs = []
    for sess in q["haystack_sessions"]:
        docs.append(" ".join(t["content"] for t in sess
                             if t.get("content") and (t["role"] == "user" or not user_only)))
    return docs

data = json.load(open(DATA))
print("dataset n =", len(data))
rows = []
rng = random.Random(20260917)
for qi, q in enumerate(data):
    if q["question_id"].endswith("_abs"):
        continue
    units = units_of(q)
    docs = [u[2] for u in units]
    order = bm25_rank(q["question"], docs)
    rank = {i: r for r, i in enumerate(order)}
    # per-session positions in units order
    pos, per_sess = 0, {}
    for si, sess in enumerate(q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                per_sess.setdefault(si, []).append((pos, t["role"]))
                pos += 1
    sdocs = session_docs(q, user_only=True)
    sorder = bm25_rank(q["question"], sdocs)
    # gold evidence
    sid2 = dict(zip(q["haystack_session_ids"], q["haystack_sessions"]))
    ev, ev_roles = [], []
    for sid in q["answer_session_ids"]:
        for t in sid2.get(sid, []):
            if t.get("has_answer") and t.get("content"):
                ev.append(t["content"]); ev_roles.append(t["role"])
    gold_sess = {i for i, s in enumerate(q["haystack_session_ids"]) if s in set(q["answer_session_ids"])}

    uord = [i for i in order if units[i][1] == "user"]
    aord = [i for i in order if units[i][1] == "assistant"]

    def sess_user(sis):
        s = set()
        for si in sis:
            for i, role in per_sess.get(si, []):
                if role == "user":
                    s.add(i)
        return s

    arms = {}
    arms["base20"] = set(order[:20])
    arms["base30"] = set(order[:30])
    arms["base50"] = set(order[:50])
    for ku in (14, 18, 20, 24, 28, 34, 40, 50, 60):
        arms[f"q{ku}u6a"] = set(uord[:ku]) | set(aord[:6])
    for ku in (14, 20, 26, 34, 44):
        arms[f"u{ku}only"] = set(uord[:ku])
    q14 = set(uord[:14]) | set(aord[:6])
    for ns in (1, 2, 3):
        arms[f"s{ns}"] = q14 | sess_user(sorder[:ns])
    # nulls / alternatives
    nsess = len(q["haystack_sessions"])
    randsess = rng.sample(range(nsess), min(1, nsess))
    arms["s1_random"] = q14 | sess_user(randsess)
    randsess2 = rng.sample(range(nsess), min(2, nsess))
    arms["s2_random"] = q14 | sess_user(randsess2)
    # session of the top-1 user turn (no session-level BM25)
    top_user_sess = None
    for si, lst in per_sess.items():
        if uord and any(i == uord[0] for i, r in lst):
            top_user_sess = si
    arms["s1_topturn"] = q14 | sess_user([top_user_sess] if top_user_sess is not None else [])
    # oracle: gold session user turns
    arms["s1_oracle"] = q14 | sess_user(sorted(gold_sess))

    # duplicate-content diagnosis: does a gold ev string appear in a non-gold turn?
    dup = 0
    ev_set = set(ev)
    goldpos = set()
    for si in gold_sess:
        for i, role in per_sess.get(si, []):
            goldpos.add(i)
    for e in ev_set:
        occ = [i for i, d in enumerate(docs) if d == e]
        if any(i not in goldpos for i in occ):
            dup += 1

    row = {"qid": q["question_id"], "type": q["question_type"], "n_units": len(units),
           "n_sess": nsess, "ev_n": len(ev), "ev_roles": ev_roles,
           "gold_sess_rank": (min(sorder.index(i) for i in gold_sess) if gold_sess else None),
           "dup_ev_strings": dup, "arms": {}}
    for name, idx in arms.items():
        got = {docs[i] for i in idx}
        row["arms"][name] = {
            "hits": sum(1 for e in ev if e in got),
            "chars": sum(len(docs[i]) for i in idx),
            "nturn": len(idx),
            "achars": sum(len(docs[i]) for i in idx if units[i][1] == "assistant"),
        }
    rows.append(row)
    if (qi + 1) % 50 == 0:
        print("  ...", qi + 1, flush=True)

json.dump(rows, open(OUT, "w"))
print("wrote", OUT, len(rows))
