"""ADVERSARIAL recompute #2b: match-rule sensitivity of the 45/42/36 trace."""
import json, os, hashlib, pickle, sys, re, itertools
sys.path.insert(0, "/root/agent_memory_iclr/code")
import lifecycle as LC

data = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
views = pickle.load(open("/root/agent_memory_iclr/verify/_views.pkl", "rb"))
ssp = [q for q in data if q["question_type"] == "single-session-preference"]

def sess_facts(q):
    out = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        text = LC._session_text(sess)
        if not text:
            continue
        key = hashlib.md5(text.encode()).hexdigest()[:16]
        p = os.path.join(LC.SESSION_CACHE_DIR, f"{key}.json")
        if os.path.exists(p):
            for e, a, v in json.load(open(p)):
                out.append((si, e, a, v))
    return out

def nrm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

MATCH = {
    "lower-substring": lambda h, n: n.lower() in h.lower(),
    "norm-substring": lambda h, n: nrm(n) in nrm(h) if nrm(n) else False,
}

for (mn_ev, mf_ev), (mn_v, mf_v), dedup in itertools.product(MATCH.items(), MATCH.items(), [False, True]):
    n = f = c = 0
    for q in ssp:
        qid = q["question_id"]
        gs = {i for i, sid in enumerate(q["haystack_session_ids"]) if sid in set(q["answer_session_ids"])}
        sid2 = dict(zip(q["haystack_session_ids"], q["haystack_sessions"]))
        ev = " ".join(t["content"] for sid in q["answer_session_ids"] for t in sid2.get(sid, [])
                      if t.get("has_answer") and t.get("content"))
        vals = [v for si, e, a, v in sess_facts(q) if si in gs and mf_ev(ev, v)]
        if dedup:
            seen, out = set(), []
            for v in vals:
                if nrm(v) not in seen:
                    seen.add(nrm(v)); out.append(v)
            vals = out
        V = views[qid]
        n += len(vals)
        f += sum(1 for v in vals if mf_v(V["full"], v))
        c += sum(1 for v in vals if mf_v(V["cap"], v))
    print(f"ev-match={mn_ev:16s} view-match={mn_v:16s} dedup={dedup!s:5s}  n={n:3d} full={f:3d} cap={c:3d}")

# also: restrict to the gold evidence TURN (has_answer) only, exclude non-user entity
print()
for min_len in (0, 10, 15, 20):
    n = f = c = 0
    for q in ssp:
        qid = q["question_id"]
        gs = {i for i, sid in enumerate(q["haystack_session_ids"]) if sid in set(q["answer_session_ids"])}
        sid2 = dict(zip(q["haystack_session_ids"], q["haystack_sessions"]))
        ev = " ".join(t["content"] for sid in q["answer_session_ids"] for t in sid2.get(sid, [])
                      if t.get("has_answer") and t.get("content") and t["role"] == "user")
        vals = [v for si, e, a, v in sess_facts(q) if si in gs and len(v) >= min_len and nrm(v) in nrm(ev)]
        V = views[qid]
        n += len(vals); f += sum(1 for v in vals if nrm(v) in nrm(V["full"]))
        c += sum(1 for v in vals if nrm(v) in nrm(V["cap"]))
    print(f"user-evidence-turn only, min_len={min_len:2d}: n={n:3d} full={f:3d} cap={c:3d}")
