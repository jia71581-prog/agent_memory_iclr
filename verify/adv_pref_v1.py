"""ADVERSARIAL recompute #2: the '45 SSP preference values' trace under every plausible
definition. Offline."""
import json, os, hashlib, pickle, sys, re
sys.path.insert(0, "/root/agent_memory_iclr/code")
import lifecycle as LC

DATA = "/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"
data = json.load(open(DATA))
views = pickle.load(open("/root/agent_memory_iclr/verify/_views.pkl", "rb"))
ssp = [q for q in data if q["question_type"] == "single-session-preference"]

def sess_facts(q):
    """[(sidx, entity, attribute, value)] from the on-disk extraction cache"""
    out = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        text = LC._session_text(sess)
        if not text:
            continue
        key = hashlib.md5(text.encode()).hexdigest()[:16]
        p = os.path.join(LC.SESSION_CACHE_DIR, f"{key}.json")
        if not os.path.exists(p):
            continue
        for e, a, v in json.load(open(p)):
            out.append((si, e, a, v))
    return out

def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def contains(hay, needle):
    return norm(needle) in norm(hay) if norm(needle) else False

PREF_ATTR = re.compile(r"prefer|favou?rite|like|love|enjoy|hobby|hobbies|interest|dislike|hate|"
                       r"style|taste|want|wish|goal|habit", re.I)

defs = {}
for q in ssp:
    qid = q["question_id"]
    gs = {i for i, sid in enumerate(q["haystack_session_ids"]) if sid in set(q["answer_session_ids"])}
    sid2 = dict(zip(q["haystack_session_ids"], q["haystack_sessions"]))
    ev_turns = [t["content"] for sid in q["answer_session_ids"] for t in sid2.get(sid, [])
                if t.get("has_answer") and t.get("content")]
    ev_text = " ".join(ev_turns)
    facts = [f for f in sess_facts(q) if f[0] in gs]
    V = views[qid]
    for si, e, a, v in facts:
        line_full = contains(V["full"], v)
        line_cap = contains(V["cap"], v)
        d = {
            "all_gold_session_values": True,
            "user_entity_only": norm(e) == "user",
            "pref_attr": bool(PREF_ATTR.search(a)),
            "value_in_gold_evidence_turn": contains(ev_text, v),
            "user_and_in_evidence": norm(e) == "user" and contains(ev_text, v),
        }
        for k, ok in d.items():
            if ok:
                defs.setdefault(k, []).append((qid, e, a, v, line_full, line_cap))

print(f"{'definition':34s} {'n':>4s} {'in_full':>8s} {'in_cap':>7s}  full%  cap%")
for k, rows in defs.items():
    n = len(rows)
    f = sum(1 for r in rows if r[4]); c = sum(1 for r in rows if r[5])
    print(f"{k:34s} {n:4d} {f:8d} {c:7d}  {f/n:.3f} {c/n:.3f}")

# one-value-per-gold-evidence-turn variant: does ANY extracted value from that turn render?
per_turn = []
for q in ssp:
    qid = q["question_id"]
    gs = {i for i, sid in enumerate(q["haystack_session_ids"]) if sid in set(q["answer_session_ids"])}
    sid2 = dict(zip(q["haystack_session_ids"], q["haystack_sessions"]))
    facts = [f for f in sess_facts(q) if f[0] in gs]
    V = views[qid]
    for sid in q["answer_session_ids"]:
        for t in sid2.get(sid, []):
            if not (t.get("has_answer") and t.get("content")):
                continue
            hits = [f for f in facts if contains(t["content"], f[3])]
            per_turn.append((qid, len(hits),
                             any(contains(V["full"], f[3]) for f in hits),
                             any(contains(V["cap"], f[3]) for f in hits)))
n = len(per_turn)
print(f"\ngold-evidence TURNS (SSP): {n}; turn has >=1 extracted value {sum(1 for r in per_turn if r[1])}; "
      f"any value in full view {sum(1 for r in per_turn if r[2])}; any value in capped view {sum(1 for r in per_turn if r[3])}")

json.dump({k: [list(r) for r in v] for k, v in defs.items()},
          open("/root/agent_memory_iclr/verify/_pref_defs.json", "w"), indent=1)
