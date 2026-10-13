"""Independent adversarial recomputation of the v5_retrieval motivating diagnostic.
No API. Own BM25 implementation (cross-checked against run_baseline.bm25_rank)."""
import json, math, re, os, hashlib, sys
from collections import Counter, defaultdict

DATA = "/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"
CACHE = "/root/agent_memory_iclr/results/lifecycle_cache/sessions_nt"
R = "/root/agent_memory_iclr/results"

_w = re.compile(r"[a-z0-9']+")
def tok(s): return _w.findall(s.lower())

def bm25_order(query, docs, k1=1.5, b=0.75):
    qt = tok(query); dt = [tok(d) for d in docs]
    avgdl = sum(len(d) for d in dt)/max(1,len(dt))
    df = Counter()
    for d in dt: df.update(set(d))
    n = len(dt); sc = []
    for d in dt:
        tf = Counter(d); s = 0.0
        for w in qt:
            if w not in tf: continue
            idf = math.log(1 + (n - df[w] + 0.5)/(df[w] + 0.5))
            s += idf*tf[w]*(k1+1)/(tf[w] + k1*(1 - b + b*len(d)/avgdl))
        sc.append(s)
    return sorted(range(n), key=lambda i: -sc[i]), sc

# session text hashing -- must match lifecycle._session_text
sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import _session_text, SESSION_CACHE_DIR
print("cache dir used by lifecycle:", SESSION_CACHE_DIR, flush=True)

def facts_of(sess):
    t = _session_text(sess)
    if not t: return []
    k = hashlib.md5(t.encode()).hexdigest()[:16]
    p = os.path.join(SESSION_CACHE_DIR, f"{k}.json")
    if not os.path.exists(p): return None
    return json.load(open(p))

data = json.load(open(DATA))
rows = []
for qi, q in enumerate(data):
    qid = q["question_id"]
    sid2i = {s: i for i, s in enumerate(q["haystack_session_ids"])}
    ev_sess = sorted({sid2i[s] for s in q["answer_session_ids"] if s in sid2i})
    units = []   # (sidx, role, content, has_answer)
    for si, sess in enumerate(q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                units.append((si, t["role"], t["content"], bool(t.get("has_answer"))))
    order, _ = bm25_order(q["question"], [u[2] for u in units])
    ev_turn_idx = [i for i, u in enumerate(units) if u[3]]
    ev_turn_in_gold = [i for i in ev_turn_idx if units[i][0] in set(ev_sess)]
    res = {"qid": qid, "type": q["question_type"], "is_abs": qid.endswith("_abs"),
           "n_sess": len(q["haystack_sessions"]), "n_turns": len(units),
           "n_ev_sess": len(ev_sess), "n_ev_turns": len(ev_turn_idx),
           "n_ev_turns_in_gold_sess": len(ev_turn_in_gold)}
    for K in (20, 30, 50):
        top = order[:K]
        tset = set(top)
        sess_hit = {units[i][0] for i in top}
        res[f"ev_sess_hit{K}"] = len(set(ev_sess) & sess_hit)
        res[f"ev_turn_hit{K}"] = sum(1 for i in ev_turn_idx if i in tset)
        res[f"n_sess_spanned{K}"] = len(sess_hit)
        res[f"full_ev{K}"] = (len(ev_turn_idx) > 0 and
                              all(i in tset for i in ev_turn_idx))
        res[f"chars{K}"] = sum(len(units[i][2]) for i in top)
    # facts per gold evidence session
    nf_ok, nf_missing_cache = 0, 0
    for si in ev_sess:
        f = facts_of(q["haystack_sessions"][si])
        if f is None: nf_missing_cache += 1
        elif len(f) > 0: nf_ok += 1
    res["ev_sess_with_fact"] = nf_ok
    res["ev_sess_cache_missing"] = nf_missing_cache
    rows.append(res)
    if (qi+1) % 50 == 0: print(" ..", qi+1, flush=True)

json.dump(rows, open("/root/agent_memory_iclr/verify/v1_rows.json","w"))
print("saved", len(rows))
