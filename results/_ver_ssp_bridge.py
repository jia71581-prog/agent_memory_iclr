"""Independent recomputation of the B5b oracle query-expansion headroom claim.

Adversarial verifier: own BM25 implementation, own tie handling, plus null controls.
"""
import json, math, re, os, sys, random, statistics as st
from collections import Counter

DATA = "/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"
RES = "/root/agent_memory_iclr/results"

_w = re.compile(r"[a-z0-9']+")
def tok(s): return _w.findall(str(s).lower())

def bm25_scores(query, docs, k1=1.5, b=0.75):
    """Independent Okapi BM25. Returns list of scores."""
    qt = tok(query)
    dt = [tok(d) for d in docs]
    N = len(dt)
    avgdl = sum(len(d) for d in dt)/max(1, N)
    df = Counter()
    for d in dt:
        for w in set(d): df[w] += 1
    out = []
    for d in dt:
        tf = Counter(d); s = 0.0; dl = len(d)
        for w in qt:
            f = tf.get(w, 0)
            if not f: continue
            idf = math.log(1 + (N - df[w] + 0.5)/(df[w] + 0.5))
            s += idf * f*(k1+1)/(f + k1*(1-b+b*dl/avgdl))
        out.append(s)
    return out

def rank_of_gold(scores, gold, tie="pessimistic"):
    """0-based rank of the best gold doc. tie='pessimistic': all equal-score
    non-gold docs counted ahead; 'optimistic': counted behind (what argsort-by-index gives
    when gold happens to come first)."""
    g = max(scores[i] for i in gold)
    better = sum(1 for i, s in enumerate(scores) if i not in gold and s > g)
    equal = sum(1 for i, s in enumerate(scores) if i not in gold and s == g)
    return better if tie == "optimistic" else better + equal

def sess_user_docs(q):
    return [" ".join(t["content"] for t in sess if t.get("content") and t["role"] == "user")
            for sess in q["haystack_sessions"]]

def main():
    data = json.load(open(DATA))
    ssp = [q for q in data if q["question_type"] == "single-session-preference"]
    sys.path.insert(0, "/root/agent_memory_iclr/code")
    from run_baseline import bm25_rank as their_rank

    rows = []
    for q in ssp:
        docs = sess_user_docs(q)
        gold = {i for i, s in enumerate(q["haystack_session_ids"]) if s in set(q["answer_session_ids"])}
        sb = bm25_scores(q["question"], docs)
        so = bm25_scores(q["question"] + " " + q["answer"], docs)
        # their ranking (index-order tie break)
        tr_b = their_rank(q["question"], docs); tr_o = their_rank(q["question"] + " " + q["answer"], docs)
        rows.append(dict(
            qid=q["question_id"], nsess=len(docs),
            bare_opt=rank_of_gold(sb, gold, "optimistic"),
            bare_pess=rank_of_gold(sb, gold, "pessimistic"),
            orac_opt=rank_of_gold(so, gold, "optimistic"),
            orac_pess=rank_of_gold(so, gold, "pessimistic"),
            their_bare=min(tr_b.index(i) for i in gold),
            their_orac=min(tr_o.index(i) for i in gold),
            bare_score=max(sb[i] for i in gold), bare_max=max(sb),
            orac_score=max(so[i] for i in gold), orac_max=max(so),
        ))
    def c(k, v=0): return sum(1 for r in rows if r[k] == v)
    def c3(k): return sum(1 for r in rows if r[k] < 3)
    print("=== B5b recomputation, n=%d SSP ===" % len(rows))
    for tag in ("their_bare", "bare_opt", "bare_pess"):
        print(f"  bare  {tag:11s} rank0={c(tag)}/30 top3={c3(tag)}/30")
    for tag in ("their_orac", "orac_opt", "orac_pess"):
        print(f"  oracle{tag:11s} rank0={c(tag)}/30 top3={c3(tag)}/30")
    print("  worst-case rows (bare rank >= 10):")
    for r in sorted(rows, key=lambda r: -r["their_bare"])[:6]:
        print("   ", r["qid"], "bare", r["their_bare"], "oracle", r["their_orac"], "nsess", r["nsess"])
    json.dump(rows, open(f"{RES}/_ver_ssp_bridge.json", "w"), indent=1)

if __name__ == "__main__":
    main()
