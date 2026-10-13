"""Adversarial independent recompute of the 'coverage retriever is shape-independent' claim."""
import json, os, re, sys
from collections import defaultdict, Counter
from multiprocessing import Pool
sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from run_baseline import bm25_rank, tokenize
from v5_shape import classify_shape

R = "/root/agent_memory_iclr/results"
K = 20

def norm(s): return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
def contains(hay, needle):
    h, n = norm(hay), norm(needle)
    if not n: return False
    if n in h: return True
    toks = [t for t in n.split() if len(t) > 2 and t not in ("the","and","for","was","were","you","your")]
    if not toks: return False
    return all(t in h for t in toks)

def build_cov(order, units, k=K, depth=2):
    best = defaultdict(list)
    for i in order: best[units[i][0]].append(i)
    rank_of = {i: r for r, i in enumerate(order)}
    sess_rank = sorted(best, key=lambda s: rank_of[best[s][0]])
    cov, p = [], 0
    while len(cov) < k and p < depth:
        for s in sess_rank:
            if len(cov) >= k: break
            if len(best[s]) > p: cov.append(best[s][p])
        p += 1
    return cov

def work(q):
    qid = q["question_id"]
    sid2idx = {s: i for i, s in enumerate(q["haystack_session_ids"])}
    ev = {sid2idx[s] for s in q["answer_session_ids"] if s in sid2idx}
    units, ansturns = [], set()
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for t in sess:
            if t.get("content"):
                if t.get("has_answer"): ansturns.add(len(units))
                units.append((si, date, t["role"], t["content"]))
    order = bm25_rank(q["question"], [u[3] for u in units])
    flat = order[:K]
    cov = build_cov(order, units)
    fs, cs = {units[i][0] for i in flat}, {units[i][0] for i in cov}
    gold = str(q["answer"])
    ftxt = " ".join(units[i][3] for i in flat)
    ctxt = " ".join(units[i][3] for i in cov)
    return {
        "qid": qid, "type": q["question_type"], "shape": classify_shape(q["question"])[0],
        "is_abs": qid.endswith("_abs"), "n_ev": len(ev), "n_sess": len(q["haystack_sessions"]),
        "n_units": len(units), "n_ansturns": len(ansturns),
        "flat_all": ev <= fs, "cov_all": ev <= cs, "subset": fs <= cs,
        "n_flat_sess": len(fs), "n_cov_sess": len(cs),
        "flat_chars": sum(len(units[i][3]) for i in flat),
        "cov_chars": sum(len(units[i][3]) for i in cov),
        "flat_toks": sum(len(tokenize(units[i][3])) for i in flat),
        "cov_toks": sum(len(tokenize(units[i][3])) for i in cov),
        "flat_evturns": sum(1 for i in flat if units[i][0] in ev),
        "cov_evturns": sum(1 for i in cov if units[i][0] in ev),
        # annotated evidence turns (has_answer)
        "ansturn_any_flat": bool(ansturns & set(flat)), "ansturn_any_cov": bool(ansturns & set(cov)),
        "ansturn_all_flat": bool(ansturns) and ansturns <= set(flat),
        "ansturn_all_cov": bool(ansturns) and ansturns <= set(cov),
        "ansturn_hit_flat": len(ansturns & set(flat)), "ansturn_hit_cov": len(ansturns & set(cov)),
        "gold_in_flat": contains(ftxt, gold), "gold_in_cov": contains(ctxt, gold),
        "gold": gold, "question": q["question"],
    }

if __name__ == "__main__":
    data = load_dataset("s_cleaned")
    with Pool(24) as p: rows = p.map(work, data, chunksize=4)
    json.dump(rows, open("/root/agent_memory_iclr/verify/adv_cov1.json", "w"), ensure_ascii=False)
    print("n =", len(rows))
    print("subset flat_sess<=cov_sess holds:", sum(r["subset"] for r in rows), "/", len(rows))
    NA = [r for r in rows if not r["is_abs"]]
    print("non-abstention:", len(NA))
    g = [r for r in NA if r["cov_all"] and not r["flat_all"]]
    l = [r for r in NA if r["flat_all"] and not r["cov_all"]]
    print(f"complete-evidence-set: gain {len(g)}  loss {len(l)}  net {len(g)-len(l):+d}")
    print("gain by shape:", dict(Counter(r["shape"] for r in g)))
