"""Independent recomputation #5: budget-matched control.
Is the +0.040 complete-evidence recall specific to session-coverage round-robin, or does
a plain flat top-k at the SAME char budget get it too (without the turn-level loss)?"""
import json, os, re, sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from run_baseline import bm25_rank
from v5_shape import classify_shape

R = "/root/agent_memory_iclr/results"
OUT = "/root/agent_memory_iclr/verify_x"
STOP = {"the", "and", "for", "was", "were", "you", "your", "with", "that", "this",
        "from", "have", "has", "had", "are", "not"}
KS = [20, 21, 22, 24, 26, 30]


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def contains(h, n):
    h, n = norm(h), norm(n)
    if not n:
        return False
    if n in h:
        return True
    t = [x for x in n.split() if len(x) > 2 and x not in STOP]
    return bool(t) and all(x in h for x in t)


def cov_select(order, sidx, top_k, per_pass=2):
    b = defaultdict(list)
    for i in order:
        b[sidx[i]].append(i)
    pos = {i: r for r, i in enumerate(order)}
    sr = sorted(b, key=lambda s: pos[b[s][0]])
    cov, p = [], 0
    while len(cov) < top_k and p < per_pass:
        for s in sr:
            if len(cov) >= top_k:
                break
            if len(b[s]) > p:
                cov.append(b[s][p])
        p += 1
    return cov


def work(q):
    sid2idx = {s: i for i, s in enumerate(q["haystack_session_ids"])}
    ev = {sid2idx[s] for s in q["answer_session_ids"] if s in sid2idx}
    units, sidx, isha = [], [], []
    for si, sess in enumerate(q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                units.append(t["content"])
                sidx.append(si)
                isha.append(bool(t.get("has_answer")) and si in ev)
    order = bm25_rank(q["question"], units)
    ha = {i for i, v in enumerate(isha) if v}
    gold = str(q["answer"])
    out = {"qid": q["question_id"], "is_abs": q["question_id"].endswith("_abs"),
           "shape": classify_shape(q["question"])[0], "n_ha": len(ha)}
    for k in KS:
        sel = order[:k]
        ss = {sidx[i] for i in sel}
        out[f"flat{k}_all"] = ev <= ss
        out[f"flat{k}_chars"] = sum(len(units[i]) for i in sel)
        out[f"flat{k}_ha_in"] = len(ha & set(sel))
        out[f"flat{k}_ha_all"] = bool(ha and ha <= set(sel))
        out[f"flat{k}_gold"] = contains(" ".join(units[i] for i in sel), gold)
    sel = cov_select(order, sidx, 20)
    ss = {sidx[i] for i in sel}
    out["cov_all"] = ev <= ss
    out["cov_chars"] = sum(len(units[i]) for i in sel)
    out["cov_ha_in"] = len(ha & set(sel))
    out["cov_ha_all"] = bool(ha and ha <= set(sel))
    out["cov_gold"] = contains(" ".join(units[i] for i in sel), gold)
    return out


if __name__ == "__main__":
    data = load_dataset("s_cleaned")
    with Pool(16) as p:
        rows = p.map(work, data, chunksize=2)
    J = {}
    for l in open(f"{R}/full500_qc.jsonl"):
        d = json.loads(l)
        J[d["qid"]] = d.get("judge")
    for r in rows:
        r["judge"] = J.get(r["qid"])
    json.dump(rows, open(f"{OUT}/x5_budget.json", "w"), ensure_ascii=False)
    print("saved", len(rows))
