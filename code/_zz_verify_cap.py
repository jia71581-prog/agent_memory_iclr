"""Pareto check: same 20-turn budget, per-session cap sweep (cap=1 is exactly the claimed
coverage round-robin; cap=None is the deployed flat BM25). Measures both the session-coverage
metric the claim uses and the gold-evidence metrics it cannot see."""
import json
import re
import sys
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from run_baseline import bm25_rank

R = "/root/agent_memory_iclr/results"
TOP_K = 20
CAPS = [1, 2, 3, 5, None]


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def contains(hay, needle):
    h, n = norm(hay), norm(needle)
    if not n:
        return False
    if n in h:
        return True
    toks = [t for t in n.split() if len(t) > 2 and t not in ("the", "and", "for", "was", "were", "you", "your")]
    return bool(toks) and all(t in h for t in toks)


def work(q):
    qid = q["question_id"]
    sid2idx = {s: i for i, s in enumerate(q["haystack_session_ids"])}
    ev = {sid2idx[s] for s in q["answer_session_ids"] if s in sid2idx}
    units = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for t in sess:
            if t.get("content"):
                units.append((si, t["content"]))
    order = bm25_rank(q["question"], [u[1] for u in units])
    gold = str(q["answer"])
    gold_turns = {i for i, u in enumerate(units) if contains(u[1], gold)}
    out = {"qid": qid, "is_abs": qid.endswith("_abs"), "has_goldturn": bool(gold_turns)}
    for cap in CAPS:
        if cap is None:
            pick = order[:TOP_K]
        else:
            pick, used = [], defaultdict(int)
            for i in order:
                if used[units[i][0]] >= cap:
                    continue
                used[units[i][0]] += 1
                pick.append(i)
                if len(pick) >= TOP_K:
                    break
        ss = {units[i][0] for i in pick}
        k = "flat" if cap is None else f"cap{cap}"
        out[f"{k}_all"] = ev <= ss
        out[f"{k}_gold"] = contains(" ".join(units[i][1] for i in pick), gold)
        out[f"{k}_goldturn"] = bool(gold_turns & set(pick))
        out[f"{k}_chars"] = sum(len(units[i][1]) for i in pick)
    return out


if __name__ == "__main__":
    data = load_dataset("s_cleaned")
    with Pool(16) as p:
        rows = p.map(work, data, chunksize=4)
    json.dump(rows, open(f"{R}/_zz_verify_cap.json", "w"))
    QC = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{R}/full500_qc.jsonl")}
    NA = [r for r in rows if not r["is_abs"]]
    for r in NA:
        r["judge"] = QC[r["qid"]]["judge"]
    G = [r for r in NA if r["has_goldturn"]]
    n, ng = len(NA), len(G)
    print(f"n non-abs={n}  with verbatim gold turn={ng}")
    print(f"{'arm':10s} {'evSessAll':>10s} {'goldStr':>8s} {'goldTurn':>9s} {'chars':>7s}")
    for cap in [None, 5, 3, 2, 1]:
        k = "flat" if cap is None else f"cap{cap}"
        lab = "flat(dep)" if cap is None else ("cap1=COV" if cap == 1 else k)
        print(f"{lab:10s} {sum(r[k+'_all'] for r in NA)/n:10.3f} {sum(r[k+'_gold'] for r in NA)/n:8.3f} "
              f"{sum(r[k+'_goldturn'] for r in G)/ng:9.3f} {sum(r[k+'_chars'] for r in NA)/n:7.0f}")
    # McNemar exact, gold-turn, cov vs flat
    from scipy.stats import binomtest
    b = sum(1 for r in G if r["cap1_goldturn"] and not r["flat_goldturn"])
    c = sum(1 for r in G if r["flat_goldturn"] and not r["cap1_goldturn"])
    print(f"\ngold-turn McNemar cov vs flat: +{b} / -{c}  exact p={binomtest(b, b+c, 0.5).pvalue:.4g}")
    b2 = sum(1 for r in G if r["cap3_goldturn"] and not r["flat_goldturn"])
    c2 = sum(1 for r in G if r["flat_goldturn"] and not r["cap3_goldturn"])
    print(f"gold-turn McNemar cap3 vs flat: +{b2} / -{c2}  exact p={binomtest(b2, max(1,b2+c2), 0.5).pvalue:.4g}")
    b3 = sum(1 for r in NA if r["cap3_all"] and not r["flat_all"])
    c3 = sum(1 for r in NA if r["flat_all"] and not r["cap3_all"])
    print(f"evSessAll  cap3 vs flat: +{b3} / -{c3}")
