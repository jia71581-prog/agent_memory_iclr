"""Adversarial controls on B5b.

C1  null: mismatched rubric (another SSP question's gold rubric) -> does length alone help?
C2  rubric-only query.
C3  leakage decomposition: drop from the rubric every token that occurs ONLY in the gold
    session (df==1 within this question's haystack) -> how much of 30/30 survives?
C4  free (no-API) ranking variants: full-session text, max-turn score, top3-turn sum,
    RM3 pseudo-relevance feedback.  If any of these close the gap, the API call is not needed.
"""
import json, math, re, sys, random
from collections import Counter

DATA = "/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"
_w = re.compile(r"[a-z0-9']+")
def tok(s): return _w.findall(str(s).lower())

def bm25_scores(qtokens, dt, k1=1.5, b=0.75):
    N = len(dt); avgdl = sum(len(d) for d in dt)/max(1, N)
    df = Counter()
    for d in dt:
        for w in set(d): df[w] += 1
    out = []
    for d in dt:
        tf = Counter(d); s = 0.0; dl = len(d)
        for w, wt in qtokens:
            f = tf.get(w, 0)
            if not f: continue
            idf = math.log(1 + (N - df[w] + 0.5)/(df[w] + 0.5))
            s += wt * idf * f*(k1+1)/(f + k1*(1-b+b*dl/avgdl))
        out.append(s)
    return out

def rank_pess(scores, gold):
    g = max(scores[i] for i in gold)
    return sum(1 for i, s in enumerate(scores) if i not in gold and s >= g)

def sess_docs(q, role=None):
    return [" ".join(t["content"] for t in sess if t.get("content") and (role is None or t["role"] == role))
            for sess in q["haystack_sessions"]]

STOP = set("""the a an and or but if of in on at to for with from by about as into over under after before
is are was were be been being am do does did have has had i you my me your we us our it its this that these
those not no so such too very can could will would should may might must just now what how when where which
who whom whose why whether there their them they he she his her would prefer responses user might general
also more most some any""".split())

def main():
    data = json.load(open(DATA))
    ssp = [q for q in data if q["question_type"] == "single-session-preference"]
    rnd = random.Random(0)
    perm = list(range(len(ssp))); rnd.shuffle(perm)
    # ensure no fixed points
    for i, j in enumerate(perm):
        if i == j:
            k = (i+1) % len(perm); perm[i], perm[k] = perm[k], perm[i]

    res = {k: [] for k in ["bare", "oracle", "mismatch", "rubric_only", "noleak", "noleak_rare",
                           "full_text", "max_turn", "top3_turn", "rm3", "full_text_oracle"]}
    detail = []
    for qi, q in enumerate(ssp):
        gold = {i for i, s in enumerate(q["haystack_session_ids"]) if s in set(q["answer_session_ids"])}
        du = [tok(d) for d in sess_docs(q, "user")]
        da = [tok(d) for d in sess_docs(q, None)]
        qt = [(w, 1.0) for w in tok(q["question"])]
        rub = tok(q["answer"])
        res["bare"].append(rank_pess(bm25_scores(qt, du), gold))
        res["oracle"].append(rank_pess(bm25_scores(qt + [(w, 1.0) for w in rub], du), gold))
        mis = tok(ssp[perm[qi]]["answer"])
        res["mismatch"].append(rank_pess(bm25_scores(qt + [(w, 1.0) for w in mis], du), gold))
        res["rubric_only"].append(rank_pess(bm25_scores([(w, 1.0) for w in rub], du), gold))

        # C3 leakage: tokens of the rubric that occur in NO non-gold session (df restricted to gold)
        dfu = Counter()
        for d in du:
            for w in set(d): dfu[w] += 1
        goldset = set()
        for i in gold: goldset |= set(du[i])
        excl = {w for w in set(rub) if w in goldset and dfu[w] <= 1}          # gold-only tokens
        rare = {w for w in set(rub) if w in goldset and dfu[w] <= 3}          # gold + <=2 others
        res["noleak"].append(rank_pess(bm25_scores(qt + [(w, 1.0) for w in rub if w not in excl], du), gold))
        res["noleak_rare"].append(rank_pess(bm25_scores(qt + [(w, 1.0) for w in rub if w not in rare], du), gold))

        # C4 free variants
        res["full_text"].append(rank_pess(bm25_scores(qt, da), gold))
        res["full_text_oracle"].append(rank_pess(bm25_scores(qt + [(w, 1.0) for w in rub], da), gold))
        # turn-level scores aggregated per session (user turns only, same channel as baseline)
        turns, owner = [], []
        for si, sess in enumerate(q["haystack_sessions"]):
            for t in sess:
                if t.get("content") and t["role"] == "user":
                    turns.append(tok(t["content"])); owner.append(si)
        ts = bm25_scores(qt, turns)
        mx = [0.0]*len(du); top3 = {}
        for s, si in zip(ts, owner):
            mx[si] = max(mx[si], s); top3.setdefault(si, []).append(s)
        res["max_turn"].append(rank_pess(mx, gold))
        s3 = [sum(sorted(top3.get(i, [0.0]), reverse=True)[:3]) for i in range(len(du))]
        res["top3_turn"].append(rank_pess(s3, gold))
        # RM3: top-3 sessions by bare, top-15 expansion terms by tf*idf, weight 0.5
        base = bm25_scores(qt, du)
        top = sorted(range(len(du)), key=lambda i: -base[i])[:3]
        cnt = Counter()
        for i in top:
            for w in du[i]:
                if w not in STOP and len(w) > 2: cnt[w] += 1
        exp = [(w, 0.5) for w, _ in cnt.most_common(15)]
        res["rm3"].append(rank_pess(bm25_scores(qt + exp, du), gold))
        detail.append(dict(qid=q["question_id"], gold_only_tokens=sorted(excl)[:14],
                           n_rub_tok=len(set(rub)), n_gold_only=len(excl),
                           bare=res["bare"][-1], oracle=res["oracle"][-1],
                           noleak=res["noleak"][-1], noleak_rare=res["noleak_rare"][-1],
                           full_text=res["full_text"][-1], max_turn=res["max_turn"][-1],
                           top3=res["top3_turn"][-1], rm3=res["rm3"][-1]))

    print("variant            rank0/30  top3/30  top5/30  median_rank")
    for k, v in res.items():
        print(f"{k:18s} {sum(1 for x in v if x==0):5d}    {sum(1 for x in v if x<3):5d}   "
              f"{sum(1 for x in v if x<5):5d}     {sorted(v)[len(v)//2]:4d}")
    print("\nper-question detail (bare / oracle / noleak(gold-only removed) / noleak_rare / full_text / max_turn / top3turn / rm3):")
    for d in sorted(detail, key=lambda d: -d["bare"]):
        print(f"  {d['qid']:10s} {d['bare']:3d} {d['oracle']:3d} {d['noleak']:3d} {d['noleak_rare']:3d} "
              f"{d['full_text']:3d} {d['max_turn']:3d} {d['top3']:3d} {d['rm3']:3d}   goldonly={d['n_gold_only']}/{d['n_rub_tok']} {d['gold_only_tokens'][:8]}")
    json.dump(detail, open("/root/agent_memory_iclr/results/_ver_ssp_controls.json", "w"), indent=1)

main()
