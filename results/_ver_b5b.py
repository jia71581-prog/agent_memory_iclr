"""Adversarial verification of B5b (SSP lexical-bridge / query-expansion headroom).

Sections:
  R  reproduce headline (bare 12/30 rank0, 19/30 top3; oracle 30/30) with own BM25 + tie audit
  C  null controls: mismatched rubric, rubric-only, leakage-stripped oracle
  F  free (no-API) alternatives: full-session text, max-turn, top3-turn, RM3 PRF
  P  predictive validity: does gold-session rank / evidence-in-context predict the judge?
  H  question-level headroom over the already-implemented no-API fix (retrieve_v5 n=3)
  B  budget: SSP raw chars, top-20 vs n=1/2/3
  X  cost: advice_shaped() fire count over the full 500
"""
import json, math, re, sys, random
from collections import Counter

sys.path.insert(0, "/root/agent_memory_iclr/code")
DATA = "/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"
RES = "/root/agent_memory_iclr/results"

_w = re.compile(r"[a-z0-9']+")
def tok(s): return _w.findall(str(s).lower())

def bm25(qterms, dt, k1=1.5, b=0.75):
    N = len(dt); avgdl = sum(len(d) for d in dt)/max(1, N)
    df = Counter()
    for d in dt:
        for w in set(d): df[w] += 1
    out = []
    for d in dt:
        tf = Counter(d); s = 0.0; dl = len(d)
        for w, wt in qterms:
            f = tf.get(w, 0)
            if not f: continue
            idf = math.log(1 + (N - df[w] + 0.5)/(df[w] + 0.5))
            s += wt*idf*f*(k1+1)/(f + k1*(1-b+b*dl/avgdl))
        out.append(s)
    return out

def rp(scores, gold):                      # pessimistic rank (ties counted ahead)
    g = max(scores[i] for i in gold)
    return sum(1 for i, s in enumerate(scores) if i not in gold and s >= g)

def sdocs(q, role):
    return [" ".join(t["content"] for t in sess if t.get("content") and (role is None or t["role"] == role))
            for sess in q["haystack_sessions"]]

def units_of(q):
    out = []
    for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        for t in sess:
            if t.get("content"): out.append((date, t["role"], t["content"]))
    return out

STOP = set("""the a an and or but if of in on at to for with from by about as into over under after before
is are was were be been being am do does did have has had i you my me your we us our it its this that these
those not no so such too very can could will would should may might must just now what how when where which
who whom whose why whether there their them they he she his her prefer responses user general also more most
some any specifically especially might""".split())

def fisher(a, b, c, d):
    """two-sided Fisher exact p for [[a,b],[c,d]]"""
    n = a+b+c+d
    def pr(x):
        return (math.comb(a+b, x)*math.comb(c+d, a+c-x))/math.comb(n, a+c)
    lo = max(0, a+c-(c+d)); hi = min(a+b, a+c)
    p0 = pr(a)
    return sum(pr(x) for x in range(lo, hi+1) if pr(x) <= p0 + 1e-12)

def main():
    data = json.load(open(DATA))
    ssp = [q for q in data if q["question_type"] == "single-session-preference"]
    from run_baseline import bm25_rank as their
    import v5_guards as G

    qc = {}
    for line in open(f"{RES}/full500_qc.jsonl"):
        r = json.loads(line); qc[r["qid"]] = r

    rnd = random.Random(0); perm = list(range(len(ssp))); rnd.shuffle(perm)
    for i, j in enumerate(perm):
        if i == j:
            k = (i+1) % len(perm); perm[i], perm[k] = perm[k], perm[i]

    rows = []
    for qi, q in enumerate(ssp):
        gold = {i for i, s in enumerate(q["haystack_session_ids"]) if s in set(q["answer_session_ids"])}
        du, da = [tok(d) for d in sdocs(q, "user")], [tok(d) for d in sdocs(q, None)]
        qt = [(w, 1.0) for w in tok(q["question"])]
        rub = tok(q["answer"])
        dfu = Counter()
        for d in du:
            for w in set(d): dfu[w] += 1
        gset = set()
        for i in gold: gset |= set(du[i])
        excl = {w for w in set(rub) if w in gset and dfu[w] <= 1}
        rare = {w for w in set(rub) if w in gset and dfu[w] <= 3}
        mis = tok(ssp[perm[qi]]["answer"])

        r = dict(qid=q["question_id"], nsess=len(du),
                 bare=rp(bm25(qt, du), gold),
                 their_bare=min(their(q["question"], sdocs(q, "user")).index(i) for i in gold),
                 oracle=rp(bm25(qt+[(w, 1.0) for w in rub], du), gold),
                 mismatch=rp(bm25(qt+[(w, 1.0) for w in mis], du), gold),
                 rubric_only=rp(bm25([(w, 1.0) for w in rub], du), gold),
                 noleak=rp(bm25(qt+[(w, 1.0) for w in rub if w not in excl], du), gold),
                 noleak_rare=rp(bm25(qt+[(w, 1.0) for w in rub if w not in rare], du), gold),
                 full_text=rp(bm25(qt, da), gold),
                 n_gold_only=len(excl), n_rub=len(set(rub)),
                 gold_only=sorted(excl)[:10])
        # turn-level aggregations over user turns
        turns, owner = [], []
        for si, sess in enumerate(q["haystack_sessions"]):
            for t in sess:
                if t.get("content") and t["role"] == "user":
                    turns.append(tok(t["content"])); owner.append(si)
        ts = bm25(qt, turns)
        mx = [0.0]*len(du); per = {}
        for s, si in zip(ts, owner):
            mx[si] = max(mx[si], s); per.setdefault(si, []).append(s)
        r["max_turn"] = rp(mx, gold)
        r["top3_turn"] = rp([sum(sorted(per.get(i, [0.0]), reverse=True)[:3]) for i in range(len(du))], gold)
        base = bm25(qt, du); top = sorted(range(len(du)), key=lambda i: -base[i])[:3]
        cnt = Counter()
        for i in top:
            for w in du[i]:
                if w not in STOP and len(w) > 2: cnt[w] += 1
        r["rm3"] = rp(bm25(qt+[(w, 0.5) for w, _ in cnt.most_common(15)], du), gold)

        # evidence in context: current pipeline (top-20 turns, all roles) vs retrieve_v5
        units = units_of(q); docs = [u[2] for u in units]
        sid2 = dict(zip(q["haystack_session_ids"], q["haystack_sessions"]))
        ev = [t["content"] for sid in q["answer_session_ids"] for t in sid2.get(sid, [])
              if t.get("has_answer") and t.get("content")]
        b20 = {docs[i] for i in their(q["question"], docs)[:20]}
        r["ev_n"] = len(ev)
        r["ev20"] = sum(1 for e in ev if e in b20)
        for ns in (1, 2, 3):
            idx, _ = G.retrieve_v5(q, n_sessions=ns)
            got = {docs[i] for i in idx}
            r[f"ev_s{ns}"] = sum(1 for e in ev if e in got)
            r[f"ch_s{ns}"] = sum(len(docs[i]) for i in idx)
        r["ch20"] = sum(len(docs[i]) for i in their(q["question"], docs)[:20])
        r["judge"] = bool(qc[q["question_id"]]["judge"]) if q["question_id"] in qc else None
        r["decline"] = G.is_decline(qc.get(q["question_id"], {}).get("pred", ""))
        rows.append(r)

    def n0(k): return sum(1 for r in rows if r[k] == 0)
    def n3(k): return sum(1 for r in rows if r[k] < 3)
    print("=== R/C/F  gold-session rank under each query formulation (n=30 SSP) ===")
    print(f"{'variant':14s} {'rank0':>6s} {'top3':>6s} {'med':>4s}  {'mean':>6s}")
    for k in ["bare", "their_bare", "oracle", "mismatch", "rubric_only", "noleak", "noleak_rare",
              "full_text", "max_turn", "top3_turn", "rm3"]:
        v = sorted(r[k] for r in rows)
        print(f"{k:14s} {n0(k):5d}/30 {n3(k):5d}/30 {v[15]:4d} {sum(v)/30:6.2f}")

    print("\n=== leakage: rubric tokens that occur in NO other session ===")
    print("  mean gold-only tokens per rubric: %.2f of %.1f content tokens"
          % (sum(r["n_gold_only"] for r in rows)/30, sum(r["n_rub"] for r in rows)/30))
    print("  questions where oracle rank0 but noleak rank>0: %d"
          % sum(1 for r in rows if r["oracle"] == 0 and r["noleak"] > 0))
    print("  questions where oracle rank0 but noleak_rare rank>0: %d"
          % sum(1 for r in rows if r["oracle"] == 0 and r["noleak_rare"] > 0))

    print("\n=== P  predictive validity vs the qc judge (n=30) ===")
    for name, fn in [("gold session rank0 (bare)", lambda r: r["bare"] == 0),
                     ("gold session top3 (bare)", lambda r: r["bare"] < 3),
                     ("all gold ev turns in top20", lambda r: r["ev20"] == r["ev_n"]),
                     ("any gold ev turn in top20", lambda r: r["ev20"] > 0),
                     ("all gold ev turns at n=3", lambda r: r["ev_s3"] == r["ev_n"])]:
        A = [r for r in rows if fn(r)]; B = [r for r in rows if not fn(r)]
        aa = sum(1 for r in A if r["judge"]); bb = sum(1 for r in B if r["judge"])
        p = fisher(aa, len(A)-aa, bb, len(B)-bb) if A and B else float("nan")
        print(f"  {name:30s} yes {aa:2d}/{len(A):<2d} ({aa/max(1,len(A)):.3f})   "
              f"no {bb:2d}/{len(B):<2d} ({bb/max(1,len(B)):.3f})   fisher p={p:.3f}")

    print("\n=== H  headroom over the no-API fix, question level ===")
    def full(r, k): return r[k] == r["ev_n"]
    cur = [r for r in rows if not full(r, "ev20")]
    print(f"  SSP lacking complete gold evidence today (top20): {len(cur)}/30; of those wrong: "
          f"{sum(1 for r in cur if not r['judge'])}")
    s3 = [r for r in rows if not full(r, "ev_s3")]
    print(f"  still lacking at retrieve_v5 n=3 (no API):        {len(s3)}/30; of those wrong: "
          f"{sum(1 for r in s3 if not r['judge'])}  -> {[r['qid'] for r in s3]}")
    print("  those questions' bare session rank / oracle rank:",
          [(r["qid"], r["bare"], r["oracle"]) for r in s3])
    print(f"  wrong today AND complete evidence already present at top20: "
          f"{sum(1 for r in rows if full(r,'ev20') and not r['judge'])}")

    print("\n=== B  SSP raw-turn budget ===")
    print("  mean chars  top20=%.0f  n=1 %.0f  n=2 %.0f  n=3 %.0f"
          % tuple(sum(r[k] for r in rows)/30 for k in ["ch20", "ch_s1", "ch_s2", "ch_s3"]))
    print("  median chars top20=%d  n=3 %d"
          % (sorted(r["ch20"] for r in rows)[15], sorted(r["ch_s3"] for r in rows)[15]))
    print("  SSP questions where n=3 context is LARGER than top20: %d/30"
          % sum(1 for r in rows if r["ch_s3"] > r["ch20"]))

    print("\n=== X  cost: advice_shaped() over the full 500 ===")
    adv = [q for q in data if G.advice_shaped(q["question"])]
    print(f"  fires on {len(adv)}/500 = {len(adv)/500:.3%}")
    print("  by type:", Counter(("ABS" if q["question_id"].endswith("_abs") else q["question_type"])
                                for q in adv))
    print("  question chars mean/max for advice questions: %.0f / %d"
          % (sum(len(q["question"]) for q in adv)/len(adv), max(len(q["question"]) for q in adv)))

    print("\n=== per-question dump ===")
    hdr = ["qid", "judge", "dec", "bare", "orac", "nolk", "full", "maxT", "rm3", "ev20", "evs3", "evn"]
    print("  " + " ".join(f"{h:>6s}" for h in hdr))
    for r in sorted(rows, key=lambda r: -r["bare"]):
        print("  " + " ".join(f"{str(r[k]):>6s}" for k in
              ["qid", "judge", "decline", "bare", "oracle", "noleak", "full_text", "max_turn",
               "rm3", "ev20", "ev_s3", "ev_n"]))
    json.dump(rows, open(f"{RES}/_ver_b5b.json", "w"), indent=1)

main()
