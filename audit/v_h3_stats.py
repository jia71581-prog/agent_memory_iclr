"""H3 verification part 2: paired stats, density, subset conditioning, loose recall."""
import pickle, re, json, math
from statistics import mean, stdev
rows = pickle.load(open("/root/agent_memory_iclr/audit/v_h3_ssp.pkl", "rb"))
W = re.compile(r"[a-z0-9]+")
def tok(s): return W.findall(str(s).lower())
STOP = set("""a an the and or but if then than that this these those of in on at to for with from by as is are was were be been being am do does did done have has had having i me my mine you your yours he him his she her hers it its we us our ours they them their theirs what which who whom when where why how all all any both each few more most other some such no nor not only own same so too very can will just don should now about into over after before again further once here there also would could may might must shall s t d ll m o re ve y""".split())

def prep(r):
    turns = r["turns"]; n = len(turns)
    df = {}
    for t in turns:
        for w in set(tok(t)):
            df[w] = df.get(w, 0) + 1
    g = set(tok(r["answer"])) - set(tok(r["question"]))
    g = {w for w in g if w not in STOP}
    distinct = {w for w in g if df.get(w, 0) / max(1, n) <= 0.02}
    loose = {w for w in set(tok(r["answer"])) if w not in STOP}  # content words, no q-removal, no df filter
    return distinct, loose, n

ctxs = {"view": lambda r: r["view"], "raw20": lambda r: r["raw20"], "raw30": lambda r: r["raw30"],
        "view+raw20": lambda r: r["view"] + "\n" + r["raw20"]}

recs = []
for r in rows:
    distinct, loose, n = prep(r)
    d = {"qid": r["qid"], "nterms": len(distinct), "n_turns": n,
         "judge_qc": r["judge_qc"], "judge_full": r["judge_full"], "judge_nolc": r["judge_nolc"],
         "f1_qc": r["f1_qc"], "f1_full": r["f1_full"], "answer": r["answer"], "question": r["question"]}
    for k, f in ctxs.items():
        T = set(tok(f(r)))
        d["cov_" + k] = len(distinct & T) / len(distinct) if distinct else float("nan")
        d["loose_" + k] = len(loose & T) / len(loose) if loose else float("nan")
        d["ntok_" + k] = len(tok(f(r)))
    recs.append(d)

print("== distinctive-term coverage (df<=2%, gold minus question, content words) ==")
for k in ctxs: print(f"  {k:11s} {mean([d['cov_'+k] for d in recs]):.4f}")
print("== loose content-word recall (all gold content words) ==")
for k in ctxs: print(f"  {k:11s} {mean([d['loose_'+k] for d in recs]):.4f}")

# paired tests
def wilcoxon(x, y):
    d = [a - b for a, b in zip(x, y) if a != b]
    n = len(d)
    if n == 0: return None, 1.0, 0
    ad = sorted((abs(v), i) for i, v in enumerate(d))
    ranks = [0.0] * n
    i = 0
    while i < len(ad):
        j = i
        while j + 1 < len(ad) and ad[j + 1][0] == ad[i][0]: j += 1
        rk = (i + j) / 2 + 1
        for t in range(i, j + 1): ranks[ad[t][1]] = rk
        i = j + 1
    wp = sum(ranks[i] for i in range(n) if d[i] > 0)
    wm = sum(ranks[i] for i in range(n) if d[i] < 0)
    W_ = min(wp, wm)
    mu = n * (n + 1) / 4; sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (W_ - mu + 0.5) / sd
    p = 2 * 0.5 * math.erfc(-z / math.sqrt(2)) if z < 0 else 2 * (1 - 0.5 * math.erfc(-z / math.sqrt(2)))
    return z, min(1.0, p), n

def paired_t(x, y):
    d = [a - b for a, b in zip(x, y)]
    n = len(d); m = mean(d); s = stdev(d)
    t = m / (s / math.sqrt(n))
    # two-sided p via normal approx + t correction (use survival of t dist by numeric integration)
    return m, s, t, n

print("\n== paired comparisons on the same 30 questions ==")
for a, b in [("view", "raw30"), ("view", "raw20"), ("view+raw20", "raw30"), ("view+raw20", "raw20")]:
    x = [d["cov_" + a] for d in recs]; y = [d["cov_" + b] for d in recs]
    m, s, t, n = paired_t(x, y)
    z, p, nz = wilcoxon(x, y)
    print(f"  cov {a:11s} - {b:10s} mean diff={m:+.4f} sd={s:.4f} t={t:+.2f} (n=30)  wilcoxon z={z:+.2f} p={p:.4f} (nonzero {nz})")

print("\n== per-1000-token density of distinctive gold terms ==")
for k in ctxs:
    dens = [1000 * d["cov_" + k] * d["nterms"] / max(1, d["ntok_" + k]) for d in recs]
    print(f"  {k:11s} hits/1k tok = {mean(dens):.3f}   mean ctx tokens={mean([d['ntok_'+k] for d in recs]):.0f}")

print("\n== conditioning on the discordant questions that produce the SSP gain ==")
groups = {"qc win / full lose": [d for d in recs if d["judge_qc"] and not d["judge_full"]],
          "full win / qc lose": [d for d in recs if d["judge_full"] and not d["judge_qc"]],
          "both correct": [d for d in recs if d["judge_qc"] and d["judge_full"]],
          "both wrong": [d for d in recs if not d["judge_qc"] and not d["judge_full"]]}
for g, G in groups.items():
    if not G: print(f"  {g:20s} n=0"); continue
    print(f"  {g:20s} n={len(G):2d}  cov view={mean([d['cov_view'] for d in G]):.3f} raw20={mean([d['cov_raw20'] for d in G]):.3f} "
          f"raw30={mean([d['cov_raw30'] for d in G]):.3f} view+raw20={mean([d['cov_view+raw20'] for d in G]):.3f}")
print("  qc acc:", sum(d["judge_qc"] for d in recs), "/30   full acc:", sum(d["judge_full"] for d in recs),
      "/30   nolc(top20 only):", sum(d["judge_nolc"] for d in recs), "/30")

json.dump(recs, open("/root/agent_memory_iclr/audit/v_h3_recs.json", "w"), default=str)
