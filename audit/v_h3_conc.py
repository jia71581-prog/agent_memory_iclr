"""H3 part 3: is the coverage measure the right probe? Concentration / phrase-level tests.
For a 'leak of gold phrasing' the relevant quantity is whether ONE compact unit restates the
answer, not how many gold tokens are scattered over the whole context."""
import pickle, re, json
from statistics import mean
rows = pickle.load(open("/root/agent_memory_iclr/audit/v_h3_ssp.pkl", "rb"))
W = re.compile(r"[a-z0-9]+")
def tok(s): return W.findall(str(s).lower())
STOP = set("""a an the and or but if then than that this these those of in on at to for with from by as is are was were be been being am do does did done have has had having i me my mine you your yours he him his she her hers it its we us our ours they them their theirs what which who whom when where why how all any both each few more most other some such no nor not only own same so too very can will just should now about into over after before again further once here there also would could may might must shall""".split())

def dset(r):
    turns = r["turns"]; n = len(turns)
    df = {}
    for t in turns:
        for w in set(tok(t)): df[w] = df.get(w, 0) + 1
    g = {w for w in set(tok(r["answer"])) - set(tok(r["question"])) if w not in STOP}
    return {w for w in g if df.get(w, 0) / max(1, n) <= 0.02}

def units(text, kind):
    if kind == "view":
        return [l for l in text.splitlines() if l.strip()]
    return [l for l in text.splitlines() if l.strip()]

def maxngram(gold, ctx):
    g = tok(gold); c = tok(ctx)
    cs = set()
    best = 0
    # greedy: longest n such that some gold n-gram appears in ctx
    cgrams = {}
    for n in range(1, min(len(g), 12) + 1):
        grams = {tuple(c[i:i+n]) for i in range(len(c) - n + 1)}
        hit = any(tuple(g[i:i+n]) in grams for i in range(len(g) - n + 1))
        if hit: best = n
        else: break
    return best

out = []
for r in rows:
    D = dset(r)
    row = {"qid": r["qid"], "judge_qc": r["judge_qc"], "judge_full": r["judge_full"], "n": len(D)}
    for k, txt, kind in [("view", r["view"], "view"), ("raw20", r["raw20"], "raw"), ("raw30", r["raw30"], "raw")]:
        us = units(txt, kind)
        best, bestu = 0.0, ""
        for u in us:
            T = set(tok(u))
            c = len(D & T) / len(D) if D else 0
            if c > best: best, bestu = c, u
        row["best_" + k] = best
        row["bestunit_" + k] = bestu[:220]
        row["ng_" + k] = maxngram(r["answer"], txt)
        row["nunits_" + k] = len(us)
    out.append(row)

print("== best SINGLE unit (one view line vs one raw turn): share of distinctive gold terms ==")
for k in ["view", "raw20", "raw30"]:
    print(f"  {k:6s} mean best-unit coverage={mean([o['best_'+k] for o in out]):.4f}   mean #units={mean([o['nunits_'+k] for o in out]):.1f}")
print("\n== longest gold n-gram reproduced verbatim in the context (tokens) ==")
for k in ["view", "raw20", "raw30"]:
    v = [o["ng_" + k] for o in out]
    print(f"  {k:6s} mean={mean(v):.2f}  max={max(v)}  >=4-gram: {sum(1 for x in v if x >= 4)}/30  >=6-gram: {sum(1 for x in v if x >= 6)}/30")

print("\n== split by outcome ==")
for lab, G in [("qc win/full lose", [o for o in out if o["judge_qc"] and not o["judge_full"]]),
               ("other", [o for o in out if not (o["judge_qc"] and not o["judge_full"])])]:
    print(f"  {lab:18s} n={len(G):2d} best-unit view={mean([o['best_view'] for o in G]):.3f} "
          f"raw30={mean([o['best_raw30'] for o in G]):.3f}  ngram view={mean([o['ng_view'] for o in G]):.2f} raw30={mean([o['ng_raw30'] for o in G]):.2f}")

print("\n== 6 examples: gold answer vs best-matching view line ==")
for r, o in list(zip(rows, out))[:6]:
    print(f"\nQ  : {r['question'][:150]}")
    print(f"GOLD: {r['answer'][:220]}")
    print(f"VIEW best line ({o['best_view']:.2f}): {o['bestunit_view']}")
    print(f"RAW30 best turn ({o['best_raw30']:.2f}): {o['bestunit_raw30'][:220]}")
    print(f"judge qc={r['judge_qc']} full={r['judge_full']}")
json.dump(out, open("/root/agent_memory_iclr/audit/v_h3_conc.json", "w"))
