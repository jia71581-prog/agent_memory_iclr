"""Independent recomputation of the abstention-twin claim.
- recompute shapes directly from dataset (not from the design agent's cached rows)
- ground-truth twin via qid pairing (X_abs -> X)
- Jaccard sensitivity to tokenisation / stoplist
"""
import json, re, sys, statistics, collections
sys.path.insert(0, "/root/agent_memory_iclr/code")
from v5_shape import classify_shape

D = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
byid = {r["question_id"]: r for r in D}
Q = [{"qid": r["question_id"], "q": r["question"], "type": r["question_type"],
      "gold": r["answer"]} for r in D]
for r in Q:
    r["is_abs"] = r["qid"].endswith("_abs")
    r["shape"], r["fired"] = classify_shape(r["q"])

# ---- cross-check vs design agent's cached rows
ROWS = {r["qid"]: r for r in json.load(open("/root/agent_memory_iclr/results/_v5_shape_rows.json"))}
mism = [r["qid"] for r in Q if ROWS[r["qid"]]["shape"] != r["shape"]]
print(f"[A] shape recompute vs cached rows: mismatches = {len(mism)}/{len(Q)} {mism[:5]}")
absm = [r["qid"] for r in Q if ROWS[r["qid"]]["is_abs"] != r["is_abs"]]
print(f"[A] is_abs flag mismatches = {len(absm)}")

AB = [r for r in Q if r["is_abs"]]
NA = [r for r in Q if not r["is_abs"]]
print(f"[A] n_abs={len(AB)} n_ans={len(NA)}")

# ---- shape distribution of abstention questions
dist = collections.Counter(r["shape"] for r in AB)
print("[B] abstention shape distribution:", dict(dist.most_common()))
tot = collections.Counter(r["shape"] for r in Q)
rates = {s: dist.get(s,0)/tot[s] for s in tot}
print("[B] within-shape abstention rate:",
      {s: f"{rates[s]:.3f} ({dist.get(s,0)}/{tot[s]})" for s,_ in dist.most_common()})
print(f"[B] max within-shape abstention rate = {max(rates.values()):.3f} at "
      f"{max(rates,key=rates.get)}")

# ---- tokenisation variants
W = re.compile(r"[a-z0-9']+")
STOP_AGENT = {"i","my","the","a","an","do","did","was","is","in","of","to","have","how"}
STOP_SK = set("""i me my myself we our ours you your the a an and or but if of at by for with
about into to from in on off over under then once here there all any both each few more most
other some such no nor not only own same so than too very s t can will just don should now do
does did is are was were be been being have has had having it its this that these those what
which who whom when where why""".split())
VARIANTS = {
    "agent_stop": lambda s: {t for t in W.findall(s.lower()) if t not in STOP_AGENT},
    "no_stop":    lambda s: set(W.findall(s.lower())),
    "sk_stop":    lambda s: {t for t in W.findall(s.lower()) if t not in STOP_SK},
    "bigram":     lambda s: set(zip(W.findall(s.lower()), W.findall(s.lower())[1:])),
}

def nn(a, tokf, pool):
    ta = tokf(a["q"]); best, bj = None, -1.0
    for n in pool:
        tn = tokf(n["q"]); j = len(ta & tn)/max(1, len(ta | tn))
        if j > bj: bj, best = j, n
    return best, bj

print()
for name, tokf in VARIANTS.items():
    same = 0; js = []; hit_true = 0; n_true = 0
    for a in AB:
        best, bj = nn(a, tokf, NA)
        same += (a["shape"] == best["shape"]); js.append(bj)
        base = a["qid"][:-4]
        if base in byid:
            n_true += 1
            hit_true += (best["qid"] == base)
    js.sort()
    print(f"[C] {name:10s} sameshape={same}/30={same/30:.1%} "
          f"median_J={statistics.median(js):.2f} min={js[0]:.2f} max={js[-1]:.2f} "
          f">=0.5: {sum(j>=0.5 for j in js)}/30 | NN==true_twin {hit_true}/{n_true}")

# ---- ground-truth twin (qid pairing), shape agreement
print()
gt = [(a, byid.get(a["qid"][:-4])) for a in AB]
have = [(a,b) for a,b in gt if b]
print(f"[D] abstention questions whose source qid is in the 500: {len(have)}/30")
sameG = 0
diffs = []
for a, b in have:
    sb, fb = classify_shape(b["question"])
    ok = (a["shape"] == sb); sameG += ok
    if not ok: diffs.append((a["qid"], a["q"], a["shape"], b["question"], sb))
print(f"[D] TRUE-twin same shape: {sameG}/{len(have)} = {sameG/len(have):.1%}")
for d in diffs:
    print("    DIFF", d[0], "|", d[1][:60], "->", d[2])
    print("         ", d[3][:60], "->", d[4])
