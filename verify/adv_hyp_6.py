"""Test 6: the strongest defense - the wider 'how many' set (n=165). Is the effect
significant there, does the permuted-table control still reproduce it, and does the
proxy predict correctness there?"""
import json, pickle, sys, random, re, math, statistics as st
sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains
import v5_chains as V

D = {x["qid"]: x for x in pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl","rb"))}
G = pickle.load(open("/root/agent_memory_iclr/results/_v5_gold.pkl","rb"))
QC = {}
for line in open("/root/agent_memory_iclr/results/full500_qc.jsonl"):
    r = json.loads(line)
    if "judge" in r: QC[r["qid"]] = r
FULL = dict(V.HYPONYMS); cats = list(FULL)
WORDNUM = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
           "ten":10,"eleven":11,"twelve":12}
def gnum_of(a):
    s = str(a).lower().replace(",", "")
    m = re.search(r"-?\d+(\.\d+)?", s)
    if m: return float(m.group(0))
    for w, v in WORDNUM.items():
        if re.search(rf"\b{w}\b", s): return float(v)
    return None

sel = []
for qid, x in D.items():
    if not V.is_aggregation_question(x["question"])[0]: continue
    g = gnum_of(x["answer"])
    if g is None or g != int(g) or g < 1: continue
    if not re.search(r"\bhow many\b|\bnumber of\b", x["question"].lower()): continue
    sel.append(qid)
print("wider set n =", len(sel))
stores, meta = {}, {}
for qid in sel:
    x = D[qid]
    stores[qid] = V.merge_chains(assemble_chains(x["facts"]))
    meta[qid] = (x["question"], set(G[qid]["gold_sidx"]), gnum_of(x["answer"]),
                 QC.get(qid, {}).get("judge"))

def flags(tbl, uh=True):
    V.HYPONYMS = tbl
    o = {}
    for qid, stt in stores.items():
        q, gs, gn, _ = meta[qid]
        kept, _ = V.aggregation_printed_groups(stt, q, uh)
        pr = [[it[1] for it in g["items"]] for g in kept]
        allit = [c for gi in pr for c in gi]
        cov = len({c.get("sidx") for c in allit} & gs) == len(gs)
        o[qid] = ((len(allit) >= gn) and cov,
                  (max((len(gi) for gi in pr), default=0) >= gn) and cov, len(allit))
    V.HYPONYMS = FULL
    return o

A, B = flags(FULL, True), flags(FULL, False)
n = len(sel)
def mcnemar(A, B, i):
    b = sum(1 for q in A if A[q][i] and not B[q][i])
    c = sum(1 for q in A if B[q][i] and not A[q][i])
    m = b + c
    p = min(1.0, 2 * sum(math.comb(m, k) for k in range(0, min(b, c) + 1)) / 2**m) if m else 1.0
    return b, c, p
for i, nm in [(0, "D1"), (1, "D2")]:
    a = sum(1 for q in A if A[q][i]); bb = sum(1 for q in B if B[q][i])
    b, c, p = mcnemar(A, B, i)
    print(f"{nm}: with {a}/{n}={a/n:.3f}  without {bb}/{n}={bb/n:.3f}  "
          f"gains {b} losses {c}  exact McNemar p={p:.4f}")
print(f"items: with {sum(v[2] for v in A.values())}  without {sum(v[2] for v in B.values())}")

print("\n--- permuted-table control on the wider set (6 seeds) ---")
pd1, pd2 = [], []
for s in range(1, 7):
    rnd = random.Random(s); tgt = cats[:]
    while True:
        rnd.shuffle(tgt)
        if all(x != y for x, y in zip(cats, tgt)): break
    P = flags({c: FULL[t] for c, t in zip(cats, tgt)})
    d1 = sum(1 for q in P if P[q][0]); d2 = sum(1 for q in P if P[q][1])
    pd1.append(d1); pd2.append(d2)
    print(f"  seed {s}: D1 {d1}/{n}={d1/n:.3f}  D2 {d2}/{n}={d2/n:.3f}  "
          f"items {sum(v[2] for v in P.values())}")
print(f"  perm mean D1 {st.mean(pd1):.1f} ({st.mean(pd1)/n:.3f})  D2 {st.mean(pd2):.1f} "
      f"({st.mean(pd2)/n:.3f})")
rd1 = sum(1 for q in A if A[q][0]); rd2 = sum(1 for q in A if A[q][1])
bd1 = sum(1 for q in B if B[q][0]); bd2 = sum(1 for q in B if B[q][1])
print(f"  real table D1 {rd1} D2 {rd2} | no table D1 {bd1} D2 {bd2}")
print(f"  fraction of the D1 gain reproduced by semantics-free permutation: "
      f"{(st.mean(pd1)-bd1)/max(1e-9,(rd1-bd1)):.2f}")
print(f"  fraction of the D2 gain reproduced by semantics-free permutation: "
      f"{(st.mean(pd2)-bd2)/max(1e-9,(rd2-bd2)):.2f}")

print("\n--- proxy validity on the wider set: does D1/D2 predict the judge? ---")
for lab, F in [("with table", A), ("without table", B)]:
    for i, nm in [(0, "D1"), (1, "D2")]:
        t = [q for q in F if F[q][i] and meta[q][3] is not None]
        f = [q for q in F if not F[q][i] and meta[q][3] is not None]
        at = sum(1 for q in t if meta[q][3]) / max(1, len(t))
        af = sum(1 for q in f if meta[q][3]) / max(1, len(f))
        print(f"  {lab:14s} {nm}: acc|proxy=True {at:.3f} (n={len(t)})  "
              f"acc|proxy=False {af:.3f} (n={len(f)})  gap {at-af:+.3f}")
