"""Independent adversarial recomputation of the 'nondeterminism floor' claim.
Own implementations of tokenF1, Wilson, Fisher, McNemar exact. No reuse of a3_flip.py.
"""
import json, math, re, difflib, os, sys, hashlib
from collections import Counter
from fractions import Fraction

R = "/root/agent_memory_iclr/results/"

def load(t):
    d = {}
    for line in open(R + f"full500_{t}.jsonl"):
        r = json.loads(line)
        if "judge" in r:          # only successful rows; last wins
            d[r["qid"]] = r
    return d

qc, nq = load("qc"), load("qc_no_qc")
assert set(qc) == set(nq), (len(qc), len(nq))
qids = sorted(qc)
print(f"n rows qc={len(qc)} no_qc={len(nq)} common={len(qids)}")

# ---------- own token F1 (SQuAD style, matches paper's stated normalisation) ----------
WORD = re.compile(r"[A-Za-z0-9]+")
STOP = {"a", "an", "the"}
def tok(s):
    return [w for w in WORD.findall(str(s).lower()) if w not in STOP]
def tf1(a, b):
    A, B = tok(a), tok(b)
    if not A or not B:
        return 0.0
    ca, cb = Counter(A), Counter(B)
    ov = sum(min(v, cb[k]) for k, v in ca.items())
    if ov == 0:
        return 0.0
    p, r = ov / len(A), ov / len(B)
    return 2 * p * r / (p + r)

# ---------- own stats ----------
def wilson(k, n, z=1.959963985):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def fisher2x2(a, b, c, d):
    """two-sided Fisher, exact rationals, point-probability method."""
    n = a + b + c + d; r1 = a + b; r2 = c + d; c1 = a + c
    def P(x):
        return Fraction(math.comb(r1, x) * math.comb(r2, c1 - x), math.comb(n, c1))
    p0 = P(a); tot = Fraction(0)
    for x in range(max(0, c1 - r2), min(r1, c1) + 1):
        px = P(x)
        if px <= p0:
            tot += px
    return min(1.0, float(tot))

def mcnemar_exact(b, c):
    """two-sided exact binomial on discordant pairs."""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)

def bino_ci(k, n):  # Clopper-Pearson
    from math import comb
    def cdf(p, k, n): return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))
    lo, hi = 0.0, 1.0
    if k > 0:
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            if 1 - cdf(m, k - 1, n) > 0.025: b = m
            else: a = m
        lo = (a + b) / 2
    if k < n:
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            if cdf(m, k, n) < 0.025: b = m
            else: a = m
        hi = (a + b) / 2
    return lo, hi

# ---------- strata ----------
same = [q for q in qids if qc[q]["view_mode"] == nq[q]["view_mode"]]
diff = [q for q in qids if qc[q]["view_mode"] != nq[q]["view_mode"]]
print(f"\nview_mode strata: same={len(same)} diff={len(diff)}")
print("  qc view_mode dist   :", dict(Counter(qc[q]["view_mode"] for q in qids)))
print("  no_qc view_mode dist:", dict(Counter(nq[q]["view_mode"] for q in qids)))
print("  qc qlabel dist      :", dict(Counter(qc[q]["qlabel"] for q in qids)))
print("  same-set qlabels    :", dict(Counter(qc[q]["qlabel"] for q in same)))

for name, ids in [("IDENTICAL-MODE", same), ("ROUTED-DIFFERENT", diff)]:
    ex = sum(1 for q in ids if qc[q]["pred"].strip() == nq[q]["pred"].strip())
    sims = [tf1(qc[q]["pred"], nq[q]["pred"]) for q in ids]
    rat = [difflib.SequenceMatcher(None, qc[q]["pred"], nq[q]["pred"]).ratio() for q in ids]
    jf = sum(1 for q in ids if qc[q]["judge"] != nq[q]["judge"])
    ff = sum(1 for q in ids if abs(qc[q]["f1"] - nq[q]["f1"]) > 1e-9)
    hi = sum(1 for s in sims if s > 0.9)
    n01 = sum(1 for q in ids if (not nq[q]["judge"]) and qc[q]["judge"])   # no_qc wrong -> qc right
    n10 = sum(1 for q in ids if nq[q]["judge"] and (not qc[q]["judge"]))
    ss = sorted(sims)
    med = ss[len(ss)//2] if len(ss) % 2 else (ss[len(ss)//2-1]+ss[len(ss)//2])/2
    print(f"\n[{name}] n={len(ids)}")
    print(f"  exact pred match   {ex}/{len(ids)} = {ex/len(ids):.4f}")
    print(f"  mean pred-pred F1  {sum(sims)/len(sims):.4f}  median {med:.4f}")
    print(f"  mean SeqMatcher    {sum(rat)/len(rat):.4f}")
    print(f"  pairs F1>0.9       {hi}/{len(ids)}")
    print(f"  gold-F1 differs    {ff}/{len(ids)} = {ff/len(ids):.4f}")
    print(f"  JUDGE FLIP         {jf}/{len(ids)} = {jf/len(ids):.4f}  Wilson{tuple(round(x,4) for x in wilson(jf,len(ids)))}"
          f"  CP{tuple(round(x,4) for x in bino_ci(jf,len(ids)))}")
    print(f"  discordant n01(no_qc0->qc1)={n01}  n10={n10}  net={n01-n10:+d}  McNemar exact p={mcnemar_exact(n01,n10):.4f}")

fs = sum(1 for q in same if qc[q]["judge"] != nq[q]["judge"])
fd = sum(1 for q in diff if qc[q]["judge"] != nq[q]["judge"])
print(f"\nFisher exact flip-rate 2x2 [[{fd},{len(diff)-fd}],[{fs},{len(same)-fs}]] p={fisher2x2(fd,len(diff)-fd,fs,len(same)-fs):.4f}")

# accuracy of each arm per stratum
for name, ids in [("same", same), ("diff", diff), ("all", qids)]:
    aq = sum(qc[q]["judge"] for q in ids) / len(ids)
    an = sum(nq[q]["judge"] for q in ids) / len(ids)
    print(f"  acc[{name:5s}] qc={aq:.4f} no_qc={an:.4f} delta={aq-an:+.4f} (n={len(ids)})")

# ---------- judge noise vs generation noise ----------
print("\n== judge behaviour on byte-identical predictions (same stratum) ==")
idp = [q for q in same if qc[q]["pred"].strip() == nq[q]["pred"].strip()]
print(f"  identical preds: {len(idp)}, judge disagreements among them: "
      f"{sum(1 for q in idp if qc[q]['judge']!=nq[q]['judge'])}")
idp_all = [q for q in qids if qc[q]["pred"].strip() == nq[q]["pred"].strip()]
print(f"  identical preds over all 500: {len(idp_all)}, judge disagreements: "
      f"{sum(1 for q in idp_all if qc[q]['judge']!=nq[q]['judge'])}")

print("\n== the identical-mode judge flips ==")
for q in same:
    if qc[q]["judge"] != nq[q]["judge"]:
        print(f"  {q} type={qc[q]['type']} qc.judge={qc[q]['judge']} nq.judge={nq[q]['judge']} "
              f"predF1={tf1(qc[q]['pred'],nq[q]['pred']):.3f} f1 qc={qc[q]['f1']} nq={nq[q]['f1']}")

# ---------- stratum comparability (confound check) ----------
print("\n== stratum composition (is 'same' a random subsample?) ==")
ts, td = Counter(qc[q]["type"] for q in same), Counter(qc[q]["type"] for q in diff)
alltypes = sorted(set(ts) | set(td))
print(f"  {'type':30s} same        diff")
for t in alltypes:
    print(f"  {t:30s} {ts.get(t,0):3d}({ts.get(t,0)/len(same):.3f}) {td.get(t,0):3d}({td.get(t,0)/len(diff):.3f})")
# chi2-ish: fisher on type composition would need RxC; report accuracy + pred length instead
for name, ids in [("same", same), ("diff", diff)]:
    L = [len(tok(qc[q]["pred"])) for q in ids]
    print(f"  {name}: mean pred tokens {sum(L)/len(L):.1f}, mean gold-F1 qc {sum(qc[q]['f1'] for q in ids)/len(ids):.3f}, "
          f"acc {sum(qc[q]['judge'] for q in ids)/len(ids):.3f}")
