"""Test 2: (a) does D1/D2 predict judge correctness? (b) budget-matched comparison
(c) per-category attribution of the recall gain (d) McNemar significance (e) subset fragility."""
import json, pickle, re, sys, os, statistics as st, itertools, math
from collections import Counter
sys.path.insert(0, "/root/agent_memory_iclr/code")
rows = json.load(open("/root/agent_memory_iclr/verify/adv_hyp_rows.json"))
n = len(rows); tgs = sum(r["ngs"] for r in rows)

def d1(r, t): return r[t]["items"] >= r["gnum"] and r[t]["gold_hit"] == r["ngs"]
def d2(r, t): return r[t]["best"] >= r["gnum"] and r[t]["gold_hit"] == r["ngs"]

print("=== (a) does D1/D2 predict answer correctness (v4 judge, n=60)? ===")
for t in ["v5", "nohyp"]:
    for name, f in [("D1", d1), ("D2", d2)]:
        tp = sum(1 for r in rows if f(r, t) and r["judge"])
        fp = sum(1 for r in rows if f(r, t) and not r["judge"])
        fn = sum(1 for r in rows if not f(r, t) and r["judge"])
        tn = sum(1 for r in rows if not f(r, t) and not r["judge"])
        # Fisher exact 2x2
        def C(a,b): return math.comb(a,b)
        rowt, rowf = tp+fp, fn+tn; colc, colw = tp+fn, fp+tn; N = n
        p = 0.0
        for k in range(0, min(rowt, colc)+1):
            if rowt-k < 0 or colc-k < 0 or rowf-(colc-k) < 0: continue
            pk = C(colc,k)*C(colw,rowt-k)/C(N,rowt) if rowt-k <= colw else 0
            if pk <= C(colc,tp)*C(colw,fp)/C(N,rowt) + 1e-12: p += pk
        print(f"{t:6s} {name}: judged-correct | proxy=True {tp}/{tp+fp}={tp/max(1,tp+fp):.3f}   "
              f"| proxy=False {fn}/{fn+tn}={fn/max(1,fn+tn):.3f}   Fisher p={p:.3f}")

print("\n=== (b) McNemar on the with/without-hyponym flips ===")
for name, f in [("D1", d1), ("D2", d2)]:
    b = sum(1 for r in rows if f(r,"v5") and not f(r,"nohyp"))
    c = sum(1 for r in rows if f(r,"nohyp") and not f(r,"v5"))
    # exact binomial two-sided
    m = b+c
    p = sum(math.comb(m,k) for k in range(0, min(b,c)+1))/2**m*2 if m else 1.0
    p = min(1.0, p)
    print(f"{name}: gains {b}, losses {c}, net {b-c} of {n}; exact McNemar p={p:.3f}")

print("\n=== (c) recall accounting ===")
for t in ["v5","nohyp","nohand","nomerge"]:
    print(f"{t:8s} gold-sessions hit {sum(r[t]['gold_hit'] for r in rows)}/{tgs} "
          f"= {sum(r[t]['gold_hit'] for r in rows)/tgs:.4f}")
gain = [(r["qid"], r["nohyp"]["gold_hit"], r["v5"]["gold_hit"], r["ngs"])
        for r in rows if r["v5"]["gold_hit"] != r["nohyp"]["gold_hit"]]
print("per-question gold-session delta v5 vs nohyp:")
for q,a,b,g in gain: print(f"   {q:15s} {a}->{b} of {g}  ({'+' if b>a else '-'}{abs(b-a)})")
print("\n=== (d) item-count inflation (D1 is monotone in item count) ===")
it5 = [r["v5"]["items"] for r in rows]; itn = [r["nohyp"]["items"] for r in rows]
print(f"items with table:  median {st.median(it5):.1f} mean {st.mean(it5):.1f} sum {sum(it5)}")
print(f"items without:     median {st.median(itn):.1f} mean {st.mean(itn):.1f} sum {sum(itn)}")
print(f"ratio of totals {sum(it5)/sum(itn):.2f}x ; median gold count "
      f"{st.median([r['gnum'] for r in rows]):.1f}")
bigger = sum(1 for r in rows if r["v5"]["items"] > r["nohyp"]["items"])
print(f"questions where the table strictly enlarges the printed set: {bigger}/{n}")
