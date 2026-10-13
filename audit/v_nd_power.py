"""Power / equivalence / confound analysis of the p=1.000 'no difference' inference."""
import json, math, random, re
from collections import Counter
from fractions import Fraction

R = "/root/agent_memory_iclr/results/"
def load(t):
    d = {}
    for line in open(R + f"full500_{t}.jsonl"):
        r = json.loads(line)
        if "judge" in r: d[r["qid"]] = r
    return d
qc, nq = load("qc"), load("qc_no_qc")
qids = sorted(qc)
same = [q for q in qids if qc[q]["view_mode"] == nq[q]["view_mode"]]
diff = [q for q in qids if qc[q]["view_mode"] != nq[q]["view_mode"]]
flip = lambda q: qc[q]["judge"] != nq[q]["judge"]
fs, fd = sum(map(flip, same)), sum(map(flip, diff))
n2, n1 = len(same), len(diff)
print(f"floor {fs}/{n2}={fs/n2:.4f}   routed {fd}/{n1}={fd/n1:.4f}")

# pred truncation sanity
mx = max(max(len(qc[q]["pred"]) for q in qids), max(len(nq[q]["pred"]) for q in qids))
print("max pred chars:", mx, "(truncation cap 4000) -> truncation artefacts:", mx >= 4000)

# ---------------- Fisher (own, exact rationals) ----------------
_fc = {}
def fisher(a, b, c, d):
    key = (a, b, c, d)
    if key in _fc: return _fc[key]
    n = a+b+c+d; r1=a+b; r2=c+d; c1=a+c
    den = math.comb(n, c1)
    def P(x): return Fraction(math.comb(r1,x)*math.comb(r2,c1-x), den)
    p0 = P(a); tot = Fraction(0)
    for x in range(max(0,c1-r2), min(r1,c1)+1):
        px = P(x)
        if px <= p0: tot += px
    v = min(1.0, float(tot)); _fc[key] = v; return v
print("observed Fisher p =", round(fisher(fd, n1-fd, fs, n2-fs), 4))

# ---------------- Newcombe hybrid-score CI for difference of proportions ----------------
def wilson(k, n, z=1.959963985):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0.,c-h), min(1.,c+h)
l1,u1 = wilson(fd,n1); l2,u2 = wilson(fs,n2)
d_hat = fd/n1 - fs/n2
print(f"\nrouted-minus-floor difference = {d_hat:+.4f}")
print(f"  Newcombe 95% CI = [{d_hat - math.sqrt((fd/n1-l1)**2+(u2-fs/n2)**2):+.4f}, "
      f"{d_hat + math.sqrt((u1-fd/n1)**2+(fs/n2-l2)**2):+.4f}]")

# ---------------- Bayesian posterior on the difference (Jeffreys priors) ----------------
random.seed(7)
def beta_rv(a,b): return random.betavariate(a,b)
N = 400000
d_s = [beta_rv(fd+0.5, n1-fd+0.5) - beta_rv(fs+0.5, n2-fs+0.5) for _ in range(N)]
d_s.sort()
print(f"  posterior median diff {d_s[N//2]:+.4f}, 95% CrI [{d_s[int(.025*N)]:+.4f}, {d_s[int(.975*N)]:+.4f}]")
print(f"  P(routed rate > floor rate | data) = {sum(1 for x in d_s if x>0)/N:.3f}")
print(f"  P(routed rate > 2x floor rate)     = "
      f"{sum(1 for _ in range(0)) or ''}", end="")
r_s = [beta_rv(fd+0.5, n1-fd+0.5)/max(1e-9, beta_rv(fs+0.5, n2-fs+0.5)) for _ in range(N)]
r_s.sort()
print(f"{sum(1 for x in r_s if x>2)/N:.3f}   ratio median {r_s[N//2]:.2f}, "
      f"95% CrI [{r_s[int(.025*N)]:.2f}, {r_s[int(.975*N)]:.2f}]")

# ---------------- Power of the Fisher design ----------------
print("\n== Monte-Carlo power of the Fisher test, floor arm n=44, routed arm n=456 ==")
print("   (true floor pi2 fixed at 0.0455, alpha=0.05 two-sided, 4000 reps)")
def binom(n,p): return sum(1 for _ in range(n) if random.random()<p)
random.seed(11)
REPS = 4000
for pi1 in [0.048, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    hit = 0
    for _ in range(REPS):
        a = binom(n1, pi1); c = binom(n2, 0.0455)
        if fisher(a, n1-a, c, n2-c) < 0.05: hit += 1
    print(f"   true routed rate {pi1:.3f} (excess {pi1-0.0455:+.3f}) -> power {hit/REPS:.3f}")

# ---------------- confound: is the floor stratum comparable? ----------------
print("\n== stratum-mix confound: type-conditional flip rates (routed stratum) ==")
tf = {}
for q in diff:
    t = qc[q]["type"]; s = tf.setdefault(t, [0,0]); s[0]+=1; s[1]+= flip(q)
mix_same = Counter(qc[q]["type"] for q in same)
exp = sum(mix_same[t]*(tf[t][1]/tf[t][0]) for t in mix_same if t in tf) / sum(mix_same[t] for t in mix_same if t in tf)
print(f"   {'type':30s} routed n  flips  rate     same-stratum n")
for t in sorted(tf):
    print(f"   {t:30s} {tf[t][0]:5d} {tf[t][1]:5d}  {tf[t][1]/tf[t][0]:.4f}   {mix_same.get(t,0):4d}")
print(f"   routed overall flip rate                        {fd/n1:.4f}")
print(f"   routed flip rate REWEIGHTED to same-stratum type mix: {exp:.4f}")
print(f"   -> the two rates being compared are drawn from different question mixes")

# ---------------- what the whole-500 paired test says ----------------
n01 = sum(1 for q in qids if not nq[q]["judge"] and qc[q]["judge"])
n10 = sum(1 for q in qids if nq[q]["judge"] and not qc[q]["judge"])
def mcn(b,c):
    n=b+c; k=min(b,c)
    return min(1.0, 2*sum(math.comb(n,i) for i in range(k+1))/2**n)
print(f"\n== paired whole-500: n01={n01} n10={n10} net={n01-n10:+d} "
      f"({(n01-n10)/500:+.4f} acc) McNemar exact p={mcn(n01,n10):.4f}")
print("   net delta by stratum:")
for nm, ids in [("identical-prompt", same), ("routed-different", diff)]:
    a = sum(1 for q in ids if not nq[q]["judge"] and qc[q]["judge"])
    b = sum(1 for q in ids if nq[q]["judge"] and not qc[q]["judge"])
    print(f"     {nm:18s} n01={a} n10={b} net={a-b:+d}  (contributes {(a-b)/500:+.4f} to the 500-question delta)")
# MDE for McNemar with 24 discordant
for nd in [24, 22]:
    m = next(mm for mm in range(nd//2, nd+1) if mcn(nd-mm, mm) < 0.05)
    print(f"   MDE with {nd} discordant pairs: need {m}/{nd-m} split -> net {2*m-nd} q = {(2*m-nd)/500:.4f} acc")
