"""Flip-set analysis + McNemar + budget accounting. Zero API."""
import json, itertools, math, statistics as st
from collections import Counter

R = "/root/agent_memory_iclr/results"
D = json.load(open(f"{R}/_ver_repro.json"))
pq = D["per_q"]
judge = {}
for arm in ("qc", "full", "qc_view_only", "qc_no_lifecycle", "full_p2", "qc_no_qc"):
    judge[arm] = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{R}/full500_{arm}.jsonl")}

FIN, OFF = pq["final"], pq["no_intext"]
SHUF = [pq[f"shuffle_{k}"] for k in range(3)]
JIT = [pq[f"jitter_{k}"] for k in range(3)]

delta_q = [k for k, v in FIN.items() if v["fam"] in ("delta_between", "delta_since")
           and OFF[k]["fam"] in ("delta_between", "delta_since")]
# keep only those with gold_vals (dTL defined) -> proxy: metric present in both
delta_q = [k for k in delta_q if FIN[k]["dTL"] is not None]
ord_q = [k for k, v in FIN.items() if v["fam"] == "ordering"]
print("n_delta", len(delta_q), "n_ord", len(ord_q))


def mcnemar(a, b):
    """exact two-sided McNemar on paired booleans."""
    n01 = sum(1 for x, y in zip(a, b) if (not x) and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return n10, n01, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return n10, n01, min(1.0, p)


# ---------- delta family: ON vs OFF, ON vs shuffle
fin = [bool(FIN[k]["dTL"]) for k in delta_q]
off = [bool(OFF[k]["dTL"]) for k in delta_q]
print(f"deltaTL  final={sum(fin)}/{len(fin)}={sum(fin)/len(fin):.4f}  "
      f"off={sum(off)}/{len(off)}={sum(off)/len(off):.4f}")
n10, n01, p = mcnemar(fin, off)
print(f"  McNemar final-vs-off: gained={n10} lost={n01} exact p={p:.4g}")
for i, S in enumerate(SHUF):
    sh = [bool(S[k]["dTL"]) for k in delta_q]
    n10s, n01s, ps = mcnemar(fin, sh)
    print(f"  vs shuffle_{i}: shuffle={sum(sh)}/{len(sh)}={sum(sh)/len(sh):.4f} "
          f"gained={n10s} lost={n01s} p={ps:.4g}")
for i, S in enumerate(JIT):
    sh = [bool(S[k]["dTL"]) for k in delta_q]
    print(f"  vs jitter_{i}:  jitter={sum(sh)}/{len(sh)}={sum(sh)/len(sh):.4f}")

flip = [k for k in delta_q if FIN[k]["dTL"] and not OFF[k]["dTL"]]
back = [k for k in delta_q if OFF[k]["dTL"] and not FIN[k]["dTL"]]
print(f"\ndelta flip ON-only n={len(flip)}  OFF-only n={len(back)}")
# how many of the flip set survive ALL 3 shuffle controls (i.e. random dates do NOT get them)
surv = [k for k in flip if not any(S[k]["dTL"] for S in SHUF)]
print(f"  flip questions NOT reproduced by any random-offset control: {len(surv)} -> {surv}")
print(f"  flip questions reproduced by >=1 random control: {len(flip)-len(surv)}")

# ---------- end-task headroom on the flip set
for arm in ("qc", "full", "qc_view_only"):
    J = judge[arm]
    got = [k for k in flip if J.get(k) and J[k]["judge"]]
    print(f"  [{arm}] deployed pipeline already CORRECT on {len(got)}/{len(flip)} flip questions")
    wrong = [k for k in flip if J.get(k) and not J[k]["judge"]]
    print(f"       still wrong: {wrong}")

# ---------- ordering
fo = [bool(FIN[k]["ordc"]) for k in ord_q]
oo = [bool(OFF[k]["ordc"]) for k in ord_q]
print(f"\nord_correct_of_all final={sum(fo)}/{len(fo)}={sum(fo)/len(fo):.4f} "
      f"off={sum(oo)}/{len(oo)}={sum(oo)/len(oo):.4f}")
n10, n01, p = mcnemar(fo, oo)
print(f"  McNemar final-vs-off: gained={n10} lost={n01} exact p={p:.4g}")
for i, S in enumerate(SHUF):
    sh = [bool(S[k]["ordc"]) for k in ord_q]
    n10s, n01s, ps = mcnemar(fo, sh)
    print(f"  vs shuffle_{i}: {sum(sh)}/{len(sh)}={sum(sh)/len(sh):.4f} gained={n10s} lost={n01s} p={ps:.4g}")
oflip = [k for k in ord_q if FIN[k]["ordc"] and not OFF[k]["ordc"]]
osurv = [k for k in oflip if not any(bool(S[k]["ordc"]) for S in SHUF)]
print(f"  ord flip ON-only n={len(oflip)}, not reproduced by any random control: {len(osurv)} {osurv}")
J = judge["qc"]
print(f"  [qc] already correct on {sum(1 for k in oflip if J[k]['judge'])}/{len(oflip)} ord-flip questions")

# same_day_split: is it correct when it fires?
sds_fin = [k for k in ord_q if FIN[k]["sds"]]
print(f"\nsame_day_split fires final={len(sds_fin)} ; random-control arms: "
      + ", ".join(str(sum(1 for k in ord_q if S[k]['sds'])) for S in SHUF + JIT))
print("  of the questions where split fires, ordering verdict:",
      Counter(str(FIN[k]["ordc"]) for k in sds_fin))

# ---------- budget accounting, per family
print("\nchars by family (median):")
fams = sorted({v["fam"] for v in FIN.values()})
for f in fams:
    ks = [k for k in FIN if FIN[k]["fam"] == f]
    a = st.median([FIN[k]["chars"] for k in ks])
    b = st.median([OFF[k]["chars"] for k in ks])
    na = st.median([FIN[k]["ndates"] for k in ks])
    nb = st.median([OFF[k]["ndates"] for k in ks])
    print(f"  {f:15s} n={len(ks):3d}  ON={a:7.0f}  OFF={b:7.0f}  ratio={a/max(1,b):.2f}   "
          f"distinct tl dates ON={na:.1f} OFF={nb:.1f}")

# ---------- deltaTL vs number of distinct dates (chance exposure)
print("\ndeltaTL rate by n distinct timeline dates (pooled over final+3 shuffles), delta family:")
buck = {}
for name, P in [("final", FIN)] + [(f"shuffle_{i}", S) for i, S in enumerate(SHUF)]:
    for k in delta_q:
        b = min(P[k]["ndates"], 9)
        s = buck.setdefault(b, [0, 0])
        s[0] += bool(P[k]["dTL"]); s[1] += 1
for b in sorted(buck):
    h, n = buck[b]
    print(f"  ndates={b}{'+' if b==9 else ''}: {h}/{n} = {h/n:.3f}")

# ---------- proxy validity: does dTL predict judged correctness?
print("\nproxy validity (deployed arms, delta family):")
for arm in ("qc", "full", "qc_no_qc", "qc_view_only"):
    J = judge[arm]
    a1 = [k for k in delta_q if FIN[k]["dTL"]]
    a0 = [k for k in delta_q if not FIN[k]["dTL"]]
    r1 = sum(bool(J[k]["judge"]) for k in a1) / len(a1)
    r0 = sum(bool(J[k]["judge"]) for k in a0) / len(a0)
    print(f"  {arm:14s} dTL=1 acc={r1:.3f} (n={len(a1)})   dTL=0 acc={r0:.3f} (n={len(a0)})  diff={r1-r0:+.3f}")
