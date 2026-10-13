#!/usr/bin/env python3
"""Main-table + statistical rigor analysis for QC-Lifecycle full-500 run.

Self-contained: exact McNemar (binomial) implemented here; numpy used only for
bootstrap resampling speed. scipy used ONLY as an independent cross-check of the
hand-rolled exact p-values (never as the reported source).
"""
import json, math, os, collections
import numpy as np

RES = "/root/agent_memory_iclr/results"
ARMS = ["full", "decay", "fifo", "qc_no_lifecycle", "qc", "qc_no_qc", "qc_view_only"]
LABEL = {
    "full": "Full (no forgetting, BM25 k=30)",
    "decay": "Time decay (MemoryBank-style)",
    "fifo": "FIFO eviction (MemoryOS-style)",
    "qc_no_lifecycle": "BM25-only k=20 (attribution control)",
    "qc": "QC-Lifecycle (ours)",
    "qc_no_qc": "-QC (view always historical)",
    "qc_view_only": "-RAW (view only, no raw turns)",
}
TYPES = ["single-session-user", "single-session-assistant", "single-session-preference",
         "multi-session", "knowledge-update", "temporal-reasoning"]
SHORT = {"single-session-user": "SSU", "single-session-assistant": "SSA",
         "single-session-preference": "SSP", "multi-session": "MS",
         "knowledge-update": "KU", "temporal-reasoning": "TR"}

# ---------------------------------------------------------------- load
D = {}
for a in ARMS:
    rows = [json.loads(l) for l in open(f"{RES}/full500_{a}.jsonl")]
    D[a] = {r["qid"]: r for r in rows}
QIDS = sorted(D["qc"].keys())
assert all(set(D[a]) == set(QIDS) for a in ARMS)
TYPE = {q: D["qc"][q]["type"] for q in QIDS}
for a in ARMS:
    assert all(D[a][q]["type"] == TYPE[q] for q in QIDS), f"type mismatch {a}"
QLABEL = {q: D["qc"][q]["qlabel"] for q in QIDS}
for a in ("qc_no_qc", "qc_view_only"):
    assert all(D[a][q]["qlabel"] == QLABEL[q] for q in QIDS), "qlabel mismatch"

ABS = [q for q in QIDS if q.endswith("_abs")]
NONABS = [q for q in QIDS if not q.endswith("_abs")]

J = {a: {q: bool(D[a][q]["judge"]) for q in QIDS} for a in ARMS}
F = {a: {q: float(D[a][q]["f1"]) for q in QIDS} for a in ARMS}

def acc(a, qs):
    return sum(J[a][q] for q in qs) / len(qs) if qs else float("nan")
def f1m(a, qs):
    return sum(F[a][q] for q in qs) / len(qs) if qs else float("nan")

# ---------------------------------------------------------------- stats
def mcnemar_exact(b, c):
    """Two-sided exact (binomial, p=0.5) McNemar on discordant pairs.
    b = ours right / theirs wrong ; c = ours wrong / theirs right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)

def mcnemar_pair(arm_a, arm_b, qs):
    """arm_a = ours. Returns dict with discordants, p, deltas."""
    b = sum(1 for q in qs if J[arm_a][q] and not J[arm_b][q])
    c = sum(1 for q in qs if not J[arm_a][q] and J[arm_b][q])
    return {"n": len(qs), "b": b, "c": c, "p": mcnemar_exact(b, c),
            "acc_a": acc(arm_a, qs), "acc_b": acc(arm_b, qs),
            "delta": acc(arm_a, qs) - acc(arm_b, qs)}

RNG = np.random.default_rng(20260917)
B_REPS = 20000

def boot_ci_diff(arm_a, arm_b, qs, metric="judge", reps=B_REPS, alpha=0.05):
    """Paired percentile bootstrap over questions for mean(a) - mean(b)."""
    if metric == "judge":
        xa = np.array([1.0 if J[arm_a][q] else 0.0 for q in qs])
        xb = np.array([1.0 if J[arm_b][q] else 0.0 for q in qs])
    else:
        xa = np.array([F[arm_a][q] for q in qs])
        xb = np.array([F[arm_b][q] for q in qs])
    d = xa - xb
    n = len(d)
    idx = RNG.integers(0, n, size=(reps, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)

def boot_ci_acc(arm, qs, reps=B_REPS, alpha=0.05):
    x = np.array([1.0 if J[arm][q] else 0.0 for q in qs])
    n = len(x)
    idx = RNG.integers(0, n, size=(reps, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)

def holm(pairs):
    """pairs: list of (key, p). Returns dict key -> (p_adj, rank)."""
    m = len(pairs)
    order = sorted(pairs, key=lambda kv: kv[1])
    out, running = {}, 0.0
    for i, (k, p) in enumerate(order):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)  # enforce monotonicity
        out[k] = (running, i + 1)
    return out

# ---------------------------------------------------------------- cross-check
try:
    from scipy.stats import binomtest
    _xchk = []
    for (b, c) in [(1, 0), (5, 2), (30, 9), (0, 0), (12, 12), (44, 21)]:
        mine = mcnemar_exact(b, c)
        if b + c == 0:
            theirs = 1.0
        else:
            theirs = binomtest(b, b + c, 0.5, alternative="two-sided").pvalue
        _xchk.append((b, c, mine, theirs, abs(mine - theirs) < 1e-12))
    XCHK = _xchk
except Exception as e:  # pragma: no cover
    XCHK = [("scipy unavailable", str(e))]

# ---------------------------------------------------------------- report data
OUT = {}
OUT["counts"] = {"total": len(QIDS), "abs": len(ABS), "nonabs": len(NONABS),
                 "per_type_all": {t: sum(1 for q in QIDS if TYPE[q] == t) for t in TYPES},
                 "per_type_nonabs": {t: sum(1 for q in NONABS if TYPE[q] == t) for t in TYPES},
                 "per_type_abs": {t: sum(1 for q in ABS if TYPE[q] == t) for t in TYPES},
                 "qlabel": dict(collections.Counter(QLABEL.values()))}

OUT["overall"] = {}
for a in ARMS:
    m, lo, hi = boot_ci_acc(a, QIDS)
    m4, lo4, hi4 = boot_ci_acc(a, NONABS)
    OUT["overall"][a] = {
        "acc500": acc(a, QIDS), "acc500_ci": (lo, hi), "f1_500": f1m(a, QIDS),
        "acc470": acc(a, NONABS), "acc470_ci": (lo4, hi4), "f1_470": f1m(a, NONABS),
        "acc_abs": acc(a, ABS), "f1_abs": f1m(a, ABS),
    }

OUT["bytype"] = {a: {t: {"n": sum(1 for q in NONABS if TYPE[q] == t),
                         "acc": acc(a, [q for q in NONABS if TYPE[q] == t]),
                         "f1": f1m(a, [q for q in NONABS if TYPE[q] == t])}
                     for t in TYPES} for a in ARMS}
OUT["bytype_abs"] = {a: {t: {"n": sum(1 for q in ABS if TYPE[q] == t),
                             "acc": acc(a, [q for q in ABS if TYPE[q] == t]),
                             "f1": f1m(a, [q for q in ABS if TYPE[q] == t])}
                         for t in TYPES if sum(1 for q in ABS if TYPE[q] == t) > 0}
                     for a in ARMS}

# arm-level paired tests (family A)
COMPS = ["full", "qc_no_lifecycle", "decay", "fifo", "qc_no_qc", "qc_view_only"]
OUT["arm_tests"] = {}
for b_arm in COMPS:
    r = mcnemar_pair("qc", b_arm, QIDS)
    d, lo, hi = boot_ci_diff("qc", b_arm, QIDS, "judge")
    df, lof, hif = boot_ci_diff("qc", b_arm, QIDS, "f1")
    r.update({"boot": (d, lo, hi), "f1_delta": df, "f1_ci": (lof, hif),
              "f1_a": f1m("qc", QIDS), "f1_b": f1m(b_arm, QIDS)})
    r470 = mcnemar_pair("qc", b_arm, NONABS)
    d7, lo7, hi7 = boot_ci_diff("qc", b_arm, NONABS, "judge")
    r["n470"] = r470
    r["boot470"] = (d7, lo7, hi7)
    OUT["arm_tests"][b_arm] = r

famA = holm([(k, v["p"]) for k, v in OUT["arm_tests"].items()])
for k, (padj, rank) in famA.items():
    OUT["arm_tests"][k]["p_holm"] = padj
    OUT["arm_tests"][k]["holm_rank"] = rank

# per-subtype paired tests (families B, C, D)
FOCUS = ["knowledge-update", "temporal-reasoning", "multi-session", "single-session-preference"]
OUT["subtype_tests"] = {}
for ref in ["full", "qc_no_lifecycle", "decay", "qc_no_qc", "qc_view_only"]:
    fam = {}
    for t in FOCUS:
        qs = [q for q in NONABS if TYPE[q] == t]
        r = mcnemar_pair("qc", ref, qs)
        d, lo, hi = boot_ci_diff("qc", ref, qs, "judge")
        r["boot"] = (d, lo, hi)
        fam[t] = r
    h = holm([(t, fam[t]["p"]) for t in FOCUS])
    for t in FOCUS:
        fam[t]["p_holm"], fam[t]["holm_rank"] = h[t]
    OUT["subtype_tests"][ref] = fam

# all six types vs full and vs qc_no_lifecycle (for completeness / neutral check)
OUT["alltype_tests"] = {}
for ref in ["full", "qc_no_lifecycle"]:
    fam = {}
    for t in TYPES:
        qs = [q for q in NONABS if TYPE[q] == t]
        r = mcnemar_pair("qc", ref, qs)
        d, lo, hi = boot_ci_diff("qc", ref, qs, "judge")
        r["boot"] = (d, lo, hi)
        fam[t] = r
    h = holm([(t, fam[t]["p"]) for t in TYPES])
    for t in TYPES:
        fam[t]["p_holm"], fam[t]["holm_rank"] = h[t]
    OUT["alltype_tests"][ref] = fam

# best-baseline per subtype (max over the four non-ours, non-ablation arms)
BASES = ["full", "decay", "fifo", "qc_no_lifecycle"]
OUT["best_baseline"] = {}
for t in TYPES:
    qs = [q for q in NONABS if TYPE[q] == t]
    best = max(BASES, key=lambda a: acc(a, qs))
    r = mcnemar_pair("qc", best, qs)
    d, lo, hi = boot_ci_diff("qc", best, qs, "judge")
    OUT["best_baseline"][t] = {"arm": best, "acc": acc(best, qs), "ours": acc("qc", qs),
                               "delta": acc("qc", qs) - acc(best, qs), "p": r["p"],
                               "b": r["b"], "c": r["c"], "n": len(qs), "boot": (d, lo, hi)}
for t in TYPES:
    qs = [q for q in ABS if TYPE[q] == t]
    if qs:
        best = max(BASES, key=lambda a: acc(a, qs))
        OUT.setdefault("best_baseline_abs", {})[t] = {
            "arm": best, "acc": acc(best, qs), "ours": acc("qc", qs), "n": len(qs)}

# qlabel-conditioned view (the classifier's own partition)
OUT["byqlabel"] = {}
for ql in ["current", "historical", "neutral"]:
    qs = [q for q in QIDS if QLABEL[q] == ql]
    cell = {"n": len(qs)}
    for a in ARMS:
        cell[a] = {"acc": acc(a, qs), "f1": f1m(a, qs)}
    for ref in ["full", "qc_no_lifecycle", "qc_no_qc"]:
        r = mcnemar_pair("qc", ref, qs)
        d, lo, hi = boot_ci_diff("qc", ref, qs, "judge")
        r["boot"] = (d, lo, hi)
        cell[f"vs_{ref}"] = r
    OUT["byqlabel"][ql] = cell

# judge/F1 disagreement diagnostics
OUT["diag"] = {}
for a in ARMS:
    j_true_lowf1 = sum(1 for q in QIDS if J[a][q] and F[a][q] < 0.2)
    j_false_hif1 = sum(1 for q in QIDS if not J[a][q] and F[a][q] > 0.6)
    OUT["diag"][a] = {"judge_true_f1_lt0.2": j_true_lowf1, "judge_false_f1_gt0.6": j_false_hif1}

# view_mode routing audit for qc
vm = collections.Counter((QLABEL[q], D["qc"][q]["view_mode"]) for q in QIDS)
OUT["routing"] = {f"{k[0]}->{k[1]}": v for k, v in sorted(vm.items())}
# accuracy by (qlabel, correct?) for the 44 historical-routed questions
hq = [q for q in QIDS if D["qc"][q]["view_mode"] == "historical"]
OUT["hist_routed"] = {"n": len(hq), "qc": acc("qc", hq), "full": acc("full", hq),
                      "qc_no_lifecycle": acc("qc_no_lifecycle", hq),
                      "types": dict(collections.Counter(TYPE[q] for q in hq))}

with open(f"{RES}/main_table_stats.json", "w") as fh:
    json.dump(OUT, fh, indent=1, default=str)

# ---------------------------------------------------------------- console
def pf(x, d=3):
    return "n/a" if x != x else f"{x:.{d}f}"

print("=== cross-check hand-rolled exact McNemar vs scipy.binomtest ===")
for row in XCHK:
    print("  ", row)
print(f"\ncounts: {OUT['counts']}\n")
print("=== overall ===")
for a in ARMS:
    o = OUT["overall"][a]
    print(f"{a:18s} acc500={pf(o['acc500'])} [{pf(o['acc500_ci'][0])},{pf(o['acc500_ci'][1])}] "
          f"f1_500={pf(o['f1_500'])} acc470={pf(o['acc470'])} f1_470={pf(o['f1_470'])} "
          f"abs30={pf(o['acc_abs'])}")
print("\n=== by type (non-abstention, n=470) : acc / f1 ===")
hdr = f"{'arm':18s}" + "".join(f"{SHORT[t]+'('+str(OUT['bytype']['qc'][t]['n'])+')':>14s}" for t in TYPES)
print(hdr)
for a in ARMS:
    print(f"{a:18s}" + "".join(f"{pf(OUT['bytype'][a][t]['acc'])+'/'+pf(OUT['bytype'][a][t]['f1']):>14s}" for t in TYPES))
print("\n=== abstention (n=30) ===")
for a in ARMS:
    cells = " ".join(f"{SHORT[t]}={pf(OUT['bytype_abs'][a][t]['acc'])}(n={OUT['bytype_abs'][a][t]['n']})"
                     for t in OUT["bytype_abs"][a])
    print(f"{a:18s} overall={pf(OUT['overall'][a]['acc_abs'])}  {cells}")
print("\n=== arm-level paired tests (qc vs X, n=500) ===")
for k, r in OUT["arm_tests"].items():
    d, lo, hi = r["boot"]
    print(f"qc vs {k:18s} acc {pf(r['acc_a'])} vs {pf(r['acc_b'])} d=+{pf(r['delta'])} "
          f"b={r['b']} c={r['c']} p={r['p']:.3e} p_holm={r['p_holm']:.3e} "
          f"CI95=[{pf(lo)},{pf(hi)}] | F1 d={pf(r['f1_delta'])} CI=[{pf(r['f1_ci'][0])},{pf(r['f1_ci'][1])}]")
print("\n=== per-subtype vs each reference (Holm within family of 4) ===")
for ref, fam in OUT["subtype_tests"].items():
    print(f"-- reference: {ref}")
    for t in FOCUS:
        r = fam[t]
        d, lo, hi = r["boot"]
        print(f"   {SHORT[t]:4s} n={r['n']:3d} {pf(r['acc_a'])} vs {pf(r['acc_b'])} "
              f"d={d:+.3f} CI=[{pf(lo)},{pf(hi)}] b={r['b']} c={r['c']} "
              f"p={r['p']:.4f} p_holm={r['p_holm']:.4f}")
print("\n=== all six types vs full / vs qc_no_lifecycle (Holm over 6) ===")
for ref, fam in OUT["alltype_tests"].items():
    print(f"-- reference: {ref}")
    for t in TYPES:
        r = fam[t]
        d, lo, hi = r["boot"]
        print(f"   {SHORT[t]:4s} n={r['n']:3d} {pf(r['acc_a'])} vs {pf(r['acc_b'])} "
              f"d={d:+.3f} CI=[{pf(lo)},{pf(hi)}] b={r['b']} c={r['c']} p={r['p']:.4f} p_holm={r['p_holm']:.4f}")
print("\n=== vs best baseline per subtype ===")
for t in TYPES:
    r = OUT["best_baseline"][t]
    d, lo, hi = r["boot"]
    print(f"   {SHORT[t]:4s} n={r['n']:3d} best={r['arm']:16s} {pf(r['acc'])} -> ours {pf(r['ours'])} "
          f"d={r['delta']:+.4f} CI=[{pf(lo)},{pf(hi)}] b={r['b']} c={r['c']} p={r['p']:.4f}")
print("\n=== by query label ===")
for ql, cell in OUT["byqlabel"].items():
    print(f"   {ql:11s} n={cell['n']:3d} " + " ".join(f"{a}={pf(cell[a]['acc'])}" for a in ARMS))
    for ref in ["full", "qc_no_lifecycle", "qc_no_qc"]:
        r = cell[f"vs_{ref}"]
        d, lo, hi = r["boot"]
        print(f"        vs {ref:16s} d={r['delta']:+.4f} CI=[{pf(lo)},{pf(hi)}] b={r['b']} c={r['c']} p={r['p']:.4f}")
print("\nrouting:", OUT["routing"])
print("hist_routed:", OUT["hist_routed"])
print("\njudge/F1 diagnostics:", json.dumps(OUT["diag"], indent=1))
