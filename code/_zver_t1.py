"""Test 1: paired flip analysis + decline-rate confound across arms."""
import json, re, sys
sys.path.insert(0, "/root/agent_memory_iclr/code")
from _zver_core import arms, load
from v5_guards import is_decline

A = arms()
qids_abs = [q for q in A["qc"] if q.endswith("_abs")]
qids_ans = [q for q in A["qc"] if not q.endswith("_abs")]
print("n_abs", len(qids_abs), "n_ans", len(qids_ans))

print("\n=== arm-level table (independent recompute) ===")
print("%-16s %-13s %-15s %-18s %-16s" % ("arm", "abs_acc", "ans_acc", "abs_decline", "ans_decline"))
rows = []
for tag, R in A.items():
    a = [R[q] for q in qids_abs]
    n = [R[q] for q in qids_ans]
    aa = sum(r["judge"] for r in a) / len(a)
    na = sum(r["judge"] for r in n) / len(n)
    ad = sum(is_decline(r["pred"]) for r in a) / len(a)
    nd = sum(is_decline(r["pred"]) for r in n) / len(n)
    rows.append((tag, aa, na, ad, nd))
    print("%-16s %2d/30=%.3f  %3d/470=%.3f  %2d/30=%.3f      %3d/470=%.3f" % (
        tag, sum(r["judge"] for r in a), aa, sum(r["judge"] for r in n), na,
        sum(is_decline(r["pred"]) for r in a), ad,
        sum(is_decline(r["pred"]) for r in n), nd))

# correlation between answerable decline rate and abs accuracy
import statistics
xs = [r[4] for r in rows]
ys = [r[1] for r in rows]
mx, my = statistics.mean(xs), statistics.mean(ys)
num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
print("\nPearson r(ans_decline_rate, abs_acc) over %d arms = %.4f" % (len(rows), num / den))
# spearman
def rank(v):
    s = sorted(range(len(v)), key=lambda i: v[i])
    r = [0] * len(v)
    for pos, i in enumerate(s):
        r[i] = pos
    return r
rx, ry = rank(xs), rank(ys)
mrx, mry = statistics.mean(rx), statistics.mean(ry)
num = sum((a - mrx) * (b - mry) for a, b in zip(rx, ry))
den = (sum((a - mrx) ** 2 for a in rx) * sum((b - mry) ** 2 for b in ry)) ** 0.5
print("Spearman rho = %.4f" % (num / den))

print("\n=== paired flips on the 30 abs questions ===")
def flips(a, b):
    ab = [q for q in qids_abs if A[a][q]["judge"] and not A[b][q]["judge"]]
    ba = [q for q in qids_abs if not A[a][q]["judge"] and A[b][q]["judge"]]
    return ab, ba

for pair in [("qc", "qc_no_lifecycle"), ("qc", "qc_view_only"), ("qc_no_lifecycle", "qc_view_only")]:
    ab, ba = flips(*pair)
    print("%s vs %s : %s-only-correct=%d %s ; %s-only-correct=%d %s" % (
        pair[0], pair[1], pair[0], len(ab), ab, pair[1], len(ba), ba))
    # exact McNemar (two-sided binomial)
    n = len(ab) + len(ba)
    from math import comb
    k = min(len(ab), len(ba))
    p = sum(comb(n, i) for i in range(0, k + 1)) * 2 / (2 ** n) if n else 1.0
    print("    exact McNemar two-sided p = %.4f (discordant n=%d)" % (min(1.0, p), n))

print("\n=== qc's 5 abs losses; who else is right ===")
losses = [q for q in qids_abs if not A["qc"][q]["judge"]]
for q in losses:
    who = [t for t in A if A[t][q]["judge"]]
    dec = [t for t in A if is_decline(A[t][q]["pred"])]
    print(" %-24s correct_in=%s" % (q, sorted(who)))
    print("   %-24s declines_in=%s" % ("", sorted(dec)))
