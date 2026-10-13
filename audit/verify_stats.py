import json, math, statistics as st
from collections import Counter

rows = json.load(open("/root/agent_memory_iclr/audit/verify_viewdiff.json"))
n = len(rows)
print("=" * 78)
print("A. ROUTING SANITY")
print("=" * 78)
bad = [r for r in rows if r["mode_qc"] != r["mode_qc_expected"]]
print("recorded qc mode != deterministic classifier mode :", len(bad))
print("no_qc mode always 'historical'                    :", all(r["mode_nq"] == "historical" for r in rows))
print("qlabel recorded != recomputed label               :", sum(1 for r in rows if r["qlabel"] != r["label_recomputed"]))
print("label distribution:", Counter(r["label_recomputed"] for r in rows))
routed = [r for r in rows if r["routed_diff"]]
same_mode = [r for r in rows if not r["routed_diff"]]
print("routed-different n =", len(routed), " same-mode n =", len(same_mode))
print("routed-different all have mode_qc=current:", all(r["mode_qc"] == "current" for r in routed))

print()
print("=" * 78)
print("B. DID THE VIEW TEXT ACTUALLY CHANGE? (routed-different subset)")
print("=" * 78)
print("view text identical anyway :", sum(1 for r in routed if r["same_view"]))
print("view text actually differs :", sum(1 for r in routed if not r["same_view"]))

def dist(vals, name):
    v = sorted(vals)
    m = len(v)
    print("%-34s n=%3d min=%d p25=%d median=%.1f p75=%d max=%d mean=%.2f" %
          (name, m, v[0], v[m // 4], st.median(v), v[3 * m // 4], v[-1], sum(v) / m))

print()
print("--- differing-line counts, MY definition (zip_longest, tail counted) ---")
dist([r["n_diff"] for r in routed], "routed-different (n=456?)")
dist([r["n_diff"] for r in rows], "all 500 questions")
print("--- differing-line counts, a6_cap.py definition (zip = common prefix only) ---")
dist([r["n_diff_prefix"] for r in routed], "routed-different")
dist([r["n_diff_prefix"] for r in rows], "all 500 questions")
print()
print("questions with 0 differing lines : %d/%d (all)   %d/%d (routed)" %
      (sum(1 for r in rows if r["n_diff"] == 0), n,
       sum(1 for r in routed if r["n_diff"] == 0), len(routed)))
print("rendered lines kept after cap    :", end=" ")
dist([r["kept_cur"] for r in rows], "kept_cur all 500")
dist([r["kept_cur"] for r in routed], "kept_cur routed")
print("mean fraction of view lines differing (all 500)   : %.4f" %
      (sum(r["n_diff"] / max(1, r["kept_cur"]) for r in rows) / n))
print("mean fraction of view lines differing (routed)    : %.4f" %
      (sum(r["n_diff"] / max(1, r["kept_cur"]) for r in routed) / len(routed)))
print("median fraction (routed)                          : %.4f" %
      st.median([r["n_diff"] / max(1, r["kept_cur"]) for r in routed]))

print()
print("=" * 78)
print("C. WHAT KIND OF DIFFERENCE? (cosmetic vs real information)")
print("=" * 78)
print("A 'cosmetic' differing line = identical alphanumeric token multiset in both views,")
print("i.e. only punctuation/separator changed ('; ' vs ' / ') -> no information difference.")
tot_d = sum(r["n_diff"] for r in routed)
tot_c = sum(r["n_diff_cosmetic"] for r in routed)
tot_r = sum(r["n_diff_real"] for r in routed)
print("routed subset: total differing lines=%d  cosmetic=%d (%.1f%%)  real=%d (%.1f%%)" %
      (tot_d, tot_c, 100 * tot_c / tot_d, tot_r, 100 * tot_r / tot_d))
dist([r["n_diff_real"] for r in routed], "REAL differing lines / question")
dist([r["n_diff_cosmetic"] for r in routed], "cosmetic differing lines / q")
print("routed questions with 0 REAL differing lines:", sum(1 for r in routed if r["n_diff_real"] == 0))
print("differing lines on multi-session chains: %d, on single-session chains: %d" %
      (sum(r["n_diff_multi"] for r in routed), sum(r["n_diff_single"] for r in routed)))
print("mean fraction of view lines with REAL difference (routed): %.4f" %
      (sum(r["n_diff_real"] / max(1, r["kept_cur"]) for r in routed) / len(routed)))

print()
print("=" * 78)
print("D. LENGTH / CAP")
print("=" * 78)
d = [r["len_hist"] - r["len_cur"] for r in routed]
print("capped length delta (hist-cur) routed: mean %+.1f median %+.1f min %d max %d" %
      (sum(d) / len(d), st.median(d), min(d), max(d)))
d2 = [r["full_len_hist"] - r["full_len_cur"] for r in rows]
print("UNcapped length delta (hist-cur) all : mean %+.1f median %+.1f min %d max %d" %
      (sum(d2) / len(d2), st.median(d2), min(d2), max(d2)))
print("questions where cap binds (uncapped current > 20000):", sum(1 for r in rows if r["full_len_cur"] > 20000))
print("questions where cap binds (uncapped hist    > 20000):", sum(1 for r in rows if r["full_len_hist"] > 20000))
print("mean chains built %.1f, mean chains kept after cap %.1f (%.1f%%)" %
      (sum(r["n_chains"] for r in rows) / n, sum(r["kept_cur"] for r in rows) / n,
       100 * sum(r["kept_cur"] for r in rows) / sum(r["n_chains"] for r in rows)))
print("mean multi-session chains built %.1f ; questions with 0 multi chains: %d" %
      (sum(r["n_multi"] for r in rows) / n, sum(1 for r in rows if r["n_multi"] == 0)))

print()
print("=" * 78)
print("E. THE NULL ITSELF (paired, exact McNemar)")
print("=" * 78)
def mcnemar(sub):
    n01 = sum(1 for r in sub if not r["judge_nq"] and r["judge_qc"])
    n10 = sum(1 for r in sub if r["judge_nq"] and not r["judge_qc"])
    m = n01 + n10
    if m == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)
    return n01, n10, p

for name, sub in [("ALL 500", rows),
                  ("routed-different (view differs)", [r for r in routed if not r["same_view"]]),
                  ("routed-different, >=1 REAL diff", [r for r in routed if r["n_diff_real"] > 0]),
                  ("same-mode (both historical)", same_mode)]:
    if not sub:
        print("  [%s] n=0" % name); continue
    a = sum(r["judge_qc"] for r in sub) / len(sub)
    b = sum(r["judge_nq"] for r in sub) / len(sub)
    n01, n10, p = mcnemar(sub)
    fa = sum(r["f1_qc"] for r in sub) / len(sub); fb = sum(r["f1_nq"] for r in sub) / len(sub)
    print("  [%-32s] n=%3d qc=%.4f no_qc=%.4f d=%+.4f | flips %d/%d exact p=%.4f | f1 %+.4f" %
          (name, len(sub), a, b, a - b, n01, n10, p, fa - fb))

# 95% CI on the paired difference (Wald on discordant pairs / n)
sub = [r for r in routed if not r["same_view"]]
diffs = [int(r["judge_qc"]) - int(r["judge_nq"]) for r in sub]
m = len(diffs); mu = sum(diffs) / m
sd = (sum((x - mu) ** 2 for x in diffs) / (m - 1)) ** 0.5
print("  paired delta on routed-different: %+.4f  95%% CI [%+.4f, %+.4f]  (se %.4f)" %
      (mu, mu - 1.96 * sd / m ** 0.5, mu + 1.96 * sd / m ** 0.5, sd / m ** 0.5))
