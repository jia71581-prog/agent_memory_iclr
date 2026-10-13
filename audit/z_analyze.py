import json, statistics as st
from collections import Counter, defaultdict

recs = json.load(open("/root/agent_memory_iclr/audit/z_recs.json"))
n = len(recs)
m = lambda k: sum(r[k] for r in recs) / n
print("n =", n)

print("\n== 1. does the cap bind? ==")
print("mean chains built            : %.1f" % m("n_chains"))
print("mean UNCAPPED current chars  : %.0f  (median %.0f, min %d, max %d)" % (
    m("cur_full_chars"), st.median(r["cur_full_chars"] for r in recs),
    min(r["cur_full_chars"] for r in recs), max(r["cur_full_chars"] for r in recs)))
print("mean CAPPED   current chars  : %.1f" % m("cur_cap_chars"))
print("mean CAPPED   hist    chars  : %.1f" % m("hist_cap_chars"))
print("questions where cap BINDS (uncapped current > 20000) : %d/%d" % (
    sum(r["cur_binds"] for r in recs), n))
print("questions where cap BINDS (uncapped historical)      : %d/%d" % (
    sum(r["hist_binds"] for r in recs), n))
print("min uncapped current chars over all questions        : %d" % min(r["cur_full_chars"] for r in recs))
lo = sorted(recs, key=lambda r: r["cur_full_chars"])[:5]
print("5 smallest uncapped current views:", [(r["qid"][:12], r["cur_full_chars"], r["n_chains"]) for r in lo])

print("\n== 2. survival ==")
print("mean chains surviving (current)   : %.1f  (%.1f%% of built)" % (m("kept_cur"), 100 * m("kept_cur") / m("n_chains")))
print("mean chains surviving (historical): %.1f" % m("kept_hist"))
per_q = [r["kept_cur"] / r["n_chains"] for r in recs if r["n_chains"]]
print("mean PER-QUESTION survival fraction (current): %.4f  (median %.4f)" % (sum(per_q) / len(per_q), st.median(per_q)))
print("mean multi-session chains built    : %.1f" % m("n_multi"))
print("mean multi-session chains surviving: %.1f  (%.1f%% ratio-of-means)" % (m("multi_kept"), 100 * m("multi_kept") / m("n_multi")))
pm = [r["multi_kept"] / r["n_multi"] for r in recs if r["n_multi"]]
print("mean per-question multi survival   : %.4f (n=%d with multi>0)" % (sum(pm) / len(pm), len(pm)))

print("\n== 3. gold containment ==")
ne = [r for r in recs if r["gold"]]
print("questions with non-empty normalized gold:", len(ne))
gf = sum(r["gold_in_full"] for r in ne); gc = sum(r["gold_in_cap"] for r in ne)
print("gold in UNCAPPED routed-mode view: %d (%.3f)" % (gf, gf / len(ne)))
print("gold in CAPPED   routed-mode view: %d (%.3f)" % (gc, gc / len(ne)))
lost = [r for r in ne if r["gold_in_full"] and not r["gold_in_cap"]]
gain = [r for r in ne if r["gold_in_cap"] and not r["gold_in_full"]]
print("destroyed by truncation: %d (%.4f of %d)" % (len(lost), len(lost) / len(ne), len(ne)))
print("gained (sanity, should be 0):", len(gain))
print("of the lost: judge_qc correct %d/%d ; view_only correct %d/%d ; no_qc %d/%d ; no_lifecycle %d/%d" % (
    sum(r["judge_qc"] for r in lost), len(lost), sum(r["judge_vo"] for r in lost), len(lost),
    sum(r["judge_nq"] for r in lost), len(lost), sum(r["judge_nl"] for r in lost), len(lost)))
print("lost type mix (raw counts):", dict(Counter(r["type"] for r in lost)))

print("\n== 4. per-type RATES (denominators matter) ==")
den = Counter(r["type"] for r in ne)
lc = Counter(r["type"] for r in lost)
for t in sorted(den, key=lambda t: -lc[t] / den[t]):
    print("  %-26s lost %2d / %3d = %.3f" % (t, lc[t], den[t], lc[t] / den[t]))

print("\n== 5. gold-match proxy quality on the lost set ==")
gl = [r["gold_len"] for r in lost]
print("gold string length among lost: min %d p25 %d median %d max %d" % (
    min(gl), sorted(gl)[len(gl) // 4], st.median(gl), max(gl)))
print("lost with gold_len <= 4 chars (spurious-substring risk):",
      sum(1 for r in lost if r["gold_len"] <= 4))
print("all recs: gold_len<=4 count:", sum(1 for r in ne if r["gold_len"] <= 4))
short = [r for r in ne if r["gold_len"] <= 6]
print("short-gold (<=6ch) containment uncapped/capped: %d/%d vs %d/%d" % (
    sum(r["gold_in_full"] for r in short), len(short), sum(r["gold_in_cap"] for r in short), len(short)))

print("\n== 6. where does the gold line sit in the alphabetical order? ==")
inf = [r for r in ne if r["gold_in_full"]]
frac = []
for r in inf:
    if r["n_chains"]:
        frac.append(min(r["gold_ranks"]) / r["n_chains"])
print("gold-line normalized rank among sorted keys: mean %.3f median %.3f (uniform would be ~0.5)" % (
    sum(frac) / len(frac), st.median(frac)))
lf = [min(r["gold_ranks"]) / r["n_chains"] for r in lost]
print("for the LOST ones: mean %.3f min %.3f (all must exceed kept fraction)" % (sum(lf) / len(lf), min(lf)))
keep_frac = [r["kept_cur"] / r["n_chains"] for r in recs if r["n_chains"]]
print("kept fraction mean %.3f median %.3f" % (sum(keep_frac) / len(keep_frac), st.median(keep_frac)))

print("\n== 7. does truncation actually correlate with accuracy? (the 'so what') ==")
def acc(rs, k): return sum(r[k] for r in rs) / max(1, len(rs))
print("view_only acc  overall %.3f | gold-in-capped %.3f (n=%d) | not %.3f (n=%d)" % (
    acc(ne, "judge_vo"), acc([r for r in ne if r["gold_in_cap"]], "judge_vo"),
    sum(r["gold_in_cap"] for r in ne), acc([r for r in ne if not r["gold_in_cap"]], "judge_vo"),
    len(ne) - sum(r["gold_in_cap"] for r in ne)))
print("qc acc: lost-set %.3f (n=%d) vs rest %.3f" % (
    acc(lost, "judge_qc"), len(lost), acc([r for r in ne if r not in lost], "judge_qc")))
