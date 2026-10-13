import json, statistics as st
from scipy.stats import fisher_exact
R = "/root/agent_memory_iclr/results"
rows = json.load(open(f"{R}/_adv6_rows3.json"))
ARMS = {}
for a in ("full500_qc", "full500_full", "full500_qc_no_lifecycle"):
    ARMS[a] = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{R}/{a}.jsonl") if "judge" in json.loads(l)}
sub = [r for r in rows if r["enum"]]
print(f"enum60 n={len(sub)}  qc accuracy {sum(1 for r in sub if ARMS['full500_qc'][r['qid']]['judge'])}/{len(sub)}")
print("\npredictive power of each 'the cluster has the material' metric (enum60, qc arm)")
print(f"{'metric':44s} {'satisfied':>10s} {'acc|sat':>8s} {'acc|unsat':>10s} {'fisher p':>9s}")
for f in ("union_dv", "max_dv", "top_dv", "u_gold_sess_distinct", "m_gold_sess_distinct",
          "u_grounded_evturn", "u_grounded_evturn_gold", "m_grounded_evturn", "t_grounded_evturn"):
    ok = [r for r in sub if r[f] >= r["gold_num"]]
    bad = [r for r in sub if r[f] < r["gold_num"]]
    a = sum(1 for r in ok if ARMS["full500_qc"][r["qid"]]["judge"])
    b = sum(1 for r in bad if ARMS["full500_qc"][r["qid"]]["judge"])
    p = fisher_exact([[a, len(ok) - a], [b, len(bad) - b]])[1] if bad and ok else float("nan")
    print(f"{f:44s} {len(ok):3d}/{len(sub):<3d}   {a/max(1,len(ok)):8.3f} {b/max(1,len(bad)):10.3f} {p:9.4g}")

print("\nsame, but predicting the LOOSER continuous signal: correlation with judge (point-biserial)")
try:
    from scipy.stats import pointbiserialr
    y = [1 if ARMS["full500_qc"][r["qid"]]["judge"] else 0 for r in sub]
    for f in ("union_dv", "u_grounded_evturn", "gold_num", "n_ev"):
        x = [r[f] for r in sub]
        rr, pp = pointbiserialr(y, x)
        print(f"  {f:24s} r={rr:+.3f} p={pp:.4g}")
except Exception as e:
    print("scipy pointbiserial missing", e)

print("\nratio (union candidates / gold N) - the 'oversize' claim")
rat = [r["union_dv"] / r["gold_num"] for r in sub]
print(f"  union/N: mean {st.mean(rat):.2f}x median {st.median(rat):.2f}x max {max(rat):.1f}x")
rat2 = [r["max_dv"] / r["gold_num"] for r in sub]
print(f"  largest-cluster/N: mean {st.mean(rat2):.2f}x median {st.median(rat2):.2f}x")
rat3 = [r["top_dv"] / r["gold_num"] for r in sub]
print(f"  top-cluster/N: mean {st.mean(rat3):.2f}x median {st.median(rat3):.2f}x")
print(f"  mean gold N {st.mean(r['gold_num'] for r in sub):.2f}; mean union {st.mean(r['union_dv'] for r in sub):.2f}; "
      f"mean largest {st.mean(r['max_dv'] for r in sub):.2f}; mean top {st.mean(r['top_dv'] for r in sub):.2f}")
