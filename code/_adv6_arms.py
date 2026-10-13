import json, re, statistics as st
from collections import Counter
from scipy.stats import fisher_exact
R = "/root/agent_memory_iclr/results"
rows = json.load(open(f"{R}/_adv6_rows2.json"))
ARMS = {}
for a in ("full500_qc", "full500_full", "full500_full_p2", "full500_qc_view_only",
          "full500_qc_no_qc", "full500_qc_no_lifecycle", "full500_fifo", "full500_decay"):
    d = {}
    for l in open(f"{R}/{a}.jsonl"):
        r = json.loads(l)
        if "judge" in r:
            d[r["qid"]] = r
    ARMS[a] = d
idx = {r["qid"]: r for r in rows}

SUBS = {
    "enum60": [r for r in rows if r["enum"]],
    "enum&AGGenum44": [r for r in rows if r["enum"] and r["shape"] == "AGG.enum"],
    "AGGenum_howmany_N1-12_52": [r for r in rows if r["shape"] == "AGG.enum" and "how many" in r["question"].lower() and r["gold_num"] is not None and r["gold_num"] == int(r["gold_num"]) and 1 <= r["gold_num"] <= 12],
}
print("=== split across all judged arms (looking for the reported 0.917 / 0.591) ===")
for sname, sub in SUBS.items():
    for a, D in ARMS.items():
        hi = [r for r in sub if r["gold_num"] > r["n_ev"]]
        lo = [r for r in sub if r["gold_num"] <= r["n_ev"]]
        ah = sum(1 for r in hi if D[r["qid"]]["judge"])
        al = sum(1 for r in lo if D[r["qid"]]["judge"])
        p = fisher_exact([[ah, len(hi)-ah], [al, len(lo)-al]])[1]
        print(f"  {sname:26s} {a:24s} hi {ah}/{len(hi)}={ah/len(hi):.3f} lo {al}/{len(lo)}={al/len(lo):.3f} p={p:.4g}")

print("\n=== type stratification (enum60, qc arm) ===")
D = ARMS["full500_qc"]
sub = SUBS["enum60"]
for t in sorted({r["type"] for r in sub}):
    s = [r for r in sub if r["type"] == t]
    hi = [r for r in s if r["gold_num"] > r["n_ev"]]; lo = [r for r in s if r["gold_num"] <= r["n_ev"]]
    ah = sum(1 for r in hi if D[r["qid"]]["judge"]); al = sum(1 for r in lo if D[r["qid"]]["judge"])
    print(f"  {t:26s} n={len(s):3d}  N>n_ev {ah}/{len(hi)}  N<=n_ev {al}/{len(lo)}")
ms = [r for r in sub if r["type"] == "multi-session"]
hi = [r for r in ms if r["gold_num"] > r["n_ev"]]; lo = [r for r in ms if r["gold_num"] <= r["n_ev"]]
ah = sum(1 for r in hi if D[r["qid"]]["judge"]); al = sum(1 for r in lo if D[r["qid"]]["judge"])
print(f"  WITHIN multi-session only: N>n_ev {ah}/{len(hi)}={ah/max(1,len(hi)):.3f}  N<=n_ev {al}/{len(lo)}={al/max(1,len(lo)):.3f}  "
      f"fisher p={fisher_exact([[ah,len(hi)-ah],[al,len(lo)-al]])[1]:.4g}")
# type-only baseline on the same subset
print("  type-only baseline accuracy on enum60:",
      {t: f"{sum(1 for r in sub if r['type']==t and D[r['qid']]['judge'])}/{sum(1 for r in sub if r['type']==t)}"
       for t in sorted({r['type'] for r in sub})})
