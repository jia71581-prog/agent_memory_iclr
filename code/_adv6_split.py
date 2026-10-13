import json, re, statistics as st
from collections import Counter
from scipy.stats import fisher_exact
R = "/root/agent_memory_iclr/results"
rows = json.load(open(f"{R}/_adv6_rows2.json"))
QC = {}
for l in open(f"{R}/full500_qc.jsonl"):
    d = json.loads(l)
    if "judge" in d:
        QC[d["qid"]] = d
for r in rows:
    r["judge"] = QC[r["qid"]]["judge"]
    r["f1"] = QC[r["qid"]].get("f1")

SUBS = {
    "enum60": [r for r in rows if r["enum"]],
    "enum&AGGenum(n44)": [r for r in rows if r["enum"] and r["shape"] == "AGG.enum"],
    "AGGenum&howmany&N1-12(n52)": [r for r in rows if r["shape"] == "AGG.enum" and "how many" in r["question"].lower() and r["gold_num"] is not None and r["gold_num"] == int(r["gold_num"]) and 1 <= r["gold_num"] <= 12],
    "AGGenum&N1-12(n57)": [r for r in rows if r["shape"] == "AGG.enum" and r["gold_num"] is not None and r["gold_num"] == int(r["gold_num"]) and 1 <= r["gold_num"] <= 12],
}
for name, sub in SUBS.items():
    hi = [r for r in sub if r["gold_num"] > r["n_ev"]]
    lo = [r for r in sub if r["gold_num"] <= r["n_ev"]]
    ah = sum(1 for r in hi if r["judge"])
    al = sum(1 for r in lo if r["judge"])
    tab = [[ah, len(hi) - ah], [al, len(lo) - al]]
    p = fisher_exact(tab)[1]
    print(f"{name:28s} N>n_ev {ah}/{len(hi)}={ah/max(1,len(hi)):.3f}  N<=n_ev {al}/{len(lo)}={al/max(1,len(lo)):.3f}  fisher p={p:.4g}")
    print(f"    n_ev mean: hi={st.mean(r['n_ev'] for r in hi):.2f} lo={st.mean(r['n_ev'] for r in lo):.2f} | "
          f"gold mean hi={st.mean(r['gold_num'] for r in hi):.2f} lo={st.mean(r['gold_num'] for r in lo):.2f} | "
          f"types hi={dict(Counter(r['type'] for r in hi))} lo={dict(Counter(r['type'] for r in lo))}")

print("\n=== confound probe on enum60: accuracy by n_ev alone ===")
sub = SUBS["enum60"]
for thr in (1, 2, 3, 4):
    a = [r for r in sub if r["n_ev"] <= thr]
    b = [r for r in sub if r["n_ev"] > thr]
    if not a or not b:
        continue
    aa = sum(1 for r in a if r["judge"]); bb = sum(1 for r in b if r["judge"])
    print(f"  n_ev<={thr}: {aa}/{len(a)}={aa/len(a):.3f}   n_ev>{thr}: {bb}/{len(b)}={bb/len(b):.3f}  fisher p={fisher_exact([[aa,len(a)-aa],[bb,len(b)-bb]])[1]:.4g}")
print("  accuracy by gold_num alone:")
for thr in (2, 3, 4, 5):
    a = [r for r in sub if r["gold_num"] <= thr]
    b = [r for r in sub if r["gold_num"] > thr]
    if not a or not b: continue
    aa = sum(1 for r in a if r["judge"]); bb = sum(1 for r in b if r["judge"])
    print(f"  N<={thr}: {aa}/{len(a)}={aa/len(a):.3f}   N>{thr}: {bb}/{len(b)}={bb/len(b):.3f}  fisher p={fisher_exact([[aa,len(a)-aa],[bb,len(b)-bb]])[1]:.4g}")

print("\n=== does the '>=N distinct' metric predict correctness? (enum60, union & top cluster) ===")
for f in ("union_dv", "max_dv", "top_dv"):
    ok = [r for r in sub if r[f] >= r["gold_num"]]
    bad = [r for r in sub if r[f] < r["gold_num"]]
    a = sum(1 for r in ok if r["judge"]); b = sum(1 for r in bad if r["judge"])
    p = fisher_exact([[a, len(ok)-a], [b, len(bad)-b]])[1] if bad else float("nan")
    print(f"  {f:9s}>=N: {a}/{len(ok)}={a/max(1,len(ok)):.3f}   <N: {b}/{len(bad)}={b/max(1,len(bad)):.3f}  fisher p={p:.4g}")

print("\n=== 2x2 of (N>n_ev) x n_ev==1, enum60 ===")
print("  N>n_ev & n_ev==1:", sum(1 for r in sub if r["gold_num"] > r["n_ev"] and r["n_ev"] == 1),
      " N>n_ev & n_ev>1:", sum(1 for r in sub if r["gold_num"] > r["n_ev"] and r["n_ev"] > 1),
      " N<=n_ev & n_ev==1:", sum(1 for r in sub if r["gold_num"] <= r["n_ev"] and r["n_ev"] == 1),
      " N<=n_ev & n_ev>1:", sum(1 for r in sub if r["gold_num"] <= r["n_ev"] and r["n_ev"] > 1))
# stratified: within n_ev>1 only, does N>n_ev still predict?
s2 = [r for r in sub if r["n_ev"] > 1]
hi = [r for r in s2 if r["gold_num"] > r["n_ev"]]; lo = [r for r in s2 if r["gold_num"] <= r["n_ev"]]
ah = sum(1 for r in hi if r["judge"]); al = sum(1 for r in lo if r["judge"])
print(f"  within n_ev>1 (n={len(s2)}): N>n_ev {ah}/{len(hi)}  N<=n_ev {al}/{len(lo)}  "
      f"fisher p={fisher_exact([[ah,len(hi)-ah],[al,len(lo)-al]])[1]:.4g}")
