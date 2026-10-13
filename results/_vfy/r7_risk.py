import json, glob, os, collections, itertools, math
R="/root/agent_memory_iclr/results"
SH={r["qid"]:r["shape"] for r in json.load(open(f"{R}/_v5_shape_rows.json"))}
arms={os.path.basename(f)[8:-6]:{json.loads(l)["qid"]:json.loads(l) for l in open(f)}
      for f in glob.glob(f"{R}/full500_*.jsonl")}
absq=[q for q in SH if q.endswith("_abs")]
print("[P] which abstention questions each arm fails (shape in brackets):")
for a in ("qc","qc_no_qc","full_p2","full"):
    bad=[(q,SH[q]) for q in absq if not arms[a][q]["judge"]]
    print(f"   {a:12s} misses {len(bad)}: {[(q[:8],s) for q,s in bad]}")
# overlap qc vs qc_no_qc on abstention
q1={q for q in absq if arms['qc'][q]['judge']}; q2={q for q in absq if arms['qc_no_qc'][q]['judge']}
b,c=len(q2-q1), len(q1-q2)
print(f"\n[P] qc vs qc_no_qc on abstention: discordant b={b} c={c} -> exact binomial p="
      f"{sum(math.comb(b+c,k) for k in range(min(b,c)+1))*2/2**(b+c) if b+c else 1:.3f}")
# answerable accuracy per shape, qc arm
print("\n[Q] full500_qc ANSWERABLE accuracy by shape (for the AGG.arith operand-rule risk):")
d=collections.defaultdict(lambda:[0,0])
for q,s in SH.items():
    if q.endswith("_abs"): continue
    d[s][0]+=arms['qc'][q]['judge']; d[s][1]+=1
for s,(a,n) in sorted(d.items(), key=lambda kv:-kv[1][1]):
    print(f"    {s:18s} {a}/{n} = {a/n:.3f}")
