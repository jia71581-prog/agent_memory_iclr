"""Measured abstention accuracy per arm and per shape from the full500 runs."""
import json, glob, collections, os
R="/root/agent_memory_iclr/results"
SH={r["qid"]: r["shape"] for r in json.load(open(f"{R}/_v5_shape_rows.json"))}
for f in sorted(glob.glob(f"{R}/full500_*.jsonl")):
    rows=[json.loads(l) for l in open(f)]
    ab=[r for r in rows if r["qid"].endswith("_abs")]
    an=[r for r in rows if not r["qid"].endswith("_abs")]
    print(f"{os.path.basename(f):30s} n={len(rows)} overall={sum(r['judge'] for r in rows)/len(rows):.3f} "
          f"abs={sum(r['judge'] for r in ab)/max(1,len(ab)):.3f} ({sum(r['judge'] for r in ab)}/{len(ab)}) "
          f"ans={sum(r['judge'] for r in an)/max(1,len(an)):.3f}")
print()
rows=[json.loads(l) for l in open(f"{R}/full500_qc.jsonl")]
byq={r["qid"]:r for r in rows}
d=collections.defaultdict(lambda:[0,0])
for r in rows:
    if r["qid"].endswith("_abs"):
        s=SH[r["qid"]]; d[s][0]+=r["judge"]; d[s][1]+=1
print("[F] full500_qc abstention accuracy by shape:")
for s,(a,b) in sorted(d.items(), key=lambda kv:-kv[1][1]):
    print(f"    {s:18s} {a}/{b} = {a/b:.3f}")
