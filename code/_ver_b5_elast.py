import json, collections, statistics as st
R="/root/agent_memory_iclr/results"
rows={r["qid"]: r for r in json.load(open(f"{R}/_verify/b5_repro.json"))}
def load(p):
    out={}
    for line in open(f"{R}/{p}"):
        d=json.loads(line)
        if "judge" in d: out[d["qid"]]=d
    return out
full=load("full500_full.jsonl")          # all turns, k=30, NO view
nolc=load("full500_qc_no_lifecycle.jsonl")  # all turns, k=20, NO view
qc=load("full500_qc.jsonl")
print("arm sizes:", len(full), len(nolc), len(qc))
def rec(r,a): return r["arms"][a]["hits"]/max(1,r["ev_n"])
common=[q for q in rows if q in full and q in nolc]
print("common non-abs:", len(common))
ev=sum(rows[q]["ev_n"] for q in common)
r20=sum(rows[q]["arms"]["base20"]["hits"] for q in common)/ev
r30=sum(rows[q]["arms"]["base30"]["hits"] for q in common)/ev
r50=sum(rows[q]["arms"]["base50"]["hits"] for q in common)/ev
a20=sum(nolc[q]["judge"] for q in common)/len(common)
a30=sum(full[q]["judge"] for q in common)/len(common)
print(f"\n=== NATURAL EXPERIMENT: same pipeline, k=20 (no_lifecycle) vs k=30 (full), no view either side ===")
print(f"evidence recall  k20 {r20:.3f} -> k30 {r30:.3f}  (delta {100*(r30-r20):+.1f} pp)   [k50 would be {r50:.3f}]")
print(f"judged accuracy  k20 {a20:.3f} -> k30 {a30:.3f}  (delta {100*(a30-a20):+.1f} pp)")
print(f"chars            k20 {st.mean([rows[q]['arms']['base20']['chars'] for q in common]):.0f} -> k30 {st.mean([rows[q]['arms']['base30']['chars'] for q in common]):.0f}")
gainers=[q for q in common if rows[q]["arms"]["base30"]["hits"]>rows[q]["arms"]["base20"]["hits"]]
print(f"\nquestions gaining >=1 evidence turn at k=30: {len(gainers)}")
w2r=[q for q in gainers if not nolc[q]["judge"] and full[q]["judge"]]
r2w=[q for q in gainers if nolc[q]["judge"] and not full[q]["judge"]]
print(f"  of these: wrong@k20 -> right@k30 : {len(w2r)}   right@k20 -> wrong@k30 : {len(r2w)}   net {len(w2r)-len(r2w)}")
nog=[q for q in common if q not in gainers]
w2r2=[q for q in nog if not nolc[q]["judge"] and full[q]["judge"]]
r2w2=[q for q in nog if nolc[q]["judge"] and not full[q]["judge"]]
print(f"  non-gainers ({len(nog)}): wrong->right {len(w2r2)}, right->wrong {len(r2w2)}, net {len(w2r2)-len(r2w2)}  <- pure noise floor")
print(f"\nper type: recall k20->k30, acc k20->k30")
for t in sorted({rows[q]['type'] for q in common}):
    sub=[q for q in common if rows[q]["type"]==t]
    e=sum(rows[q]["ev_n"] for q in sub)
    print(f"  {t:28s} n={len(sub):3d} rec {sum(rows[q]['arms']['base20']['hits'] for q in sub)/e:.3f}->{sum(rows[q]['arms']['base30']['hits'] for q in sub)/e:.3f}"
          f"  acc {sum(nolc[q]['judge'] for q in sub)/len(sub):.3f}->{sum(full[q]['judge'] for q in sub)/len(sub):.3f}"
          f"  gainers={sum(1 for q in sub if rows[q]['arms']['base30']['hits']>rows[q]['arms']['base20']['hits']):2d}"
          f"  flips(w->r/r->w)={sum(1 for q in sub if rows[q]['arms']['base30']['hits']>rows[q]['arms']['base20']['hits'] and not nolc[q]['judge'] and full[q]['judge'])}/"
          f"{sum(1 for q in sub if rows[q]['arms']['base30']['hits']>rows[q]['arms']['base20']['hits'] and nolc[q]['judge'] and not full[q]['judge'])}")
# regression-style estimate: P(correct) as a function of recall bucket, within type, in the qc arm
print("\n=== qc arm: accuracy by base20 evidence-recall bucket (within type) ===")
for t in sorted({rows[q]['type'] for q in rows}):
    sub=[q for q in rows if rows[q]["type"]==t and q in qc]
    b={}
    for q in sub:
        k = "full" if rec(rows[q],"base20")>=1 else ("partial" if rec(rows[q],"base20")>0 else "none")
        b.setdefault(k,[]).append(qc[q]["judge"])
    print(f"  {t:28s} " + "  ".join(f"{k}: {sum(v)}/{len(v)}={sum(v)/len(v):.2f}" for k,v in sorted(b.items())))
