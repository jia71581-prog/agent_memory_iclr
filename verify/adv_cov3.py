import json, sys
from collections import Counter
sys.path.insert(0,"/root/agent_memory_iclr/code")
R="/root/agent_memory_iclr/results"
rows={r["qid"]:r for r in json.load(open("/root/agent_memory_iclr/verify/adv_cov1.json"))}
QC={json.loads(l)["qid"]: json.loads(l) for l in open(f"{R}/full500_qc.jsonl")}
CD={r["qid"]:r for r in json.load(open(f"{R}/coverage_diag.json"))}
NA=[r for r in rows.values() if not r["is_abs"]]
for r in NA: r["judge"]=QC[r["qid"]]["judge"]

print("="*72); print("D. DOES 'complete evidence SESSION set' PREDICT ANSWER CORRECTNESS?"); print("="*72)
def rate(p): return sum(r["judge"] for r in p)/len(p) if p else float('nan')
a=[r for r in NA if r["flat_all"]]; b=[r for r in NA if not r["flat_all"]]
print(f"  flat_all=True  n={len(a):3d} acc={rate(a):.3f}")
print(f"  flat_all=False n={len(b):3d} acc={rate(b):.3f}   gap={rate(a)-rate(b):+.3f}")
try:
    from scipy.stats import fisher_exact
    t=[[sum(r['judge'] for r in a),len(a)-sum(r['judge'] for r in a)],
       [sum(r['judge'] for r in b),len(b)-sum(r['judge'] for r in b)]]
    print(f"  Fisher p = {fisher_exact(t)[1]:.4f}   (table {t})")
except Exception as e: print("  scipy:",e)
for k in ("gold_in_flat","ansturn_any_flat","ansturn_all_flat"):
    a=[r for r in NA if r[k]]; b=[r for r in NA if not r[k]]
    print(f"  {k:18s} True n={len(a):3d} acc={rate(a):.3f} | False n={len(b):3d} acc={rate(b):.3f}  gap={rate(a)-rate(b):+.3f}")

print(); print("="*72); print("E. AUDIT OF THE 7 'RETRIEVAL-RECOVERABLE' ERRORS"); print("="*72)
rec=[r for r in NA if r["cov_all"] and not r["flat_all"] and not r["judge"]]
print(f"  n = {len(rec)}")
print(f"  {'qid':10s} {'shape':17s} {'gold':26s} {'inView':6s} {'inRaw20':7s} {'goldCov':7s} {'ansCov':6s} {'ansFlat':7s}")
for r in sorted(rec,key=lambda x:x["shape"]):
    cd=CD[r["qid"]]
    print(f"  {r['qid']:10s} {r['shape']:17s} {str(r['gold'])[:26]:26s} "
          f"{str(cd['in_view']):6s} {str(cd['in_raw20']):7s} {str(r['gold_in_cov']):7s} "
          f"{str(r['ansturn_any_cov']):6s} {str(r['ansturn_any_flat']):7s}")
already=sum(1 for r in rec if CD[r["qid"]]["in_view"] or CD[r["qid"]]["in_raw20"])
print(f"  gold ALREADY in the shipped context (lifecycle view or BM25 top-20): {already}/{len(rec)}")
gc=sum(1 for r in rec if r["gold_in_cov"]); ac=sum(1 for r in rec if r["ansturn_any_cov"])
print(f"  coverage retrieval actually puts the gold string in context:      {gc}/{len(rec)}")
print(f"  coverage retrieval actually puts an annotated answer turn in ctx: {ac}/{len(rec)}")
print(f"  numeric / derived gold among the 7: {sum(1 for r in rec if str(r['gold']).strip().replace('.','',1).isdigit())}/{len(rec)}  "
      f"golds={[str(r['gold'])[:18] for r in rec]}")

print(); print("="*72); print("F. THE 19 'GAIN' QUESTIONS: WHAT IS ACTUALLY GAINED?"); print("="*72)
g=[r for r in NA if r["cov_all"] and not r["flat_all"]]
print(f"  n={len(g)}  currently correct {sum(r['judge'] for r in g)}  currently wrong {sum(1 for r in g if not r['judge'])}")
print(f"  of these 19, coverage puts gold string in ctx: {sum(r['gold_in_cov'] for r in g)}  "
      f"(flat already had it: {sum(r['gold_in_flat'] for r in g)})")
print(f"  of these 19, coverage puts an annotated answer turn in ctx: {sum(r['ansturn_any_cov'] for r in g)} "
      f"(flat: {sum(r['ansturn_any_flat'] for r in g)})")
print(f"  gold already in lifecycle view for these 19: {sum(1 for r in g if CD[r['qid']]['in_view'])}")

print(); print("="*72); print("G. COLLATERAL: THE 91 QUESTIONS THAT LOSE THEIR ANNOTATED ANSWER TURN"); print("="*72)
l=[r for r in NA if r["ansturn_any_flat"] and not r["ansturn_any_cov"]]
print(f"  n={len(l)}  currently CORRECT {sum(r['judge'] for r in l)}  (= questions put at risk)")
print(f"  by shape: {dict(Counter(r['shape'] for r in l))}")
print(f"  ratio risk:reward  = {sum(r['judge'] for r in l)} at-risk-correct  vs  {sum(1 for r in g if not r['judge'])} recoverable-wrong")
lg=[r for r in NA if r["gold_in_flat"] and not r["gold_in_cov"]]
print(f"  gold-string losers n={len(lg)}  currently CORRECT {sum(r['judge'] for r in lg)}")
