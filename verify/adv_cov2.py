import json, random, statistics as st
from collections import Counter, defaultdict
R="/root/agent_memory_iclr/results"
rows=json.load(open("/root/agent_memory_iclr/verify/adv_cov1.json"))
NA=[r for r in rows if not r["is_abs"]]
QC={json.loads(l)["qid"]: json.loads(l) for l in open(f"{R}/full500_qc.jsonl")}
CD={r["qid"]: r for r in json.load(open(f"{R}/coverage_diag.json"))}
for r in rows: r["judge"]=QC[r["qid"]].get("judge")

print("="*70)
print("A. IS 'no shape loses' A THEOREM?  (monotone-by-construction check)")
print("="*70)
print(f"  flat_sess subset-of cov_sess : {sum(r['subset'] for r in rows)}/{len(rows)} questions")
print(f"  questions where cov_sess is the FULL top-20-session set: "
      f"{sum(1 for r in rows if r['n_cov_sess']==20)}/{len(rows)}")
print(f"  n_sess >= 20 (so pass-0 alone fills the budget): {sum(1 for r in rows if r['n_sess']>=20)}/{len(rows)}")
print(f"  cov depth per session: mean turns-per-session = "
      f"{st.mean([20/r['n_cov_sess'] for r in rows]):.3f}  (1.000 => exactly one turn per session)")
# zero-power demo: any partition gives 0 losses
random.seed(0)
worst=0
for trial in range(2000):
    lab=[random.randrange(12) for _ in NA]
    d=defaultdict(lambda:[0,0])
    for r,k in zip(NA,lab):
        d[k][0]+= (r["cov_all"] and not r["flat_all"]); d[k][1]+= (r["flat_all"] and not r["cov_all"])
    worst=max(worst, max(v[1] for v in d.values()))
print(f"  2000 RANDOM 12-way partitions of the same 470 questions: max per-cell loss = {worst}")
print(f"  -> 'delta>=0 for all 12 shapes' carries the same information as 'delta>=0 for 12 random cells'")

print()
print("="*70)
print("B. THE SAME SWITCH ON NON-MONOTONE (answer-bearing) METRICS")
print("="*70)
def tally(name, kf, kc, pool=NA):
    g=[r for r in pool if r[kc] and not r[kf]]; l=[r for r in pool if r[kf] and not r[kc]]
    print(f"  {name:38s} flat {sum(r[kf] for r in pool):3d}  cov {sum(r[kc] for r in pool):3d}   "
          f"gain {len(g):3d}  LOSS {len(l):3d}  net {len(g)-len(l):+d}")
    return g,l
tally("complete evidence SESSION set", "flat_all","cov_all")
g_gold,l_gold = tally("gold string in retrieved turns", "gold_in_flat","gold_in_cov")
g_any,l_any = tally("any annotated has_answer turn", "ansturn_any_flat","ansturn_any_cov")
g_all,l_all = tally("ALL annotated has_answer turns", "ansturn_all_flat","ansturn_all_cov")
print(f"  evidence-session turn depth: flat mean {st.mean([r['flat_evturns'] for r in NA]):.2f} turns "
      f"vs cov mean {st.mean([r['cov_evturns'] for r in NA]):.2f}  "
      f"(questions losing depth: {sum(1 for r in NA if r['cov_evturns']<r['flat_evturns'])}/470)")

print()
print("  per-shape delta, gold-string metric (the one that is NOT monotone):")
print(f"  {'shape':18s} {'n':>4s} {'flat':>5s} {'cov':>5s} {'gain':>5s} {'loss':>5s} {'delta':>6s}")
shapes=sorted({r["shape"] for r in NA})
nloss_shapes=0
for s in shapes:
    p=[r for r in NA if r["shape"]==s]
    gg=sum(1 for r in p if r["gold_in_cov"] and not r["gold_in_flat"])
    ll=sum(1 for r in p if r["gold_in_flat"] and not r["gold_in_cov"])
    nloss_shapes += ll>0
    print(f"  {s:18s} {len(p):4d} {sum(r['gold_in_flat'] for r in p):5d} {sum(r['gold_in_cov'] for r in p):5d} "
          f"{gg:5d} {ll:5d} {(gg-ll)/len(p):+6.3f}")
print(f"  shapes with >=1 loss on the gold-string metric: {nloss_shapes}/{len(shapes)}")

print()
print("  per-shape delta, annotated-evidence-turn metric (ansturn_any):")
nloss_shapes=0
for s in shapes:
    p=[r for r in NA if r["shape"]==s]
    gg=sum(1 for r in p if r["ansturn_any_cov"] and not r["ansturn_any_flat"])
    ll=sum(1 for r in p if r["ansturn_any_flat"] and not r["ansturn_any_cov"])
    nloss_shapes += ll>0
    print(f"  {s:18s} {len(p):4d} {sum(r['ansturn_any_flat'] for r in p):5d} {sum(r['ansturn_any_cov'] for r in p):5d} "
          f"{gg:5d} {ll:5d} {(gg-ll)/len(p):+6.3f}")
print(f"  shapes with >=1 loss on the annotated-evidence-turn metric: {nloss_shapes}/{len(shapes)}")

print()
print("="*70)
print("C. BUDGET MATCH")
print("="*70)
fc=st.mean([r["flat_chars"] for r in NA]); cc=st.mean([r["cov_chars"] for r in NA])
ft=st.mean([r["flat_toks"] for r in NA]); ct=st.mean([r["cov_toks"] for r in NA])
print(f"  20 turns either way. chars: flat {fc:.0f} vs cov {cc:.0f}  ({(cc-fc)/fc:+.1%})")
print(f"                       toks : flat {ft:.0f} vs cov {ct:.0f}  ({(ct-ft)/ft:+.1%})")
print(f"  questions where cov ships MORE chars: {sum(1 for r in NA if r['cov_chars']>r['flat_chars'])}/470")
