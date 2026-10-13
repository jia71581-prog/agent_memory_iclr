import json, sys, statistics as st
from collections import Counter
sys.path.insert(0,"/root/agent_memory_iclr/code")
R="/root/agent_memory_iclr/results"
rows=json.load(open("/root/agent_memory_iclr/verify/adv_cov1.json"))
CD={r["qid"]:r for r in json.load(open(f"{R}/coverage_diag.json"))}
QC={json.loads(l)["qid"]: json.loads(l) for l in open(f"{R}/full500_qc.jsonl")}
NA=[r for r in rows if not r["is_abs"]]; AB=[r for r in rows if r["is_abs"]]

print("H. dead branch check")
print(f"  min n_sess over 500 = {min(r['n_sess'] for r in rows)}  -> pass-1 (2nd turn/session) fires on "
      f"{sum(1 for r in rows if r['n_sess']<20)}/500 questions")
print(f"  docstring says 'each session its top-2 turns'; realised turns/session = "
      f"{st.mean([20/r['n_cov_sess'] for r in rows]):.2f} everywhere")
print(f"  questions with 0 annotated has_answer turns: {sum(1 for r in rows if r['n_ansturns']==0)}/500")
print(f"  mean annotated answer turns per question: {st.mean([r['n_ansturns'] for r in rows]):.2f}")

print("\nI. abstention arm (30 _abs, excluded from the claim's 470)")
print(f"  ansturn_any: flat {sum(r['ansturn_any_flat'] for r in AB)} cov {sum(r['ansturn_any_cov'] for r in AB)}")
print(f"  n_ansturns==0 among abs: {sum(1 for r in AB if r['n_ansturns']==0)}/30 (abstention Qs have no evidence turn by design)")
print(f"  flat_all {sum(r['flat_all'] for r in AB)} cov_all {sum(r['cov_all'] for r in AB)}")

print("\nJ. is session breadth in the RAW channel even the binding constraint?")
na=[CD[r['qid']] for r in NA]
n=len(na)
print(f"  gold in lifecycle view       {sum(x['in_view'] for x in na)/n:.3f}")
print(f"  gold in BM25 top-20 raw      {sum(x['in_raw20'] for x in na)/n:.3f}")
print(f"  gold in either (shipped ctx) {sum(x['in_view'] or x['in_raw20'] for x in na)/n:.3f}")
print(f"  raw-channel-unique gold      {sum(1 for x in na if x['in_raw20'] and not x['in_view'])}/470"
      f"  <- the raw channel's only irreplaceable contribution, and it is a DEPTH contribution")
errs=[r for r in NA if not QC[r['qid']]['judge']]
print(f"  errors={len(errs)}; of them evidence already in shipped ctx: "
      f"{sum(1 for r in errs if CD[r['qid']]['in_view'] or CD[r['qid']]['in_raw20'])}/{len(errs)}")

print("\nK. net effect on the shipped context if coverage were applied unconditionally")
g=[r for r in NA if r["ansturn_any_cov"] and not r["ansturn_any_flat"]]
l=[r for r in NA if r["ansturn_any_flat"] and not r["ansturn_any_cov"]]
print(f"  annotated answer turn: +{len(g)} / -{len(l)}  => net {len(g)-len(l):+d} questions on 470")
print(f"  binomial sign test on the {len(g)+len(l)} discordant pairs:")
try:
    from scipy.stats import binomtest
    print(f"    p = {binomtest(len(g), len(g)+len(l), 0.5, alternative='two-sided').pvalue:.3e}  (direction: WORSE)")
except Exception as e: print("   ",e)
