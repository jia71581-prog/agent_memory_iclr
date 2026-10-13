"""(1) turn-level completeness of flat-20 / cov-20 / flat-50 / cov-turnlevel for the 62 errors
   (2) character budget of flat-20 vs cov-20 (matched-budget check)
   (3) how many of the 62 have a truncated (20k-capped) lifecycle view"""
import json,sys
from collections import defaultdict,Counter
sys.path.insert(0,"/root/agent_memory_iclr/code"); sys.path.insert(0,"/root/agent_memory_iclr/results")
import _recon_ctx as RC
from common import load_dataset
from run_baseline import bm25_rank
R="/root/agent_memory_iclr/results"
V={r['qid']:r for r in json.load(open(f"{R}/_v5_offline_val.json"))}
ROWS=json.load(open(f"{R}/_v5_shape_rows.json"))
NA=[r for r in ROWS if not r['is_abs']]
ERR=[r for r in NA if not r['judge']]
errset={r['qid']:r['shape'] for r in ERR}
data={q['question_id']:q for q in load_dataset("s_cleaned")}
K=20
res={}; chars_flat=[]; chars_cov=[]
for qid in list(errset)+[r['qid'] for r in NA if r['qid'] not in errset]:
    q=data[qid]
    units=[]
    for si,(date,sess) in enumerate(zip(q["haystack_dates"],q["haystack_sessions"])):
        for t in sess:
            if t.get("content"): units.append((si,t["content"]))
    docs=[u[1] for u in units]
    order=bm25_rank(q["question"],docs)
    flat=order[:K]
    best=defaultdict(list)
    for i in order: best[units[i][0]].append(i)
    sess_rank=sorted(best,key=lambda s: order.index(best[s][0]))
    cov=[];p=0
    while len(cov)<K and p<2:
        for s in sess_rank:
            if len(cov)>=K: break
            if len(best[s])>p: cov.append(best[s][p])
        p+=1
    cf=sum(len(docs[i]) for i in flat); cc=sum(len(docs[i]) for i in cov)
    chars_flat.append(cf); chars_cov.append(cc)
    if qid in errset:
        ev=set(RC.evidence_turns(q))
        g=lambda idxs: sum(1 for i in idxs if docs[i] in ev)
        res[qid]={"shape":errset[qid],"flat_all":V[qid]['flat_all'],"cov_all":V[qid]['cov_all'],
                  "n_ev":len(ev),"ev_in_flat20":g(flat),"ev_in_cov20":g(cov),
                  "ev_in_flat50":g(order[:50]),"ev_in_flat100":g(order[:100]),
                  "chars_flat20":cf,"chars_cov20":cc,"chars_flat50":sum(len(docs[i]) for i in order[:50])}
json.dump(res,open(f"{R}/_vrf/v3_budget.json","w"))
import statistics as st
print("470 non-abs: mean chars flat20=%.0f cov20=%.0f (ratio %.3f) median %.0f/%.0f"%(
  st.mean(chars_flat),st.mean(chars_cov),st.mean(chars_cov)/st.mean(chars_flat),
  st.median(chars_flat),st.median(chars_cov)))
comp=[v for v in res.values() if v['flat_all']]
inc=[v for v in res.values() if not v['flat_all']]
def f(G,key): return sum(1 for v in G if v[key]>=v['n_ev'])
for nm,G in (("45 flat_all",comp),("17 not flat_all",inc)):
    print(f"{nm}: n={len(G)} turn-level complete: flat20={f(G,'ev_in_flat20')} cov20={f(G,'ev_in_cov20')} flat50={f(G,'ev_in_flat50')} flat100={f(G,'ev_in_flat100')}")
print("\ncov-fixed 7 (session-level) -> turn level:")
for qid,v in res.items():
    if (not v['flat_all']) and v['cov_all']:
        print("  ",qid,v['shape'],"n_ev",v['n_ev'],"flat20",v['ev_in_flat20'],"cov20",v['ev_in_cov20'])
