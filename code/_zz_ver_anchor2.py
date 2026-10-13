"""Adversarial round 2: is the date window SELECTIVE (precision), and does a null date do as well?
Also: does complete-recall predict correctness within TEMP.anchor?"""
import json, sys, random, math
from datetime import datetime, timedelta
from collections import Counter
sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from v5_shape import classify_shape, resolve_relative_dates, parse_qdate

R = "/root/agent_memory_iclr/results"
rows = {r["qid"]: r for r in json.load(open(f"{R}/_zz_ver_rows.json"))}
data = load_dataset("s_cleaned")
print("dataset loaded", len(data), flush=True)
BY = {q["question_id"]: q for q in data}

anc = [r for r in rows.values() if r["shape"] == "TEMP.anchor"]
anc.sort(key=lambda r: r["qid"])

print("\n=== A. WINDOW SELECTIVITY: how many haystack sessions fall inside the resolved window ===")
print(f"{'qid':18s} {'nsess':>5s} {'span_d':>6s} {'tol':>4s} {'inWin':>5s} {'evInWin':>7s} {'rank_of_gold':>12s} {'judge':>5s} {'flat_all':>8s}")
sel = []
for r in anc:
    q = BY[r["qid"]]
    rel = resolve_relative_dates(r["q"], r["qdate"])
    if not rel:
        continue
    dates = [parse_qdate(d) for d in q["haystack_dates"]]
    span = (max(dates) - min(dates)).days
    sid2idx = {s: i for i, s in enumerate(q["haystack_session_ids"])}
    evidx = {sid2idx[s] for s in q["answer_session_ids"] if s in sid2idx}
    # union over all resolved expressions
    inwin = set()
    for _, d, tol in rel:
        for i, dd in enumerate(dates):
            if abs((dd - d).days) <= tol:
                inwin.add(i)
    # rank of the nearest evidence session when sorting all sessions by |date - resolved|
    d0 = rel[0][1]
    order = sorted(range(len(dates)), key=lambda i: (abs((dates[i] - d0).days), i))
    rank_gold = min(order.index(i) for i in evidx) + 1 if evidx else None
    tol0 = rel[0][2]
    sel.append(dict(qid=r["qid"], nsess=len(dates), span=span, tol=tol0,
                    inwin=len(inwin), ev_inwin=len(inwin & evidx), n_ev=len(evidx),
                    rank_gold=rank_gold, judge=r["judge"], flat_all=r["flat_all"],
                    hit=bool(inwin & evidx)))
    print(f"{r['qid']:18s} {len(dates):5d} {span:6d} {tol0:4d} {len(inwin):5d} "
          f"{len(inwin & evidx):3d}/{len(evidx):<3d} {rank_gold:12d} {str(r['judge']):>5s} {str(r['flat_all']):>8s}")

import statistics as st
print(f"\n  median sessions inside window = {st.median([x['inwin'] for x in sel]):.1f}  "
      f"mean = {st.mean([x['inwin'] for x in sel]):.2f}  max = {max(x['inwin'] for x in sel)}")
print(f"  windows containing 0 sessions: {sum(1 for x in sel if x['inwin']==0)}")
print(f"  windows containing >=2 sessions: {sum(1 for x in sel if x['inwin']>=2)}/{len(sel)}")
print(f"  PRECISION of window (ev sessions / sessions in window): "
      f"{sum(x['ev_inwin'] for x in sel)}/{sum(x['inwin'] for x in sel)} = "
      f"{sum(x['ev_inwin'] for x in sel)/max(1,sum(x['inwin'] for x in sel)):.3f}")
print(f"  COMPLETE ev coverage by window (all ev sessions inside): "
      f"{sum(1 for x in sel if x['ev_inwin']==x['n_ev'])}/{len(sel)} = "
      f"{sum(1 for x in sel if x['ev_inwin']==x['n_ev'])/len(sel):.3f}")
print(f"  rank-1 of nearest ev session (nearest-date ranking): "
      f"{sum(1 for x in sel if x['rank_gold']==1)}/{len(sel)} = {sum(1 for x in sel if x['rank_gold']==1)/len(sel):.3f}")

print("\n=== B. NULL MODELS for the 19/20 'within tolerance' hit rate ===")
random.seed(20260917)
NREP = 2000
def hitrate(datefn, tolfn=None):
    """fraction of the 20 resolved questions whose surrogate date lands within tol of an ev date"""
    hits = 0
    for r in anc:
        q = BY[r["qid"]]
        rel = resolve_relative_dates(r["q"], r["qdate"])
        if not rel: continue
        evs = [parse_qdate(d) for d in r["ev_dates"]]
        ok = False
        for expr, d, tol in rel:
            dd = datefn(r, q, d, tol)
            if dd is None: continue
            if any(abs((e - dd).days) <= tol for e in evs):
                ok = True
        hits += ok
    return hits

n_res = sum(1 for r in anc if resolve_relative_dates(r["q"], r["qdate"]))
print(f"  TRUE resolved-date hit rate: {hitrate(lambda r,q,d,tol: d)}/{n_res}")

# N1: date = question_date itself (trivial 'most recent' anchor)
print(f"  N1 qdate itself (no resolution at all): {hitrate(lambda r,q,d,tol: parse_qdate(r['qdate']))}/{n_res}")
# N2: random date uniform over the haystack span
def n2(r,q,d,tol):
    dates=[parse_qdate(x) for x in q["haystack_dates"]]
    lo,hi=min(dates),max(dates)
    return lo + timedelta(days=random.randint(0,(hi-lo).days))
tot=0
for _ in range(200): tot += hitrate(n2)
print(f"  N2 uniform-random date in haystack span: mean {tot/200:.2f}/{n_res} = {tot/200/n_res:.3f}")
# N3: permuted offsets -- take the (qdate - resolved) offset from a DIFFERENT anchor question
offs=[]
for r in anc:
    rel = resolve_relative_dates(r["q"], r["qdate"])
    for _,d,tol in rel: offs.append((parse_qdate(r["qdate"])-d).days)
print(f"  offsets used (days back): {sorted(offs)}")
def n3(r,q,d,tol):
    return parse_qdate(r["qdate"]) - timedelta(days=random.choice(offs))
tot=0
for _ in range(500): tot += hitrate(n3)
print(f"  N3 permuted offset from another anchor Q: mean {tot/500:.2f}/{n_res} = {tot/500/n_res:.3f}")
# N4: random haystack SESSION date (i.e. 'pick any session')
def n4(r,q,d,tol):
    return random.choice([parse_qdate(x) for x in q["haystack_dates"]])
tot=0
for _ in range(500): tot += hitrate(n4)
print(f"  N4 random haystack session date: mean {tot/500:.2f}/{n_res} = {tot/500/n_res:.3f}")

print("\n=== C. does complete-recall predict correctness inside TEMP.anchor? ===")
a=sum(1 for r in anc if r["flat_all"] and r["judge"]); b=sum(1 for r in anc if r["flat_all"] and not r["judge"])
c=sum(1 for r in anc if not r["flat_all"] and r["judge"]); d=sum(1 for r in anc if not r["flat_all"] and not r["judge"])
print(f"  flat_all=T: {a} correct / {b} wrong (acc {a/(a+b):.3f})")
print(f"  flat_all=F: {c} correct / {d} wrong (acc {c/(c+d):.3f})")
try:
    from scipy.stats import fisher_exact
    print(f"  Fisher exact p = {fisher_exact([[a,b],[c,d]])[1]:.4f}")
except ImportError: pass
print(f"  => {c}/{c+d} of the 'retrieval-failed' anchor questions are answered CORRECTLY anyway")

json.dump(sel, open(f"{R}/_zz_ver_sel.json","w"))
