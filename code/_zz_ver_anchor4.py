"""Adversarial round 4: packing-variant check for the 0.23@39k figure; binomial vs null;
n_ev confound; where the date window would rank vs BM25 on the two addressable errors."""
import json, sys, statistics as st
from collections import Counter, defaultdict
sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from run_baseline import bm25_rank
from v5_shape import resolve_relative_dates, parse_qdate

R="/root/agent_memory_iclr/results"
rows={r["qid"]:r for r in json.load(open(f"{R}/_zz_ver_rows.json"))}
data=load_dataset("s_cleaned"); BY={q["question_id"]:q for q in data}
anc=sorted([r for r in rows.values() if r["shape"]=="TEMP.anchor"], key=lambda r:r["qid"])

def sc_(s): return sum(len(t.get("content") or "") for t in s)
BUD=[39_000,125_000]
print("=== packing variants for the budget curve (TEMP.anchor, n=26) ===")
for variant in ["break_on_overflow","skip_and_continue"]:
    res={b:0 for b in BUD}
    for r in anc:
        q=BY[r["qid"]]
        sid2idx={s:i for i,s in enumerate(q["haystack_session_ids"])}
        ev={sid2idx[s] for s in q["answer_session_ids"] if s in sid2idx}
        units=[]
        for si,(d,s) in enumerate(zip(q["haystack_dates"],q["haystack_sessions"])):
            for t in s:
                if t.get("content"): units.append((si,t["content"]))
        order=bm25_rank(q["question"],[u[1] for u in units])
        best={}
        for k,i in enumerate(order): best.setdefault(units[i][0],k)
        srank=sorted(best,key=lambda s:best[s]); sc=[sc_(s) for s in q["haystack_sessions"]]
        for b in BUD:
            used,tk=0,set()
            for s in srank:
                if used+sc[s]>b:
                    if variant=="break_on_overflow": break
                    continue
                tk.add(s); used+=sc[s]
            res[b]+= (ev<=tk)
    print(f"  {variant:20s} " + "  ".join(f"{b//1000}k={res[b]}/26={res[b]/26:.3f}" for b in BUD))

print("\n=== binomial: is 19/20 above the 'random haystack session date' null (p0=0.567)? ===")
try:
    from scipy.stats import binomtest, fisher_exact
    print(f"  one-sided p = {binomtest(19,20,0.567,alternative='greater').pvalue:.2e}  -> resolver carries real signal")
except Exception as e: print(" scipy:",e)

print("\n=== n_ev confound: is TEMP.anchor's low recall explained by having more evidence sessions? ===")
allr=list(rows.values())
print(f"  {'shape':20s} {'n':>4s} {'mean n_ev':>9s} {'perEvRecall':>11s}")
for s in sorted({r['shape'] for r in allr}, key=lambda s: -st.mean([r['n_ev'] for r in allr if r['shape']==s])):
    g=[r for r in allr if r['shape']==s]
    print(f"  {s:20s} {len(g):4d} {st.mean([r['n_ev'] for r in g]):9.2f} "
          f"{st.mean([r['flat_hit']/max(1,r['n_ev']) for r in g]):11.3f}")
# matched comparison: n_ev==2 subset
print("\n  matched on n_ev==2:")
for s in sorted({r['shape'] for r in allr}):
    g=[r for r in allr if r['shape']==s and r['n_ev']==2]
    if len(g)>=5: print(f"    {s:20s} n={len(g):3d} flat_all={sum(r['flat_all'] for r in g)/len(g):.3f}")

print("\n=== the 2 potentially-addressable errors: does date-order beat BM25-order for the GOLD-BEARING session? ===")
for qid in ["gpt4_4929293b","eac54add"]:
    r=rows[qid]; q=BY[qid]
    rel=resolve_relative_dates(r["q"],r["qdate"]); _,d0,tol=rel[0]
    sid2idx={s:i for i,s in enumerate(q["haystack_session_ids"])}
    ev=sorted(sid2idx[s] for s in q["answer_session_ids"] if s in sid2idx)
    ds=[parse_qdate(d) for d in q["haystack_dates"]]
    gold=str(r["gold"]).lower()
    print(f"\n  {qid}  Q={r['q'][:95]}")
    print(f"    qdate={r['qdate']}  resolved={d0.date()} +/-{tol}   gold={gold[:60]}")
    for i in ev:
        txt=" ".join((t.get('content') or '') for t in q["haystack_sessions"][i]).lower()
        # crude: does this session mention distinctive gold tokens?
        toks=[w for w in gold.replace("'s"," ").split() if len(w)>3]
        hitw=[w for w in toks if w in txt]
        print(f"    ev session idx={i} date={q['haystack_dates'][i]} dist_to_resolved={abs((ds[i]-d0).days)}d "
              f"in_window={abs((ds[i]-d0).days)<=tol} gold_tokens_present={hitw}")
