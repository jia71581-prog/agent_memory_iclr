"""Adversarial round 3: verify date density is real; recompute the context-budget curve;
test whether a shape-agnostic alternative (session-coverage retrieval) already gets the gain."""
import json, sys, statistics as st
from collections import Counter, defaultdict
sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from run_baseline import bm25_rank
from v5_shape import classify_shape, resolve_relative_dates, parse_qdate

R = "/root/agent_memory_iclr/results"
rows = {r["qid"]: r for r in json.load(open(f"{R}/_zz_ver_rows.json"))}
data = load_dataset("s_cleaned")
BY = {q["question_id"]: q for q in data}
anc = sorted([r for r in rows.values() if r["shape"] == "TEMP.anchor"], key=lambda r: r["qid"])
print("loaded", len(data), "anchor", len(anc), flush=True)

print("\n=== D. is the date density real? (sessions per distinct date) ===")
for r in anc[:6]:
    q = BY[r["qid"]]
    c = Counter(d.split(" (")[0] for d in q["haystack_dates"])
    ds = [parse_qdate(d) for d in q["haystack_dates"]]
    print(f"  {r['qid']:14s} nsess={len(ds)} distinct_days={len(c)} span={(max(ds)-min(ds)).days}d "
          f"max_sess_per_day={max(c.values())} median_per_day={st.median(c.values()):.1f}")
allsp=[]; alldens=[]
for r in anc:
    q=BY[r["qid"]]; ds=[parse_qdate(d) for d in q["haystack_dates"]]
    sp=(max(ds)-min(ds)).days; allsp.append(sp); alldens.append(len(ds)/max(1,sp))
print(f"  over 26 anchor Qs: median span {st.median(allsp):.0f} days, median density "
      f"{st.median(alldens):.2f} sessions/day -> a +/-3d window (7 days) covers ~{7*st.median(alldens):.0f} sessions")

print("\n=== E. CONTEXT-BUDGET CURVE: whole-session expansion, TEMP.anchor ===")
def sess_chars(sess):
    return sum(len(t.get("content") or "") for t in sess)
BUD = [10_000, 20_000, 39_000, 60_000, 80_000, 125_000, 200_000, 400_000, 10**9]
curve = {b: [] for b in BUD}
flatchars = []
full_hay = []
for r in anc:
    q = BY[r["qid"]]
    sid2idx = {s: i for i, s in enumerate(q["haystack_session_ids"])}
    ev = {sid2idx[s] for s in q["answer_session_ids"] if s in sid2idx}
    units = []
    for si,(date,sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for t in sess:
            if t.get("content"): units.append((si, t["content"]))
    order = bm25_rank(q["question"], [u[1] for u in units])
    flatchars.append(sum(len(units[i][1]) for i in order[:20]))
    sc = [sess_chars(s) for s in q["haystack_sessions"]]
    full_hay.append(sum(sc))
    # session ranking by best turn score (= order position of first turn of that session)
    best = {}
    for rank,i in enumerate(order):
        best.setdefault(units[i][0], rank)
    srank = sorted(best, key=lambda s: best[s])
    for b in BUD:
        used, taken = 0, set()
        for s in srank:
            if used + sc[s] > b: break
            taken.add(s); used += sc[s]
        curve[b].append(ev <= taken)
print(f"  flat top-20 turns: median {st.median(flatchars):.0f} chars, mean {st.mean(flatchars):.0f}; "
      f"complete recall {sum(r['flat_all'] for r in anc)/len(anc):.3f}")
print(f"  FULL haystack: median {st.median(full_hay):.0f} chars (mean {st.mean(full_hay):.0f})")
print(f"  {'budget':>10s} {'completeRecall':>14s}")
for b in BUD:
    print(f"  {('full' if b>10**8 else f'{b//1000}k'):>10s} {sum(curve[b])/len(anc):14.3f}")

print("\n=== F. shape-AGNOSTIC alternative: same-budget session-coverage retrieval ===")
TOP_K=20
res=defaultdict(lambda: [0,0,0])
for r in rows.values():
    res[r["shape"]][0]+=1; res[r["shape"]][1]+=r["flat_all"]; res[r["shape"]][2]+=r["cov_all"]
print(f"  {'shape':20s} {'n':>4s} {'flat_all':>9s} {'cov_all':>9s} {'delta':>7s}")
for s,(n,f,c) in sorted(res.items(), key=lambda kv: kv[1][1]/kv[1][0]):
    print(f"  {s:20s} {n:4d} {f/n:9.3f} {c/n:9.3f} {(c-f)/n:+7.3f}")
a=[r for r in anc]
print(f"  TEMP.anchor: flat_all {sum(r['flat_all'] for r in a)}/{len(a)}  -> cov_all {sum(r['cov_all'] for r in a)}/{len(a)}")
print(f"  TEMP.anchor multi-ev: flat {sum(r['flat_all'] for r in a if r['n_ev']>1)}/{sum(1 for r in a if r['n_ev']>1)}"
      f" -> cov {sum(r['cov_all'] for r in a if r['n_ev']>1)}/{sum(1 for r in a if r['n_ev']>1)}")
errs=[r for r in a if not r["judge"]]
print(f"  the 5 errors under cov_all: {[(r['qid'], r['flat_all'], r['cov_all']) for r in errs]}")

print("\n=== G. does the DATE WINDOW beat same-budget coverage retrieval on the 5 errors? ===")
for r in errs:
    q=BY[r["qid"]]
    rel=resolve_relative_dates(r["q"], r["qdate"])
    sid2idx={s:i for i,s in enumerate(q["haystack_session_ids"])}
    ev={sid2idx[s] for s in q["answer_session_ids"] if s in sid2idx}
    ds=[parse_qdate(d) for d in q["haystack_dates"]]
    sc=[sess_chars(s) for s in q["haystack_sessions"]]
    if not rel:
        print(f"  {r['qid']:14s} NO RESOLVED DATE (rule {r['fired']}) -> strategy inapplicable"); continue
    _,d0,tol=rel[0]
    win=[i for i in range(len(ds)) if abs((ds[i]-d0).days)<=tol]
    print(f"  {r['qid']:14s} window={len(win)} sess ({sum(sc[i] for i in win)} chars), ev in window "
          f"{len(set(win)&ev)}/{len(ev)}, ev complete in window={ev<=set(win)}")
