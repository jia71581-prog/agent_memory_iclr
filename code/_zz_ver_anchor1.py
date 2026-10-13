"""Adversarial verification round 1: cheap checks off existing intermediates."""
import json, sys, os
from collections import Counter, defaultdict
sys.path.insert(0, "/root/agent_memory_iclr/code")
from v5_shape import classify_shape, resolve_relative_dates, parse_qdate

R = "/root/agent_memory_iclr/results"
QS = json.load(open(f"{R}/_questions500.json"))
V = {r["qid"]: r for r in json.load(open(f"{R}/_v5_offline_val.json"))}
QC = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{R}/full500_qc.jsonl")}

print("questions500 keys:", list(QS[0].keys()))

# independent re-classification (do not trust _v5_shape_rows.json)
rows = []
for q in QS:
    sh, fired = classify_shape(q["question"])
    v = V[q["qid"]]
    rows.append(dict(qid=q["qid"], q=q["question"], shape=sh, fired=fired,
                     is_abs=q["qid"].endswith("_abs"), type=q["type"],
                     gold=q["answer"], judge=QC[q["qid"]]["judge"],
                     n_ev=v["n_ev"], n_sess=v["n_sess"],
                     flat_all=v["flat_all"], flat_hit=v["flat_hit"],
                     cov_all=v["cov_all"], cov_hit=v["cov_hit"],
                     ev_dates=v["ev_dates"], qdate=v["qdate"]))
json.dump(rows, open(f"{R}/_zz_ver_rows.json", "w"), ensure_ascii=False)

NA = [r for r in rows if not r["is_abs"]]
print(f"\nn={len(rows)} non-abs={len(NA)} abs={len(rows)-len(NA)}")

# ---- per-shape complete recall (flat top-20), on ALL 500 and on NA
print("\n=== per-shape flat top-20 COMPLETE recall (all 500) ===")
print(f"{'shape':20s} {'n':>4s} {'flat_all':>9s} {'mean n_ev':>10s} {'multiEv n':>9s} {'flat_all|mEv':>13s} {'perEvRecall':>12s} {'acc(NA)':>8s}")
tab = {}
for s in sorted({r["shape"] for r in rows}):
    g = [r for r in rows if r["shape"] == s]
    m = [r for r in g if r["n_ev"] > 1]
    na = [r for r in g if not r["is_abs"]]
    fa = sum(r["flat_all"] for r in g)/len(g)
    fam = sum(r["flat_all"] for r in m)/len(m) if m else float('nan')
    per = sum(r["flat_hit"]/max(1,r["n_ev"]) for r in g)/len(g)
    acc = sum(r["judge"] for r in na)/len(na) if na else float('nan')
    tab[s] = (len(g), fa, fam, len(m), per, acc)
    print(f"{s:20s} {len(g):4d} {fa:9.3f} {sum(r['n_ev'] for r in g)/len(g):10.2f} {len(m):9d} {fam:13.3f} {per:12.3f} {acc:8.3f}")
g = rows
print(f"{'ALL':20s} {len(g):4d} {sum(r['flat_all'] for r in g)/len(g):9.3f} "
      f"{sum(r['n_ev'] for r in g)/len(g):10.2f} "
      f"{sum(1 for r in g if r['n_ev']>1):9d} "
      f"{sum(r['flat_all'] for r in g if r['n_ev']>1)/sum(1 for r in g if r['n_ev']>1):13.3f} "
      f"{sum(r['flat_hit']/max(1,r['n_ev']) for r in g)/len(g):12.3f} "
      f"{sum(r['judge'] for r in NA)/len(NA):8.3f}")

# rank shapes by flat_all
print("\n=== ranking by flat_all (worst first), shapes with n>=10 ===")
for s,(n,fa,fam,nm,per,acc) in sorted(tab.items(), key=lambda kv: kv[1][1]):
    if n >= 10:
        print(f"  {s:20s} n={n:3d} flat_all={fa:.3f}  mEv={nm:3d} flat_all|mEv={fam:.3f} perEv={per:.3f} acc={acc:.3f}")

# ---- TEMP.anchor detail
anc = [r for r in rows if r["shape"] == "TEMP.anchor"]
print(f"\n=== TEMP.anchor n={len(anc)} (abs={sum(r['is_abs'] for r in anc)}) ===")
print(f"  flat_all = {sum(r['flat_all'] for r in anc)}/{len(anc)} = {sum(r['flat_all'] for r in anc)/len(anc):.4f}")
m = [r for r in anc if r["n_ev"]>1]
print(f"  multi-evidence n={len(m)} flat_all = {sum(r['flat_all'] for r in m)}/{len(m)} = {sum(r['flat_all'] for r in m)/len(m):.4f}")
na = [r for r in anc if not r["is_abs"]]
print(f"  non-abs n={len(na)} acc={sum(r['judge'] for r in na)}/{len(na)} = {sum(r['judge'] for r in na)/len(na):.4f}")
print(f"  errors: {[r['qid'] for r in na if not r['judge']]}")

# ---- date resolution
print("\n=== resolve_relative_dates on TEMP.anchor ===")
res_ok, res_hit = [], []
for r in anc:
    rel = resolve_relative_dates(r["q"], r["qdate"])
    if not rel:
        print(f"  UNRESOLVED {r['qid']} fired={r['fired']} | {r['q'][:90]}")
        continue
    res_ok.append(r)
    evs = [parse_qdate(d) for d in r["ev_dates"]]
    hit = any(abs((e-d).days) <= tol for _,d,tol in rel for e in evs)
    if hit: res_hit.append(r)
    r["_rel"] = [(e, d.strftime("%Y/%m/%d"), tol) for e,d,tol in rel]
    r["_hit"] = hit
print(f"  resolved {len(res_ok)}/{len(anc)}; hit-within-tolerance {len(res_hit)}/{len(res_ok)} = {len(res_hit)/len(res_ok):.4f}")
print(f"  misses: {[(r['qid'], r['q'][:60], r['_rel']) for r in res_ok if not r['_hit']]}")
errs = [r for r in anc if not r["is_abs"] and not r["judge"]]
print(f"  of {len(errs)} errors: resolved={sum(1 for r in errs if r.get('_rel'))} correctly-hit={sum(1 for r in errs if r.get('_hit'))}")
for r in errs:
    print(f"    ERR {r['qid']} hit={r.get('_hit')} n_ev={r['n_ev']} flat_all={r['flat_all']} rel={r.get('_rel')}")
    print(f"        Q: {r['q'][:120]}")
    print(f"        gold: {str(r['gold'])[:100]}  evdates={r['ev_dates']}")
json.dump([{k:v for k,v in r.items()} for r in anc], open(f"{R}/_zz_ver_anchor.json","w"), ensure_ascii=False, default=str)
