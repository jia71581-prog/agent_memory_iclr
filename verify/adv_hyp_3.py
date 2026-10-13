"""Test 3: budget-matched control + permuted-table control.
Q: is the D1/D2 gain semantics (correct hyponyms) or just a larger printed set?
Controls:
  A. nohyp_wide: no hyponym table, but rel_keep/max_groups relaxed so the printed set is
     >= the hyponym arm's size. If D1/D2 match, the table adds nothing beyond volume.
  B. permuted table: same 21 categories, but each category gets ANOTHER category's
     hyponym list (identical size/token statistics, semantics destroyed).
  C. truncated: hyponym arm truncated per-question to the nohyp item count.
"""
import json, pickle, re, sys, os, statistics as st, random
sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains
import v5_chains as V

D = {x["qid"]: x for x in pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl","rb"))}
G = pickle.load(open("/root/agent_memory_iclr/results/_v5_gold.pkl","rb"))
base = json.load(open("/root/agent_memory_iclr/verify/adv_hyp_rows.json"))
keep = {r["qid"]: r for r in base}
FULL = dict(V.HYPONYMS)
cats = list(FULL)

def permuted(seed):
    rnd = random.Random(seed)
    tgt = cats[:]
    while True:
        rnd.shuffle(tgt)
        if all(a != b for a, b in zip(cats, tgt)):
            break
    return {c: FULL[t] for c, t in zip(cats, tgt)}

def run(store, q, gold_sidx, tbl, uh=True, **kw):
    V.HYPONYMS = tbl
    kept, _ = V.aggregation_printed_groups(store, q, uh, **kw)
    V.HYPONYMS = FULL
    printed = [[it[1] for it in g["items"]] for g in kept]
    allit = [c for gi in printed for c in gi]
    return {"items": len(allit), "best": max((len(gi) for gi in printed), default=0),
            "gold_hit": len({c.get("sidx") for c in allit} & gold_sidx),
            "printed": printed}

out = {}
perms = [permuted(s) for s in (1, 2, 3)]
for qid, r in keep.items():
    x = D[qid]; gs = set(G[qid]["gold_sidx"])
    st5 = V.merge_chains(assemble_chains(x["facts"]))
    rec = {"gnum": r["gnum"], "ngs": r["ngs"], "judge": r["judge"]}
    rec["v5"] = run(st5, x["question"], gs, FULL, True)
    rec["nohyp"] = run(st5, x["question"], gs, FULL, False)
    # A: no table, relaxed retention so more groups/items survive
    for tag, kw in [("wide_rk15", {"rel_keep": 0.15}), ("wide_rk05", {"rel_keep": 0.05}),
                    ("wide_rk05_g16", {"rel_keep": 0.05, "max_groups": 16}),
                    ("wide_dfc", {"rel_keep": 0.05, "max_groups": 16, "df_cap": 0.60})]:
        rec[tag] = run(st5, x["question"], gs, FULL, False, **kw)
    # B: permuted tables
    for i, p in enumerate(perms):
        rec[f"perm{i}"] = run(st5, x["question"], gs, p, True)
    # C: truncate hyponym arm to nohyp item budget (keep renderer's own group order)
    budget = rec["nohyp"]["items"]
    tr, cnt, best = [], 0, 0
    for gi in rec["v5"]["printed"]:
        take = gi[:max(0, budget - cnt)]
        cnt += len(take); best = max(best, len(take))
        tr += take
        if cnt >= budget: break
    rec["trunc"] = {"items": len(tr), "best": best,
                    "gold_hit": len({c.get("sidx") for c in tr} & gs)}
    for k in list(rec):
        if isinstance(rec[k], dict): rec[k].pop("printed", None)
    out[qid] = rec

json.dump(out, open("/root/agent_memory_iclr/verify/adv_hyp_ctrl.json", "w"), indent=1)
rows = list(out.values()); n = len(rows); tgs = sum(r["ngs"] for r in rows)
def d1(r,t): return r[t]["items"] >= r["gnum"] and r[t]["gold_hit"] == r["ngs"]
def d2(r,t): return r[t]["best"] >= r["gnum"] and r[t]["gold_hit"] == r["ngs"]
print(f"n={n} gold sessions={tgs}\n{'arm':16s} {'D1':>11s} {'D2':>11s} {'recall':>14s} {'items sum':>9s} {'med':>5s}")
for t in ["v5","nohyp","wide_rk15","wide_rk05","wide_rk05_g16","wide_dfc",
          "perm0","perm1","perm2","trunc"]:
    a=sum(d1(r,t) for r in rows); b=sum(d2(r,t) for r in rows); c=sum(r[t]["gold_hit"] for r in rows)
    print(f"{t:16s} {a/n:7.3f}({a:2d}) {b/n:7.3f}({b:2d}) {c/tgs:8.3f}({c:3d}) "
          f"{sum(r[t]['items'] for r in rows):9d} {st.median([r[t]['items'] for r in rows]):5.1f}")
