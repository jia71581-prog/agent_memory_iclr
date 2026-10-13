"""Adversarial stress tests on the v5 part(a) vs part(b) decomposition claim.

(1) apples-to-apples v4 arm: full current view AND the deployed 20k-char capped view,
    scored with the SAME two-part criterion (items >= gold AND all gold sessions present).
(2) inflation / exact-count diagnostics: how far the printed set is from the gold count.
(3) statistics: McNemar exact + bootstrap CI on the merge-vs-nomerge deltas.
(4) proxy validity: does D1/D2 predict the v4 pipeline's judge on these 60 questions?
(5) hyperparameter sensitivity of the merge-vs-nomerge delta.
(6) fact loss caused by merge_chains, split by gold session.
No API.
"""
import hashlib
import itertools
import json
import math
import os
import pickle
import random
import re
import statistics as st
import sys
from collections import Counter

sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains, render_view, _norm
import v5_chains as V
from v5_chains import merge_chains, aggregation_printed_groups, aggregation_kind

D = {x["qid"]: x for x in pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl", "rb"))}
G = pickle.load(open("/root/agent_memory_iclr/results/_v5_gold.pkl", "rb"))
ROWS = json.load(open("/root/agent_memory_iclr/results/_adv_recompute.json"))
ENUM = [r for r in ROWS if r["enum"]]
ENUMQ = {r["qid"] for r in ENUM}
CAP = 20000   # run_qc.py VIEW_CHAR_CAP


def cap_view(txt, cap=CAP):
    if len(txt) <= cap:
        return txt
    kept, total = [], 0
    for line in txt.splitlines():
        if total + len(line) + 1 > cap:
            break
        kept.append(line)
        total += len(line) + 1
    return "\n".join(kept)


def view_arm(store, gold_sidx, mode="current", cap=None):
    """Score the plain lifecycle view with the same 2-part criterion.
    A rendered LINE maps back to the facts it prints, so gold-session coverage is exact."""
    txt = render_view(store, mode)
    full_lines = txt.splitlines()
    kept = set(cap_view(txt, cap).splitlines()) if cap else set(full_lines)
    vals, sids = set(), set()
    for k in sorted(store):
        chain = store[k]
        if chain[0]["relation"] == "accumulative":
            vis = chain
        elif mode == "current":
            last = (chain[-1]["date"], chain[-1].get("sidx"))
            vis = [c for c in chain if (c["date"], c.get("sidx")) == last]
        else:
            vis = chain
        # find this chain's rendered line to know whether the cap kept it
        line = None
        for l in full_lines:
            if chain[0]["attribute"] in l and chain[0]["entity"] in l:
                line = l
                break
        if cap and line is not None and line not in kept:
            continue
        for c in vis:
            vals.add(_norm(c["value"]))
            sids.add(c.get("sidx"))
    return {"items": len(vals), "gold_hit": len(sids & gold_sidx), "chars": len(cap_view(txt, cap) if cap else txt)}


def d1(r, t):
    return r[f"{t}_items"] >= r["gold_num"] and r[f"{t}_gold_hit"] == r["n_gold_sess"]


def d2(r, t):
    return r[f"{t}_best_group"] >= r["gold_num"] and r[f"{t}_gold_hit"] == r["n_gold_sess"]


# ---------------------------------------------------------------- (1) v4 arms
print("=== (1) apples-to-apples v4 view arms on the same 60 enum questions ===")
v4 = {}
for r in ENUM:
    x = D[r["qid"]]
    gs = set(G[r["qid"]]["gold_sidx"])
    s4 = assemble_chains(x["facts"])
    s5 = merge_chains(s4)
    v4[r["qid"]] = {
        "cur_full": view_arm(s4, gs, "current", None),
        "cur_cap": view_arm(s4, gs, "current", CAP),
        "hist_cap": view_arm(s4, gs, "historical", CAP),
        "cur_full_m": view_arm(s5, gs, "current", None),
        "cur_cap_m": view_arm(s5, gs, "current", CAP),
    }
tot_gs = sum(r["n_gold_sess"] for r in ENUM)
print(f"{'arm':38s} {'D1':>7s} {'recall':>7s} {'items_med':>9s} {'chars_med':>9s}")
for lab, key in [("v4 current view, UNCAPPED", "cur_full"),
                 ("v4 current view, deployed 20k cap", "cur_cap"),
                 ("v4 historical view, 20k cap", "hist_cap"),
                 ("MERGED current view, uncapped", "cur_full_m"),
                 ("MERGED current view, 20k cap", "cur_cap_m")]:
    ok = sum(1 for r in ENUM if v4[r["qid"]][key]["items"] >= r["gold_num"]
             and v4[r["qid"]][key]["gold_hit"] == r["n_gold_sess"])
    rec = sum(v4[r["qid"]][key]["gold_hit"] for r in ENUM) / tot_gs
    print(f"{lab:38s} {ok/len(ENUM):7.3f} {rec:7.3f} "
          f"{st.median([v4[r['qid']][key]['items'] for r in ENUM]):9.1f} "
          f"{st.median([v4[r['qid']][key]['chars'] for r in ENUM]):9.0f}")
for t, lab in [("v5", "v5 agg block (merge)"), ("nomerge", "v5 agg block (no merge)")]:
    print(f"{lab:38s} {sum(d1(r,t) for r in ENUM)/len(ENUM):7.3f} "
          f"{sum(r[f'{t}_gold_hit'] for r in ENUM)/tot_gs:7.3f} "
          f"{st.median([r[f'{t}_items'] for r in ENUM]):9.1f} "
          f"{st.median([r[f'{t}_chars'] for r in ENUM]):9.0f}")

# ---------------------------------------------------------------- (2) inflation
print("\n=== (2) inflation of the printed item set vs the gold count ===")
for t in ("v5", "nomerge"):
    ratios = [r[f"{t}_items"] / r["gold_num"] for r in ENUM]
    bratio = [r[f"{t}_best_group"] / r["gold_num"] for r in ENUM]
    exact = sum(1 for r in ENUM if r[f"{t}_best_group"] == r["gold_num"])
    exact_un = sum(1 for r in ENUM if r[f"{t}_items"] == r["gold_num"])
    print(f"{t:8s} union/gold median {st.median(ratios):5.2f} mean {st.mean(ratios):5.2f} "
          f"max {max(ratios):5.1f} | best_group/gold median {st.median(bratio):5.2f} | "
          f"exact best_group==gold {exact}/{len(ENUM)} union==gold {exact_un}/{len(ENUM)}")
print("D1 passes with union >= 3x gold:",
      sum(1 for r in ENUM if d1(r, "v5") and r["v5_items"] >= 3 * r["gold_num"]), "of",
      sum(1 for r in ENUM if d1(r, "v5")))

# ---------------------------------------------------------------- (3) statistics
print("\n=== (3) statistics on the merge-vs-nomerge delta ===")


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


for name, fn in [("D1", d1), ("D2", d2)]:
    b = sum(1 for r in ENUM if fn(r, "v5") and not fn(r, "nomerge"))
    c = sum(1 for r in ENUM if fn(r, "nomerge") and not fn(r, "v5"))
    print(f"{name}: merge-only wins b={b}, nomerge-only wins c={c}, "
          f"McNemar exact p={mcnemar_exact(b,c):.3f}, delta={(b-c)/len(ENUM):+.4f}")
random.seed(0)
d_boot = []
for _ in range(20000):
    s = [ENUM[random.randrange(len(ENUM))] for _ in ENUM]
    d_boot.append(sum(d1(r, "v5") for r in s) / len(s) - sum(d1(r, "nomerge") for r in s) / len(s))
d_boot.sort()
print(f"D1 delta bootstrap 95% CI: [{d_boot[int(0.025*len(d_boot))]:+.4f}, "
      f"{d_boot[int(0.975*len(d_boot))]:+.4f}] (point {sum(d1(r,'v5') for r in ENUM)/len(ENUM)-sum(d1(r,'nomerge') for r in ENUM)/len(ENUM):+.4f})")
rb = []
for _ in range(20000):
    s = [ENUM[random.randrange(len(ENUM))] for _ in ENUM]
    g = sum(r["n_gold_sess"] for r in s)
    rb.append(sum(r["v5_gold_hit"] for r in s) / g - sum(r["nomerge_gold_hit"] for r in s) / g)
rb.sort()
print(f"recall delta bootstrap 95% CI: [{rb[int(0.025*len(rb))]:+.4f}, {rb[int(0.975*len(rb))]:+.4f}]")
# how much of each delta comes from the single question bf659f65
sub = [r for r in ENUM if r["qid"] != "bf659f65"]
gsub = sum(r["n_gold_sess"] for r in sub)
print(f"drop qid bf659f65 -> D1 {sum(d1(r,'v5') for r in sub)/len(sub):.3f} vs "
      f"{sum(d1(r,'nomerge') for r in sub)/len(sub):.3f}; recall "
      f"{sum(r['v5_gold_hit'] for r in sub)/gsub:.3f} vs {sum(r['nomerge_gold_hit'] for r in sub)/gsub:.3f}")

# ---------------------------------------------------------------- (4) proxy validity
print("\n=== (4) does the derivability proxy track answered-correctness (v4 judge)? ===")
lab = [(d1(r, "v5"), r["judge"]) for r in ENUM if r["judge"] is not None]
tt = sum(1 for a, b in lab if a and b)
tf = sum(1 for a, b in lab if a and not b)
ft = sum(1 for a, b in lab if not a and b)
ff = sum(1 for a, b in lab if not a and not b)
print(f"n={len(lab)}  D1&judge={tt}  D1&!judge={tf}  !D1&judge={ft}  !D1&!judge={ff}")


def fisher(a, b, c, d):
    n = a + b + c + d
    def p(a_):
        b_, c_, d_ = a + b - a_, a + c - a_, d - (a - a_)
        if min(b_, c_, d_) < 0:
            return 0.0
        return (math.comb(a + b, a_) * math.comb(c + d, c_)) / math.comb(n, a + c)
    obs = p(a)
    return sum(p(x) for x in range(0, min(a + b, a + c) + 1) if p(x) <= obs + 1e-12)


print(f"Fisher exact two-sided p={fisher(tt,tf,ft,ff):.3f} "
      f"(judge rate | D1 = {tt/max(1,tt+tf):.3f}, | not D1 = {ft/max(1,ft+ff):.3f})")
lab2 = [(d2(r, "v5"), r["judge"]) for r in ENUM if r["judge"] is not None]
a2 = sum(1 for x, y in lab2 if x and y); b2 = sum(1 for x, y in lab2 if x and not y)
c2 = sum(1 for x, y in lab2 if not x and y); e2 = sum(1 for x, y in lab2 if not x and not y)
print(f"D2: judge rate | D2 = {a2/max(1,a2+b2):.3f}, | not D2 = {c2/max(1,c2+e2):.3f}, "
      f"Fisher p={fisher(a2,b2,c2,e2):.3f}")
# also: does the plain v4 uncapped view's own derivability predict judge?
lab3 = [(v4[r["qid"]]["cur_cap"]["items"] >= r["gold_num"]
         and v4[r["qid"]]["cur_cap"]["gold_hit"] == r["n_gold_sess"], r["judge"])
        for r in ENUM if r["judge"] is not None]
a3 = sum(1 for x, y in lab3 if x and y); b3 = sum(1 for x, y in lab3 if x and not y)
c3 = sum(1 for x, y in lab3 if not x and y); e3 = sum(1 for x, y in lab3 if not x and not y)
print(f"v4-capped-view derivability vs judge: rate|D={a3/max(1,a3+b3):.3f} "
      f"rate|!D={c3/max(1,c3+e3):.3f} Fisher p={fisher(a3,b3,c3,e3):.3f}")

json.dump({"v4arms": v4}, open("/root/agent_memory_iclr/results/_adv_v4arms.json", "w"), indent=1)
