"""ADVERSARIAL recompute #3:
 (a) gold-fact loss under the view the PRODUCTION pipeline actually renders (20k cap),
 (b) does the hidden-fact rate predict judge correctness on the 227 triggered questions?
 (c) budget: characters added by the aggregation block vs the current view it supplements.
"""
import json, pickle, sys, math, statistics as st
from collections import Counter
sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains, render_view
from v5_chains import merge_chains, is_aggregation_question, render_aggregation_view

D = {x["qid"]: x for x in pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl", "rb"))}
G = pickle.load(open("/root/agent_memory_iclr/results/_v5_gold.pkl", "rb"))
QC = {json.loads(l)["qid"]: json.loads(l) for l in open("/root/agent_memory_iclr/results/full500_qc.jsonl")}
CAP = 20000


def visible(chain):
    if chain[0]["relation"] == "accumulative":
        return chain
    last = (chain[-1]["date"], chain[-1].get("sidx"))
    return [c for c in chain if (c["date"], c.get("sidx")) == last]


rows = []
T = Counter()
for qid, x in D.items():
    if not is_aggregation_question(x["question"])[0]:
        continue
    gold_sidx = set(G[qid]["gold_sidx"])
    s5 = merge_chains(assemble_chains(x["facts"]))
    s4 = assemble_chains(x["facts"])

    # which chain KEYS survive the alphabetical 20k line cap, for each store
    def surviving_keys(store):
        total, keep = 0, set()
        txt = render_view(store, "current")
        # render_view iterates sorted(store) and emits exactly one line per chain
        for k, line in zip(sorted(store), txt.splitlines()):
            if total + len(line) + 1 > CAP:
                break
            keep.add(k)
            total += len(line) + 1
        return keep, len(txt)

    keep5, chars5 = surviving_keys(s5)
    keep4, chars4 = surviving_keys(s4)

    def count(store, keep):
        gf = uncapped = capped = 0
        for k, chain in store.items():
            vis = visible(chain)
            for c in chain:
                if c.get("sidx") in gold_sidx:
                    gf += 1
                    if c in vis:
                        uncapped += 1
                        if k in keep:
                            capped += 1
        return gf, uncapped, capped

    gf5, un5, cp5 = count(s5, keep5)
    gf4, un4, cp4 = count(s4, keep4)
    T["gf5"] += gf5; T["un5"] += un5; T["cp5"] += cp5
    T["gf4"] += gf4; T["un4"] += un4; T["cp4"] += cp4
    agg = render_aggregation_view(s5, x["question"])
    rows.append({"qid": qid, "gf5": gf5, "un5": un5, "cp5": cp5,
                 "gf4": gf4, "un4": un4, "cp4": cp4,
                 "judge": QC.get(qid, {}).get("judge"),
                 "cur_chars": chars5, "agg_chars": len(agg),
                 "hidden_rate": 1 - un5 / gf5 if gf5 else 0.0})

print("=== (a) what the PRODUCTION view actually renders (run_qc.py VIEW_CHAR_CAP=20000) ===")
print(f"v4 UNMERGED store : gold facts {T['gf4']}  uncapped-current {T['un4']} "
      f"({T['un4']/T['gf4']:.3f})  production-capped {T['cp4']} ({T['cp4']/T['gf4']:.3f})")
print(f"v5 MERGED store   : gold facts {T['gf5']}  uncapped-current {T['un5']} "
      f"({T['un5']/T['gf5']:.3f})  production-capped {T['cp5']} ({T['cp5']/T['gf5']:.3f})")
print(f"  claimed loss channel (supersession truncation, v4 store) : {1-T['un4']/T['gf4']:.1%}")
print(f"  unclaimed loss channel (20k char cap, v4 store, on top)  : "
      f"{(T['un4']-T['cp4'])/T['gf4']:.1%}")
print(f"  TOTAL v4 production loss                                 : {1-T['cp4']/T['gf4']:.1%}")

print("\n=== (b) does the hidden-fact rate predict judge correctness? (n=227, v4 QC arm) ===")
j = [r for r in rows if r["judge"] is not None]
hs = sorted(j, key=lambda r: r["hidden_rate"])
half = len(hs) // 2
lo, hi = hs[:half], hs[half:]
print(f"judged {len(j)}; overall acc {sum(r['judge'] for r in j)/len(j):.3f}")
print(f"  LOW-hiding half  (median rate {st.median([r['hidden_rate'] for r in lo]):.3f}): "
      f"acc {sum(r['judge'] for r in lo)/len(lo):.3f}  n={len(lo)}")
print(f"  HIGH-hiding half (median rate {st.median([r['hidden_rate'] for r in hi]):.3f}): "
      f"acc {sum(r['judge'] for r in hi)/len(hi):.3f}  n={len(hi)}")
zero = [r for r in j if r["hidden_rate"] == 0]
nz = [r for r in j if r["hidden_rate"] > 0]
print(f"  ZERO hidden facts  : acc {sum(r['judge'] for r in zero)/max(1,len(zero)):.3f} n={len(zero)}")
print(f"  >0  hidden facts   : acc {sum(r['judge'] for r in nz)/max(1,len(nz)):.3f} n={len(nz)}")
# point-biserial correlation
xs = [r["hidden_rate"] for r in j]; ys = [1.0 if r["judge"] else 0.0 for r in j]
mx, my = st.mean(xs), st.mean(ys)
num = sum((a-mx)*(b-my) for a, b in zip(xs, ys))
den = math.sqrt(sum((a-mx)**2 for a in xs) * sum((b-my)**2 for b in ys))
r_pb = num/den if den else 0.0
n = len(j)
t = r_pb*math.sqrt((n-2)/max(1e-12, 1-r_pb**2))
print(f"  point-biserial r(hidden_rate, correct) = {r_pb:+.4f}  t={t:+.2f}  n={n}")

print("\n=== (c) budget ===")
print(f"median current-view chars {st.median([r['cur_chars'] for r in rows]):.0f} "
      f"(production-capped to {CAP})")
print(f"median agg-block chars    {st.median([r['agg_chars'] for r in rows]):.0f}  "
      f"mean {st.mean([r['agg_chars'] for r in rows]):.0f}  max {max(r['agg_chars'] for r in rows)}")
print(f"agg block is PREPENDED, so context grows by that much unless the cap is re-cut: "
      f"mean +{st.mean([r['agg_chars'] for r in rows])/CAP:.1%} of the view budget")
json.dump(rows, open("/root/agent_memory_iclr/results/_adv_v3.json", "w"))
