"""Test 4: (a) rel_keep sweep for both arms - is the gain knob-specific?
(b) D1 decomposition: count-threshold vs coverage.
(c) inspect the groups that satisfy D2 in the flipped cases - correct category or spurious?"""
import json, pickle, sys, statistics as st
sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains
import v5_chains as V

D = {x["qid"]: x for x in pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl","rb"))}
G = pickle.load(open("/root/agent_memory_iclr/results/_v5_gold.pkl","rb"))
base = json.load(open("/root/agent_memory_iclr/verify/adv_hyp_rows.json"))
FULL = dict(V.HYPONYMS)

stores, meta = {}, {}
for r in base:
    x = D[r["qid"]]
    stores[r["qid"]] = V.merge_chains(assemble_chains(x["facts"]))
    meta[r["qid"]] = (x["question"], set(G[r["qid"]]["gold_sidx"]), r["gnum"], r["judge"])

def measure(uh, rel_keep, max_groups=8):
    ok1 = ok2 = rec = tot = cf = vf = items = 0
    for qid, stt in stores.items():
        q, gs, gnum, _ = meta[qid]
        kept, _ = V.aggregation_printed_groups(stt, q, uh, rel_keep=rel_keep,
                                               max_groups=max_groups)
        pr = [[it[1] for it in g["items"]] for g in kept]
        allit = [c for gi in pr for c in gi]
        gh = len({c.get("sidx") for c in allit} & gs)
        best = max((len(gi) for gi in pr), default=0)
        cnt_ok, cov_ok = len(allit) >= gnum, gh == len(gs)
        ok1 += cnt_ok and cov_ok; ok2 += (best >= gnum) and cov_ok
        cf += not cnt_ok; vf += not cov_ok
        rec += gh; tot += len(gs); items += len(allit)
    n = len(stores)
    return dict(D1=ok1/n, D2=ok2/n, recall=rec/tot, items=items,
                count_fail=cf, cov_fail=vf, n1=ok1, n2=ok2, nrec=rec)

print("=== (a) rel_keep sweep, both arms (max_groups=8 default) ===")
print(f"{'rel_keep':>9s} | {'D1 hyp':>10s} {'D1 nohyp':>10s} | {'D2 hyp':>10s} {'D2 nohyp':>10s} "
      f"| {'rec hyp':>9s} {'rec nohyp':>9s} | {'items hyp':>9s} {'nohyp':>7s}")
for rk in [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.70]:
    a = measure(True, rk); b = measure(False, rk)
    print(f"{rk:9.2f} | {a['D1']:6.3f}({a['n1']:2d}) {b['D1']:6.3f}({b['n1']:2d}) "
          f"| {a['D2']:6.3f}({a['n2']:2d}) {b['D2']:6.3f}({b['n2']:2d}) "
          f"| {a['recall']:9.3f} {b['recall']:9.3f} | {a['items']:9d} {b['items']:7d}")

print("\n=== (b) D1 failure decomposition at the reported setting (rel_keep=0.35) ===")
for uh, lab in [(True, "with table"), (False, "without table")]:
    m = measure(uh, 0.35)
    print(f"{lab:15s} count-threshold failures {m['count_fail']:2d}  "
          f"coverage failures {m['cov_fail']:2d}  D1 pass {m['n1']}/60")

print("\n=== (c) the group that satisfies D2 in flipped cases ===")
for qid in ["21436231", "gpt4_f2262a51", "603deb26", "2ce6a0f2", "af8d2e46"]:
    q, gs, gnum, judge = meta[qid]
    for uh in (False, True):
        kept, _ = V.aggregation_printed_groups(stores[qid], q, uh)
        best = max(kept, key=lambda g: len(g["items"])) if kept else None
        tag = "with" if uh else "wout"
        if best:
            vals = [it[1]["value"][:46] for it in best["items"][:6]]
            print(f"{qid:14s} {tag:4s} gold={gnum:.0f} bestgroup=[{best['term']}] "
                  f"n={len(best['items'])} judge={judge}")
            print(f"    sample: {vals}")
    print(f"    Q: {q[:120]}\n")
