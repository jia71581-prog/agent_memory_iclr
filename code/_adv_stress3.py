"""Final adversarial pass: strict (non-inflation-rewarding) metrics for merge vs nomerge,
broader subsets, and item-level precision against gold sessions."""
import json
import pickle
import statistics as st
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains
from v5_chains import merge_chains, aggregation_printed_groups

D = {x["qid"]: x for x in pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl", "rb"))}
G = pickle.load(open("/root/agent_memory_iclr/results/_v5_gold.pkl", "rb"))
ROWS = json.load(open("/root/agent_memory_iclr/results/_adv_recompute.json"))
ENUM = [r for r in ROWS if r["enum"]]

print("=== strict metrics on the 60 enum questions ===")
res = {}
for tag, fn in [("merge", lambda x: merge_chains(assemble_chains(x["facts"]))),
                ("nomerge", lambda x: assemble_chains(x["facts"]))]:
    exact = within1 = 0
    prec, absdev = [], []
    for r in ENUM:
        x = D[r["qid"]]
        gs = set(G[r["qid"]]["gold_sidx"])
        kept, _ = aggregation_printed_groups(fn(x), x["question"])
        items = [it[1] for g in kept for it in g["items"]]
        best = max((len(g["items"]) for g in kept), default=0)
        exact += best == r["gold_num"]
        within1 += abs(best - r["gold_num"]) <= 1
        if items:
            prec.append(sum(1 for c in items if c.get("sidx") in gs) / len(items))
        absdev.append(abs(best - r["gold_num"]) / r["gold_num"])
    res[tag] = (exact, within1, st.mean(prec), st.median(absdev))
    print(f"{tag:9s} best_group==gold {exact:2d}/60  |diff|<=1 {within1:2d}/60  "
          f"item precision (from a gold session) mean {st.mean(prec):.3f}  "
          f"median |best-gold|/gold {st.median(absdev):.2f}")

print("\n=== broader subsets: gold-session recall only (gold count often absent) ===")
for lab, subset in [("all 227 triggered", ROWS),
                    ("184 count-kind", [r for r in ROWS if r["kind"] == "count"]),
                    ("43 amount-kind", [r for r in ROWS if r["kind"] == "amount"]),
                    ("167 triggered non-enum", [r for r in ROWS if not r["enum"]])]:
    tg = sum(r["n_gold_sess"] for r in subset)
    m = sum(r["v5_gold_hit"] for r in subset) / tg
    n = sum(r["nomerge_gold_hit"] for r in subset) / tg
    nq_m = sum(1 for r in subset if r["v5_gold_hit"] == r["n_gold_sess"])
    nq_n = sum(1 for r in subset if r["nomerge_gold_hit"] == r["n_gold_sess"])
    diff = sum(1 for r in subset if r["v5_gold_hit"] != r["nomerge_gold_hit"])
    print(f"{lab:26s} recall merge {m:.3f} vs nomerge {n:.3f} | full-coverage questions "
          f"{nq_m}/{len(subset)} vs {nq_n}/{len(subset)} | questions differing: {diff}")

print("\n=== D1 on the 227 restricted to rows with a parseable gold number ===")
sub = [r for r in ROWS if r["gold_num"] is not None]
for t in ("v5", "nomerge"):
    d1 = sum(1 for r in sub if r[f"{t}_items"] >= r["gold_num"] and r[f"{t}_gold_hit"] == r["n_gold_sess"])
    print(f"  {t:8s} n={len(sub)} D1={d1/len(sub):.3f}")
