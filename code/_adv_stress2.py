"""More adversarial checks: fact loss under merging (gold-session split), the single
discordant question, hyperparameter/subset sensitivity of the merge-vs-nomerge delta."""
import json
import pickle
import statistics as st
import sys
from collections import Counter

sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains, _norm
from v5_chains import merge_chains, cluster_keys, aggregation_printed_groups, render_aggregation_view

D = {x["qid"]: x for x in pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl", "rb"))}
G = pickle.load(open("/root/agent_memory_iclr/results/_v5_gold.pkl", "rb"))
ROWS = json.load(open("/root/agent_memory_iclr/results/_adv_recompute.json"))
ENUM = [r for r in ROWS if r["enum"]]

print("=== fact loss caused by merge_chains, over the 227 triggered questions ===")
lost_gold = 0
lost_tot = 0
examples = []
for r in ROWS:
    x = D[r["qid"]]
    gs = set(G[r["qid"]]["gold_sidx"])
    s4 = assemble_chains(x["facts"])
    s5 = merge_chains(s4)

    def bag(store, orig=False):
        out = Counter()
        for ch in store.values():
            for c in ch:
                e = c.get("ent_orig", c["entity"]) if orig else c["entity"]
                a = c.get("attr_orig", c["attribute"]) if orig else c["attribute"]
                out[(e, a, c["value"], c["date"], c.get("sidx"))] += 1
        return out
    b4, b5 = bag(s4), bag(s5, orig=True)
    for k, v in b4.items():
        d = v - b5.get(k, 0)
        if d > 0:
            lost_tot += d
            if k[4] in gs:
                lost_gold += d
            if len(examples) < 12:
                examples.append((r["qid"], k[1], k[2], k[4] in gs))
print("facts dropped:", lost_tot, "of which from gold sessions:", lost_gold)
for e in examples:
    print("   ", e)

print("\n=== the single question that carries the whole reported delta: bf659f65 ===")
x = D["bf659f65"]
gs = set(G["bf659f65"]["gold_sidx"])
print("Q:", x["question"], "| gold:", x["answer"], "| gold sessions:", sorted(gs))
s4 = assemble_chains(x["facts"])
s5 = merge_chains(s4)
print("chains v4:", len(s4), "-> merged:", len(s5))
for tag, store in [("nomerge", s4), ("merge", s5)]:
    kept, meta = aggregation_printed_groups(store, x["question"])
    print(f"-- {tag}: {len(kept)} group(s), focus weights {sorted(meta[0].items(), key=lambda kv:-kv[1])[:6]}")
    for g in kept:
        print(f"   [{g['term']}] {g['entity']}: {len(g['items'])} items")
        for it in g["items"][:25]:
            c = it[1]
            print(f"      {c['sidx']:>3} {'G' if c['sidx'] in gs else ' '} {it[2]} = {c['value'][:70]}")
clusters, edges = cluster_keys(s4)
big = [c for c in clusters if len(c) > 1]
print("merged clusters here:", len(big))
for c in sorted(big, key=lambda c: -len(c))[:8]:
    print("   ", c)

print("\n=== hyperparameter / subset sensitivity of the merge-vs-nomerge delta ===")


def score(store_fn, subset, **kw):
    d1 = d2 = 0
    gh = tg = 0
    for r in subset:
        x = D[r["qid"]]
        gsx = set(G[r["qid"]]["gold_sidx"])
        store = store_fn(x)
        kept, _ = aggregation_printed_groups(store, x["question"], **kw)
        items = [it[1] for g in kept for it in g["items"]]
        best = max((len(g["items"]) for g in kept), default=0)
        cov = len({c.get("sidx") for c in items} & gsx)
        full = cov == len(gsx)
        d1 += (len(items) >= r["gold_num"]) and full
        d2 += (best >= r["gold_num"]) and full
        gh += cov
        tg += len(gsx)
    n = len(subset)
    return d1 / n, d2 / n, gh / tg


M = lambda x: merge_chains(assemble_chains(x["facts"]))
M_j = lambda x: merge_chains(assemble_chains(x["facts"]), jaccard=0.6)
N4 = lambda x: assemble_chains(x["facts"])
count_kind = [r for r in ROWS if r["kind"] == "count"]
print(f"{'setting':44s} {'merge D1/D2/rec':>26s} {'nomerge D1/D2/rec':>26s}")
for lab, subset, kw in [
        ("enum60 default", ENUM, {}),
        ("enum60 df_cap=0.15", ENUM, {"df_cap": 0.15}),
        ("enum60 df_cap=0.40", ENUM, {"df_cap": 0.40}),
        ("enum60 rel_keep=0.20", ENUM, {"rel_keep": 0.20}),
        ("enum60 rel_keep=0.60", ENUM, {"rel_keep": 0.60}),
        ("enum60 max_items=15", ENUM, {"max_items_per_group": 15}),
        ("enum60 no hyponyms", ENUM, {"use_hyponyms": False}),
        ("all 184 count-kind", count_kind, {}),
        ("all 227 triggered", ROWS, {}),
]:
    a = score(M, subset, **kw)
    b = score(N4, subset, **kw)
    print(f"{lab:44s} {a[0]:8.3f}{a[1]:9.3f}{a[2]:9.3f} {b[0]:8.3f}{b[1]:9.3f}{b[2]:9.3f}")
a = score(M_j, ENUM)
print(f"{'enum60 merge with R3 jaccard=0.6':44s} {a[0]:8.3f}{a[1]:9.3f}{a[2]:9.3f}")
