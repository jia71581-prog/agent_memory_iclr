"""Measurement 1: fragmentation before/after merge_chains over all 500 questions (OFFLINE).
Also dumps a reproducible random sample of 30 merge clusters for hand-checking, and view sizes."""
import json
import pickle
import random
import statistics as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifecycle import assemble_chains, render_view
from v5_chains import merge_chains, cluster_keys

D = pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl", "rb"))
rows = []
all_edges = []          # (qid, k1, k2, rule)
all_clusters = []       # (qid, [attrs], rule_set, sizes)
for x in D:
    store = assemble_chains(x["facts"])
    clusters, edges = cluster_keys(store)
    merged = merge_chains(store)
    n_before, n_after = len(store), len(merged)
    sing_b = sum(1 for c in store.values() if len(c) == 1)
    sing_a = sum(1 for c in merged.values() if len(c) == 1)
    v_cur_b, v_cur_a = len(render_view(store, "current")), len(render_view(merged, "current"))
    v_his_b, v_his_a = len(render_view(store, "historical")), len(render_view(merged, "historical"))
    facts_b = sum(len(c) for c in store.values())
    facts_a = sum(len(c) for c in merged.values())
    rows.append({"qid": x["qid"], "type": x["type"], "n_facts": len(x["facts"]),
                 "chains_before": n_before, "chains_after": n_after,
                 "singletons_before": sing_b, "singletons_after": sing_a,
                 "facts_in_store_before": facts_b, "facts_in_store_after": facts_a,
                 "maxchain_before": max((len(c) for c in store.values()), default=0),
                 "maxchain_after": max((len(c) for c in merged.values()), default=0),
                 "view_cur_before": v_cur_b, "view_cur_after": v_cur_a,
                 "view_hist_before": v_his_b, "view_hist_after": v_his_a,
                 "n_merge_edges": len(edges)})
    for k1, k2, rule in edges:
        all_edges.append((x["qid"], k1, k2, rule))
    for cl in clusters:
        if len(cl) > 1:
            all_clusters.append((x["qid"], cl))

with open("/root/agent_memory_iclr/results/_v5_frag.json", "w") as f:
    json.dump(rows, f)


def p(vals, q):
    v = sorted(vals)
    return v[min(len(v) - 1, int(q * len(v)))]


def summ(name, key):
    v = [r[key] for r in rows]
    return f"{name:26s} median {st.median(v):8.1f}  mean {st.mean(v):8.1f}  p90 {p(v,0.9):8.1f}  min {min(v)}  max {max(v)}"


print(f"n questions {len(rows)}")
for nm, k in [("chains BEFORE", "chains_before"), ("chains AFTER", "chains_after"),
              ("singletons BEFORE", "singletons_before"), ("singletons AFTER", "singletons_after"),
              ("max chain len BEFORE", "maxchain_before"), ("max chain len AFTER", "maxchain_after"),
              ("view current BEFORE", "view_cur_before"), ("view current AFTER", "view_cur_after"),
              ("view hist BEFORE", "view_hist_before"), ("view hist AFTER", "view_hist_after"),
              ("merge edges", "n_merge_edges")]:
    print(summ(nm, k))
red = [1 - r["chains_after"] / r["chains_before"] for r in rows if r["chains_before"]]
print(f"chain-count reduction: median {st.median(red):.3f} mean {st.mean(red):.3f} "
      f"min {min(red):.3f} max {max(red):.3f}")
sb = sum(r["singletons_before"] for r in rows) / sum(r["chains_before"] for r in rows)
sa = sum(r["singletons_after"] for r in rows) / sum(r["chains_after"] for r in rows)
print(f"singleton share of chains: before {sb:.3f} after {sa:.3f}")
fb = sum(r["facts_in_store_before"] for r in rows)
fa = sum(r["facts_in_store_after"] for r in rows)
print(f"facts retained in store: before {fb} after {fa} (delta {fa-fb}, "
      f"{(fa-fb)/fb:+.3%} - negative = duplicate values collapsed by merging)")
vb = sum(r["view_cur_before"] for r in rows) / len(rows)
va = sum(r["view_cur_after"] for r in rows) / len(rows)
print(f"mean current-view chars {vb:.0f} -> {va:.0f} ({(va-vb)/vb:+.2%})")
vhb = sum(r["view_hist_before"] for r in rows) / len(rows)
vha = sum(r["view_hist_after"] for r in rows) / len(rows)
print(f"mean historical-view chars {vhb:.0f} -> {vha:.0f} ({(vha-vhb)/vhb:+.2%})")
print(f"total merge edges {len(all_edges)}, clusters with >1 key {len(all_clusters)}")
from collections import Counter
print("edges by rule:", dict(Counter(e[3] for e in all_edges)))

# reproducible sample of 30 clusters for hand-check
random.seed(20260917)
samp = random.sample(all_clusters, 30)
with open("/root/agent_memory_iclr/results/_v5_merge_sample.txt", "w") as f:
    for i, (qid, cl) in enumerate(samp, 1):
        f.write(f"--- {i} qid={qid}\n")
        for k in cl:
            f.write(f"    {k}\n")
with open("/root/agent_memory_iclr/results/_v5_merge_sample.json", "w") as f:
    json.dump(samp, f)
print("wrote merge sample -> results/_v5_merge_sample.txt")
