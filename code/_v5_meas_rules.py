"""Measurement 1b: merge-rule configurations, head-merge ceiling, and rule-attributed
cluster samples for hand-checking. OFFLINE."""
import json
import pickle
import random
import statistics as st
import sys
import os
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifecycle import assemble_chains, render_view
from v5_chains import merge_chains, cluster_keys, _canon_tokens, _head, _ent_canon

D = pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl", "rb"))

CONFIGS = {
    "v5 default(R1+R2+R4')": dict(jaccard=None, ratio=0.85),
    "R1+R2 only": dict(jaccard=None, ratio=2.0),
    "+R3 jaccard 0.6": dict(jaccard=0.6, ratio=0.85),
    "+R3 jaccard 0.5": dict(jaccard=0.5, ratio=0.85),
}

stores = {}
for x in D:
    stores[x["qid"]] = assemble_chains(x["facts"])

print(f"{'config':24s} {'chains med':>11s} {'p90':>6s} {'reduction':>10s} {'edges':>8s} "
      f"{'view_cur med':>13s}")
base_med = st.median([len(stores[x['qid']]) for x in D])
for name, kw in CONFIGS.items():
    ns, reds, edges, views = [], [], 0, []
    for x in D:
        s = stores[x["qid"]]
        m = merge_chains(s, **kw)
        _, e = cluster_keys(s, **kw)
        ns.append(len(m))
        reds.append(1 - len(m) / len(s))
        edges += len(e)
        views.append(len(render_view(m, "current")))
    sn = sorted(ns)
    print(f"{name:24s} {st.median(ns):11.1f} {sn[int(0.9*len(sn))]:6d} {st.mean(reds):9.3f} "
          f"{edges:8d} {st.median(views):13.0f}")
print(f"{'v4 (no merge)':24s} {base_med:11.1f} "
      f"{sorted(len(stores[x['qid']]) for x in D)[450]:6d} {0.0:9.3f} {0:8d} "
      f"{st.median([len(render_view(stores[x['qid']],'current')) for x in D]):13.0f}")

# ---- head-merge ceiling: merge every chain sharing (canonical entity, head token)
ceil_ns, ceil_dist = [], []
for x in D:
    s = stores[x["qid"]]
    heads = Counter()
    for k, ch in s.items():
        heads[(_ent_canon(ch[0]["entity"]), _head(_canon_tokens(ch[0]["attribute"])))] += 1
    ceil_ns.append(len(heads))
    ceil_dist.append(max(heads.values()))
print(f"\nhead-merge CEILING (merge all same (entity,head)): chains median {st.median(ceil_ns):.1f} "
      f"(v4 {base_med:.1f}, reduction {1-st.median(ceil_ns)/base_med:.3f}); "
      f"largest head bucket median {st.median(ceil_dist):.0f}")

# how much of fragmentation is unique heads (irreducible under any head-bucketed rule)?
uniq_head_share = []
for x in D:
    s = stores[x["qid"]]
    heads = Counter()
    for k, ch in s.items():
        heads[(_ent_canon(ch[0]["entity"]), _head(_canon_tokens(ch[0]["attribute"])))] += 1
    uniq_head_share.append(sum(1 for v in heads.values() if v == 1) / len(s))
print(f"share of v4 chains whose (entity,head) is UNIQUE in that question: "
      f"median {st.median(uniq_head_share):.3f}  -> upper bound on any head-based merging")

# ---- rule-attributed cluster samples
def sample_clusters(kw, tag, n=30, only_rules=None, seed=20260917):
    pool = []
    for x in D:
        s = stores[x["qid"]]
        clusters, edges = cluster_keys(s, **kw)
        by_key = defaultdict(set)
        for k1, k2, rule in edges:
            by_key[k1].add(rule)
            by_key[k2].add(rule)
        for cl in clusters:
            if len(cl) < 2:
                continue
            rules = set()
            for k in cl:
                rules |= by_key[k]
            if only_rules and not (rules & only_rules):
                continue
            pool.append({"qid": x["qid"], "keys": cl, "rules": sorted(rules),
                         "sizes": [len(s[k]) for k in cl]})
    random.seed(seed)
    samp = random.sample(pool, min(n, len(pool)))
    with open(f"/root/agent_memory_iclr/results/_v5_sample_{tag}.json", "w") as f:
        json.dump({"pool_size": len(pool), "sample": samp}, f, indent=1)
    print(f"\n=== sample {tag}: pool {len(pool)} clusters, drew {len(samp)} ===")
    for i, c in enumerate(samp, 1):
        print(f"[{i}] {c['qid']} rules={','.join(r.split('_')[0] for r in c['rules'])} "
              f"sizes={c['sizes']}")
        for k in c["keys"]:
            print(f"      {k}")
    return samp

sample_clusters(CONFIGS["v5 default(R1+R2+R4')"], "FROZEN", 30, seed=20260924)
# clusters that exist ONLY because of R3 (to price the jaccard rule separately)
r3_only = []
for x in D:
    s = stores[x["qid"]]
    cl_def, e_def = cluster_keys(s, jaccard=0.6, ratio=0.85)
    cl_no3, _ = cluster_keys(s, jaccard=None, ratio=0.85)
    small = {k: i for i, cl in enumerate(cl_no3) for k in cl}
    for cl in cl_def:
        if len({small[k] for k in cl}) > 1:      # R3 glued >=2 no-R3 clusters together
            r3_only.append({"qid": x["qid"], "keys": cl, "sizes": [len(s[k]) for k in cl]})
random.seed(20260917)
r3s = random.sample(r3_only, 20)
with open("/root/agent_memory_iclr/results/_v5_sample_r3only.json", "w") as f:
    json.dump({"pool_size": len(r3_only), "sample": r3s}, f, indent=1)
print(f"\n=== R3-attributable clusters: pool {len(r3_only)}, drew {len(r3s)} ===")
for i, c in enumerate(r3s, 1):
    print(f"[{i}] {c['qid']} sizes={c['sizes']}")
    for k in c["keys"]:
        print(f"      {k}")
