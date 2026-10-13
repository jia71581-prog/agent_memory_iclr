"""ADVERSARIAL VERIFIER step 5: merge-precision audit on the clusters that actually matter.
The design agent hand-audited 30 RANDOM clusters. Random sampling under-weights the big
clusters, which is where a false merge does the most damage (they drive maxchain 13 -> 18)."""
import json
import pickle
import random
import statistics as st
import sys
from collections import Counter

sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains
from v5_chains import cluster_keys, merge_chains

MINE = pickle.load(open("/root/agent_memory_iclr/results/_verify_facts.pkl", "rb"))

pool = []
sizehist = Counter()
pairs = Counter()
for x in MINE:
    s = assemble_chains(x["facts"])
    clusters, edges = cluster_keys(s)
    for cl in clusters:
        if len(cl) > 1:
            pool.append({"qid": x["qid"], "keys": cl,
                         "nkeys": len(cl), "nfacts": sum(len(s[k]) for k in cl)})
            sizehist[len(cl)] += 1
    for k1, k2, r in edges:
        a, b = sorted((k1.split("::", 1)[1], k2.split("::", 1)[1]))
        pairs[(a, b)] += 1

print(f"multi-key clusters {len(pool)}")
print("cluster size histogram (keys per cluster):", dict(sorted(sizehist.items())))
print(f"chains absorbed = {sum(c['nkeys']-1 for c in pool)}")
print(f"facts inside merged clusters {sum(c['nfacts'] for c in pool)}")

print("\n=== 30 LARGEST clusters (by #keys) ===")
for c in sorted(pool, key=lambda c: -c["nkeys"])[:30]:
    print(f"[{c['qid']}] {c['nkeys']} keys, {c['nfacts']} facts")
    print("    " + " | ".join(k.split("::", 1)[1] for k in c["keys"]))

print("\n=== 40 most frequent merged attribute PAIRS across the corpus ===")
for (a, b), n in pairs.most_common(40):
    print(f"{n:5d}  {a!r}  <->  {b!r}")

print("\n=== 40 RANDOM clusters, verifier seed (independent of theirs) ===")
random.seed(777001)
for c in random.sample(pool, 40):
    print(f"[{c['qid']}] " + " | ".join(k.split("::", 1)[1] for k in c["keys"]))
