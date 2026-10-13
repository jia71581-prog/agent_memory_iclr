"""Adversarial control 3: do the stemmer/synonym fixes change CHAIN MERGING (which affects
all 500 questions, not just the 60 aggregation ones), and are the newly created merges
correct?  Dumps every merge edge that exists only under the fixed stemmer. OFFLINE."""
import json
import pickle
import random
import statistics as st
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifecycle import assemble_chains
import v5_chains as V
from _adv_v5_bugs import D, apply_arm

ALL = {"ii", "iii", "iv"}          # stemmer + synonym bugs (dedupe does not affect merging)


def edges_for(bugs):
    apply_arm(bugs, "exact")
    per_q = {}
    nch = []
    for qid, x in D.items():
        store4 = assemble_chains(x["facts"])
        cl, ed = V.cluster_keys(store4)
        per_q[qid] = {(a, b, r) for a, b, r in ed}
        nch.append((len(store4), len(cl)))
    apply_arm(set())
    return per_q, nch


new, nch_new = edges_for(set())
old, nch_old = edges_for(ALL)

n_new = sum(len(v) for v in new.values())
n_old = sum(len(v) for v in old.values())
added, removed = [], []
for qid in D:
    a = new[qid] - old[qid]
    r = old[qid] - new[qid]
    for e in a:
        added.append((qid, e))
    for e in r:
        removed.append((qid, e))
print("=== (c) effect of the stemmer/SYN fixes on CHAIN MERGING, all 500 questions ===")
print(f"merge edges: fixed stemmer {n_new}, buggy stemmer {n_old} "
      f"(net {n_new - n_old:+d}); edges added by the fixes {len(added)}, removed {len(removed)}")
print(f"chains after merge (median over 500): fixed "
      f"{st.median([c for _, c in nch_new])}, buggy {st.median([c for _, c in nch_old])}, "
      f"v4 unmerged {st.median([s for s, _ in nch_new])}")
print("added edges by rule:", dict(Counter(e[2] for _, e in added)))
print("removed edges by rule:", dict(Counter(e[2] for _, e in removed)))

random.seed(0)
samp = random.sample(added, min(60, len(added)))
print(f"\n--- random sample of {len(samp)} merge edges CREATED by the four fixes "
      f"(attribute pairs, for hand audit) ---")
for qid, (a, b, r) in sorted(samp):
    print(f"  [{r}] {a.split('::',1)[1][:44]:44s} <=> {b.split('::',1)[1][:44]}")

if removed:
    samp2 = random.sample(removed, min(25, len(removed)))
    print(f"\n--- random sample of {len(samp2)} merge edges DESTROYED by the four fixes ---")
    for qid, (a, b, r) in sorted(samp2):
        print(f"  [{r}] {a.split('::',1)[1][:44]:44s} <=> {b.split('::',1)[1][:44]}")

with open("/root/agent_memory_iclr/results/_adv_v5_mergediff.json", "w") as f:
    json.dump({"added": [[q, list(e)] for q, e in added],
               "removed": [[q, list(e)] for q, e in removed]}, f, indent=1)
