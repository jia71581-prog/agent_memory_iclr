"""ADVERSARIAL VERIFIER #2 step 4: (a) prove 0 API calls, (b) timing distribution,
(c) are the hand tables (HYPONYMS/SYN) fitted to the 500 evaluation questions?"""
import json
import pickle
import statistics as st
import sys
import time

sys.path.insert(0, "/root/agent_memory_iclr/code")
import common

CALLS = []
_real = common.chat


def _trap(*a, **k):
    CALLS.append(k.get("op", "?"))
    raise AssertionError("API CALL ATTEMPTED")


common.chat = _trap
import lifecycle
lifecycle.chat = _trap

from lifecycle import assemble_chains, render_view
from query_classifier import classify_query
import v5_chains
from v5_chains import (merge_chains, render_aggregation_view, question_focus, HYPONYMS,
                       _stem, _stem_tokens, aggregation_kind, aggregation_printed_groups)

D = pickle.load(open("/root/agent_memory_iclr/results/_adv2_facts.pkl", "rb"))

# ---- (a)+(b): run the whole v5 path with chat() trapped
tm, ta = [], []
for x in D:
    label, _ = classify_query(x["question"])
    mode = "historical" if label == "historical" else "current"
    s4 = assemble_chains(x["facts"])
    t0 = time.perf_counter(); s5 = merge_chains(s4); tm.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); blk = render_aggregation_view(s5, x["question"]); ta.append(time.perf_counter() - t0)
    render_view(s5, mode)
print(f"(a) API calls attempted during the full 500-question v5 path: {len(CALLS)}  -> claim '0 API calls' "
      f"{'HOLDS' if not CALLS else 'FALSE'}")
tot = [a + b for a, b in zip(tm, ta)]
print(f"(b) merge+block ms/question: mean {st.mean(tot)*1000:.1f} median {st.median(tot)*1000:.1f} "
      f"p95 {sorted(tot)[int(.95*len(tot))]*1000:.1f} max {max(tot)*1000:.1f}")
print(f"    merge only mean {st.mean(tm)*1000:.1f} ms ; block only mean {st.mean(ta)*1000:.1f} ms")

# ---- (c) hand-table fitting
print("\n(c) HYPONYMS coverage over the 500 evaluation questions")
qs = [(x["qid"], x["question"]) for x in D]
per_cat = {}
for cat in HYPONYMS:
    cstem = _stem(cat)
    hst = {_stem(h) for h in HYPONYMS[cat]}
    hit = []
    for qid, q in qs:
        toks = set(_stem_tokens(q))
        if cstem in toks or (toks & hst):
            hit.append(qid)
    per_cat[cat] = hit
for cat, h in sorted(per_cat.items(), key=lambda kv: len(kv[1])):
    print(f"    {cat:12s} matches {len(h):3d}/500 questions  {h[:4]}")
print(f"    categories matching 0 questions: {[c for c,h in per_cat.items() if not h]}")
print(f"    categories matching <=2 questions: {[c for c,h in per_cat.items() if 0 < len(h) <= 2]}")

# does the block change without hyponyms?
n_fire_h = n_fire_n = 0
diff = same = 0
grp_h, grp_n = [], []
for x in D:
    if aggregation_kind(x["question"]) is None:
        continue
    s5 = merge_chains(assemble_chains(x["facts"]))
    bh = render_aggregation_view(s5, x["question"], use_hyponyms=True)
    bn = render_aggregation_view(s5, x["question"], use_hyponyms=False)
    n_fire_h += bool(bh); n_fire_n += bool(bn)
    if bh != bn:
        diff += 1
    else:
        same += 1
    kh, _ = aggregation_printed_groups(s5, x["question"], use_hyponyms=True)
    kn, _ = aggregation_printed_groups(s5, x["question"], use_hyponyms=False)
    grp_h.append(sum(len(g["items"]) for g in kh))
    grp_n.append(sum(len(g["items"]) for g in kn))
print(f"\n    block fires: with hyponyms {n_fire_h}, without {n_fire_n}")
print(f"    block text differs on {diff} of {diff+same} aggregation questions")
print(f"    median union size: with {st.median(grp_h):.0f} without {st.median(grp_n):.0f}")

# SYN table usage
print("\n    SYN table: which entries ever fire on a question token?")
from v5_chains import SYN
used = {}
alltok = set()
for _, q in qs:
    alltok |= set(_stem_tokens(q))
for k in SYN:
    if k in alltok or _stem(k) in alltok:
        used[k] = 1
print(f"    SYN keys reachable from some question: {len(used)}/{len(SYN)} -> {sorted(set(SYN)-set(used))}")
