"""Adversarial control 5: regression check the claim never ran.  The stemmer/SYN fixes add
~3.9k merge edges over all 500 questions, so they change the MAIN 'current' view for every
question type, not just the 60 aggregation ones.  More merging => more facts labelled
supersession => more gold-session facts hidden by current-mode rendering.  Measure it."""
import statistics as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifecycle import assemble_chains, render_view
import v5_chains as V
from _adv_v5_bugs import D, G, apply_arm

ALL = {"ii", "iii_nodbl", "iv"}      # stemmer/SYN bugs, keeping the double-consonant branch


def run(bugs, label):
    apply_arm(bugs, "exact")
    gf = gr = 0
    hit = tot = 0
    chars = []
    nch = []
    for qid, x in D.items():
        gs = set(G[qid]["gold_sidx"])
        store = V.merge_chains(assemble_chains(x["facts"]))
        nch.append(len(store))
        seen = set()
        for chain in store.values():
            if chain[0]["relation"] == "accumulative":
                vis = chain
            else:
                last = (chain[-1]["date"], chain[-1].get("sidx"))
                vis = [c for c in chain if (c["date"], c.get("sidx")) == last]
            for c in vis:
                seen.add(c.get("sidx"))
            for c in chain:
                if c.get("sidx") in gs:
                    gf += 1
                    if c in vis:
                        gr += 1
        hit += len(seen & gs)
        tot += len(gs)
        chars.append(len(render_view(store, "current")))
    apply_arm(set())
    print(f"{label:34s} chains med {st.median(nch):6.1f} | gold-session facts "
          f"{gr}/{gf} = {gr/gf:.4f} survive current-mode rendering "
          f"(hidden {1-gr/gf:.2%}) | gold sessions represented {hit}/{tot} = {hit/tot:.4f} "
          f"| view chars med {st.median(chars):.0f}")


print("=== (e) MAIN-VIEW regression over ALL 500 questions (not just the 60) ===")
run(set(), "FIXED stemmer/SYN (shipped)")
run(ALL, "BUGGY stemmer/SYN (pre-fix)")
