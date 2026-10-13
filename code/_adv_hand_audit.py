"""Audit: do the hand-verdict evidence quotes actually appear in the CURRENT rendered block?"""
import json, pickle, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifecycle import assemble_chains
from v5_chains import merge_chains, render_aggregation_view

D = {x["qid"]: x for x in pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl", "rb"))}
G = pickle.load(open("/root/agent_memory_iclr/results/_v5_gold.pkl", "rb"))
HV = json.load(open("/root/agent_memory_iclr/results/_v5_hand_verdicts.json"))

def norm(s):
    return re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())

out = []
for h in HV:
    qid = h["qid"]
    x = D[qid]
    st5 = merge_chains(assemble_chains(x["facts"]))
    txt = render_aggregation_view(st5, x["question"])
    ntxt = norm(txt)
    # quoted spans in the evidence field
    quotes = re.findall(r"'([^']+)'", h["evidence"])
    miss = [q for q in quotes if norm(q).strip() and norm(q).strip() not in ntxt]
    gs = G[qid]["gold_sidx"]
    gs = set(eval(gs)) if isinstance(gs, str) else set(gs)
    out.append({"qid": qid, "verdict": h["hand_verdict"], "n_quotes": len(quotes),
                "n_missing": len(miss), "missing": miss[:3], "block_chars": len(txt)})
    print(f"{qid:16s} {h['hand_verdict']:8s} quotes={len(quotes):2d} missing={len(miss):2d} "
          f"{'<-- EVIDENCE NOT IN BLOCK' if miss else ''}")
    for m in miss[:3]:
        print(f"      MISSING: {m[:90]}")
json.dump(out, open("/root/agent_memory_iclr/results/_adv_handaudit.json", "w"), indent=1)
print("\ntotal verdicts with at least one unverifiable quote:",
      sum(1 for o in out if o["n_missing"]), "/", len(out))
