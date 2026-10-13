"""Adversarial check 3: metric granularity. The claim's metric is 'gold evidence SESSION
represented' (>=1 printed item shares the session index). Recompute at FACT level and measure
the block's precision, plus the real deployed prompt cost. OFFLINE."""
import json
import pickle
import statistics as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifecycle import assemble_chains, render_view
from v5_chains import (merge_chains, render_aggregation_view, aggregation_printed_groups,
                       is_aggregation_question)
from run_baseline import bm25_rank

CAP = 20000
D = {x["qid"]: x for x in pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl", "rb"))}
G = pickle.load(open("/root/agent_memory_iclr/results/_v5_gold.pkl", "rb"))

rows = []
for qid, x in D.items():
    if not is_aggregation_question(x["question"])[0]:
        continue
    gold = set(G[qid]["gold_sidx"])
    s5 = merge_chains(assemble_chains(x["facts"]))
    kept, _ = aggregation_printed_groups(s5, x["question"])
    printed = [it[1] for g in kept for it in g["items"]]
    blk = render_aggregation_view(s5, x["question"])

    # gold-session facts in the store, and how many the block / the current view expose
    ident = lambda c: (c.get("sidx"), c["attribute"], c["value"])
    gold_facts = {ident(c) for c in x["facts"] if c.get("sidx") in gold}
    blk_facts = {ident(c) for c in printed if c.get("sidx") in gold}
    view_facts = set()
    for ch in s5.values():
        if ch[0]["relation"] == "accumulative":
            vis = ch
        else:
            last = (ch[-1]["date"], ch[-1].get("sidx"))
            vis = [c for c in ch if (c["date"], c.get("sidx")) == last]
        for c in vis:
            if c.get("sidx") in gold:
                view_facts.add(ident(c))
    # deployed prompt cost: capped view + BM25 top-20 raw turns
    vt = render_view(s5, "current")
    if len(vt) > CAP:
        kk, tot = [], 0
        for line in vt.splitlines():
            if tot + len(line) + 1 > CAP:
                break
            kk.append(line)
            tot += len(line) + 1
        vt = "\n".join(kk)
    raw_chars = sum(len(t[2]) for t in x["raw20"])
    rows.append({"qid": qid, "n_gold_sess": len(gold),
                 "gold_facts": len(gold_facts), "blk_gold_facts": len(blk_facts),
                 "view_gold_facts": len(view_facts),
                 "n_items": len(printed),
                 "blk_items_from_gold": sum(1 for c in printed if c.get("sidx") in gold),
                 "blk_chars": len(blk), "sent_view_chars": len(vt), "raw_chars": raw_chars})

gf = sum(r["gold_facts"] for r in rows)
print(f"n={len(rows)} triggered questions; facts extracted from gold sessions: {gf}")
print(f"  exposed by 'current' view (uncapped):  {sum(r['view_gold_facts'] for r in rows)/gf:.3f}")
print(f"  exposed by the v5 aggregation block :  {sum(r['blk_gold_facts'] for r in rows)/gf:.3f}")
print(f"\nblock precision: printed items {sum(r['n_items'] for r in rows)}, of which from a gold "
      f"session {sum(r['blk_items_from_gold'] for r in rows)} "
      f"({sum(r['blk_items_from_gold'] for r in rows)/sum(r['n_items'] for r in rows):.3f})")
print(f"median items/block {st.median([r['n_items'] for r in rows]):.0f}, median items from gold "
      f"{st.median([r['blk_items_from_gold'] for r in rows]):.0f}")
print(f"\ndeployed prompt (run_qc.py): capped view median {st.median([r['sent_view_chars'] for r in rows]):.0f} "
      f"+ BM25 top-20 raw turns median {st.median([r['raw_chars'] for r in rows]):.0f} chars")
tot_ctx = [r["sent_view_chars"] + r["raw_chars"] for r in rows]
print(f"total context median {st.median(tot_ctx):.0f} chars; block adds median "
      f"{st.median([r['blk_chars'] for r in rows]):.0f} = "
      f"+{st.median([r['blk_chars'] for r in rows])/st.median(tot_ctx):.2%} on top (prepended, "
      f"nothing removed)")
json.dump(rows, open("/root/agent_memory_iclr/results/_ver_conc3.json", "w"), indent=1)
