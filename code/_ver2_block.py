"""Independent rebuild of the v5 store from the session cache (no API) for the 227
triggered questions + independent measurement of the aggregation block's cost/benefit.
Also cross-checks the design agent's _v5_facts.pkl against a from-scratch cache rebuild.
"""
import hashlib
import json
import os
import pickle
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifecycle import _session_text, assemble_chains, render_view, SESSION_CACHE_DIR
from v5_chains import merge_chains, render_aggregation_view, aggregation_printed_groups

ROOT = "/root/agent_memory_iclr"
DATA = {q["question_id"]: q for q in json.load(open(f"{ROOT}/data/longmemeval_s_cleaned.json"))}
rows = json.load(open(f"{ROOT}/results/_ver2_stated.json"))
PKL = {x["qid"]: x for x in pickle.load(open(f"{ROOT}/results/_v5_facts.pkl", "rb"))}
G = pickle.load(open(f"{ROOT}/results/_v5_gold.pkl", "rb"))


def build(q):
    facts, miss = [], 0
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        text = _session_text(sess)
        if not text:
            continue
        key = hashlib.md5(text.encode()).hexdigest()[:16]
        p = os.path.join(SESSION_CACHE_DIR, f"{key}.json")
        if not os.path.exists(p):
            miss += 1
            continue
        for t in json.load(open(p)):
            if len(t) == 3 and t[0] and t[2]:
                facts.append({"entity": t[0], "attribute": t[1], "value": t[2],
                              "date": date, "sidx": si})
    return facts, miss


out, miss_tot, pkl_mismatch = [], 0, 0
for r in rows:
    qid = r["qid"]
    q = DATA[qid]
    facts, miss = build(q)
    miss_tot += miss
    if len(facts) != len(PKL[qid]["facts"]):
        pkl_mismatch += 1
    store4 = assemble_chains(facts)
    store5 = merge_chains(store4)
    txt = render_aggregation_view(store5, q["question"])
    view = render_view(store5, "current")
    kept, _ = aggregation_printed_groups(store5, q["question"])
    items = [it[1] for g in kept for it in g["items"]]
    gs = set(G[qid]["gold_sidx"])
    out.append({**r, "n_facts": len(facts), "miss": miss, "n_chains4": len(store4),
                "n_chains5": len(store5), "agg_chars": len(txt), "view_chars": len(view),
                "n_items": len(items), "n_groups": len(kept),
                "agg_gold_hit": len({c.get("sidx") for c in items} & gs),
                "n_gold_sess": len(gs)})

json.dump(out, open(f"{ROOT}/results/_ver2_block.json", "w"), ensure_ascii=False, indent=1)
print(f"cache misses across the 227 rebuilt stores: {miss_tot}; "
      f"fact-count mismatches vs _v5_facts.pkl: {pkl_mismatch}/{len(rows)}")

enum = [r for r in out if r["enum"]]
non = [r for r in out if not r["enum"]]


def blk(lab, rs):
    ne = [r for r in rs if r["agg_chars"] > 0]
    print(f"{lab:26s} n={len(rs):3d}  block non-empty {len(ne):3d} ({len(ne)/len(rs):.3f})  "
          f"median block {st.median([r['agg_chars'] for r in rs]):6.0f} ch  "
          f"median view {st.median([r['view_chars'] for r in rs]):7.0f} ch  "
          f"block/view {st.median([r['agg_chars'] for r in rs])/st.median([r['view_chars'] for r in rs]):6.2%}  "
          f"median items {st.median([r['n_items'] for r in rs]):5.1f}")


print("\n== aggregation block cost, recomputed from the cache ==")
blk("enum-count subset (60)", enum)
blk("other triggered (167)", non)
blk("all triggered (227)", out)
tot_gs = sum(r["n_gold_sess"] for r in out)
print(f"\nblock gold-session recall: enum "
      f"{sum(r['agg_gold_hit'] for r in enum)/sum(r['n_gold_sess'] for r in enum):.3f}  "
      f"non-enum {sum(r['agg_gold_hit'] for r in non)/sum(r['n_gold_sess'] for r in non):.3f}  "
      f"all {sum(r['agg_gold_hit'] for r in out)/tot_gs:.3f}")
print(f"extra context added by the block over the 227: "
      f"{sum(r['agg_chars'] for r in out):,} chars on top of "
      f"{sum(r['view_chars'] for r in out):,} chars of view "
      f"(+{sum(r['agg_chars'] for r in out)/sum(r['view_chars'] for r in out):.2%})")
print(f"  of that, spent on the 167 non-enumeration questions: "
      f"{sum(r['agg_chars'] for r in non):,} chars "
      f"({sum(r['agg_chars'] for r in non)/max(1,sum(r['agg_chars'] for r in out)):.1%} of the block budget)")
