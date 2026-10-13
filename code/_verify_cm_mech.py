"""Mechanism check: on the questions where count_mode changes the selected turns, is the
count-mode session head actually pointing at gold evidence sessions?"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from lifecycle import assemble_chains, _session_text, SESSION_CACHE_DIR
from query_classifier import classify_query
import v5_retrieval as V5

R = "/root/agent_memory_iclr/results"
rows = json.load(open(f"{R}/_verify_countmode.json"))
changed = {r["qid"] for r in rows
           if r["v"]["v5"]["raw_chars"] != r["v"]["v5_nocount"]["raw_chars"]
           or r["v"]["v5"]["ev_hit"] != r["v"]["v5_nocount"]["ev_hit"]}
DATA = {q["question_id"]: q for q in load_dataset("s_cleaned")}


def cached(sess):
    text = _session_text(sess)
    if not text:
        return []
    p = os.path.join(SESSION_CACHE_DIR, hashlib.md5(text.encode()).hexdigest()[:16] + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


agg = {"cm_head_prec": [], "plain_prec": [], "cm_head_rec": [], "plain_rec": [],
       "chain_is_ev": 0, "chain_tot": 0}
for qid in sorted(changed):
    q = DATA[qid]
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    facts = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for e, a, v in (cached(sess) or []):
            facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
    store = assemble_chains(facts)
    units = V5.build_units(q)
    bm = V5.bm25_scores(q["question"], [u[3] for u in units])
    scored = V5.score_chains(store, q["question"])
    evs = {i for i, sid in enumerate(q["haystack_session_ids"]) if sid in set(q["answer_session_ids"])}

    _, ord_cm, _ = V5.select_turns(q, store, q["question"], scored, 8, bm_scores=bm, units=units,
                                   count_mode=True)
    _, ord_pl, _ = V5.select_turns(q, store, q["question"], scored, 8, bm_scores=bm, units=units,
                                   count_mode=False)
    for name, o in (("cm_head", ord_cm), ("plain", ord_pl)):
        hit = len(set(o) & evs)
        agg[f"{name}_prec"].append(hit / max(1, len(o)))
        agg[f"{name}_rec"].append(hit / max(1, len(evs)))
    css = V5.chain_session_sets(store, scored)
    if css:
        agg["chain_tot"] += 1
        agg["chain_is_ev"] += bool(set(css[0][2]) & evs)

n = len(changed)
print(f"questions analysed: {n}")
for k in ("cm_head_prec", "plain_prec", "cm_head_rec", "plain_rec"):
    print(f"  {k:14s} mean {sum(agg[k]) / n:.4f}")
print(f"  best multi-session chain touches >=1 gold evidence session: "
      f"{agg['chain_is_ev']}/{agg['chain_tot']}")
