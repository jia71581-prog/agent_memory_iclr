"""Measurement 2: aggregation questions. Does render_aggregation_view expose enough distinct
items to derive the gold count, and are the gold evidence sessions all represented? OFFLINE."""
import json
import pickle
import re
import statistics as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifecycle import assemble_chains, render_view
from v5_chains import (merge_chains, render_aggregation_view, aggregation_groups,
                       is_aggregation_question)

D = {x["qid"]: x for x in pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl", "rb"))}
G = pickle.load(open("/root/agent_memory_iclr/results/_v5_gold.pkl", "rb"))
QC = {}
for line in open("/root/agent_memory_iclr/results/full500_qc.jsonl"):
    r = json.loads(line)
    if "judge" in r:
        QC[r["qid"]] = r

WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
           "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
           "nineteen": 19, "twenty": 20}


def gold_number(ans):
    s = str(ans).lower()
    m = re.search(r"-?\d+(\.\d+)?", s.replace(",", ""))
    if m:
        return float(m.group(0))
    for w, v in WORDNUM.items():
        if re.search(rf"\b{w}\b", s):
            return float(v)
    return None


def current_view_items(store, question, use_hyponyms=True):
    """How many distinct values of the SAME matched chains does the v4 'current' view show?
    (current mode prints only the last session group per chain)."""
    groups, _ = aggregation_groups(store, question, use_hyponyms)
    shown = 0
    for g in groups:
        chain = store[g["key"]]
        if chain[0]["relation"] == "accumulative":
            shown += len({c["value"].lower() for c in chain})
        else:
            last = (chain[-1]["date"], chain[-1].get("sidx"))
            shown += len({c["value"].lower() for c in chain
                          if (c["date"], c.get("sidx")) == last})
    return shown, sum(len(g["items"]) for g in groups), len(groups)


rows = []
for qid, x in D.items():
    is_agg, fired = is_aggregation_question(x["question"])
    if not is_agg:
        continue
    store4 = assemble_chains(x["facts"])
    store5 = merge_chains(store4)
    gnum = gold_number(x["answer"])
    gold_sidx = set(G[qid]["gold_sidx"])

    out = {}
    for tag, store, hyp in [("v5", store5, True), ("v5_nohyp", store5, False),
                            ("v5_nomerge", store4, True)]:
        groups, _ = aggregation_groups(store, x["question"], hyp)
        # global dedupe, exactly as the renderer does
        seen, items = [], []
        from v5_chains import _value_key
        for g in groups[:25]:
            for c in g["items"][:25]:
                vk = _value_key(c["value"])
                if any(vk == p or vk <= p for p in seen):
                    continue
                seen.append(vk)
                items.append(c)
        cov = {c.get("sidx") for c in items} & gold_sidx
        out[tag] = {"n_items": len(items), "n_groups": len(groups),
                    "gold_sess_hit": len(cov), "gold_sess_total": len(gold_sidx)}
    cur_shown, agg_items, ngroups = current_view_items(store5, x["question"])
    txt = render_aggregation_view(store5, x["question"])
    rows.append({
        "qid": qid, "type": x["type"], "question": x["question"], "gold": x["answer"],
        "gold_num": gnum, "judge": QC.get(qid, {}).get("judge"),
        "pred": (QC.get(qid, {}).get("pred") or "")[:300],
        "fired": fired, "n_gold_sess": len(gold_sidx),
        "v5_items": out["v5"]["n_items"], "v5_groups": out["v5"]["n_groups"],
        "v5_gold_sess_hit": out["v5"]["gold_sess_hit"],
        "nohyp_items": out["v5_nohyp"]["n_items"],
        "nohyp_gold_sess_hit": out["v5_nohyp"]["gold_sess_hit"],
        "nomerge_items": out["v5_nomerge"]["n_items"],
        "nomerge_gold_sess_hit": out["v5_nomerge"]["gold_sess_hit"],
        "v4_current_shown": cur_shown,
        "agg_chars": len(txt),
        "view_cur_chars": len(render_view(store5, "current")),
        "snippet": txt[:2500],
    })

with open("/root/agent_memory_iclr/results/_v5_agg.json", "w") as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)

print(f"aggregation-detected questions: {len(rows)} / 500")
from collections import Counter
print("by type:", dict(Counter(r["type"] for r in rows)))
print("judge on these (excluding None):",
      f"{sum(1 for r in rows if r['judge'])}/{sum(1 for r in rows if r['judge'] is not None)}")
numeric = [r for r in rows if r["gold_num"] is not None]
print(f"with numeric gold: {len(numeric)}")


def derivable(r, tag="v5"):
    if r["gold_num"] is None:
        return None
    return (r[f"{tag}_items"] >= r["gold_num"]) and (r[f"{tag}_gold_sess_hit"] == r["n_gold_sess"])


for tag, label in [("v5", "v5 (merge + hyponyms)"), ("nohyp", "v5 no hyponym table"),
                   ("nomerge", "v5 aggregation on UNMERGED store")]:
    d = [derivable(r, tag) for r in numeric]
    print(f"{label:34s} count-derivable {sum(1 for v in d if v)}/{len(d)} = "
          f"{sum(1 for v in d if v)/len(d):.3f}")
d4 = [(r["v4_current_shown"] >= r["gold_num"]) for r in numeric]
print(f"{'v4 current view (same chains)':34s} items>=gold      {sum(d4)}/{len(d4)} = "
      f"{sum(d4)/len(d4):.3f}   (no gold-session check: current view drops earlier items)")
print(f"\nitems exposed: v5 median {st.median([r['v5_items'] for r in numeric])}, "
      f"v4 current-view median {st.median([r['v4_current_shown'] for r in numeric])}, "
      f"gold count median {st.median([r['gold_num'] for r in numeric])}")
print(f"gold-session recall: v5 "
      f"{sum(r['v5_gold_sess_hit'] for r in rows)}/{sum(r['n_gold_sess'] for r in rows)} = "
      f"{sum(r['v5_gold_sess_hit'] for r in rows)/sum(r['n_gold_sess'] for r in rows):.3f}; "
      f"nomerge {sum(r['nomerge_gold_sess_hit'] for r in rows)/sum(r['n_gold_sess'] for r in rows):.3f}; "
      f"nohyp {sum(r['nohyp_gold_sess_hit'] for r in rows)/sum(r['n_gold_sess'] for r in rows):.3f}")
print(f"aggregation block chars: median {st.median([r['agg_chars'] for r in rows])}, "
      f"p90 {sorted(r['agg_chars'] for r in rows)[int(0.9*len(rows))]}, "
      f"max {max(r['agg_chars'] for r in rows)}")

print("\n=== per-question (numeric gold) ===")
print(f"{'qid':18s} {'type':6s} {'gold':>6s} {'v5it':>5s} {'v4cur':>6s} {'gsess':>7s} "
      f"{'deriv':>6s} {'judge':>6s}  question")
for r in sorted(numeric, key=lambda r: (r["type"], r["qid"])):
    print(f"{r['qid']:18s} {r['type'][:6]:6s} {r['gold_num']:6.0f} {r['v5_items']:5d} "
          f"{r['v4_current_shown']:6d} {r['v5_gold_sess_hit']}/{r['n_gold_sess']:<5d} "
          f"{str(derivable(r)):>6s} {str(r['judge']):>6s}  {r['question'][:70]}")
