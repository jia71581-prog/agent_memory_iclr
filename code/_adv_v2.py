"""ADVERSARIAL recompute #2: is the 9.7% metric question-RELEVANT, and does the
aggregation block actually recover the hidden facts?

Splits the hidden gold-session facts into:
  - relevant  : the fact matches >=1 surviving question focus term (the same
                _fuzzy_terms / focus_weights machinery the agg block itself uses)
  - irrelevant: everything else (a gold session yields ~20 facts; most are noise)
Then measures, for the RELEVANT hidden facts, whether render_aggregation_view prints them.
Also measures what the production 20k char cap (run_qc.VIEW_CHAR_CAP) does to the same facts.
"""
import json, pickle, sys, os, statistics as st
from collections import Counter
sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains, render_view
from v5_chains import (merge_chains, is_aggregation_question, focus_weights,
                       _fuzzy_terms, _stem_tokens, _canon_tokens,
                       aggregation_printed_groups, render_aggregation_view)

D = {x["qid"]: x for x in pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl", "rb"))}
G = pickle.load(open("/root/agent_memory_iclr/results/_v5_gold.pkl", "rb"))
QC = {}
for line in open("/root/agent_memory_iclr/results/full500_qc.jsonl"):
    r = json.loads(line)
    if "judge" in r:
        QC[r["qid"]] = r

VIEW_CHAR_CAP = 20000


def visible(chain):
    if chain[0]["relation"] == "accumulative":
        return chain
    last = (chain[-1]["date"], chain[-1].get("sidx"))
    return [c for c in chain if (c["date"], c.get("sidx")) == last]


def cap_lines(txt, cap=VIEW_CHAR_CAP):
    """exact replica of run_qc.py's line-boundary truncation"""
    if len(txt) <= cap:
        return txt
    kept, total = [], 0
    for line in txt.splitlines():
        if total + len(line) + 1 > cap:
            break
        kept.append(line)
        total += len(line) + 1
    return "\n".join(kept)


tot = Counter()
per_q = []
for qid, x in D.items():
    if not is_aggregation_question(x["question"])[0]:
        continue
    gold_sidx = set(G[qid]["gold_sidx"])
    s4 = assemble_chains(x["facts"])
    s5 = merge_chains(s4)
    w, order, label = focus_weights(s5, x["question"], True)

    # what the aggregation block actually PRINTS (value strings, per the real renderer)
    kept, _ = aggregation_printed_groups(s5, x["question"])
    printed_ids = set()
    for g in kept:
        for it in g["items"]:
            printed_ids.add((it[1].get("sidx"), it[1]["value"]))

    # what survives the production 20k cap of the current view
    full_cur = render_view(s5, "current")
    capped_cur = cap_lines(full_cur)
    n_lines_full = len(full_cur.splitlines())
    n_lines_cap = len(capped_cur.splitlines())

    rel_h = rel_v = irr_h = irr_v = 0
    rec_rel_h = rec_rel_v = 0
    for k in sorted(s5):
        chain = s5[k]
        vis = visible(chain)
        attr = chain[0]["attribute"]
        a_terms = _fuzzy_terms(set(_canon_tokens(attr)) | set(_stem_tokens(attr)), w)
        for c in chain:
            if c.get("sidx") not in gold_sidx:
                continue
            v_terms = _fuzzy_terms(set(_stem_tokens(c["value"])), w)
            relevant = bool(a_terms or v_terms)
            shown = c in vis
            if relevant:
                if shown:
                    rel_v += 1
                else:
                    rel_h += 1
                    if (c.get("sidx"), c["value"]) in printed_ids:
                        rec_rel_h += 1
            else:
                if shown:
                    irr_v += 1
                else:
                    irr_h += 1
    tot["rel_h"] += rel_h; tot["rel_v"] += rel_v
    tot["irr_h"] += irr_h; tot["irr_v"] += irr_v
    tot["rec_rel_h"] += rec_rel_h
    tot["lines_full"] += n_lines_full; tot["lines_cap"] += n_lines_cap
    per_q.append({"qid": qid, "rel_h": rel_h, "rel_v": rel_v, "irr_h": irr_h, "irr_v": irr_v,
                  "rec_rel_h": rec_rel_h, "judge": QC.get(qid, {}).get("judge"),
                  "cur_chars": len(full_cur), "capped": len(full_cur) > VIEW_CHAR_CAP,
                  "lines_full": n_lines_full, "lines_cap": n_lines_cap,
                  "agg_chars": len(render_aggregation_view(s5, x["question"]))})

n = len(per_q)
rel_tot = tot["rel_h"] + tot["rel_v"]
irr_tot = tot["irr_h"] + tot["irr_v"]
print(f"questions {n}")
print(f"ALL gold facts            : {rel_tot+irr_tot}  hidden {tot['rel_h']+tot['irr_h']} "
      f"({(tot['rel_h']+tot['irr_h'])/(rel_tot+irr_tot):.2%})")
print(f"QUESTION-RELEVANT gold fx : {rel_tot}  hidden {tot['rel_h']} ({tot['rel_h']/max(1,rel_tot):.2%})")
print(f"IRRELEVANT gold facts     : {irr_tot}  hidden {tot['irr_h']} ({tot['irr_h']/max(1,irr_tot):.2%})")
print(f"  -> share of the 'hidden' mass that is question-irrelevant: "
      f"{tot['irr_h']/(tot['rel_h']+tot['irr_h']):.1%}")
print(f"RECOVERY: of {tot['rel_h']} relevant hidden facts, the aggregation block prints "
      f"{tot['rec_rel_h']} ({tot['rec_rel_h']/max(1,tot['rel_h']):.1%})")
print()
ncap = sum(1 for r in per_q if r["capped"])
print(f"PRODUCTION 20k CHAR CAP: {ncap}/{n} ({ncap/n:.1%}) of triggered questions have their "
      f"current view truncated by run_qc.py")
print(f"  view lines kept: {tot['lines_cap']}/{tot['lines_full']} = "
      f"{tot['lines_cap']/tot['lines_full']:.3f} -> the cap alone drops "
      f"{1-tot['lines_cap']/tot['lines_full']:.1%} of ALL chain lines")
print(f"  median current-view chars {st.median([r['cur_chars'] for r in per_q]):.0f}, "
      f"median agg-block chars {st.median([r['agg_chars'] for r in per_q]):.0f}")
json.dump(per_q, open("/root/agent_memory_iclr/results/_adv_v2.json", "w"))
