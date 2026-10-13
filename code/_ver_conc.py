"""ADVERSARIAL VERIFICATION of the v5 aggregation-block 'concentration device' claim.
Independent recomputation + budget-matched baselines. OFFLINE, no API.
Writes /root/agent_memory_iclr/results/_ver_conc.json
"""
import json
import math
import pickle
import random
import re
import statistics as st
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifecycle import assemble_chains, render_view, _group_by_session
from v5_chains import (merge_chains, render_aggregation_view, aggregation_printed_groups,
                       is_aggregation_question, aggregation_kind, _stem_tokens, QSTOP, _syn)

CAP = 20000

D = {x["qid"]: x for x in pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl", "rb"))}
G = pickle.load(open("/root/agent_memory_iclr/results/_v5_gold.pkl", "rb"))
QC = {}
for line in open("/root/agent_memory_iclr/results/full500_qc.jsonl"):
    r = json.loads(line)
    if "judge" in r:
        QC[r["qid"]] = r


def qterms(q):
    out = []
    for t in _stem_tokens(q):
        t = _syn(t)
        if t in QSTOP or len(t) < 3:
            continue
        out.append(t)
    return set(out)


def cap_lines(txt, cap=CAP):
    """Same truncation rule as _v5_meas_size.capped / run_qc VIEW_CHAR_CAP."""
    if len(txt) <= cap:
        return txt, None
    kept, total = [], 0
    for i, line in enumerate(txt.splitlines()):
        if total + len(line) + 1 > cap:
            return "\n".join(kept), i
        kept.append(line)
        total += len(line) + 1
    return "\n".join(kept), None


def visible_facts(chain):
    """Facts a 'current'-mode view line exposes for this chain."""
    if chain[0]["relation"] == "accumulative":
        return list(chain)
    last = (chain[-1]["date"], chain[-1].get("sidx"))
    return [c for c in chain if (c["date"], c.get("sidx")) == last]


def render_fact_lines(facts):
    return [f"  - {c['attribute']} = {c['value']} ({str(c['date']).split()[0]})" for c in facts]


def budget_pick(facts, budget):
    """Take facts in the given order until the rendered char budget is exhausted."""
    out, tot = [], 0
    for c in facts:
        ln = len(f"  - {c['attribute']} = {c['value']} ({str(c['date']).split()[0]})") + 1
        if tot + ln > budget:
            break
        out.append(c)
        tot += ln
    return out


rows = []
rng = random.Random(0)
for qid, x in D.items():
    if not is_aggregation_question(x["question"])[0]:
        continue
    gold = set(G[qid]["gold_sidx"])
    store4 = assemble_chains(x["facts"])
    store5 = merge_chains(store4)

    # ---- v5 block
    kept, _meta = aggregation_printed_groups(store5, x["question"])
    printed = [it[1] for g in kept for it in g["items"]]
    blk = render_aggregation_view(store5, x["question"])
    blk_chars = len(blk)
    blk_cov = len({c.get("sidx") for c in printed} & gold)

    # ---- full 'current' view, uncapped (their arm) and capped at 20000 (what run_qc sends)
    view_txt = render_view(store5, "current")
    view_cov_sidx = set()
    for ch in store5.values():
        for c in visible_facts(ch):
            view_cov_sidx.add(c.get("sidx"))
    view_cov = len(view_cov_sidx & gold)
    # capped: line i of render_view corresponds to sorted(store5)[i]
    keys = sorted(store5)
    capped_txt, cut_at = cap_lines(view_txt)
    n_lines_kept = len(keys) if cut_at is None else cut_at
    capped_sidx = set()
    for k in keys[:n_lines_kept]:
        for c in visible_facts(store5[k]):
            capped_sidx.add(c.get("sidx"))
    capped_cov = len(capped_sidx & gold)

    # ---- budget-matched baselines at exactly the block's char budget
    qt = qterms(x["question"])
    facts = x["facts"]
    N = max(1, len(facts))
    dfc = Counter()
    ftoks = []
    for c in facts:
        tt = {_syn(t) for t in _stem_tokens(c["attribute"] + " " + c["value"])}
        ftoks.append(tt)
        for t in tt & qt:
            dfc[t] += 1
    idf = {t: math.log(N / (1 + dfc[t])) for t in qt}

    def score(i):
        return sum(idf[t] for t in (ftoks[i] & qt))

    order_bm = sorted(range(len(facts)), key=lambda i: (-score(i), i))
    bm_facts = [facts[i] for i in order_bm if score(i) > 0]
    # B1 keyword/IDF top facts (no merging, no grouping, no hyponyms, no fuzzy prefix)
    b1 = budget_pick(bm_facts, blk_chars)
    b1_cov = len({c.get("sidx") for c in b1} & gold)
    # B2 plain any-question-token match, chronological
    b2src = [c for i, c in enumerate(facts) if ftoks[i] & qt]
    b2 = budget_pick(sorted(b2src, key=lambda c: (c["date"], c.get("sidx", 0))), blk_chars)
    b2_cov = len({c.get("sidx") for c in b2} & gold)
    # B3 recency only
    b3 = budget_pick(sorted(facts, key=lambda c: (c["date"], c.get("sidx", 0)), reverse=True),
                     blk_chars)
    b3_cov = len({c.get("sidx") for c in b3} & gold)
    # B4 random facts
    idx = list(range(len(facts)))
    rng.shuffle(idx)
    b4 = budget_pick([facts[i] for i in idx], blk_chars)
    b4_cov = len({c.get("sidx") for c in b4} & gold)
    # B5 head of the full view truncated to the block budget
    vt, cut = cap_lines(view_txt, blk_chars)
    nk = len(keys) if cut is None else cut
    b5_sidx = set()
    for k in keys[:nk]:
        for c in visible_facts(store5[k]):
            b5_sidx.add(c.get("sidx"))
    b5_cov = len(b5_sidx & gold)

    # block text features
    n_star = blk.count("\n  *-") + blk.count("\n  *- ")
    stars = len(re.findall(r"^  \*- ", blk, re.M))
    warns = blk.count("[may restate an earlier line")
    rows.append({
        "qid": qid, "type": x["type"], "question": x["question"], "gold": x["answer"],
        "n_gold": len(gold), "n_sess": x["n_sessions"], "n_facts": len(facts),
        "kind": aggregation_kind(x["question"]),
        "judge": QC.get(qid, {}).get("judge"),
        "blk_chars": blk_chars, "view_chars": len(view_txt), "capped_chars": len(capped_txt),
        "blk_cov": blk_cov, "view_cov": view_cov, "capped_cov": capped_cov,
        "b1_cov": b1_cov, "b2_cov": b2_cov, "b3_cov": b3_cov, "b4_cov": b4_cov,
        "b5_cov": b5_cov,
        "b1_n": len(b1), "b2_n": len(b2), "b3_n": len(b3), "b4_n": len(b4),
        "n_items": len(printed), "n_groups": len(kept),
        "stars": stars, "warns": warns,
        "chars_per_item": (blk_chars / len(printed)) if printed else None,
    })

json.dump(rows, open("/root/agent_memory_iclr/results/_ver_conc.json", "w"), indent=1)

tg = sum(r["n_gold"] for r in rows)
print(f"triggered questions {len(rows)}, total gold sessions {tg}, "
      f"median sessions/question {st.median([r['n_sess'] for r in rows]):.0f}")
print(f"\n{'arm':44s} {'gold-sess repr':>15s} {'median chars':>13s}")


def line(nm, ck, cc):
    print(f"{nm:44s} {sum(r[ck] for r in rows)/tg:15.3f} "
          f"{st.median([r[cc] for r in rows]):13.0f}")


line("full current view, UNCAPPED (their denom)", "view_cov", "view_chars")
line("full current view, capped at 20000 (as sent)", "capped_cov", "capped_chars")
line("v5 aggregation block", "blk_cov", "blk_chars")
line("B1 IDF keyword facts @ block budget", "b1_cov", "blk_chars")
line("B2 any-q-token facts, chrono @ budget", "b2_cov", "blk_chars")
line("B3 most-recent facts @ budget", "b3_cov", "blk_chars")
line("B4 random facts @ budget", "b4_cov", "blk_chars")
line("B5 head of view truncated @ budget", "b5_cov", "blk_chars")
print(f"\nblock size: median {st.median([r['blk_chars'] for r in rows]):.0f}, "
      f"p90 {sorted(r['blk_chars'] for r in rows)[int(0.9*len(rows))]}, "
      f"max {max(r['blk_chars'] for r in rows)}")
print(f"uncapped view median {st.median([r['view_chars'] for r in rows]):.0f} -> ratio "
      f"{st.median([r['blk_chars'] for r in rows])/st.median([r['view_chars'] for r in rows]):.3%}; "
      f"capped view median {st.median([r['capped_chars'] for r in rows]):.0f} -> ratio "
      f"{st.median([r['blk_chars'] for r in rows])/st.median([r['capped_chars'] for r in rows]):.3%}")
print(f"blocks with >=1 (*) marker: {sum(1 for r in rows if r['stars'])}/{len(rows)}; "
      f"blocks with >=1 restatement warning: {sum(1 for r in rows if r['warns'])}/{len(rows)}; "
      f"total warnings {sum(r['warns'] for r in rows)}, total (*) {sum(r['stars'] for r in rows)}")
print(f"items per block: median {st.median([r['n_items'] for r in rows]):.0f}, "
      f"max {max(r['n_items'] for r in rows)}; groups median "
      f"{st.median([r['n_groups'] for r in rows]):.0f}")
print(f"blocks that are EMPTY (never fire despite trigger): "
      f"{sum(1 for r in rows if r['blk_chars']==0)}")

# per-question win/loss of block vs baselines
for b in ["b1", "b2", "b3", "b4", "b5"]:
    w = sum(1 for r in rows if r["blk_cov"] > r[f"{b}_cov"])
    l = sum(1 for r in rows if r["blk_cov"] < r[f"{b}_cov"])
    print(f"block vs {b}: block better on {w}, worse on {l}, tie {len(rows)-w-l}")
