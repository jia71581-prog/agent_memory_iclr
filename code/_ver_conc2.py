"""Adversarial check 2: (a) baselines with ZERO dependence on v5_chains machinery,
(b) item-matched and item-char-matched budgets, (c) is gold-session coverage the binding
constraint on answer correctness at all?  OFFLINE."""
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
from lifecycle import assemble_chains, render_view
from v5_chains import (merge_chains, render_aggregation_view, aggregation_printed_groups,
                       is_aggregation_question)

# ---- a completely generic tokenizer/stopword list, independent of v5_chains
STOP = set("""a an the this that these those i me my mine we our us you your he she it they them
their his her its is are was were be been being am do does did have has had will would can could
should shall may might must of in on at to for with about from by as into and or but if then than
so how what which who whom when where why many much often more most some any all each every no not
number total times time there here now today currently still ever last past recent recently over
since during after before get got make made take took give gave go went come came want need use
used using one two three do did done things thing stuff""".split())


def toks(s):
    out = set()
    for t in re.findall(r"[a-z0-9]+", str(s).lower()):
        if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
            t = t[:-1]
        if t in STOP or len(t) < 3:
            continue
        out.add(t)
    return out


D = {x["qid"]: x for x in pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl", "rb"))}
G = pickle.load(open("/root/agent_memory_iclr/results/_v5_gold.pkl", "rb"))
QC = {}
for line in open("/root/agent_memory_iclr/results/full500_qc.jsonl"):
    r = json.loads(line)
    if "judge" in r:
        QC[r["qid"]] = r

rng = random.Random(7)
rows = []
for qid, x in D.items():
    if not is_aggregation_question(x["question"])[0]:
        continue
    gold = set(G[qid]["gold_sidx"])
    s5 = merge_chains(assemble_chains(x["facts"]))
    kept, _ = aggregation_printed_groups(s5, x["question"])
    printed = [it[1] for g in kept for it in g["items"]]
    blk = render_aggregation_view(s5, x["question"])
    item_lines = [l for l in blk.splitlines() if re.match(r"^  [\* ]- ", l)]
    item_chars = sum(len(l) + 1 for l in item_lines)
    n_items = len(printed)
    blk_cov = len({c.get("sidx") for c in printed} & gold)

    qt = toks(x["question"])
    facts = x["facts"]
    N = max(1, len(facts))
    ft = [toks(c["attribute"] + " " + c["value"]) for c in facts]
    df = Counter()
    for tt in ft:
        for t in tt & qt:
            df[t] += 1
    idf = {t: math.log(N / (1 + df[t])) for t in qt}
    sc = [sum(idf[t] for t in (ft[i] & qt)) for i in range(len(facts))]
    order = sorted(range(len(facts)), key=lambda i: (-sc[i], i))
    hit = [facts[i] for i in order if sc[i] > 0]

    def pick_chars(seq, budget):
        out, tot = [], 0
        for c in seq:
            ln = len(f"  - {c['attribute']} = {c['value']} ({str(c['date']).split()[0]})") + 1
            if tot + ln > budget:
                break
            out.append(c)
            tot += ln
        return out

    # C1 naive IDF keyword, full block char budget
    c1 = pick_chars(hit, len(blk))
    # C2 naive IDF keyword, only the block's ITEM-line char budget (header/footer charged to block)
    c2 = pick_chars(hit, item_chars)
    # C3 naive IDF keyword, exactly the same NUMBER of items as the block
    c3 = hit[:n_items]
    # C4 same as C2 but value-deduplicated first (cheap dedupe, still no merging/grouping)
    seen, ded = set(), []
    for c in hit:
        k = re.sub(r"[^a-z0-9]", "", c["value"].lower())
        if k in seen:
            continue
        seen.add(k)
        ded.append(c)
    c4 = pick_chars(ded, item_chars)
    cov = lambda L: len({c.get("sidx") for c in L} & gold)
    rows.append({"qid": qid, "n_gold": len(gold), "judge": QC.get(qid, {}).get("judge"),
                 "blk_cov": blk_cov, "blk_chars": len(blk), "item_chars": item_chars,
                 "n_items": n_items,
                 "c1": cov(c1), "c2": cov(c2), "c3": cov(c3), "c4": cov(c4),
                 "c1n": len(c1), "c2n": len(c2), "c3n": len(c3), "c4n": len(c4),
                 "hitpool": len(hit)})

tg = sum(r["n_gold"] for r in rows)
print(f"n={len(rows)} triggered, gold sessions {tg}")
print(f"block                                  {sum(r['blk_cov'] for r in rows)/tg:.3f} "
      f"@ median {st.median([r['blk_chars'] for r in rows]):.0f} chars "
      f"(item lines only: {st.median([r['item_chars'] for r in rows]):.0f}), "
      f"median {st.median([r['n_items'] for r in rows]):.0f} items")
for k, lab in [("c1", "naive IDF keyword @ full block chars"),
               ("c2", "naive IDF keyword @ block ITEM chars"),
               ("c3", "naive IDF keyword @ same #items   "),
               ("c4", "naive IDF + value dedupe @ item chars")]:
    w = sum(1 for r in rows if r["blk_cov"] > r[k])
    l = sum(1 for r in rows if r["blk_cov"] < r[k])
    print(f"{lab:38s} {sum(r[k] for r in rows)/tg:.3f}  "
          f"(median {st.median([r[k+'n'] for r in rows]):.0f} items; "
          f"block better {w}, worse {l}, tie {len(rows)-w-l})")

print("\n=== is gold-session coverage the binding constraint on correctness? ===")
J = [r for r in rows if r["judge"] is not None]
print(f"v4 pipeline judge accuracy on the {len(J)} triggered questions with a judge: "
      f"{sum(1 for r in J if r['judge'])/len(J):.3f}")
full = [r for r in J if r["blk_cov"] == r["n_gold"]]
part = [r for r in J if r["blk_cov"] < r["n_gold"]]
print(f"block covers ALL gold sessions: {len(full)} questions, v4 correct "
      f"{sum(1 for r in full if r['judge'])/max(1,len(full)):.3f}")
print(f"block misses >=1 gold session:  {len(part)} questions, v4 correct "
      f"{sum(1 for r in part if r['judge'])/max(1,len(part)):.3f}")
wrong = [r for r in J if not r["judge"]]
print(f"among the {len(wrong)} v4-WRONG triggered questions, the block already covers every "
      f"gold session in {sum(1 for r in wrong if r['blk_cov']==r['n_gold'])} of them "
      f"({sum(1 for r in wrong if r['blk_cov']==r['n_gold'])/max(1,len(wrong)):.1%})")
json.dump(rows, open("/root/agent_memory_iclr/results/_ver_conc2.json", "w"), indent=1)
