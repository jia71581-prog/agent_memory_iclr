"""Adversarial part 3: direction-aware ordering verdict (the module's own metric mishandles
'most recently' questions) + count of degenerate item->question-date defaults."""
import json, re, sys, statistics as st
from collections import Counter
sys.path.insert(0, "/root/agent_memory_iclr/code")
import v5_temporal as V
from common import load_dataset
from v5_temporal_measure import _match_event, ordering_items, norm, contains
from v5_temporal_val import build_store_cached

R = "/root/agent_memory_iclr/results"
data = {q["question_id"]: q for q in load_dataset("s_cleaned")}
qc = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{R}/full500_qc.jsonl")}
rows = json.load(open(f"{R}/_ver3_v5t_rows.json"))
o = [r for r in rows if r["family"] == "ordering"]

LATEST = re.compile(r"most recent|recently|latest|last(?:est)?\b|most current", re.I)
EARLIEST = re.compile(r"\bfirst\b|earliest|initially|originally", re.I)

res = []
for r in o:
    q = data[r["qid"]]
    store = build_store_cached(q)[0]
    ev = V.extract_events(q, store, q["question"])
    q_day = V.day0(V.parse_date(q["question_date"]))
    gseq, cands = ordering_items(q["question"], q["answer"])
    dmap = {c: _match_event(c, ev) for c in cands}
    n_at_qdate = sum(1 for c, d in dmap.items() if d == q_day)
    matched = [c for c in cands if dmap[c]]
    want_latest = bool(LATEST.search(q["question"])) and not EARLIEST.search(q["question"])
    verdict = None
    if len(cands) >= 2 and len(matched) == len(cands) and gseq:
        if len(gseq) >= 3:
            got = sorted(cands, key=lambda c: dmap[c])
            verdict = [norm(x) for x in got] == [norm(x) for x in gseq]
        else:
            got = sorted(cands, key=lambda c: dmap[c], reverse=want_latest)
            verdict = contains(got[0], gseq[0]) or contains(gseq[0], got[0])
    ties = len({dmap[c] for c in matched}) < len(matched)
    res.append({"qid": r["qid"], "n_items": len(cands), "want_latest": want_latest,
                "verdict": verdict, "orig": r.get("ordering_item_correct"),
                "n_at_qdate": n_at_qdate, "ties": ties,
                "live": bool(qc[r["qid"]]["judge"])})

comp = [x for x in res if x["verdict"] is not None]
print("=== direction-aware ordering verdict (my correction of the module's metric) ===")
print(f"ordering n={len(res)} computable={len(comp)} "
      f"correct={sum(x['verdict'] for x in comp)}/{len(comp)} = "
      f"{sum(x['verdict'] for x in comp)/len(comp):.3f}")
print(f"module's own metric: {sum(bool(x['orig']) for x in comp)}/{len(comp)}")
print(f"questions where 'most recently/latest' direction applies: "
      f"{sum(x['want_latest'] for x in res)} (module's metric always scored these as 'earliest')")
bi = [x for x in comp if x["n_items"] == 2]
print(f"binary subfamily {sum(x['verdict'] for x in bi)}/{len(bi)} = "
      f"{sum(x['verdict'] for x in bi)/len(bi):.3f} (chance 0.5)")
import math
k, n = sum(x["verdict"] for x in bi), len(bi)
print(f"  one-sided binomial p = {sum(math.comb(n,i) for i in range(k,n+1))/2**n:.3f}")

print("\n=== regression exposure, direction-corrected ===")
lok = [x for x in res if x["live"]]
bad = [x for x in lok if x["verdict"] is False]
non = [x for x in lok if x["verdict"] is None]
up = [x for x in res if not x["live"] and x["verdict"] is True]
print(f"live-correct ordering questions: {len(lok)}/{len(res)}")
print(f"  scaffold asserts a WRONG order on: {len(bad)} -> {[x['qid'] for x in bad]}")
print(f"  scaffold cannot place all named items (order asserted anyway, warning DISABLED): {len(non)}")
print(f"live-wrong where scaffold order is right (upside): {len(up)} -> {[x['qid'] for x in up]}")

print("\n=== degenerate default: named item silently mapped to the question-date session ===")
print(f"ordering questions with >=1 named item defaulted to the question date: "
      f"{sum(x['n_at_qdate']>0 for x in res)}/{len(res)}")
print(f"  ... with ALL named items defaulted to the question date: "
      f"{sum(x['n_at_qdate']==x['n_items'] and x['n_items']>0 for x in res)}")
print(f"ordering questions with tied dates among matched items: {sum(x['ties'] for x in res)}")
print(f"EMIT_MISSING_ITEM_NOTE default = {V.EMIT_MISSING_ITEM_NOTE}")
json.dump(res, open(f"{R}/_ver3_v5t_ord.json", "w"), indent=1)
