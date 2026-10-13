"""ADVERSARIAL independent recomputation of the v5 part(a)/part(b) decomposition claim.
Rebuilds facts straight from the session cache + dataset (does NOT trust _v5_facts.pkl),
rebuilds gold_sidx from the dataset, then recomputes D1/D2/gold-session recall for
merge vs nomerge, with per-question flips, block sizes, and McNemar.
No API.
"""
import hashlib
import json
import os
import pickle
import re
import statistics as st
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains, render_view, _session_text, SESSION_CACHE_DIR
from v5_chains import (merge_chains, render_aggregation_view, aggregation_printed_groups,
                       is_aggregation_question, aggregation_kind)

DATA = "/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"

WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
           "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
           "nineteen": 19, "twenty": 20}
GOLD_UNIT = re.compile(r"\b(dollars?|\$|hours?|minutes?|miles?|pounds?|lbs?|kg|pages?|"
                       r"calories|percent|%|days?|weeks?|months?|years?|nights?|dozen|"
                       r"degrees?|stars?|followers?|points?|steps?|reps?|km|liters?|"
                       r"cups?|ounces?|oz|gallons?)\b")
ARITH_Q = re.compile(r"\bhow many (days?|weeks?|months?|years?|hours?|minutes?|nights?)\b|"
                     r"\b(ago|older|younger|since I|before my|before the|left to|remaining)\b")


def gold_number(ans):
    s = str(ans).lower().replace(",", "")
    m = re.search(r"-?\d+(\.\d+)?", s)
    if m:
        return float(m.group(0))
    for w, v in WORDNUM.items():
        if re.search(rf"\b{w}\b", s):
            return float(v)
    return None


def is_enum_count(q, gold, gnum):
    ql = q.lower()
    if not re.search(r"\bhow many\b|\bnumber of\b", ql):
        return False
    if gnum is None or gnum != int(gnum) or gnum > 12 or gnum < 1:
        return False
    if GOLD_UNIT.search(str(gold).lower()):
        return False
    if re.search(r"\bhow much\b", ql) or ARITH_Q.search(ql):
        return False
    return True


def cached_facts(sess):
    text = _session_text(sess)
    if not text:
        return []
    key = hashlib.md5(text.encode()).hexdigest()[:16]
    p = os.path.join(SESSION_CACHE_DIR, f"{key}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


QC = {}
for line in open("/root/agent_memory_iclr/results/full500_qc.jsonl"):
    r = json.loads(line)
    if "judge" in r:
        QC[r["qid"]] = r

print("loading dataset...", flush=True)
data = json.load(open(DATA))
print("questions", len(data), flush=True)

PK = {x["qid"]: x for x in pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl", "rb"))}

rows = []
fact_mismatch = []
for q in data:
    qid = q["question_id"]
    if not is_aggregation_question(q["question"])[0]:
        continue
    facts, missing = [], False
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        tri = cached_facts(sess)
        if tri is None:
            missing = True
            continue
        for e, a, v in tri:
            facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
    # independence check vs the prepared pickle
    pk = PK.get(qid)
    if pk is not None:
        mine = [(f["entity"], f["attribute"], f["value"], f["date"], f["sidx"]) for f in facts]
        theirs = [(f["entity"], f["attribute"], f["value"], f["date"], f["sidx"]) for f in pk["facts"]]
        if mine != theirs:
            fact_mismatch.append((qid, len(mine), len(theirs)))
    hs = q.get("haystack_session_ids", [])
    ans = set(q.get("answer_session_ids", []))
    gold_sidx = {i for i, sid in enumerate(hs) if sid in ans}
    gnum = gold_number(q["answer"])
    store4 = assemble_chains(facts)
    store5 = merge_chains(store4)
    # fact conservation under merging
    def bag(store):
        out = {}
        for ch in store.values():
            for c in ch:
                k = (c.get("ent_orig", c["entity"]), c.get("attr_orig", c["attribute"]),
                     c["value"], c["date"], c.get("sidx"))
                out[k] = out.get(k, 0) + 1
        return out
    b4, b5 = bag(store4), bag(store5)
    arms = {}
    for tag, store, hyp in [("v5", store5, True), ("nohyp", store5, False),
                            ("nomerge", store4, True)]:
        kept, _ = aggregation_printed_groups(store, q["question"], hyp)
        printed = [[it[1] for it in g["items"]] for g in kept]
        strong = [[it[1] for it in g["items"] if it[3] >= 2] for g in kept]
        allitems = [c for gi in printed for c in gi]
        cov = {c.get("sidx") for c in allitems} & gold_sidx
        txt = render_aggregation_view(store, q["question"], hyp)
        arms[tag] = {"items": len(allitems), "groups": len(printed),
                     "best_group": max((len(gi) for gi in printed), default=0),
                     "best_strong": max((len(gi) for gi in strong), default=0),
                     "gold_hit": len(cov), "chars": len(txt),
                     "n_chains": len(store)}
    rows.append({
        "qid": qid, "type": q["question_type"], "question": q["question"],
        "gold": q["answer"], "gold_num": gnum,
        "enum": is_enum_count(q["question"], q["answer"], gnum),
        "kind": aggregation_kind(q["question"]),
        "judge": QC.get(qid, {}).get("judge"),
        "n_gold_sess": len(gold_sidx), "cache_missing": missing,
        "facts_lost_by_merge": sum(max(0, v - b5.get(k, 0)) for k, v in b4.items()),
        "facts_added_by_merge": sum(max(0, v - b4.get(k, 0)) for k, v in b5.items()),
        "view_chars_cur": len(render_view(store5, "current")),
        **{f"{t}_{k}": v for t, d in arms.items() for k, v in d.items()},
    })

json.dump(rows, open("/root/agent_memory_iclr/results/_adv_recompute.json", "w"), indent=1)
print("fact-pickle mismatches:", len(fact_mismatch), fact_mismatch[:5])
print("cache_missing questions among triggered:", sum(1 for r in rows if r["cache_missing"]))
print("triggered", len(rows), "count-kind", sum(1 for r in rows if r["kind"] == "count"),
      "amount", sum(1 for r in rows if r["kind"] == "amount"))
enum = [r for r in rows if r["enum"]]
print("ENUM subset", len(enum))


def d1(r, t):
    return r[f"{t}_items"] >= r["gold_num"] and r[f"{t}_gold_hit"] == r["n_gold_sess"]


def d2(r, t):
    return r[f"{t}_best_group"] >= r["gold_num"] and r[f"{t}_gold_hit"] == r["n_gold_sess"]


tot_gs = sum(r["n_gold_sess"] for r in enum)
print(f"\n{'arm':28s} {'D1':>7s} {'D2':>7s} {'recall':>7s} {'items_med':>9s} {'chars_med':>9s}")
for t in ("v5", "nohyp", "nomerge"):
    print(f"{t:28s} {sum(d1(r,t) for r in enum)/len(enum):7.3f} "
          f"{sum(d2(r,t) for r in enum)/len(enum):7.3f} "
          f"{sum(r[f'{t}_gold_hit'] for r in enum)/tot_gs:7.3f} "
          f"{st.median([r[f'{t}_items'] for r in enum]):9.1f} "
          f"{st.median([r[f'{t}_chars'] for r in enum]):9.0f}")

print("\n-- D1 discordant pairs (merge vs nomerge) --")
for r in enum:
    a, b = d1(r, "v5"), d1(r, "nomerge")
    if a != b:
        print(f"  {r['qid']:16s} merge={a} nomerge={b} gold={r['gold_num']:.0f} "
              f"items {r['v5_items']}/{r['nomerge_items']} "
              f"gs {r['v5_gold_hit']}/{r['nomerge_gold_hit']}/{r['n_gold_sess']} "
              f"judge={r['judge']}  {r['question'][:60]}")
print("-- D2 discordant pairs --")
for r in enum:
    a, b = d2(r, "v5"), d2(r, "nomerge")
    if a != b:
        print(f"  {r['qid']:16s} merge={a} nomerge={b} gold={r['gold_num']:.0f} "
              f"best {r['v5_best_group']}/{r['nomerge_best_group']}  {r['question'][:60]}")
print("-- gold-session recall deltas --")
for r in enum:
    if r["v5_gold_hit"] != r["nomerge_gold_hit"]:
        print(f"  {r['qid']:16s} {r['v5_gold_hit']} vs {r['nomerge_gold_hit']} of {r['n_gold_sess']}"
              f"  chars {r['v5_chars']}/{r['nomerge_chars']}")
print("\nmerge fact loss total:", sum(r["facts_lost_by_merge"] for r in rows),
      "added:", sum(r["facts_added_by_merge"] for r in rows))
print("chars enum: merge sum", sum(r["v5_chars"] for r in enum),
      "nomerge sum", sum(r["nomerge_chars"] for r in enum))
print("items enum: merge sum", sum(r["v5_items"] for r in enum),
      "nomerge sum", sum(r["nomerge_items"] for r in enum))
