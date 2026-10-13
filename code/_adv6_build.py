"""Adversarial verifier: independent rebuild of per-question facts straight from the
lifecycle cache (no API), then the AGG.enum item-count subset metrics.
Nothing here reads the design agent's _v5_facts.pkl / _v5_agg*.json.
"""
import hashlib
import json
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import _session_text, assemble_chains
from v5_chains import (merge_chains, aggregation_printed_groups, aggregation_groups,
                       is_aggregation_question, aggregation_kind, _value_key)
from v5_shape import classify_shape

CACHE = "/root/agent_memory_iclr/results/lifecycle_cache/sessions_nt"
R = "/root/agent_memory_iclr/results"

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


def facts_from_cache(q):
    facts, miss, hit = [], 0, 0
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        text = _session_text(sess)
        if not text:
            continue
        key = hashlib.md5(text.encode()).hexdigest()[:16]
        p = os.path.join(CACHE, f"{key}.json")
        if not os.path.exists(p):
            miss += 1
            continue
        hit += 1
        try:
            tri = json.load(open(p))
        except Exception:
            miss += 1
            continue
        for t in tri:
            if len(t) == 3:
                facts.append({"entity": t[0], "attribute": t[1], "value": t[2],
                              "date": date, "sidx": si})
    return facts, hit, miss


def work(q):
    qid = q["question_id"]
    facts, hit, miss = facts_from_cache(q)
    store4 = assemble_chains(facts)
    store5 = merge_chains(store4)
    shape, fired = classify_shape(q["question"])
    is_agg, _ = is_aggregation_question(q["question"])
    gnum = gold_number(q["answer"])
    sid2idx = {s: i for i, s in enumerate(q["haystack_session_ids"])}
    gold_sidx = {sid2idx[s] for s in q["answer_session_ids"] if s in sid2idx}

    kept, _meta = aggregation_printed_groups(store5, q["question"])
    printed = [[it[1] for it in g["items"]] for g in kept]
    allitems = [c for gi in printed for c in gi]
    # distinct values across the printed union (surface-normalised)
    distinct_vals = len({re.sub(r"[^a-z0-9]+", " ", c["value"].lower()).strip() for c in allitems})
    best_group = max((len(gi) for gi in printed), default=0)
    cov = {c.get("sidx") for c in allitems} & gold_sidx
    # facts extracted from gold sessions at all (recall floor of the cache itself)
    n_facts_gold_sess = sum(1 for f in facts if f["sidx"] in gold_sidx)
    return {
        "qid": qid, "type": q["question_type"], "shape": shape, "is_agg": is_agg,
        "kind": aggregation_kind(q["question"]),
        "question": q["question"], "gold": str(q["answer"]), "gold_num": gnum,
        "enum": is_enum_count(q["question"], q["answer"], gnum),
        "n_ev": len(gold_sidx), "n_sess": len(q["haystack_sessions"]),
        "cache_hit": hit, "cache_miss": miss,
        "n_facts": len(facts), "n_facts_gold_sess": n_facts_gold_sess,
        "n_chains4": len(store4), "n_chains5": len(store5),
        "union_items": len(allitems), "union_distinct": distinct_vals,
        "best_group": best_group, "n_groups": len(printed),
        "gold_sess_hit": len(cov),
        "group_sizes": [len(gi) for gi in printed],
    }


if __name__ == "__main__":
    data = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
    with Pool(16) as p:
        rows = p.map(work, data, chunksize=2)
    json.dump(rows, open(f"{R}/_adv6_rows.json", "w"), ensure_ascii=False)
    print("rows", len(rows))
    print("cache misses total", sum(r["cache_miss"] for r in rows),
          "questions with any miss", sum(1 for r in rows if r["cache_miss"]))
    print("shape counts", Counter(r["shape"] for r in rows))
    enum = [r for r in rows if r["enum"]]
    print("is_enum_count subset (all 500):", len(enum))
    print("  of which is_agg:", sum(1 for r in enum if r["is_agg"]))
    print("  of which shape AGG.enum:", sum(1 for r in enum if r["shape"] == "AGG.enum"))
    print("  enum & is_agg n =", len([r for r in enum if r["is_agg"]]))
