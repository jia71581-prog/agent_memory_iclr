"""Independent adversarial recompute of the aggregation-scope claim.
Offline only: dataset JSON + full500_*.jsonl. No API, no cache needed for the counts.
"""
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v5_chains import is_aggregation_question, aggregation_kind

ROOT = "/root/agent_memory_iclr"
DATA = json.load(open(f"{ROOT}/data/longmemeval_s_cleaned.json"))
QC = {}
for line in open(f"{ROOT}/results/full500_qc.jsonl"):
    r = json.loads(line)
    QC[r["qid"]] = r

# ---- re-implement the declared enum rule verbatim from _v5_meas_agg2.py -------------
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


rows = []
for x in DATA:
    qid, q, gold = x["question_id"], x["question"], x["answer"]
    agg, fired = is_aggregation_question(q)
    gnum = gold_number(gold)
    rows.append({"qid": qid, "type": x["question_type"], "question": q, "gold": gold,
                 "agg": agg, "fired": fired, "kind": aggregation_kind(q), "gnum": gnum,
                 "enum": bool(agg and is_enum_count(q, gold, gnum)),
                 "judge": QC.get(qid, {}).get("judge")})

trig = [r for r in rows if r["agg"]]
enum = [r for r in trig if r["enum"]]
print(f"DATASET questions: {len(rows)}   (unique qids {len({r['qid'] for r in rows})})")
print(f"trigger fires: {len(trig)}/{len(rows)}  count-kind "
      f"{sum(1 for r in trig if r['kind']=='count')}  amount-kind "
      f"{sum(1 for r in trig if r['kind']=='amount')}")
print(f"enum-count subset: {len(enum)}  types {dict(Counter(r['type'] for r in enum))}")
jn = [r for r in enum if r["judge"] is not None]
print(f"v4 (full500_qc) judge on enum: {sum(1 for r in jn if r['judge'])}/{len(jn)} "
      f"= {sum(1 for r in jn if r['judge'])/len(jn):.4f}")
allj = [r for r in rows if r["judge"] is not None]
print(f"v4 overall: {sum(1 for r in allj if r['judge'])}/{len(allj)} "
      f"= {sum(1 for r in allj if r['judge'])/len(allj):.4f}")
nonenum = [r for r in allj if not r["enum"]]
print(f"v4 on the complement (not enum): {sum(1 for r in nonenum if r['judge'])}/{len(nonenum)} "
      f"= {sum(1 for r in nonenum if r['judge'])/len(nonenum):.4f}")
trigj = [r for r in trig if r["judge"] is not None]
print(f"v4 on all 227 triggered: {sum(1 for r in trigj if r['judge'])}/{len(trigj)} "
      f"= {sum(1 for r in trigj if r['judge'])/len(trigj):.4f}")
notrig = [r for r in allj if not r["agg"]]
print(f"v4 on non-triggered: {sum(1 for r in notrig if r['judge'])}/{len(notrig)} "
      f"= {sum(1 for r in notrig if r['judge'])/len(notrig):.4f}")

json.dump(rows, open(f"{ROOT}/results/_ver2_rows.json", "w"), ensure_ascii=False, indent=1)

# ---- pattern census over the triggered set -----------------------------------------
print("\n-- fired patterns among triggered --")
for p, n in Counter(p for r in trig for p in r["fired"]).most_common():
    print(f"  {n:4d}  {p}")
print("\n-- type census --")
print("all:", dict(Counter(r["type"] for r in rows)))
print("trig:", dict(Counter(r["type"] for r in trig)))
