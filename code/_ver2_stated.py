"""Independent probe: is the gold answer's number LITERALLY STATED in the gold evidence
sessions (-> 'stated total', no enumeration needed), or must it be DERIVED by counting?
Offline: uses _v5_gold.pkl (gold session text) only.
"""
import json
import pickle
import re
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ver2_recompute import gold_number, GOLD_UNIT, ARITH_Q, WORDNUM

ROOT = "/root/agent_memory_iclr"
rows = json.load(open(f"{ROOT}/results/_ver2_rows.json"))
G = pickle.load(open(f"{ROOT}/results/_v5_gold.pkl", "rb"))
NUMWORD = {v: k for k, v in WORDNUM.items()}


def stated(gnum, text):
    """gold number appears verbatim (digits, comma-grouped digits, or number word)."""
    if gnum is None:
        return None
    t = text.lower()
    n = int(gnum) if gnum == int(gnum) else gnum
    pats = [rf"\b{re.escape(str(n))}\b"]
    if isinstance(n, int) and n >= 1000:
        pats.append(rf"\b{n//1000},{n%1000:03d}\b")
    if isinstance(n, int) and n in NUMWORD:
        pats.append(rf"\b{NUMWORD[n]}\b")
    return any(re.search(p, t) for p in pats)


out = []
for r in rows:
    if not r["agg"]:
        continue
    g = G[r["qid"]]
    txt = "\n".join(s[1] for s in g["gold_sessions"])
    r2 = dict(r)
    r2["stated"] = stated(r["gnum"], txt)
    r2["n_gold_sess"] = len(g["gold_sidx"])
    out.append(r2)

json.dump(out, open(f"{ROOT}/results/_ver2_stated.json", "w"), ensure_ascii=False, indent=1)

enum = [r for r in out if r["enum"]]
non = [r for r in out if not r["enum"]]


def summ(lab, rs):
    ng = [r for r in rs if r["gnum"] is not None]
    st = sum(1 for r in ng if r["stated"])
    j = [r for r in rs if r["judge"] is not None]
    print(f"{lab:38s} n={len(rs):3d}  parseable-gold-num={len(ng):3d}  "
          f"gold number literally in gold session: {st:3d} ({st/max(1,len(ng)):.3f})  "
          f"v4 acc {sum(1 for r in j if r['judge'])/max(1,len(j)):.3f}")


print("== 'is the answer already stated in the evidence?' ==")
summ("enum-count subset (60)", enum)
summ("other triggered (167)", non)
summ("  of those: how-many/number-of", [r for r in non
                                        if re.search(r"\bhow many\b|\bnumber of\b", r["question"].lower())])
summ("  of those: how-much/total-phrases", [r for r in non
                                            if not re.search(r"\bhow many\b|\bnumber of\b", r["question"].lower())])

print("\n== the 167, by declared bucket (claim: stated-total / magnitude / date-arith) ==")


def bucket(r):
    ql, gl = r["question"].lower(), str(r["gold"]).lower()
    if ARITH_Q.search(ql):
        return "date/interval-arithmetic"
    if re.search(r"\bhow much\b", ql) or GOLD_UNIT.search(gl):
        return "magnitude (money/time/length/unit)"
    if r["gnum"] is None:
        return "gold not a number"
    if r["stated"]:
        return "stated total (number verbatim in evidence)"
    if r["gnum"] > 12:
        return "large count, NOT verbatim in evidence"
    return "other"


c = Counter(bucket(r) for r in non)
for k, v in c.most_common():
    print(f"  {v:4d}  {k}")

print("\n== residual: non-enum, how-many/number-of, gold number NOT stated verbatim ==")
res = [r for r in non if re.search(r"\bhow many\b|\bnumber of\b", r["question"].lower())
       and r["gnum"] is not None and not r["stated"] and not ARITH_Q.search(r["question"].lower())
       and not GOLD_UNIT.search(str(r["gold"]).lower())]
print(f"n={len(res)}")
for r in sorted(res, key=lambda r: -(r["gnum"] or 0)):
    print(f"  {r['qid']:14s} {r['type'][:10]:10s} gold={str(r['gold'])[:22]:24s} "
          f"j={str(r['judge'])[:5]:5s} {r['question'][:78]}")

print("\n== enum subset: gold number stated verbatim anyway? (would mean not derived) ==")
for r in enum:
    if r["stated"]:
        print(f"  {r['qid']:14s} gold={str(r['gold'])[:18]:20s} j={str(r['judge'])[:5]:5s} "
              f"{r['question'][:76]}")
