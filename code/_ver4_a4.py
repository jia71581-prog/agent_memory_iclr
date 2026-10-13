"""Adversarial recompute of the A4 absence-certificate claim. No API calls."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_guards as G

DATA = "/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"
RES = "/root/agent_memory_iclr/results"

data = json.load(open(DATA))
print("n questions:", len(data))
qc = {r["qid"]: r for r in (json.loads(l) for l in open(f"{RES}/full500_qc.jsonl"))}
print("qc rows:", len(qc), "with judge:", sum(1 for r in qc.values() if "judge" in r))

rows = []
for q in data:
    qid = q["question_id"]
    u = G.units_of(q)
    c1 = G.absence_certificate(q["question"], u)                      # D1|D2 only
    c4 = G.absence_certificate(q["question"], u, use_d3=True, use_d4=True)
    proper, uni, bi = G.question_terms(q["question"])
    hay = " ".join(G.norm(x[2]) for x in u)
    d1 = [t for t in proper if t not in hay]
    d2 = [t for t in uni if t not in hay]
    rows.append({
        "qid": qid, "type": q["question_type"], "is_abs": qid.endswith("_abs"),
        "question": q["question"],
        "d1": bool(d1), "d2": bool(d2), "d12": bool(c1["missing_terms"]),
        "d3": bool(c4["missing_pairs"]), "d4": c4["unlinked_pair"] is not None,
        "union": bool(c4["fired"]),
        "terms": c1["missing_terms"], "pairs": c4["missing_pairs"],
        "unlinked": c4["unlinked_pair"],
        "advice": G.advice_shaped(q["question"]),
        "meta": G.meta_conversational(q["question"]),
        "judge": qc.get(qid, {}).get("judge"),
        "pred": qc.get(qid, {}).get("pred", ""),
    })

A = [r for r in rows if r["is_abs"]]
N = [r for r in rows if not r["is_abs"]]
print(f"abs={len(A)} answerable={len(N)}")


def rp(name, fn):
    ta = sum(1 for r in A if fn(r))
    fa = sum(1 for r in N if fn(r))
    print(f"{name:46s} abs {ta:2d}/{len(A)} ({ta/len(A):.3f})  ans {fa:3d}/{len(N)} "
          f"({fa/len(N):.3f})  prec {ta/max(1,ta+fa):.3f}")


print("\n--- detector rates (independent recompute) ---")
rp("D1 missing proper noun (alone)", lambda r: r["d1"])
rp("D2 missing content noun (alone)", lambda r: r["d2"])
rp("D1|D2", lambda r: r["d12"])
rp("D3 missing bigram, unigrams present", lambda r: r["d3"])
rp("D4 two rarest never co-occur", lambda r: r["d4"])
rp("D2|D3|D4", lambda r: r["d2"] or r["d3"] or r["d4"])
rp("D1|D2|D3|D4", lambda r: r["union"])

print("\n--- router gate ---")
rp("raw D1|D2", lambda r: r["d12"])
rp("gated off advice", lambda r: r["d12"] and not r["advice"])
rp("gated off advice + meta", lambda r: r["d12"] and not r["advice"] and not r["meta"])
print("abs questions that are advice_shaped:", [r["qid"] for r in A if r["advice"]])
print("abs questions that are meta_conversational:", [r["qid"] for r in A if r["meta"]])

print("\n--- the 5 qc abstention losses (from data, not from the docstring) ---")
losses = sorted(r["qid"] for r in A if r["judge"] is False)
print("qc abs judge==False:", losses)
print("qc abs acc:", sum(1 for r in A if r["judge"]), "/", len(A))
claimed = ["80ec1f4f_abs", "a96c20ee_abs", "09ba9854_abs", "gpt4_93159ced_abs", "031748ae_abs"]
print("matches docstring set:", sorted(claimed) == losses)
for r in A:
    if r["qid"] in claimed:
        print(f"  {r['qid']:20s} gatedD12={r['d12'] and not r['advice'] and not r['meta']!s:5s} "
              f"union={r['union']!s:5s} terms={r['terms']} pairs={r['pairs'][:2]} unlinked={r['unlinked']} "
              f"advice={r['advice']} meta={r['meta']}")

print("\n--- residual risk of the gated note: 42 answerable false alarms ---")
fa = [r for r in N if r["d12"] and not r["advice"] and not r["meta"]]
print("count:", len(fa), " currently correct:", sum(1 for r in fa if r["judge"]),
      " currently wrong:", sum(1 for r in fa if r["judge"] is False))
from collections import Counter
print("by type:", Counter(r["type"] for r in fa))
print("already declining (is_decline on qc pred):",
      sum(1 for r in fa if G.is_decline(r["pred"])))

print("\n--- ungated: the two SSP declines the note would have reinforced ---")
for qid in ("0edc2aef", "09d032c9"):
    r = next(x for x in rows if x["qid"] == qid)
    print(f"  {qid} type={r['type']} judge={r['judge']} advice={r['advice']} meta={r['meta']} "
          f"d12={r['d12']} terms={r['terms']} decline={G.is_decline(r['pred'])}")
    print(f"    pred[:220]: {r['pred'][:220]!r}")

print("\n--- SSP false alarms that the advice gate suppresses ---")
sup = [r for r in N if r["d12"] and (r["advice"] or r["meta"])]
print("suppressed count:", len(sup), " of which currently correct:",
      sum(1 for r in sup if r["judge"]), " wrong:", sum(1 for r in sup if r["judge"] is False))
print("by type:", Counter(r["type"] for r in sup))

json.dump(rows, open(f"{RES}/_ver4_a4.json", "w"))
print("\nsaved", f"{RES}/_ver4_a4.json")
