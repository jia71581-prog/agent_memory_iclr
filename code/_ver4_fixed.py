"""The false alarms look like a normalization artifact, not presupposition failure.

question_terms() collapses a hyphen/apostrophe token to one string
  "5-day"  -> "5day"      "Lola's" -> "lolas"     "pre-approval" -> "preapproval"
while norm() on the memory turns the same characters into a SPACE
  "a 5-day trip" -> "a 5 day trip"
so any hyphenated/possessive question term is *guaranteed* to be reported as
"never appears anywhere in the stored conversations".  Plurals are hit too
(the memory says "doctor", the question says "doctors").

This script measures how much of the reported detector table that bug produces,
and what the same detector scores once the two channels are normalised alike.
"""
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_guards as G

DATA = "/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"
RES = "/root/agent_memory_iclr/results"
data = json.load(open(DATA))
qc = {r["qid"]: r for r in (json.loads(l) for l in open(f"{RES}/full500_qc.jsonl"))}


def stem(w):
    for s in ("ies", "ing", "es", "ed", "s"):
        if w.endswith(s) and len(w) - len(s) >= 4:
            return w[: len(w) - len(s)]
    return w


out = []
for q in data:
    hay = " ".join(G.norm(u[2]) for u in G.units_of(q))
    words = set(hay.split())
    stems = {stem(w) for w in words}
    proper, uni, _ = G.question_terms(q["question"])
    terms = list(dict.fromkeys(list(uni) + list(proper)))

    module_missing = [t for t in terms if t not in hay]            # exactly the module's test
    # artifact classes among the module's "missing" terms
    art_split, art_plural, real = [], [], []
    for t in module_missing:
        # did the original question token carry a hyphen or apostrophe?
        raw_hit = [w for w in re.findall(r"[A-Za-z0-9'\-]+", q["question"])
                   if re.sub(r"[^a-z0-9]", "", w.lower()) == t]
        hyph = any(("-" in w) or ("'" in w) or ("’" in w) for w in raw_hit)
        pieces = [p for p in re.split(r"[^a-z0-9]+", " ".join(raw_hit).lower()) if p]
        if hyph and all(p in words for p in pieces if len(p) > 1):
            art_split.append(t)
        elif stem(t) in stems:
            art_plural.append(t)
        else:
            real.append(t)

    # bug-fixed detector: split the question term the same way the memory is normalised,
    # require every piece (len>2) to be absent, and allow a morphological match
    fixed_missing = []
    for t in real:
        if stem(t) not in stems and t not in words:
            fixed_missing.append(t)

    out.append({"qid": q["question_id"], "is_abs": q["question_id"].endswith("_abs"),
                "type": q["question_type"], "judge": qc[q["question_id"]]["judge"],
                "advice": G.advice_shaped(q["question"]), "meta": G.meta_conversational(q["question"]),
                "module": module_missing, "art_split": art_split, "art_plural": art_plural,
                "fixed": fixed_missing})

A = [r for r in out if r["is_abs"]]
N = [r for r in out if not r["is_abs"]]
g = lambda r: (not r["advice"]) and (not r["meta"])


def rp(name, fn):
    ta = sum(1 for r in A if fn(r)); fa = sum(1 for r in N if fn(r))
    print(f"  {name:56s} abs {ta:2d}/30 ({ta/30:.3f})  ans {fa:3d}/470 ({fa/470:.3f})  "
          f"prec {ta/max(1,ta+fa):.3f}")


print("=== module detector vs the same detector with both channels normalised alike ===")
rp("gated D1|D2, module as written", lambda r: bool(r["module"]) and g(r))
rp("gated D1|D2, hyphen/possessive artifacts removed",
   lambda r: bool(r["art_plural"] or r["fixed"]) and g(r))
rp("gated D1|D2, artifacts + plurals removed (fair detector)",
   lambda r: bool(r["fixed"]) and g(r))
rp("ungated D1|D2, fair detector", lambda r: bool(r["fixed"]))

print("\n=== how much of the reported table is the bug? ===")
for label, subset in [("abs firings", [r for r in A if r["module"] and g(r)]),
                      ("answerable firings", [r for r in N if r["module"] and g(r)])]:
    nt = sum(len(r["module"]) for r in subset)
    ns = sum(len(r["art_split"]) for r in subset)
    npl = sum(len(r["art_plural"]) for r in subset)
    nr = sum(len(r["fixed"]) for r in subset)
    print(f"  {label:20s} {len(subset):3d} questions, {nt:3d} asserted-absent terms: "
          f"{ns} hyphen/possessive artifacts, {npl} plural-only, {nr} genuinely absent")
    dropped = [r["qid"] for r in subset if not r["fixed"]]
    print(f"      firings that vanish under the fair detector: {len(dropped)}")

print("\n=== fair detector: reach on the 5 qc abstention losses ===")
losses = {"80ec1f4f_abs", "a96c20ee_abs", "09ba9854_abs", "gpt4_93159ced_abs", "031748ae_abs"}
for r in out:
    if r["qid"] in losses:
        print(f"  {r['qid']:20s} module={r['module']} fair={r['fixed']}")

print("\n=== fair detector: upside/downside ===")
fa = [r for r in N if r["fixed"] and g(r)]
ta = [r for r in A if r["fixed"] and g(r)]
print(f"  abs firings {len(ta)}; on actual losses {[r['qid'] for r in ta if not r['judge']]}")
print(f"  answerable firings {len(fa)}: currently correct {sum(1 for r in fa if r['judge'])}, "
      f"wrong {sum(1 for r in fa if not r['judge'])}")
print("  by type:", Counter(r["type"] for r in fa))
print("  examples:", [(r["qid"], r["fixed"]) for r in fa[:12]])
print("\n=== the two SSP declines under the fair detector ===")
for r in out:
    if r["qid"] in ("0edc2aef", "09d032c9"):
        print(f"  {r['qid']:12s} module={r['module']} fair={r['fixed']} advice={r['advice']}")
json.dump(out, open(f"{RES}/_ver4_fixed.json", "w"))
