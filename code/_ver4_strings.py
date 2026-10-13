"""Does the low precision survive a better string test? Stem/lemma-tolerant and
word-boundary variants of D1|D2, plus a per-question cache so this is fast."""
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
    proper, uni, bi = G.question_terms(q["question"])
    terms = list(dict.fromkeys(list(uni) + list(proper)))
    sub_missing = [t for t in terms if t not in hay]                       # module's test
    wb_missing = [t for t in terms if t not in words]                      # word-boundary
    stem_missing = [t for t in terms if stem(t) not in stems and t not in words]
    out.append({"qid": q["question_id"], "is_abs": q["question_id"].endswith("_abs"),
                "advice": G.advice_shaped(q["question"]), "meta": G.meta_conversational(q["question"]),
                "sub": bool(sub_missing), "wb": bool(wb_missing), "stem": bool(stem_missing),
                "sub_terms": sub_missing, "stem_terms": stem_missing,
                "judge": qc[q["question_id"]]["judge"], "type": q["question_type"]})

A = [r for r in out if r["is_abs"]]
N = [r for r in out if not r["is_abs"]]


def rp(name, fn):
    ta = sum(1 for r in A if fn(r)); fa = sum(1 for r in N if fn(r))
    print(f"  {name:52s} abs {ta:2d}/30 ({ta/30:.3f})  ans {fa:3d}/470 ({fa/470:.3f})  "
          f"prec {ta/max(1,ta+fa):.3f}")


print("=== string-test variants of the D1|D2 detector, all router-gated ===")
g = lambda r: (not r["advice"]) and (not r["meta"])
rp("substring (module's own test)", lambda r: r["sub"] and g(r))
rp("word-boundary (stricter: any exact token absent)", lambda r: r["wb"] and g(r))
rp("stem-tolerant (fairer: no morphological match)", lambda r: r["stem"] and g(r))
print()
print("=== ungated, for reference ===")
rp("substring ungated", lambda r: r["sub"])
rp("word-boundary ungated", lambda r: r["wb"])
rp("stem-tolerant ungated", lambda r: r["stem"])

print("\n=== stem-tolerant gated detector: reach on the 5 qc abstention losses ===")
losses = {"80ec1f4f_abs", "a96c20ee_abs", "09ba9854_abs", "gpt4_93159ced_abs", "031748ae_abs"}
for r in out:
    if r["qid"] in losses:
        print(f"  {r['qid']:20s} sub={r['sub']!s:5s} stem={r['stem']!s:5s} "
              f"sub_terms={r['sub_terms']} stem_terms={r['stem_terms']}")

print("\n=== stem-tolerant gated: upside/downside ledger ===")
fa = [r for r in N if r["stem"] and g(r)]
ta = [r for r in A if r["stem"] and g(r)]
print(f"  abs firings {len(ta)} (on actual losses: {[r['qid'] for r in ta if not r['judge']]})")
print(f"  answerable firings {len(fa)}: currently correct {sum(1 for r in fa if r['judge'])}, "
      f"wrong {sum(1 for r in fa if not r['judge'])}")
print("  by type:", Counter(r["type"] for r in fa))
json.dump(out, open(f"{RES}/_ver4_strings.json", "w"))
