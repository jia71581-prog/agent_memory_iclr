"""Adversarial probes on the A4 claim: truthfulness of the absence statement,
statistical significance of each detector, token cost, and the ship decision's EV."""
import json
import math
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_guards as G

DATA = "/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"
RES = "/root/agent_memory_iclr/results"
data = json.load(open(DATA))
byid = {q["question_id"]: q for q in data}
rows = json.load(open(f"{RES}/_ver4_a4.json"))
qc = {r["qid"]: r for r in (json.loads(l) for l in open(f"{RES}/full500_qc.jsonl"))}


def fisher(a, b, c, d):
    """one-sided p (detector enriched in abs) for table [[a,b],[c,d]]"""
    def lc(n, k):
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    n = a + b + c + d
    r1, c1 = a + b, a + c
    p = 0.0
    for x in range(a, min(r1, c1) + 1):
        p += math.exp(lc(c1, x) + lc(n - c1, r1 - x) - lc(n, r1))
    return p


A = [r for r in rows if r["is_abs"]]
N = [r for r in rows if not r["is_abs"]]
print("=== 1. per-detector enrichment vs chance (one-sided Fisher) ===")
for name, fn in [("D1 proper", lambda r: r["d1"]), ("D2 content", lambda r: r["d2"]),
                 ("D3 bigram", lambda r: r["d3"]), ("D4 unlinked", lambda r: r["d4"]),
                 ("union", lambda r: r["union"]),
                 ("gated D1|D2", lambda r: r["d12"] and not r["advice"] and not r["meta"])]:
    ta = sum(1 for r in A if fn(r)); fa = sum(1 for r in N if fn(r))
    lift = (ta / len(A)) / max(1e-9, fa / len(N))
    print(f"  {name:12s} abs {ta:2d}/30  ans {fa:3d}/470  recall/FPR lift {lift:5.2f}x  "
          f"p={fisher(ta, len(A)-ta, fa, len(N)-fa):.4f}")

print("\n=== 2. is D1 a subset of D2 (why 'D2' == 'D1|D2')? ===")
bad = 0
for q in data:
    proper, uni, bi = G.question_terms(q["question"])
    hay = " ".join(G.norm(u[2]) for u in G.units_of(q))
    d1 = {t for t in proper if t not in hay}
    d2 = {t for t in uni if t not in hay}
    if d1 - d2:
        bad += 1
print(f"  questions with a missing proper noun absent from the missing-content set: {bad}/500")

print("\n=== 3. TRUTHFULNESS of the note: are 'never appears' terms morphologically present? ===")
SUF = ["s", "es", "ies", "ing", "ed", "er", "ers", "y", "ly", "d", "n", "e"]


def stem_present(t, hayset, hay):
    """morphological / near-miss neighbour of t already in memory"""
    if t in hay:
        return None
    cands = set()
    for s in SUF:
        if t.endswith(s) and len(t) - len(s) >= 4:
            cands.add(t[: len(t) - len(s)])
    if t.endswith("ies") and len(t) > 4:
        cands.add(t[:-3] + "y")
    for s in ("s", "es", "ing", "ed"):
        cands.add(t + s)
    for c in cands:
        if c and c in hay:
            return c
    # any memory word sharing a >=5-char prefix (dog/dogs, hotel/hotels, run/running)
    pre = t[:5]
    if len(t) >= 5:
        for w in hayset:
            if len(w) >= 5 and w[:5] == pre:
                return w
    return None


def report(subset, label):
    tot = miss = misleading = 0
    ex = []
    for r in subset:
        q = byid[r["qid"]]
        hay = " ".join(G.norm(u[2]) for u in G.units_of(q))
        hayset = set(hay.split())
        for t in r["terms"]:
            tot += 1
            nb = stem_present(t, hayset, hay)
            if nb:
                misleading += 1
                if len(ex) < 12:
                    ex.append((r["qid"], t, nb))
            else:
                miss += 1
    print(f"  {label}: {tot} asserted-absent terms; {misleading} have a morphological/prefix "
          f"neighbour IN memory ({misleading/max(1,tot):.0%}) -> the note's sentence is misleading there")
    for e in ex:
        print(f"      {e[0]:20s} note says '{e[1]}' never appears; memory contains '{e[2]}'")
    return tot, misleading


gated_ans = [r for r in N if r["d12"] and not r["advice"] and not r["meta"]]
gated_abs = [r for r in A if r["d12"]]
report(gated_abs, "abs firings (12)")
report(gated_ans, "answerable firings (42)")

print("\n=== 4. questions whose note is entirely spurious (every term has a neighbour) ===")
for label, subset in [("abs", gated_abs), ("answerable", gated_ans)]:
    n_all_spur = 0
    for r in subset:
        q = byid[r["qid"]]
        hay = " ".join(G.norm(u[2]) for u in G.units_of(q))
        hayset = set(hay.split())
        if r["terms"] and all(stem_present(t, hayset, hay) for t in r["terms"]):
            n_all_spur += 1
    print(f"  {label}: {n_all_spur}/{len(subset)} firings where EVERY asserted-absent term "
          f"has a neighbour in memory")

print("\n=== 5. token cost of the note (claim: ~25 prompt tokens) ===")
lens = []
for r in gated_abs + gated_ans:
    q = byid[r["qid"]]
    cert = G.absence_certificate(q["question"], G.units_of(q))
    lens.append(len(G.presupposition_note(cert)))
lens.sort()
print(f"  n={len(lens)} notes, chars min/median/max = {lens[0]}/{lens[len(lens)//2]}/{lens[-1]}")
print(f"  ~chars/4 tokens: median {lens[len(lens)//2]/4:.0f}, i.e. {lens[len(lens)//2]/4/25:.1f}x "
      f"the stated 25 tokens")

print("\n=== 6. upside/downside ledger of shipping the gated note ===")
help_abs = [r for r in gated_abs if r["judge"] is False]
noop_abs = [r for r in gated_abs if r["judge"] is True]
risk_ans = [r for r in gated_ans if r["judge"] is True]
wrong_ans = [r for r in gated_ans if r["judge"] is False]
print(f"  abs firings on questions ALREADY correct (no upside, possible no-op): {len(noop_abs)}")
print(f"  abs firings on actual losses (max upside):                          {len(help_abs)} "
      f"{[r['qid'] for r in help_abs]}")
print(f"  answerable firings on ALREADY-CORRECT answers (downside exposure):  {len(risk_ans)}")
print(f"  answerable firings already wrong (nothing to lose):                 {len(wrong_ans)}")
print(f"  of the {len(risk_ans)} at-risk correct answers, how many already decline? "
      f"{sum(1 for r in risk_ans if G.is_decline(r['pred']))} "
      f"-> the model currently does NOT treat term-absence as a reason to refuse")
be = len(help_abs) / max(1, len(risk_ans))
print(f"  break-even flip rate on the at-risk set: {be:.1%} "
      f"(above this the note is net-negative on judge accuracy over 500)")

print("\n=== 7. robustness: can ANY threshold of this family reach gate-worthy precision? ===")
for k in (1, 2, 3):
    for use in ("d12",):
        ta = sum(1 for r in A if len(r["terms"]) >= k and not r["advice"] and not r["meta"])
        fa = sum(1 for r in N if len(r["terms"]) >= k and not r["advice"] and not r["meta"])
        print(f"  gated D1|D2 with >={k} missing terms: abs {ta:2d}/30 ans {fa:3d}/470 "
              f"prec {ta/max(1,ta+fa):.3f}")
ta = sum(1 for r in A if r["d1"] and not r["advice"] and not r["meta"])
fa = sum(1 for r in N if r["d1"] and not r["advice"] and not r["meta"])
print(f"  gated D1 (proper nouns only):          abs {ta:2d}/30 ans {fa:3d}/470 prec {ta/max(1,ta+fa):.3f}")
ta = sum(1 for r in A if r["d12"] and r["d4"] and not r["advice"] and not r["meta"])
fa = sum(1 for r in N if r["d12"] and r["d4"] and not r["advice"] and not r["meta"])
print(f"  gated D1|D2 AND D4 (conjunction):      abs {ta:2d}/30 ans {fa:3d}/470 prec {ta/max(1,ta+fa):.3f}")
