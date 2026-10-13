"""Test 8: (a) 'highest assistant-number share' under a stricter numeral tokenizer;
(b) B7 relevance-ordered truncation counterfactual for 09ba9854 -- does the taxi line come back?"""
import json, re, sys
sys.path.insert(0, "/root/agent_memory_iclr/code")
from _zver_core import load, build_ctx
from run_baseline import bm25_rank

D = {q["question_id"]: q for q in load()}
absq = sorted(qid for qid in D if qid.endswith("_abs"))
TOKNUM = re.compile(r"\b\d+(?:[.,]\d+)*\b")   # standalone numeral tokens only
rows = []
ctxs = {}
for qid in absq:
    c = build_ctx(D[qid])
    ctxs[qid] = c
    nu = sum(len(TOKNUM.findall(x[2])) for x in c["raw_units"] if x[1] == "user")
    na = sum(len(TOKNUM.findall(x[2])) for x in c["raw_units"] if x[1] == "assistant")
    rows.append((qid, na, nu, na / max(1, na + nu)))
rows.sort(key=lambda r: (-r[3], -r[1]))
print("=== assistant numeral share, strict standalone-token count ===")
for i, r in enumerate(rows[:12], 1):
    m = "  <-- traced case" if r[0] == "09ba9854_abs" else ""
    print(" %2d. %-24s asstNum=%4d userNum=%3d share=%.4f%s" % (i, r[0], r[1], r[2], r[3], m))
rk = [r[0] for r in rows].index("09ba9854_abs") + 1
print("09ba9854 rank = %d of %d ; questions with ZERO user numerals (share 1.000) = %d"
      % (rk, len(rows), sum(1 for r in rows if r[2] == 0)))

print("\n=== B7 counterfactual on 09ba9854: relevance-ordered 20k truncation ===")
q = D["09ba9854_abs"]
c = ctxs["09ba9854_abs"]
lines = c["view_full"].splitlines()
order = bm25_rank(q["question"], lines)
kept, tot, keptset = [], 0, set()
for i in order:
    if tot + len(lines[i]) + 1 > 20000:
        continue
    kept.append(i); keptset.add(i); tot += len(lines[i]) + 1
NARROW = re.compile(r"\bbus\b|\btaxi\b|\bfare\b|limousine|\bcab\b", re.I)
alpha_keep = set()
tot = 0
for i, l in enumerate(lines):
    if tot + len(l) + 1 > 20000:
        break
    alpha_keep.add(i); tot += len(l) + 1
print("alphabetical cap keeps %d/%d lines; relevance cap keeps %d/%d" % (len(alpha_keep), len(lines), len(keptset), len(lines)))
for i, l in enumerate(lines):
    if NARROW.search(l) or "60" in l and "taxi" in l.lower():
        print("  [alpha=%s relevance=%s] %s" % ("KEEP" if i in alpha_keep else "CUT ",
                                                "KEEP" if i in keptset else "CUT ", l[:150]))
tax = [i for i, l in enumerate(lines) if "taxi" in l.lower()]
print("\ntaxi lines: %d ; kept by alphabetical cap: %d ; kept by relevance cap: %d"
      % (len(tax), sum(1 for i in tax if i in alpha_keep), sum(1 for i in tax if i in keptset)))
