"""Test 6: is 09ba9854 really the extreme assistant-number case among the 30 abs questions,
and does the abs-question raw channel generally look assistant-dominated?"""
import json, re, sys, collections
sys.path.insert(0, "/root/agent_memory_iclr/code")
from _zver_core import load, build_ctx, MISS

D = {q["question_id"]: q for q in load()}
absq = sorted(qid for qid in D if qid.endswith("_abs"))
NUM = re.compile(r"\d[\d,\.]*")
rows = []
for qid in absq:
    c = build_ctx(D[qid])
    u = [x for x in c["raw_units"] if x[1] == "user"]
    a = [x for x in c["raw_units"] if x[1] == "assistant"]
    cu, ca = sum(len(x[2]) for x in u), sum(len(x[2]) for x in a)
    nu = sum(len(NUM.findall(x[2])) for x in u)
    na = sum(len(NUM.findall(x[2])) for x in a)
    rows.append((qid, len(a), len(u), ca / max(1, ca + cu), na, nu, na / max(1, na + nu)))
rows.sort(key=lambda r: -r[6])
print("%-24s %4s %4s %8s %5s %5s %8s" % ("qid", "n_a", "n_u", "charShr", "numA", "numU", "numShr"))
for r in rows:
    mark = " <-- traced case" if r[0] == "09ba9854_abs" else ""
    print("%-24s %4d %4d %8.3f %5d %5d %8.3f%s" % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], mark))
print("\ncache misses:", len(MISS))
rank = [r[0] for r in rows].index("09ba9854_abs") + 1
print("09ba9854 rank by assistant-number share: %d of %d" % (rank, len(rows)))
print("median assistant char share over 30 abs questions: %.3f" % sorted(r[3] for r in rows)[15])
print("questions with assistant number share >= 0.90: %d" % sum(1 for r in rows if r[6] >= 0.90))
