"""v5 shape 分类器的离线测量(无 API):分布 / 交叉表 / 错误分区 / 与旧时态轴对比。"""
import json
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/root/agent_memory_iclr/code")
from v5_shape import classify_shape, FAMILY, SHAPES
from query_classifier import classify_query

R = "/root/agent_memory_iclr/results"
QS = json.load(open(f"{R}/_questions500.json"))
QC = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{R}/full500_qc.jsonl")}
CD = {r["qid"]: r for r in json.load(open(f"{R}/coverage_diag.json"))}

TSHORT = {"single-session-user": "SSU", "single-session-assistant": "SSA",
          "single-session-preference": "SSP", "multi-session": "MS",
          "knowledge-update": "KU", "temporal-reasoning": "TR"}

rows = []
for q in QS:
    shape, fired = classify_shape(q["question"])
    tense, tfired = classify_query(q["question"])
    r = QC[q["qid"]]
    rows.append({"qid": q["qid"], "q": q["question"], "type": TSHORT[q["type"]],
                 "is_abs": q["qid"].endswith("_abs"), "shape": shape,
                 "family": FAMILY[shape], "fired": fired, "tense": tense,
                 "judge": r["judge"], "gold": q["answer"],
                 "in_view": CD[q["qid"]]["in_view"], "in_raw20": CD[q["qid"]]["in_raw20"],
                 "in_hay": CD[q["qid"]]["in_hay"]})
json.dump(rows, open(f"{R}/_v5_shape_rows.json", "w"), ensure_ascii=False)

NA = [r for r in rows if not r["is_abs"]]
AB = [r for r in rows if r["is_abs"]]
ERR = [r for r in NA if not r["judge"]]
print(f"n=500  non-abstention={len(NA)}  abstention={len(AB)}  non-abs errors={len(ERR)}")

print("\n=== 1a. shape 分布 + 每 shape 准确率(非弃答)===")
print(f"{'shape':18s} {'n500':>5s} {'nNA':>4s} {'acc':>6s} {'err':>4s} {'err%ofall':>9s}")
order = [s for s in SHAPES if any(r["shape"] == s for r in rows)]
for s in order:
    g = [r for r in NA if r["shape"] == s]
    n5 = sum(1 for r in rows if r["shape"] == s)
    e = sum(1 for r in g if not r["judge"])
    acc = (len(g) - e) / len(g) if g else float("nan")
    print(f"{s:18s} {n5:5d} {len(g):4d} {acc:6.3f} {e:4d} {e/len(ERR):9.1%}")
print(f"{'-- family --':18s}")
for fam in ["AGGREGATE", "TEMPORAL", "LOOKUP", "RECALL", "PREFERENCE"]:
    g = [r for r in NA if r["family"] == fam]
    n5 = sum(1 for r in rows if r["family"] == fam)
    e = sum(1 for r in g if not r["judge"])
    print(f"{fam:18s} {n5:5d} {len(g):4d} {(len(g)-e)/len(g):6.3f} {e:4d} {e/len(ERR):9.1%}")

print("\n=== 1b. shape x gold type(全 500)===")
types = ["SSU", "SSA", "SSP", "MS", "KU", "TR"]
print(f"{'shape':18s} " + " ".join(f"{t:>5s}" for t in types) + "   tot")
for s in order:
    c = Counter(r["type"] for r in rows if r["shape"] == s)
    print(f"{s:18s} " + " ".join(f"{c.get(t,0):5d}" for t in types) + f"   {sum(c.values()):4d}")
print(f"{'TOTAL':18s} " + " ".join(f"{sum(1 for r in rows if r['type']==t):5d}" for t in types))

print("\n=== 1c. 每 type 的 shape 纯度(最大 shape 占比)===")
for t in types:
    g = [r for r in rows if r["type"] == t]
    c = Counter(r["shape"] for r in g)
    top, n = c.most_common(1)[0]
    print(f"{t:4s} n={len(g):3d}  dominant={top:18s} {n/len(g):5.1%}  #shapes={len(c)}")

print("\n=== 2. 与旧时态轴的表面积对比 ===")
tc = Counter(r["tense"] for r in rows)
old_nondefault = tc["historical"] + tc["current"]
print(f"旧时态轴: historical={tc['historical']} current={tc['current']} neutral={tc['neutral']}")
print(f"  非默认决策 = {old_nondefault}/500 ({old_nondefault/500:.1%});真正 historical 渲染 = {tc['historical']}/500")
terr = Counter(r["tense"] for r in ERR)
print(f"  62 错题分区: " + " ".join(f"{k}={v}" for k, v in terr.items()))
print(f"  → 旧轴把 {terr['historical']}/{len(ERR)} 错题划入非默认分支")
new_nondefault = sum(1 for r in rows if r["shape"] != "LOOKUP.plain")
print(f"新 shape 轴: 非默认(≠LOOKUP.plain) = {new_nondefault}/500 ({new_nondefault/500:.1%})")
print(f"  错题非默认占比 = {sum(1 for r in ERR if r['shape']!='LOOKUP.plain')}/{len(ERR)}")
print("  错题按 shape 分区(降序):")
for s, n in Counter(r["shape"] for r in ERR).most_common():
    g = [r for r in NA if r["shape"] == s]
    print(f"    {s:18s} err={n:2d}  ({n/len(ERR):5.1%} of errors)  shape acc={1-n/len(g):.3f}  n={len(g)}")

print("\n=== 2b. 错误分区的集中度(Herfindahl / top-k 覆盖)===")
cnt = sorted(Counter(r["shape"] for r in ERR).values(), reverse=True)
print(f"  top-1 shape 覆盖 {cnt[0]/len(ERR):.1%}, top-2 {sum(cnt[:2])/len(ERR):.1%}, top-3 {sum(cnt[:3])/len(ERR):.1%}")
tcnt = sorted(Counter(r["tense"] for r in ERR).values(), reverse=True)
print(f"  旧轴 top-1 {tcnt[0]/len(ERR):.1%}(=neutral 默认桶,不构成条件化)")

print("\n=== 2c. 证据在场/缺失 x shape(错题)===")
print(f"{'shape':18s} {'err':>4s} {'evid_present':>12s} {'evid_absent':>11s}")
for s, n in Counter(r["shape"] for r in ERR).most_common():
    g = [r for r in ERR if r["shape"] == s]
    p = sum(1 for r in g if r["in_view"] or r["in_raw20"])
    print(f"{s:18s} {n:4d} {p:12d} {n-p:11d}")

print("\n=== 3. 弃答题的 shape 分布(检验 existence shape 是否可判)===")
for s, n in Counter(r["shape"] for r in AB).most_common():
    na = [r for r in NA if r["shape"] == s]
    print(f"  {s:18s} abs={n:2d}   同 shape 非弃答题={len(na)}  → 该 shape 内弃答占比 {n/(n+len(na)):.1%}")
