"""D. 弃答题词面孪生检验:existence/abstention 能否作为 shape 由规则判出?
对每道弃答题,在 470 道可答题中找 token-Jaccard 最近邻,报告 shape 是否相同。"""
import json
import re
import sys
from collections import Counter

sys.path.insert(0, "/root/agent_memory_iclr/code")
from v5_shape import classify_shape

R = "/root/agent_memory_iclr/results"
ROWS = json.load(open(f"{R}/_v5_shape_rows.json"))
W = re.compile(r"[a-z0-9']+")
STOP = {"i", "my", "the", "a", "an", "do", "did", "was", "is", "in", "of", "to", "have", "how"}


def toks(s):
    return {t for t in W.findall(s.lower()) if t not in STOP}


AB = [r for r in ROWS if r["is_abs"]]
NA = [r for r in ROWS if not r["is_abs"]]
same_shape = 0
print(f"{'shape(abs)':18s} {'J':>4s} {'shape(twin)':18s} same  abs-question / nearest answerable twin")
for a in AB:
    ta = toks(a["q"])
    best, bj = None, -1.0
    for n in NA:
        tn = toks(n["q"])
        j = len(ta & tn) / max(1, len(ta | tn))
        if j > bj:
            bj, best = j, n
    ok = a["shape"] == best["shape"]
    same_shape += ok
    print(f"{a['shape']:18s} {bj:4.2f} {best['shape']:18s} {'Y' if ok else 'n':4s}  {a['q'][:62]}\n{'':49s}| {best['q'][:62]}")
print(f"\n最近邻 shape 相同: {same_shape}/{len(AB)} = {same_shape/len(AB):.1%}")
js = []
for a in AB:
    ta = toks(a["q"])
    js.append(max(len(ta & toks(n["q"])) / max(1, len(ta | toks(n["q"]))) for n in NA))
js.sort()
print(f"最近邻 Jaccard: median={js[len(js)//2]:.2f} min={js[0]:.2f} max={js[-1]:.2f}  >=0.5 的题数={sum(1 for j in js if j>=0.5)}/30")

# PREF gold 串的可检验性(gold 是 rubric 而非事实串 → in_hay 类指标失效)
pref = [r for r in ROWS if r["shape"] == "PREF.advice"]
rub = sum(1 for r in pref if r["gold"].lower().startswith(("the user would prefer", "considering")))
print(f"\nPREF.advice: n={len(pref)}, gold 是 rubric 句式的 {rub}/{len(pref)};in_hay=True 的 {sum(r['in_hay'] for r in pref)}/{len(pref)}")
print("→ 该 shape 上 gold-string coverage 指标恒为假,错误归因表中的 'evid_absent' 是度量伪影,不是检索缺失")
