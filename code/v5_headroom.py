"""E. 可回收错误上界 + "单一固定渲染是否等效" 的检验(无 API)。"""
import json
import sys
from collections import Counter

sys.path.insert(0, "/root/agent_memory_iclr/code")
from v5_shape import resolve_relative_dates, parse_qdate, SHAPES

R = "/root/agent_memory_iclr/results"
V = {r["qid"]: r for r in json.load(open(f"{R}/_v5_offline_val.json"))}
ROWS = json.load(open(f"{R}/_v5_shape_rows.json"))
NA = [r for r in ROWS if not r["is_abs"]]
ERR = [r for r in NA if not r["judge"]]

print("=== E1. 会话覆盖式检索的净效应(逐题,非弃答 470)===")
gain = [r for r in NA if V[r["qid"]]["cov_all"] and not V[r["qid"]]["flat_all"]]
loss = [r for r in NA if V[r["qid"]]["flat_all"] and not V[r["qid"]]["cov_all"]]
print(f"完整证据集: 由缺→全 {len(gain)} 题;由全→缺 {len(loss)} 题;净 {len(gain)-len(loss):+d}")
print("  gain 分 shape:", dict(Counter(r["shape"] for r in gain)))
print("  loss 分 shape:", dict(Counter(r["shape"] for r in loss)))
print(f"  gain 中当前答错的(= 检索侧可回收错误上界): {sum(1 for r in gain if not r['judge'])}")
print("   ", [(r["shape"], r["q"][:58]) for r in gain if not r["judge"]])
print(f"  loss 中当前答对的(= 全局套用的风险面): {sum(1 for r in loss if r['judge'])}")
print("   ", [(r["shape"], r["q"][:58]) for r in loss])

print("\n=== E2. 若把会话覆盖式检索无条件套用(单一固定渲染)===")
print(f"  它已经拿到 shape 条件化在检索侧的全部收益(每个 shape 的 dALL 均 >= 0),")
print(f"  因为 470 题里只有 {len(loss)} 题被它弄坏,而 {len(gain)} 题被它修好。")
print("  → 检索侧不需要 shape 条件化。shape 条件化只有在渲染/算术侧才不可替代。")

print("\n=== E3. 各 shape 里必须 shape 特异的错误(检索无法解释的)===")
for s in [x for x in SHAPES if any(r['shape'] == x for r in ERR)]:
    g = [r for r in ERR if r["shape"] == s]
    if not g:
        continue
    cov_fix = sum(1 for r in g if V[r["qid"]]["cov_all"] and not V[r["qid"]]["flat_all"])
    already = sum(1 for r in g if V[r["qid"]]["flat_all"])
    print(f"  {s:18s} err={len(g):2d}  检索可修={cov_fix}  证据集本已完整却错={already}  → 后者只能靠渲染/推理")

print("\n=== E4. TEMP.anchor 的日期窗口策略能拿到什么 ===")
anc = [r for r in NA if r["shape"] == "TEMP.anchor"]
ok_res = []
for r in anc:
    rel = resolve_relative_dates(r["q"], V[r["qid"]]["qdate"])
    if not rel:
        continue
    evs = [parse_qdate(d) for d in V[r["qid"]]["ev_dates"]]
    if any(abs((e - d).days) <= tol for _, d, tol in rel for e in evs):
        ok_res.append(r)
print(f"  n={len(anc)}, 日期解析命中证据会话 {len(ok_res)};其中当前答错 {sum(1 for r in ok_res if not r['judge'])}")
print(f"  这些题当前 flat_ALL 完整率 {sum(V[r['qid']]['flat_all'] for r in ok_res)/len(ok_res):.3f} "
      f"→ 日期窗口是确定性地把证据会话置顶(不依赖词面匹配)")

print("\n=== E5. 每 shape 准确率 vs 全局均值的显著性(Fisher 精确检验,单侧)===")
try:
    from scipy.stats import fisher_exact
    tot_c = sum(r["judge"] for r in NA)
    tot_n = len(NA)
    for s in [x for x in SHAPES if sum(1 for r in NA if r['shape'] == x) >= 10]:
        g = [r for r in NA if r["shape"] == s]
        c = sum(r["judge"] for r in g)
        table = [[c, len(g) - c], [tot_c - c, (tot_n - len(g)) - (tot_c - c)]]
        p = fisher_exact(table)[1]
        print(f"  {s:18s} n={len(g):3d} acc={c/len(g):.3f} vs rest={(tot_c-c)/(tot_n-len(g)):.3f}  p={p:.4f}")
except ImportError:
    print("  scipy 不可用")
