"""A/B/C 的汇总报表(读 _v5_offline_val.json,无 API)。"""
import json
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/root/agent_memory_iclr/code")
from v5_shape import resolve_relative_dates, parse_qdate, days_between, SHAPES

R = "/root/agent_memory_iclr/results"
V = json.load(open(f"{R}/_v5_offline_val.json"))
NA = [r for r in V if not r["is_abs"]]

print("=== A. 完整证据集召回(非弃答 470;同 20 轮预算)===")
print(f"{'shape':18s} {'n':>4s} {'med_ev':>6s} | {'flat_any':>8s} {'flat_ALL':>8s} | {'cov_any':>7s} {'cov_ALL':>7s} | {'dALL':>6s}")
order = [s for s in SHAPES if any(r['shape'] == s for r in NA)]
for s in order + ["__ALL__"]:
    g = NA if s == "__ALL__" else [r for r in NA if r["shape"] == s]
    if not g:
        continue
    n = len(g)
    med = sorted(r["n_ev"] for r in g)[n // 2]
    fa, fl = sum(r["flat_any"] for r in g) / n, sum(r["flat_all"] for r in g) / n
    ca, cl = sum(r["cov_any"] for r in g) / n, sum(r["cov_all"] for r in g) / n
    print(f"{s:18s} {n:4d} {med:6d} | {fa:8.3f} {fl:8.3f} | {ca:7.3f} {cl:7.3f} | {cl-fl:+6.3f}")

print("\n=== A2. 多证据题(n_ev>=2)上的完整召回 ===")
m = [r for r in NA if r["n_ev"] >= 2]
print(f"n={len(m)}  flat_ALL={sum(r['flat_all'] for r in m)/len(m):.3f}  cov_ALL={sum(r['cov_all'] for r in m)/len(m):.3f}")
for s in order:
    g = [r for r in m if r["shape"] == s]
    if len(g) < 5:
        continue
    print(f"  {s:18s} n={len(g):3d} flat_ALL={sum(r['flat_all'] for r in g)/len(g):.3f} cov_ALL={sum(r['cov_all'] for r in g)/len(g):.3f} acc={sum(r['judge'] for r in g)/len(g):.3f}")

print("\n=== A3. 完整召回 与 正确率 的关联(多证据非弃答题)===")
for lab, sub in [("flat_ALL=True", [r for r in m if r["flat_all"]]), ("flat_ALL=False", [r for r in m if not r["flat_all"]])]:
    print(f"  {lab:15s} n={len(sub):3d} acc={sum(r['judge'] for r in sub)/max(1,len(sub)):.3f}")
agg = [r for r in m if r["shape"].startswith("AGG")]
for lab, sub in [("AGG* flat_ALL=T", [r for r in agg if r["flat_all"]]), ("AGG* flat_ALL=F", [r for r in agg if not r["flat_all"]])]:
    print(f"  {lab:15s} n={len(sub):3d} acc={sum(r['judge'] for r in sub)/max(1,len(sub)):.3f}")

print("\n=== B. TEMP.offset:gold 数值能否由纯日期算术导出 ===")
NUM = re.compile(r"(\d+(?:\.\d+)?)")
off = [r for r in NA if r["shape"] == "TEMP.offset"]
elig = [r for r in off if NUM.search(r["gold"]) and r["ev_dates"]]
print(f"TEMP.offset n={len(off)},gold 含数字且有证据日期 n={len(elig)}")
hit_d = hit_w = hit_m = 0
rowsB = []
for r in elig:
    golds = {float(x) for x in NUM.findall(r["gold"])}
    qd = parse_qdate(r["qdate"])
    evs = sorted(parse_qdate(d) for d in r["ev_dates"])
    cands = {}
    cands["ago_days"] = abs((qd - evs[-1]).days)
    cands["ago_days_first"] = abs((qd - evs[0]).days)
    if len(evs) >= 2:
        cands["span_days"] = abs((evs[-1] - evs[0]).days)
    ok = {}
    for k, v in cands.items():
        ok[k] = any(abs(v - g) <= 1 for g in golds) or any(abs(v / 7 - g) <= 0.5 for g in golds) or any(abs(v / 30 - g) <= 0.5 for g in golds)
    rowsB.append((r, cands, ok))
    hit_d += ok.get("ago_days", False) or ok.get("ago_days_first", False)
    hit_w += ok.get("span_days", False)
    hit_m += any(ok.values())
print(f"  gold 可由 |qdate - 证据会话日期| 导出(±1 天 / ±0.5 周 / ±0.5 月): {hit_d}/{len(elig)} = {hit_d/len(elig):.1%}")
print(f"  gold 可由 |两证据会话日期之差| 导出: {hit_w}/{len(elig)}")
print(f"  两者任一命中: {hit_m}/{len(elig)} = {hit_m/len(elig):.1%}")
bad = [(r, c, o) for r, c, o in rowsB if not any(o.values())]
print(f"  不可导出 {len(bad)} 例(需要会话内的细粒度日期,而非会话日期):")
for r, c, o in bad[:8]:
    print(f"    gold={r['gold'][:34]:34s} cands={c} judge={'Y' if r['judge'] else 'N'} | {r['question'][:66]}")
wrong = [(r, c, o) for r, c, o in rowsB if any(o.values()) and not r["judge"]]
print(f"  可导出但当前系统答错: {len(wrong)}(= 渲染期预算术能救的上限)")
for r, c, o in wrong:
    print(f"    gold={r['gold'][:30]:30s} cands={c} | {r['question'][:70]}")

print("\n=== C. TEMP.anchor:相对日期解析命中证据会话日期 ===")
anc = [r for r in NA if r["shape"] == "TEMP.anchor"]
res_n = hit = 0
miss = []
for r in anc:
    rel = resolve_relative_dates(r["question"], r["qdate"])
    if not rel:
        continue
    res_n += 1
    evs = [parse_qdate(d) for d in r["ev_dates"]]
    ok = any(abs((e - d).days) <= tol for _, d, tol in rel for e in evs)
    hit += ok
    if not ok:
        miss.append((r, [(x[0], x[1].strftime('%m/%d'), x[2]) for x in rel], [e.strftime('%m/%d') for e in evs]))
print(f"TEMP.anchor n={len(anc)};其中相对日期可解析 {res_n};解析日期落在证据会话日期容差内 {hit}/{res_n} = {hit/max(1,res_n):.1%}")
print(f"  解析失败(无可解析表达)= {len(anc)-res_n}")
for r, rel, evs in miss:
    print(f"    MISS resolved={rel} ev={evs} judge={'Y' if r['judge'] else 'N'} | {r['question'][:60]}")
anc_err = [r for r in anc if not r["judge"]]
print(f"  TEMP.anchor 错题 {len(anc_err)},其中解析命中的 {sum(1 for r in anc_err if any(abs((parse_qdate(d)-dd).days)<=tol for _,dd,tol in resolve_relative_dates(r['question'],r['qdate']) for d in r['ev_dates']) if resolve_relative_dates(r['question'],r['qdate']))}")
