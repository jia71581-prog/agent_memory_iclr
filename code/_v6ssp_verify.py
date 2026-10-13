"""QC_V6_SSP 验证(无 API;lifecycle 全缓存命中):
 1. 关旗 → 上下文与 v5 逐字节相同
 2. 门外(非建议型问句)→ 上下文逐字节相同,char delta 恒 0
 3. 门内 → 字符预算 <= v5(by construction),报实测 delta
 4. 败例上确认发生了预期的上下文变化:视图行数、命中轮所在会话的深度、助手轮数、
    上下文集中度(最大会话占比)
gold 位置只用于诊断输出,不进入方法。
"""
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_KEEP = {k: v for k, v in os.environ.items() if k.startswith("QC_V6_SSP_")}

DATA = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
LOSS = {"32260d93", "6b7dfb22", "57f827a0", "1da05512", "b6025781", "1c0ddc50"}
BOTH = {"0edc2aef", "195a1a1b", "09d032c9", "95228167", "75f70248", "d6233ab6"}


def build_ctx(mod, q):
    """复刻 answer_question 的上下文组装(不调 LLM,不走 v6-AGG/INF 分支)。"""
    from lifecycle import build_lifecycle, render_view
    from query_classifier import classify_query
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    store = build_lifecycle(q, None)
    view_txt = (mod._rank_view_lines(store, mode, q["question"], mod.VIEW_CHAR_CAP)
                if mod.V5 else render_view(store, mode))
    units, sel = mod._select_turns_v5(q, q["question"], mod.TOP_K)
    raw_txt = mod._v6ssp_render_units(units, sel)
    gated = False
    if getattr(mod, "V6SSP", False) and mod.V5 and mod.ABLATE is None and mod._v6ssp_is_advice(q["question"]):
        gated = True
        budget = len(view_txt) + len(raw_txt)
        view_txt = mod._rank_view_lines(store, mode, q["question"], mod.V6SSP_VIEW_CAP)
        raw_txt = mod._v6ssp_expand_turns(q["question"], units, sel, len(view_txt), budget)
    parts = []
    if view_txt:
        head = ("Current-state fact view (latest value per fact; [CONFLICT] marks values reported "
                "at the same time that could not be ordered)" if mode == "current" else
                "Fact evolution view (full value history per fact, oldest -> newest)")
        parts.append(f"### {head}\n{view_txt}")
    if raw_txt:
        parts.append(f"### Retrieved raw conversation turns\n{raw_txt}")
    return "\n\n".join(parts), dict(gated=gated, view=len(view_txt), raw=len(raw_txt),
                                    raw_txt=raw_txt, units=units, sel=sel, mode=mode)


def load(v5="1", v6=None, **kw):
    for k in list(os.environ):
        if k.startswith("QC_"):
            del os.environ[k]
    os.environ["QC_V5"] = v5
    if v6:
        os.environ["QC_V6_SSP"] = v6
    for k, v in kw.items():
        os.environ[k] = str(v)
    for k in ("QC_V6_SSP_NEIGHBORS", "QC_V6_SSP_VIEW_CAP"):
        if k in _KEEP:
            os.environ[k] = _KEEP[k]
    sys.modules.pop("run_qc", None)
    import run_qc
    return importlib.reload(run_qc)


def turn_idx(mod, units, raw_txt):
    """从渲染文本反查 unit 下标。助手轮正文含换行,不能按行匹配,故用整段包含判定。"""
    return [i for i, u in enumerate(units) if f"[{u[0]}] [{u[1]}] {u[2]}" in raw_txt]


def stats(units, idxs, gold_sess):
    by = {}
    for i in idxs:
        by.setdefault(units[i][3], []).append(i)
    top = max((len(v) for v in by.values()), default=0)
    chars = {s: sum(len(units[i][2]) for i in v) for s, v in by.items()}
    tot = sum(chars.values()) or 1
    return dict(n=len(idxs), n_sess=len(by), max_sess_turns=top,
                top_sess_share=round(max(chars.values(), default=0) / tot, 3),
                asst=sum(1 for i in idxs if units[i][1] == "assistant"),
                gold_turns=sum(1 for i in idxs if units[i][3] in gold_sess),
                gold_asst=sum(1 for i in idxs if units[i][3] in gold_sess and units[i][1] == "assistant"))


def main():
    m5 = load(v5="1")
    assert not m5.V6SSP, "flag off 时 V6SSP 必须 False"
    m6 = load(v5="1", v6="1")
    assert m6.V6SSP

    import collections
    adv = [q for q in DATA if m6._v6ssp_is_advice(q["question"])]
    print("gate fires on", len(adv), "of", len(DATA), ":",
          dict(collections.Counter(q["question_type"] for q in adv)))

    ssp = [q for q in DATA if q["question_type"] == "single-session-preference"]
    off, seen = [], collections.Counter()
    for q in DATA:
        if m6._v6ssp_is_advice(q["question"]):
            continue
        if seen[q["question_type"]] < 10:
            seen[q["question_type"]] += 1
            off.append(q)

    print("\n--- 门外一致性(应逐字节相同)---")
    bad = 0
    for q in off:
        c5, _ = build_ctx(m5, q)
        c6, i6 = build_ctx(m6, q)
        if c5 != c6 or i6["gated"]:
            bad += 1
            print("  MISMATCH", q["question_id"], q["question_type"])
    print(f"  {len(off)-bad}/{len(off)} identical (每型 10 题抽样), delta = 0 chars")

    print("\n--- 门内(建议型;SSP 30 题全跑)---")
    rows = []
    for q in ssp:
        qid = q["question_id"]
        hs = q.get("haystack_session_ids") or []
        gs = {hs.index(i) for i in (q.get("answer_session_ids") or []) if i in hs}
        c5, i5 = build_ctx(m5, q)
        c6, i6 = build_ctx(m6, q)
        u = i6["units"]
        s5 = stats(u, turn_idx(m6, u, i5["raw_txt"]), gs)
        s6 = stats(u, turn_idx(m6, u, i6["raw_txt"]), gs)
        r = {"qid": qid, "gated": i6["gated"], "v5_chars": len(c5), "v6_chars": len(c6),
             "delta_pct": round(100 * (len(c6) - len(c5)) / len(c5), 1),
             "v5_view": i5["view"], "v6_view": i6["view"],
             "v5": s5, "v6": s6, "gold_sess": sorted(gs)}
        rows.append(r)
        tag = "LOSS" if qid in LOSS else ("BOTH" if qid in BOTH else "ok  ")
        print(f"  {tag} {qid} g={i6['gated']:d} {len(c5):6d}->{len(c6):6d} ({r['delta_pct']:+5.1f}%) "
              f"view {i5['view']:5d}->{i6['view']:5d} | turns {s5['n']:2d}->{s6['n']:2d} "
              f"sess {s5['n_sess']:2d}->{s6['n_sess']:2d} deepest {s5['max_sess_turns']:2d}->{s6['max_sess_turns']:2d} "
              f"asst {s5['asst']:2d}->{s6['asst']:2d} | goldsess turns {s5['gold_turns']}->{s6['gold_turns']} "
              f"asst {s5['gold_asst']}->{s6['gold_asst']}")

    n = len(rows)
    a = lambda f: sum(f(r) for r in rows) / n
    print(f"\n  gated {sum(r['gated'] for r in rows)}/{n}")
    print(f"  chars  v5 {a(lambda r: r['v5_chars']):.0f} -> v6 {a(lambda r: r['v6_chars']):.0f} "
          f"({100*(sum(r['v6_chars'] for r in rows)-sum(r['v5_chars'] for r in rows))/sum(r['v5_chars'] for r in rows):+.1f}%), "
          f"max single-question delta {max(r['delta_pct'] for r in rows):+.1f}% (硬约束 <= 0)")
    print(f"  view   {a(lambda r: r['v5_view']):.0f} -> {a(lambda r: r['v6_view']):.0f} chars")
    print(f"  turns  {a(lambda r: r['v5']['n']):.1f} -> {a(lambda r: r['v6']['n']):.1f}; "
          f"sessions {a(lambda r: r['v5']['n_sess']):.1f} -> {a(lambda r: r['v6']['n_sess']):.1f}; "
          f"deepest session {a(lambda r: r['v5']['max_sess_turns']):.1f} -> {a(lambda r: r['v6']['max_sess_turns']):.1f} turns; "
          f"assistant turns {a(lambda r: r['v5']['asst']):.1f} -> {a(lambda r: r['v6']['asst']):.1f}")
    print(f"  top-session char share {a(lambda r: r['v5']['top_sess_share']):.3f} -> {a(lambda r: r['v6']['top_sess_share']):.3f}")
    print(f"  [诊断] gold 会话轮 {a(lambda r: r['v5']['gold_turns']):.2f} -> {a(lambda r: r['v6']['gold_turns']):.2f}; "
          f"其中助手轮 {a(lambda r: r['v5']['gold_asst']):.2f} -> {a(lambda r: r['v6']['gold_asst']):.2f}")
    L = [r for r in rows if r["qid"] in LOSS]
    b = lambda f: sum(f(r) for r in L) / len(L)
    print(f"  [诊断|6 个 LOSS] gold 会话轮 {b(lambda r: r['v5']['gold_turns']):.2f} -> {b(lambda r: r['v6']['gold_turns']):.2f}; "
          f"助手轮 {b(lambda r: r['v5']['gold_asst']):.2f} -> {b(lambda r: r['v6']['gold_asst']):.2f}; "
          f"chars {b(lambda r: r['v5_chars']):.0f} -> {b(lambda r: r['v6_chars']):.0f}")
    json.dump(rows, open("/tmp/v6ssp_verify.json", "w"), indent=1)


if __name__ == "__main__":
    main()
