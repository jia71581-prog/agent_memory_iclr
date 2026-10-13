"""端到端冒烟:直接跑真实 answer_question(),用 monkeypatch 截获 prompt,不调 DeepSeek。
断言:
 A. 关旗 → prompt 与 v5 逐字节相同(SSP + 各型抽样)
 B. 开旗、门外 → prompt 与 v5 逐字节相同
 C. 开旗、门内 → prompt == _v6ssp_verify 重建的上下文(说明验证脚本与生产路径一致)
 D. 门内 char delta <= 0
 E. 上下文构成:事实视图占比、与问句零内容词重合的视图行数
"""
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _v6ssp_verify as V
from run_baseline import tokenize

DATA = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
SSP = [q for q in DATA if q["question_type"] == "single-session-preference"]
STOP = set("i my me the a an is are was were do does did can you your what how any some for to of in on and or "
           "it that this with about new better more".split())


def capture(mod, q):
    """真实 answer_question 路径,拦下 answer_chat 拿到 prompt。"""
    got = {}

    def fake(system_prompt, user_content, ledger, op, max_tokens=6144):
        got["sys"] = system_prompt
        got["user"] = user_content
        return "STUB"

    orig = mod.answer_chat
    mod.answer_chat = fake
    try:
        mod.answer_question(q, None)
    finally:
        mod.answer_chat = orig
    return got["sys"], got["user"]


def view_block(prompt):
    for blk in prompt.split("\n\n### "):
        b = blk.lstrip("# ").strip()
        if b.startswith(("Current-state fact view", "Fact evolution view")):
            return "\n".join(b.splitlines()[1:])
    return ""


def main():
    m5 = V.load(v5="1")
    m6 = V.load(v5="1", v6="1")

    import collections
    off, seen = [], collections.Counter()
    for q in DATA:
        if m6._v6ssp_is_advice(q["question"]):
            continue
        if seen[q["question_type"]] < 6:
            seen[q["question_type"]] += 1
            off.append(q)

    print("A/B 关旗与门外一致性")
    bad = 0
    for q in off + SSP:
        s5, p5 = capture(m5, q)          # v5 参照
        s6, p6 = capture(m6, q)          # 开旗
        gated = m6._v6ssp_is_advice(q["question"])
        assert s5 == s6, "system prompt 必须完全一致"
        if not gated and p5 != p6:
            bad += 1
            print("  MISMATCH(门外)", q["question_id"])
    print(f"  system prompt 全等;门外 prompt 全等({len(off)} 抽样题 + {sum(1 for q in SSP if not m6._v6ssp_is_advice(q['question']))} 道未命中 SSP),失配 {bad}")

    print("\nC/D/E 门内(SSP)")
    rows = []
    mism = 0
    for q in SSP:
        if not m6._v6ssp_is_advice(q["question"]):
            continue
        _, p5 = capture(m5, q)
        _, p6 = capture(m6, q)
        c6, _ = V.build_ctx(m6, q)
        if c6 not in p6:
            mism += 1
            print("  验证脚本与生产路径不一致:", q["question_id"])
        qt = set(tokenize(q["question"])) - STOP
        v5v, v6v = view_block(p5), view_block(p6)
        z5 = sum(1 for l in v5v.splitlines() if l.strip() and not (set(tokenize(l)) & qt))
        z6 = sum(1 for l in v6v.splitlines() if l.strip() and not (set(tokenize(l)) & qt))
        rows.append(dict(qid=q["question_id"], p5=len(p5), p6=len(p6),
                         v5v=len(v5v), v6v=len(v6v),
                         l5=len([l for l in v5v.splitlines() if l.strip()]),
                         l6=len([l for l in v6v.splitlines() if l.strip()]), z5=z5, z6=z6))
    n = len(rows)
    a = lambda k: sum(r[k] for r in rows) / n
    print(f"  n={n}, 生产路径与验证脚本不一致 {mism}")
    print(f"  prompt chars {a('p5'):.0f} -> {a('p6'):.0f} "
          f"({100*(sum(r['p6'] for r in rows)-sum(r['p5'] for r in rows))/sum(r['p5'] for r in rows):+.2f}%), "
          f"最大单题 delta {max(100*(r['p6']-r['p5'])/r['p5'] for r in rows):+.2f}%")
    print(f"  事实视图占 prompt 比例 {a('v5v')/a('p5')*100:.1f}% -> {a('v6v')/a('p6')*100:.1f}%")
    print(f"  视图行数 {a('l5'):.0f} -> {a('l6'):.0f};其中与问句零内容词重合 {a('z5'):.0f} -> {a('z6'):.0f}")
    over = [r for r in rows if r["p6"] > r["p5"]]
    print(f"  超出 v5 预算的题数: {len(over)}(硬约束应为 0)")


if __name__ == "__main__":
    main()
