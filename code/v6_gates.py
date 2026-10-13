"""门控集测定 + 开关关闭等同性验证(离线,零 API)。

对每个 v6 flag:用真实 answer_question 路径构造上下文(answer_chat 打桩,chat 设为抛异常
以确保不会有任何 API 调用),对比 flag 开/关时的 user_content:
  - 不同的题 = 该 flag 的门控命中集(只有这些题需要在线重跑)
  - 相同的题 = 与 v5 逐字节一致,可直接复用 v5 已有结果
输出 results/v6_gates.json: {flag: {"fired": [qid...], "chars_v5": ..., "chars_v6": ...}}
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

R = "/root/agent_memory_iclr/results"
FLAGS = {"QC_V6_AGG": "1", "QC_V6_SSP": "1", "QC_V6_TR": "1", "QC_V6_TR2": "2"}


def build_all(flag, value):
    """返回 {qid: user_content}。flag=None 表示纯 v5。"""
    for k in ("QC_V6_AGG", "QC_V6_SSP", "QC_V6_TR"):
        os.environ.pop(k, None)
    os.environ["QC_V5"] = "1"
    if flag:
        os.environ[flag.replace("QC_V6_TR2", "QC_V6_TR")] = value
    for m in list(sys.modules):
        if m in ("run_qc", "v6_agg", "v6_tr", "common", "lifecycle"):
            del sys.modules[m]
    import common
    captured = {}

    def fake_answer_chat(system_prompt, user_content, ledger, op, max_tokens=6144):
        captured["last"] = user_content
        return "STUB"

    def raiser(*a, **k):
        raise AssertionError("API call attempted during offline gate probe")

    common.answer_chat = fake_answer_chat
    common.chat = raiser
    import lifecycle
    lifecycle.chat = raiser
    import run_qc
    run_qc.answer_chat = fake_answer_chat
    data = common.load_dataset("s_cleaned")
    out = {}
    for i, q in enumerate(data):
        try:
            run_qc.answer_question(q, None)
            out[q["question_id"]] = captured["last"]
        except Exception as e:
            out[q["question_id"]] = f"__ERROR__ {e}"
        if (i + 1) % 100 == 0:
            print(f"  [{flag or 'v5'}] {i+1}/500", flush=True)
    return out


def main():
    base = build_all(None, "")
    errs = {k: v for k, v in base.items() if v.startswith("__ERROR__")}
    if errs:
        print(f"!!! v5 基线构造出错 {len(errs)} 题: {list(errs.items())[:2]}")
    res = {}
    for flag, val in FLAGS.items():
        cur = build_all(flag, val)
        bad = {k: v for k, v in cur.items() if v.startswith("__ERROR__")}
        fired = [k for k in base if base[k] != cur.get(k)]
        cv5 = sum(len(base[k]) for k in fired) / max(1, len(fired))
        cv6 = sum(len(cur[k]) for k in fired) / max(1, len(fired))
        allv5 = sum(len(v) for v in base.values()) / len(base)
        allv6 = sum(len(v) for v in cur.values()) / len(cur)
        res[flag] = {"fired": fired, "n_fired": len(fired), "n_error": len(bad),
                     "chars_fired_v5": round(cv5), "chars_fired_v6": round(cv6),
                     "chars_all_v5": round(allv5), "chars_all_v6": round(allv6),
                     "ratio_fired": round(cv6 / max(1, cv5), 4),
                     "ratio_all": round(allv6 / max(1, allv5), 4)}
        print(f"{flag}={val}: 命中 {len(fired)}/500, 错误 {len(bad)}, "
              f"门内字符 {cv5:.0f}->{cv6:.0f} ({cv6/max(1,cv5):.3f}x), 全局 {allv6/max(1,allv5):.3f}x")
        if bad:
            print(f"   !!! 构造出错 {len(bad)} 题: {list(bad.items())[:1]}")
    with open(f"{R}/v6_gates.json", "w") as f:
        json.dump(res, f, indent=1)
    print("saved", f"{R}/v6_gates.json")


if __name__ == "__main__":
    main()
