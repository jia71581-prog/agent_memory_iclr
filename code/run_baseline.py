"""W1 基线跑批:全量记忆 / FIFO / 时间衰减 三基线 + 共享 BM25 检索器(MemDelta 教训:固定检索器做单变量对照)。

记忆单元 = 会话轮(role+content+session_date)。三基线只在"记忆库里留什么"上不同:
  full   : 全部轮
  fifo   : 只留最近 K 个会话的轮(容量驱逐)
  decay  : MemoryBank 式,按会话距今天数的指数衰减保留概率决定去留(确定性阈值版)
检索 = BM25 top-k 轮 + 所在会话日期标注 → 组装上下文 → deepseek-v4-pro 回答 → 双指标。

用法: python3 run_baseline.py <system> <N题> [--full500]
"""
import json
import math
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset, answer_chat, token_f1, llm_judge, CostLedger, SHARED_ANSWER_PROMPT

SYSTEM = "full"
N = 60
TOP_K = 30          # 检索轮数
FIFO_SESSIONS = 15  # fifo 保留最近会话数
DECAY_HALFLIFE = 30.0  # 天

_word = re.compile(r"[a-z0-9']+")


def tokenize(s):
    return _word.findall(s.lower())


def bm25_rank(query, docs, k1=1.5, b=0.75):
    """极简 BM25。docs: list[str]。返回按分降序的索引。"""
    qt = tokenize(query)
    dt = [tokenize(d) for d in docs]
    avgdl = sum(len(d) for d in dt) / max(1, len(dt))
    df = {}
    for d in dt:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    ndocs = len(dt)
    scores = []
    for d in dt:
        tf = {}
        for w in d:
            tf[w] = tf.get(w, 0) + 1
        s = 0.0
        for w in qt:
            if w not in tf:
                continue
            idf = math.log(1 + (ndocs - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avgdl))
        scores.append(s)
    return sorted(range(ndocs), key=lambda i: -scores[i])


def build_memory(q, system):
    """返回记忆单元列表 [(date, role, content)]。"""
    units = []
    for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                units.append((date, t["role"], t["content"]))
    if system == "full":
        return units
    if system == "fifo":
        dates = sorted(set(q["haystack_dates"]))[-FIFO_SESSIONS:]
        keep = set(dates)
        return [u for u in units if u[0] in keep]
    if system == "decay":
        qd = datetime.strptime(q["question_date"].split()[0], "%Y/%m/%d")
        out = []
        for u in units:
            d = datetime.strptime(u[0].split()[0], "%Y/%m/%d")
            age = max(0.0, (qd - d).days)
            if math.exp(-age * math.log(2) / DECAY_HALFLIFE) >= 0.05:
                out.append(u)
        return out
    raise ValueError(system)


def answer_question(q, system, ledger):
    units = build_memory(q, system)
    docs = [f"{c}" for (_, _, c) in units]
    if not docs:
        return "I don't know.", 0
    order = bm25_rank(q["question"], docs)[:TOP_K]
    order = sorted(order)  # 恢复时间顺序
    ctx = "\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in order)
    ans = answer_chat(
        SHARED_ANSWER_PROMPT,
        f"Retrieved memory (each line: [session date] [speaker] text):\n{ctx}\n\nQuestion (asked on {q['question_date']}): {q['question']}",
        ledger, op=f"answer/{system}")
    return ans, len(units)


def stratified_sample(data, n):
    by_type = {}
    for q in data:
        by_type.setdefault(q["question_type"], []).append(q)
    out = []
    idx = 0
    while len(out) < n:
        added = False
        for t in sorted(by_type):
            qs = by_type[t]
            if idx < len(qs) and len(out) < n:
                out.append(qs[idx])
                added = True
        if not added:
            break
        idx += 1
    return out


def main():
    data = load_dataset("s_cleaned")
    sample = data if "--full500" in sys.argv else stratified_sample(data, N)
    ledger = CostLedger()
    results = []
    for i, q in enumerate(sample):
        try:
            ans, mem_size = answer_question(q, SYSTEM, ledger)
            f1 = token_f1(ans, q["answer"])
            correct = llm_judge(q["question"], q["question_type"], q["answer"], ans, ledger=ledger)
        except Exception as e:
            print(f"[{i+1}] ERROR {q['question_id']}: {e}", flush=True)
            results.append({"qid": q["question_id"], "type": q["question_type"], "error": str(e)})
            continue
        results.append({"qid": q["question_id"], "type": q["question_type"],
                        "mem_units": mem_size, "f1": round(f1, 3), "judge": bool(correct),
                        "pred": ans[:200]})
        print(f"[{i+1}/{len(sample)}] {q['question_type'][:22]:22s} judge={'Y' if correct else 'N'} f1={f1:.2f} mem={mem_size}", flush=True)

    ok = [r for r in results if "judge" in r]
    by_type = {}
    for r in ok:
        s = by_type.setdefault(r["type"], {"n": 0, "judge": 0, "f1": 0.0})
        s["n"] += 1
        s["judge"] += r["judge"]
        s["f1"] += r["f1"]
    print(f"\n=== {SYSTEM} | n={len(ok)} ===")
    for t, s in sorted(by_type.items()):
        print(f"{t:28s} n={s['n']:3d} judge_acc={s['judge']/s['n']:.3f} f1={s['f1']/s['n']:.3f}")
    total_acc = sum(r["judge"] for r in ok) / max(1, len(ok))
    print(f"总体 judge_acc={total_acc:.3f}")
    os.makedirs("/root/agent_memory_iclr/results", exist_ok=True)
    out_path = f"/root/agent_memory_iclr/results/baseline_{SYSTEM}_n{len(sample)}.json"
    with open(out_path, "w") as f:
        json.dump({"system": SYSTEM, "results": results,
                   "by_type": {t: {"n": s["n"], "judge_acc": s["judge"]/s["n"], "f1": s["f1"]/s["n"]}
                                for t, s in by_type.items()},
                   "total_judge_acc": total_acc,
                   "cost": ledger.summary()}, f, ensure_ascii=False, indent=1)
    print("saved:", out_path)


if __name__ == "__main__":
    SYSTEM = sys.argv[1] if len(sys.argv) > 1 else "full"
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        N = int(sys.argv[2])
    main()
