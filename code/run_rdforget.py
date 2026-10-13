"""RD-Forget 式基线(计划书系统 4):按 2609.10263 全文精读(13_competitor_deepread.md)复现其精神。

与本方法(QC-Lifecycle)的差异 = 论文的差异化主张,必须如实实现其设计:
 1. slot 替代链:同 (entity, attribute) 链只保留 active(最新)值,superseded 条目**默认排除**
    → 无 contradiction 关系、无多值属性词表(其 curator 靠自由裁量,我们这里等价为"全部按替代处理")。
 2. 查询意图 h(q) = 二值(historical / 非historical),且**吃 benchmark 的 gold task-type 标签**
    (原文 §4.2:词法线索 + gold task-type;这是它的协议缺陷,复现时如实保留并在论文中点名)。
    historical 意图时才做 lifecycle expansion(放行被替代的旧值)。
 3. 词法兜底(其 <=20 行 lexical fallback)= BM25 原文轮,与本方法同 k 以保证可比。
 4. 单点固定预算 B(原文 Eq.3 的 B=2048 token 单点,无 R-D 扫描);本复现用等价字符预算,
    并与本方法同总预算以保证比较公平。

用法: python3 run_rdforget.py <workers=8> [tag]
结果写 results/full500_rdforget<tag>.jsonl,与其他臂同 runner 口径(断点续跑/统一 prompt/同 judge)。
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset, answer_chat, token_f1, llm_judge, CostLedger, SHARED_ANSWER_PROMPT
from lifecycle import build_lifecycle
from run_baseline import bm25_rank
from run_qc import TOP_K, VIEW_CHAR_CAP, _rank_view_lines

WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
OUT = f"/root/agent_memory_iclr/results/full500_rdforget{TAG}.jsonl"
_lock = threading.Lock()

# RD-Forget §4.2 的 h(q):词法线索 + gold task-type 标签(标签泄漏,如实复现)
_HIST_LEX = ("previously", "before", "originally", "used to", "earlier", "first time",
             "initially", "former", "at first", "back then", "history of")
_HIST_TASK_TYPES = {"knowledge-update", "temporal-reasoning"}  # 吃 benchmark 元数据


def intent_is_historical(q):
    ql = q["question"].lower()
    if any(w in ql for w in _HIST_LEX):
        return True
    return q["question_type"] in _HIST_TASK_TYPES  # 标签泄漏分支


def render_slots(store, expand):
    """slot 视图:默认只渲染 active(链尾)值;expand=True 时放行被替代的旧值。
    无 contradiction 标注、无多值累积语义 —— 一律按替代链处理(RD-Forget 精神)。"""
    lines = []
    for k in sorted(store):
        chain = store[k]
        ent, attr = chain[0]["entity"], chain[0]["attribute"]
        if expand and len(chain) > 1:
            hist = " -> ".join(f"{c['value']} ({c['date'].split()[0]})" for c in chain)
            lines.append(f"{ent} | {attr}: {hist}")
        else:
            latest = chain[-1]
            lines.append(f"[{latest['date']}] {ent} | {attr} = {latest['value']}")
    return "\n".join(lines)


def answer_question(q, ledger):
    store = build_lifecycle(q, ledger)
    expand = intent_is_historical(q)
    # 预算内按相关性保留 slot(等价于其 rate-distortion 视图构造的单点预算 B)
    full = render_slots(store, expand)
    if len(full) > VIEW_CHAR_CAP:
        keys = list(store)
        texts = [" ".join(f"{c['entity']} {c['attribute']} {c['value']}" for c in store[k]) for k in keys]
        order = bm25_rank(q["question"], texts)
        kept, total = [], 0
        for i in order:
            line = render_slots({keys[i]: store[keys[i]]}, expand)
            if total + len(line) + 1 > VIEW_CHAR_CAP:
                continue
            kept.append((i, line))
            total += len(line) + 1
        kept.sort(key=lambda x: x[0])
        view = "\n".join(l for _, l in kept)
    else:
        view = full

    units = []
    for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                units.append((date, t["role"], t["content"]))
    order = sorted(bm25_rank(q["question"], [u[2] for u in units])[:TOP_K])
    raw = "\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in order)

    head = ("Memory view (value history per slot, oldest -> newest)" if expand else
            "Memory view (active value per slot; superseded values excluded)")
    ctx = f"### {head}\n{view}\n\n### Retrieved raw conversation turns\n{raw}"
    ans = answer_chat(SHARED_ANSWER_PROMPT,
                      f"{ctx}\n\nQuestion (asked on {q['question_date']}): {q['question']}",
                      ledger, op="answer/rdforget")
    return ans, expand


def process(q, ledger):
    ans, expand = answer_question(q, ledger)
    return {"qid": q["question_id"], "type": q["question_type"], "judge":
            bool(llm_judge(q["question"], q["question_type"], q["answer"], ans, ledger=ledger)),
            "f1": round(token_f1(ans, q["answer"]), 3), "intent_hist": expand, "pred": ans[:4000]}


def main():
    data = load_dataset("s_cleaned")
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "judge" in r:
                done.add(r["qid"])
    todo = [q for q in data if q["question_id"] not in done]
    print(f"rdforget: total {len(data)}, done {len(done)}, todo {len(todo)}", flush=True)
    ledger = CostLedger()
    n = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(process, q, ledger): q for q in todo}
        for fut in as_completed(futs):
            q = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"qid": q["question_id"], "type": q["question_type"], "error": str(e)[:200]}
            with _lock:
                with open(OUT, "a") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
            if n % 25 == 0:
                print(f"progress {n}/{len(todo)}", flush=True)
    ledger.save(f"/root/agent_memory_iclr/results/full500_rdforget{TAG}_ledger_{int(time.time())}.json")
    latest = {}
    for line in open(OUT):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "judge" in r or r.get("qid") not in latest:
            latest[r["qid"]] = r
    ok = [r for r in latest.values() if "judge" in r]
    errs = [r for r in latest.values() if "judge" not in r]
    by = {}
    for r in ok:
        s = by.setdefault(r["type"], [0, 0])
        s[0] += 1
        s[1] += r["judge"]
    print("=== rdforget full500 ===")
    for t, (nn, c) in sorted(by.items()):
        print(f"{t:28s} n={nn:3d} acc={c/nn:.3f}")
    print(f"TOTAL n={len(ok)} acc={sum(r['judge'] for r in ok)/max(1,len(ok)):.3f}")
    print(f"intent_historical 命中 {sum(r.get('intent_hist', False) for r in ok)}/{len(ok)}")
    if errs:
        print(f"!!! {len(errs)} error rows, 重跑可续")
    print("RDFORGET DONE", flush=True)


if __name__ == "__main__":
    main()
