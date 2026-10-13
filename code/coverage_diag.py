"""覆盖率诊断:定位准确率损失是"证据没进上下文"还是"进了但没答对"。

对 qc 臂每题计算:
  gold_in_view   - gold 答案串是否出现在生命周期视图里
  gold_in_raw    - gold 是否出现在 BM25 top-20 原文轮里
  gold_in_raw30  - 提到 top-30 是否能捞到(检索预算敏感性)
  gold_in_hay    - gold 是否存在于全部会话原文(天花板)
再与 judge 结果交叉,分题型报告:
  A. 证据缺失型错误(gold 既不在视图也不在原文轮)→ 靠检索/抽取改进
  B. 证据在场但答错(gold 在上下文里却判错)→ 靠推理/呈现改进
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from lifecycle import assemble_chains, render_view, extract_session_facts, _session_text, SESSION_CACHE_DIR
from query_classifier import classify_query
from run_baseline import bm25_rank
import hashlib

R = "/root/agent_memory_iclr/results"


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def contains(hay, needle):
    """gold 命中判定:规范化后子串,或 gold 的所有实义 token 都出现。"""
    h, n = norm(hay), norm(needle)
    if not n:
        return False
    if n in h:
        return True
    toks = [t for t in n.split() if len(t) > 2 and t not in ("the", "and", "for", "was", "were", "you", "your")]
    if not toks:
        return False
    return all(t in h for t in toks)


def cached_facts(sess):
    text = _session_text(sess)
    if not text:
        return []
    key = hashlib.md5(text.encode()).hexdigest()[:16]
    p = os.path.join(SESSION_CACHE_DIR, f"{key}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    data = {q["question_id"]: q for q in load_dataset("s_cleaned")}
    qc = {}
    for line in open(f"{R}/full500_qc.jsonl"):
        r = json.loads(line)
        if "judge" in r:
            qc[r["qid"]] = r

    rows = []
    for qid, res in qc.items():
        q = data[qid]
        gold = q["answer"]
        label, _ = classify_query(q["question"])
        mode = "historical" if label == "historical" else "current"

        all_facts, missing = [], False
        for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
            tri = cached_facts(sess)
            if tri is None:
                missing = True
                continue
            for e, a, v in tri:
                all_facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
        view = render_view(assemble_chains(all_facts), mode)

        units = []
        for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
            for t in sess:
                if t.get("content"):
                    units.append((date, t["role"], t["content"]))
        order = bm25_rank(q["question"], [u[2] for u in units])
        raw20 = " ".join(units[i][2] for i in order[:20])
        raw30 = " ".join(units[i][2] for i in order[:30])
        raw50 = " ".join(units[i][2] for i in order[:50])
        hay = " ".join(u[2] for u in units)

        rows.append({
            "qid": qid, "type": q["type"] if "type" in q else q["question_type"],
            "judge": res["judge"], "qlabel": label,
            "n_chains": len(assemble_chains(all_facts)), "view_chars": len(view),
            "in_view": contains(view, gold), "in_raw20": contains(raw20, gold),
            "in_raw30": contains(raw30, gold), "in_raw50": contains(raw50, gold),
            "in_hay": contains(hay, gold), "cache_missing": missing,
            "is_abs": qid.endswith("_abs"),
        })

    with open(f"{R}/coverage_diag.json", "w") as f:
        json.dump(rows, f, ensure_ascii=False)

    nonabs = [r for r in rows if not r["is_abs"]]
    print(f"非弃答题 n={len(nonabs)}\n")
    print(f"{'type':26s} {'n':>4s} {'acc':>6s} {'inView':>7s} {'inRaw20':>8s} {'inCtx':>6s} {'inRaw30':>8s} {'inRaw50':>8s} {'inHay':>6s}")
    for t in sorted({r["type"] for r in nonabs}):
        g = [r for r in nonabs if r["type"] == t]
        n = len(g)
        f_ = lambda k: sum(r[k] for r in g) / n
        inctx = sum(r["in_view"] or r["in_raw20"] for r in g) / n
        print(f"{t:26s} {n:4d} {sum(r['judge'] for r in g)/n:6.3f} {f_('in_view'):7.3f} {f_('in_raw20'):8.3f} {inctx:6.3f} {f_('in_raw30'):8.3f} {f_('in_raw50'):8.3f} {f_('in_hay'):6.3f}")
    n = len(nonabs)
    f_ = lambda k: sum(r[k] for r in nonabs) / n
    inctx = sum(r["in_view"] or r["in_raw20"] for r in nonabs) / n
    print(f"{'ALL':26s} {n:4d} {sum(r['judge'] for r in nonabs)/n:6.3f} {f_('in_view'):7.3f} {f_('in_raw20'):8.3f} {inctx:6.3f} {f_('in_raw30'):8.3f} {f_('in_raw50'):8.3f} {f_('in_hay'):6.3f}")

    print("\n=== 错误归因(非弃答错题)===")
    errs = [r for r in nonabs if not r["judge"]]
    a = [r for r in errs if not (r["in_view"] or r["in_raw20"])]
    b = [r for r in errs if r["in_view"] or r["in_raw20"]]
    print(f"错题总数 {len(errs)}")
    print(f"  A 证据缺失(gold 不在视图也不在 top-20 原文): {len(a)} ({len(a)/max(1,len(errs)):.0%})")
    print(f"     其中 top-30 能捞到: {sum(r['in_raw30'] for r in a)}, top-50 能捞到: {sum(r['in_raw50'] for r in a)}, 全 haystack 有: {sum(r['in_hay'] for r in a)}")
    print(f"  B 证据在场却答错: {len(b)} ({len(b)/max(1,len(errs)):.0%})")
    from collections import Counter
    print("  A 分题型:", dict(Counter(r["type"] for r in a)))
    print("  B 分题型:", dict(Counter(r["type"] for r in b)))

    print("\n=== 视图潜在增益上限 ===")
    only_view = [r for r in nonabs if r["in_view"] and not r["in_raw20"]]
    only_raw = [r for r in nonabs if r["in_raw20"] and not r["in_view"]]
    print(f"仅视图有 gold: {len(only_view)}(视图不可替代的贡献面);仅原文轮有: {len(only_raw)}")
    print(f"视图 gold 覆盖率 {f_('in_view'):.3f} vs 原文轮 {f_('in_raw20'):.3f} vs 全库 {f_('in_hay'):.3f}")
    miss = [r for r in nonabs if r["cache_missing"]]
    print(f"缓存缺失题数: {len(miss)}")


if __name__ == "__main__":
    main()
