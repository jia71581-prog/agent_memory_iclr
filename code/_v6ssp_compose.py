"""SSP 上下文构成测量:视图行与问句的词面重合、会话级 BM25 命中、会话全文长度。
gold 位置仅用于诊断输出,不进入任何方法代码。"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_qc as R
from run_baseline import bm25_rank, tokenize
from lifecycle import build_lifecycle
from query_classifier import classify_query

D = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
SSP = [q for q in D if q["question_type"] == "single-session-preference"]
LOSS = {"32260d93", "6b7dfb22", "57f827a0", "1da05512", "b6025781", "1c0ddc50"}
STOP = set("i my me the a an is are was were do does did can you your what how any some for to of in on and or "
           "it that this with about me some new better more".split())

rows = []
for q in SSP:
    qid = q["question_id"]
    qt = set(tokenize(q["question"])) - STOP
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    store = build_lifecycle(q, None)
    view = R._rank_view_lines(store, mode, q["question"], R.VIEW_CHAR_CAP)
    lines = [l for l in view.splitlines() if l.strip()]
    zero = sum(1 for l in lines if not (set(tokenize(l)) & qt))
    # 会话级 BM25:用户文本(v5 现行)vs 全文
    hs = q.get("haystack_session_ids") or []
    gs = {hs.index(i) for i in (q.get("answer_session_ids") or []) if i in hs}
    utxt, ftxt, slen = [], [], []
    for sess in q["haystack_sessions"]:
        u = " ".join(t["content"] for t in sess if t.get("content") and t["role"] != "assistant")
        f = " ".join(t["content"] for t in sess if t.get("content"))
        utxt.append(u)
        ftxt.append(f)
        slen.append(sum(len(t["content"]) + 20 for t in sess if t.get("content")))
    ru = bm25_rank(q["question"], utxt)
    rf = bm25_rank(q["question"], ftxt)
    gu = min((ru.index(g) for g in gs), default=-1)
    gf = min((rf.index(g) for g in gs), default=-1)
    rows.append({"qid": qid, "loss": qid in LOSS, "n_lines": len(lines), "zero_overlap": zero,
                 "zero_frac": round(zero / max(1, len(lines)), 3),
                 "view_chars": len(view), "gold_rank_user": gu, "gold_rank_full": gf,
                 "top1_full_sess_chars": slen[rf[0]], "gold_sess_chars": slen[min(gs)] if gs else 0,
                 "top1_is_gold_user": ru[0] in gs, "top1_is_gold_full": rf[0] in gs})
    print(json.dumps(rows[-1]), flush=True)

json.dump(rows, open("/tmp/ssp_compose.json", "w"), indent=1)
n = len(rows)
f = lambda k: sum(r[k] for r in rows) / n
print(f"\nALL n={n}: view lines={f('n_lines'):.0f} zero-overlap lines={f('zero_overlap'):.0f} "
      f"({f('zero_frac')*100:.0f}%) view_chars={f('view_chars'):.0f}")
print(f"top1 session == gold session: user-text {sum(r['top1_is_gold_user'] for r in rows)}/{n}  "
      f"full-text {sum(r['top1_is_gold_full'] for r in rows)}/{n}")
print(f"gold session rank: user-text mean {f('gold_rank_user'):.1f}  full-text mean {f('gold_rank_full'):.1f}")
print(f"top1 full-text session size mean {f('top1_full_sess_chars'):.0f} chars, "
      f"max {max(r['top1_full_sess_chars'] for r in rows)}")
L = [r for r in rows if r["loss"]]
print(f"\nLOSS n={len(L)}: zero-overlap {sum(r['zero_overlap'] for r in L)/len(L):.0f}/"
      f"{sum(r['n_lines'] for r in L)/len(L):.0f} lines; top1_full==gold "
      f"{sum(r['top1_is_gold_full'] for r in L)}/{len(L)}; top1_user==gold {sum(r['top1_is_gold_user'] for r in L)}/{len(L)}")
