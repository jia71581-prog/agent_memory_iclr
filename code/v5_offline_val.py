"""v5 shape 的离线验证(无 API):
  A. 完整证据集召回率(每 shape):flat BM25 top-20 轮 vs 同预算的会话覆盖式检索
  B. TEMP.offset:gold 数值能否由 (question_date, 证据会话日期) 纯日期算术导出
  C. TEMP.anchor:问句相对日期解析出的绝对日期是否命中证据会话日期
  D. 弃答题的词面孪生检验(existence shape 是否规则可判)
"""
import json
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from run_baseline import bm25_rank, tokenize
from v5_shape import classify_shape, resolve_relative_dates, parse_qdate, days_between

R = "/root/agent_memory_iclr/results"
ROWS = {r["qid"]: r for r in json.load(open(f"{R}/_v5_shape_rows.json"))}
data = load_dataset("s_cleaned")
print("dataset loaded", len(data), flush=True)

TOP_K = 20
out = []
for qi, q in enumerate(data):
    qid = q["question_id"]
    shape = ROWS[qid]["shape"]
    sid2idx = {s: i for i, s in enumerate(q["haystack_session_ids"])}
    ev = {sid2idx[s] for s in q["answer_session_ids"] if s in sid2idx}
    units = []            # (sidx, date, role, content)
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for t in sess:
            if t.get("content"):
                units.append((si, date, t["role"], t["content"]))
    docs = [u[3] for u in units]
    order = bm25_rank(q["question"], docs)

    # (1) 现状:flat top-20 轮
    flat = order[:TOP_K]
    flat_sess = {units[i][0] for i in flat}

    # (2) 同预算会话覆盖式:按会话最高分排序会话,每会话取其最高分的 2 轮,轮询到 20 轮
    best_by_sess = defaultdict(list)
    for rank, i in enumerate(order):
        best_by_sess[units[i][0]].append(i)
    sess_rank = sorted(best_by_sess, key=lambda s: order.index(best_by_sess[s][0]))
    cov, si_ptr = [], 0
    while len(cov) < TOP_K and si_ptr < 2:
        for s in sess_rank:
            if len(cov) >= TOP_K:
                break
            if len(best_by_sess[s]) > si_ptr:
                cov.append(best_by_sess[s][si_ptr])
        si_ptr += 1
    cov_sess = {units[i][0] for i in cov}

    out.append({"qid": qid, "shape": shape, "n_ev": len(ev), "n_sess": len(q["haystack_sessions"]),
                "flat_all": ev <= flat_sess, "flat_any": bool(ev & flat_sess),
                "flat_hit": len(ev & flat_sess),
                "cov_all": ev <= cov_sess, "cov_any": bool(ev & cov_sess),
                "cov_hit": len(ev & cov_sess),
                "ev_dates": [q["haystack_dates"][i] for i in sorted(ev)],
                "qdate": q["question_date"], "gold": str(q["answer"]),
                "question": q["question"], "judge": ROWS[qid]["judge"],
                "is_abs": qid.endswith("_abs")})
    if (qi + 1) % 100 == 0:
        print("  ...", qi + 1, flush=True)

json.dump(out, open(f"{R}/_v5_offline_val.json", "w"), ensure_ascii=False)
print("saved _v5_offline_val.json")
