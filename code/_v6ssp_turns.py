"""看 v5 原文轮在败例上的实际构成:命中轮 / 邻接缺失 / 会话分布。"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_qc as R

D = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
Q = {q["question_id"]: q for q in D}

for qid in sys.argv[1:]:
    q = Q[qid]
    hs = q.get("haystack_session_ids") or []
    gs = {hs.index(i) for i in (q.get("answer_session_ids") or []) if i in hs}
    # 重建 units 带 (si, ti)
    meta = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for ti, t in enumerate(sess):
            if t.get("content"):
                meta.append((si, ti, t["role"], t.get("has_answer", False), len(t["content"])))
    units, sel = R._select_turns_v5(q, q["question"], R.TOP_K)
    print("=" * 100)
    print(qid, "|", q["question"])
    print(f"selected {len(sel)} turns; gold sessions {sorted(gs)}")
    bysess = {}
    for i in sel:
        bysess.setdefault(meta[i][0], []).append(i)
    for si in sorted(bysess):
        tot = sum(1 for m in meta if m[0] == si)
        marks = ", ".join(f"t{meta[i][1]}/{meta[i][2][:4]}{'*' if meta[i][3] else ''}" for i in bysess[si])
        print(f"  sess {si:2d} {'GOLD' if si in gs else '    '} ({len(bysess[si])}/{tot} turns): {marks}")
