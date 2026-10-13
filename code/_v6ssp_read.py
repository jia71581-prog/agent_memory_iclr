"""逐案读原文:gold 证据轮 vs v5 事实视图内容。"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_qc as R
from lifecycle import build_lifecycle
from query_classifier import classify_query

D = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
Q = {q["question_id"]: q for q in D}
TARGET = sys.argv[1:] or ["1c0ddc50"]

for qid in TARGET:
    q = Q[qid]
    print("=" * 110)
    print("QID", qid, "|", q["question"])
    print("GOLD:", q["answer"])
    hs = q.get("haystack_session_ids") or []
    gs = {hs.index(i) for i in (q.get("answer_session_ids") or []) if i in hs}
    print("-" * 50, "GOLD SESSION TURNS (verbatim)")
    for si, sess in enumerate(q["haystack_sessions"]):
        if si not in gs:
            continue
        for t in sess:
            if not t.get("content"):
                continue
            mark = " <<<HAS_ANSWER" if t.get("has_answer") else ""
            print(f"  [{t['role']}]{mark} {t['content'][:700]}")
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    store = build_lifecycle(q, None)
    view = R._rank_view_lines(store, mode, q["question"], R.VIEW_CHAR_CAP)
    lines = view.splitlines()
    print("-" * 50, f"V5 FACT VIEW ({len(lines)} lines / {len(store)} chains / {len(view)} chars) first 60:")
    for l in lines[:60]:
        print("   ", l[:200])
