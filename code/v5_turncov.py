"""F. 轮级证据覆盖:会话进了上下文 != 会话里的条目都进了上下文。
对每题算 (a) 证据会话的轮里有多少进了 flat top-20,(b) 若改成"整会话展开"策略的字符代价。"""
import json
import sys
from collections import Counter

sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from run_baseline import bm25_rank

R = "/root/agent_memory_iclr/results"
ROWS = {r["qid"]: r for r in json.load(open(f"{R}/_v5_shape_rows.json"))}
data = load_dataset("s_cleaned")
print("loaded", flush=True)
TOP_K = 20
out = []
for qi, q in enumerate(data):
    qid = q["question_id"]
    sid2idx = {s: i for i, s in enumerate(q["haystack_session_ids"])}
    ev = {sid2idx[s] for s in q["answer_session_ids"] if s in sid2idx}
    units = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for t in sess:
            if t.get("content"):
                units.append((si, t["content"]))
    order = bm25_rank(q["question"], [u[1] for u in units])
    flat = order[:TOP_K]
    ev_turns = [i for i, u in enumerate(units) if u[0] in ev]
    ev_in = [i for i in flat if units[i][0] in ev]
    flat_chars = sum(len(units[i][1]) for i in flat)
    # 整会话展开:取 top-3 会话(按最高分轮),给其全部轮
    seen, sess_rank = set(), []
    for i in order:
        s = units[i][0]
        if s not in seen:
            seen.add(s)
            sess_rank.append(s)
    top3 = set(sess_rank[:3])
    exp = [i for i, u in enumerate(units) if u[0] in top3]
    out.append({"qid": qid, "shape": ROWS[qid]["shape"], "is_abs": qid.endswith("_abs"),
                "judge": ROWS[qid]["judge"], "n_ev_turns": len(ev_turns), "n_ev_in": len(ev_in),
                "flat_chars": flat_chars, "exp_turns": len(exp),
                "exp_chars": sum(len(units[i][1]) for i in exp),
                "exp_ev_turns_in": sum(1 for i in exp if units[i][0] in ev),
                "ev_sess_in_top3": len(ev & top3), "n_ev_sess": len(ev)})
    if (qi + 1) % 100 == 0:
        print(" ...", qi + 1, flush=True)
json.dump(out, open(f"{R}/_v5_turncov.json", "w"))
print("saved")
