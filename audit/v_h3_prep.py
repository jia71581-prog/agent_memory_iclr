"""INDEPENDENT VERIFICATION of H3 (view leaks gold phrasing).
Step 1: rebuild every context offline for the SSP-30 subset (and, for cross-checks, all 500).
No API: chat() is monkeypatched to raise, so any cache miss is loud."""
import json, os, sys, pickle, re
sys.path.insert(0, "/root/agent_memory_iclr/code")
import common
_calls = {"n": 0}
def _no_api(*a, **k):
    _calls["n"] += 1
    raise RuntimeError("CACHE MISS -> would call API")
common.chat = _no_api
import lifecycle
lifecycle.chat = _no_api
from lifecycle import build_lifecycle, render_view
from common import load_dataset
from query_classifier import classify_query
from run_baseline import bm25_rank

R = "/root/agent_memory_iclr/results/"
def load(t):
    return {r["qid"]: r for r in (json.loads(l) for l in open(R + f"full500_{t}.jsonl"))}
qc, full, nolc = load("qc"), load("full"), load("qc_no_lifecycle")

data = load_dataset("s_cleaned")
byid = {q["question_id"]: q for q in data}
print("dataset n =", len(data))

from collections import Counter
print(Counter(q["question_type"] for q in data))

TARGET = sys.argv[1] if len(sys.argv) > 1 else "ssp"
if TARGET == "ssp":
    qids = [q["question_id"] for q in data if q["question_type"] == "single-session-preference"]
elif TARGET == "all":
    qids = [q["question_id"] for q in data]
else:
    qids = [q["question_id"] for q in data if q["question_type"] == TARGET]
print("subset", TARGET, "n =", len(qids))

VIEW_CAP = 20000
def cap(txt, CAP=VIEW_CAP):
    if len(txt) <= CAP:
        return txt
    kept, tot = [], 0
    for line in txt.splitlines():
        if tot + len(line) + 1 > CAP:
            break
        kept.append(line); tot += len(line) + 1
    return "\n".join(kept)

rows = []
for i, qid in enumerate(qids):
    q = byid[qid]
    units = []
    for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                units.append((date, t["role"], t["content"]))
    order_all = bm25_rank(q["question"], [u[2] for u in units])
    def raw(k):
        o = sorted(order_all[:k])
        return "\n".join(f"[{units[j][0]}] [{units[j][1]}] {units[j][2]}" for j in o)
    store = build_lifecycle(q, None)
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    view_qc = cap(render_view(store, mode))
    view_hist = cap(render_view(store, "historical"))
    rows.append({
        "qid": qid, "type": q["question_type"], "question": q["question"],
        "answer": q["answer"], "qlabel": label, "mode": mode,
        "view": view_qc, "view_hist": view_hist,
        "raw20": raw(20), "raw30": raw(30),
        "n_turns": len(units),
        "turns": [u[2] for u in units],
        "judge_qc": qc[qid]["judge"], "judge_full": full[qid]["judge"],
        "judge_nolc": nolc[qid]["judge"],
        "pred_qc": qc[qid]["pred"], "pred_full": full[qid]["pred"],
        "f1_qc": qc[qid]["f1"], "f1_full": full[qid]["f1"],
        "mode_run": qc[qid]["view_mode"],
    })
    if (i + 1) % 10 == 0:
        print("  ...", i + 1, flush=True)

print("api call attempts (must be 0):", _calls["n"])
print("rows:", len(rows))
assert all(r["mode"] == r["mode_run"] for r in rows), "mode mismatch vs run"
with open(f"/root/agent_memory_iclr/audit/v_h3_{TARGET}.pkl", "wb") as f:
    pickle.dump(rows, f)
print("saved")
