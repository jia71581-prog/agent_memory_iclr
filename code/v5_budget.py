"""G. 会话展开预算曲线:取 top-N 会话的全部轮,证据会话完整召回率 vs 字符代价(每 shape)。"""
import json
import sys
from collections import defaultdict

sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from run_baseline import bm25_rank

R = "/root/agent_memory_iclr/results"
ROWS = {r["qid"]: r for r in json.load(open(f"{R}/_v5_shape_rows.json"))}
data = load_dataset("s_cleaned")
print("loaded", flush=True)
NS = [1, 2, 3, 4, 6, 8, 10]
out = []
for qi, q in enumerate(data):
    qid = q["question_id"]
    sid2idx = {s: i for i, s in enumerate(q["haystack_session_ids"])}
    ev = {sid2idx[s] for s in q["answer_session_ids"] if s in sid2idx}
    units, by_sess = [], defaultdict(list)
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for t in sess:
            if t.get("content"):
                by_sess[si].append(len(units))
                units.append((si, t["content"]))
    order = bm25_rank(q["question"], [u[1] for u in units])
    seen, sess_rank = set(), []
    for i in order:
        s = units[i][0]
        if s not in seen:
            seen.add(s)
            sess_rank.append(s)
    row = {"qid": qid, "shape": ROWS[qid]["shape"], "is_abs": qid.endswith("_abs"),
           "judge": ROWS[qid]["judge"], "n_ev": len(ev)}
    for N in NS:
        top = set(sess_rank[:N])
        idx = [i for s in top for i in by_sess[s]]
        row[f"all{N}"] = ev <= top
        row[f"chars{N}"] = sum(len(units[i][1]) for i in idx)
        row[f"turns{N}"] = len(idx)
    out.append(row)
    if (qi + 1) % 100 == 0:
        print(" ...", qi + 1, flush=True)
json.dump(out, open(f"{R}/_v5_budget.json", "w"))

NA = [r for r in out if not r["is_abs"]]
print(f"\n=== G. top-N 会话整展开:证据会话完整召回 / 平均字符 (非弃答 {len(NA)}) ===")
hdr = "  ".join(f"N={N}" for N in NS)
print(f"{'shape':18s} {'n':>4s}  " + "  ".join(f"{'N='+str(N):>12s}" for N in NS))
shapes = sorted({r["shape"] for r in NA}) + ["__ALL__"]
for s in shapes:
    g = NA if s == "__ALL__" else [r for r in NA if r["shape"] == s]
    cells = []
    for N in NS:
        cells.append(f"{sum(r['all'+str(N)] for r in g)/len(g):.2f}/{sum(r['chars'+str(N)] for r in g)/len(g)/1000:.0f}k")
    print(f"{s:18s} {len(g):4d}  " + "  ".join(f"{c:>12s}" for c in cells))
print("\n(单元格 = 完整证据会话召回率 / 平均上下文字符;现状 flat top-20 轮 = 0.90 / 20.5k)")
