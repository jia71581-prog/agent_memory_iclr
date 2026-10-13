"""H. AGG.enum 的计数机制离线检验(无 API):
  H1 gold 计数 N 与证据会话数 n_ev 的关系(N == n_ev 时"每会话一条目"标注即可给出计数)
  H2 lifecycle 缓存里问句匹配的事实簇能否装下 N 个不同条目(count-annotated cluster 的可行上界)
"""
import hashlib
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from lifecycle import assemble_chains, SESSION_CACHE_DIR, _session_text, _norm

R = "/root/agent_memory_iclr/results"
ROWS = {r["qid"]: r for r in json.load(open(f"{R}/_v5_shape_rows.json"))}
V = {r["qid"]: r for r in json.load(open(f"{R}/_v5_offline_val.json"))}

NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20}


def gold_int(g):
    g = g.lower().replace(",", "")
    m = re.match(r"^\s*\$?(\d+)\b", g)
    if m:
        return int(m.group(1))
    for w, v in NUMW.items():
        if re.match(rf"^\s*{w}\b", g):
            return v
    return None


STOP = set("how many much did i do have the a an of in on to my me for and or is are was were what "
           "different total times number all across including both currently now got get".split())
W = re.compile(r"[a-z0-9']+")


def qtok(s):
    return {t for t in W.findall(s.lower()) if t not in STOP and len(t) > 2}


def cached(sess):
    t = _session_text(sess)
    if not t:
        return []
    p = os.path.join(SESSION_CACHE_DIR, f"{hashlib.md5(t.encode()).hexdigest()[:16]}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


data = {q["question_id"]: q for q in load_dataset("s_cleaned")}
print("loaded", flush=True)

targets = [qid for qid, r in ROWS.items() if r["shape"] in ("AGG.enum", "AGG.latest_count") and not r["is_abs"]]
print(f"AGG.enum + AGG.latest_count 非弃答题 {len(targets)}")

h1 = []
h2 = []
for qid in targets:
    q = data[qid]
    r = ROWS[qid]
    N = gold_int(r["gold"])
    n_ev = V[qid]["n_ev"]
    if N is not None:
        h1.append((r["shape"], N, n_ev, r["judge"], r["q"]))
    # H2: 问句匹配事实簇
    facts, miss = [], False
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        tri = cached(sess)
        if tri is None:
            miss = True
            continue
        for e, a, v in tri:
            facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
    qt = qtok(r["q"])
    pool = [f for f in facts if qt & (qtok(f["attribute"]) | qtok(f["entity"]) | qtok(f["value"]))]
    vals = {_norm(f["value"]) for f in pool}
    if N is not None:
        h2.append((r["shape"], N, len(pool), len(vals), r["judge"], miss, r["q"]))

print("\n=== H1. gold 计数 N vs 证据会话数 n_ev ===")
json.dump({"h1": [list(x) for x in h1], "h2": [list(x) for x in h2]},
          open(f"{R}/_v5_aggmech.json", "w"), ensure_ascii=False)
# 只保留"条目计数"题:gold 无货币符号、N<=12、问句以 how many 开头
CNT = re.compile(r"^how many\b")
h1 = [x for x in h1 if x[1] <= 12 and "$" not in ROWS[[k for k in targets if ROWS[k]['q'] == x[4]][0]]["gold"] and CNT.match(x[4].lower())]
h2 = [x for x in h2 if x[1] <= 12 and CNT.match(x[6].lower())]
print(f"(限定为条目计数题:H1 n={len(h1)}, H2 n={len(h2)})")
for s in ("AGG.enum", "AGG.latest_count"):
    g = [x for x in h1 if x[0] == s]
    eq = sum(1 for x in g if x[1] == x[2])
    le = sum(1 for x in g if x[1] <= x[2])
    print(f"  {s:18s} n={len(g):3d}  N==n_ev {eq} ({eq/len(g):.1%})  N<=n_ev {le} ({le/len(g):.1%})  "
          f"mean N={sum(x[1] for x in g)/len(g):.1f} mean n_ev={sum(x[2] for x in g)/len(g):.1f}")
g = [x for x in h1 if x[0] == "AGG.enum"]
print("  AGG.enum 里 N>n_ev(单会话内含多条目,必须轮级完整而非会话级)的题数:",
      sum(1 for x in g if x[1] > x[2]), f"/{len(g)}")
print("  这些题的准确率:", f"{sum(x[3] for x in g if x[1] > x[2])/max(1,sum(1 for x in g if x[1]>x[2])):.3f}",
      " vs N<=n_ev 的:", f"{sum(x[3] for x in g if x[1] <= x[2])/max(1,sum(1 for x in g if x[1]<=x[2])):.3f}")

print("\n=== H2. lifecycle 事实簇能否装下 N 个不同条目 ===")
for s in ("AGG.enum", "AGG.latest_count"):
    g = [x for x in h2 if x[0] == s]
    ok = sum(1 for x in g if x[3] >= x[1])
    print(f"  {s:18s} n={len(g):3d}  簇内不同 value 数 >= N: {ok} ({ok/len(g):.1%})  "
          f"mean pool={sum(x[2] for x in g)/len(g):.1f} mean distinct={sum(x[3] for x in g)/len(g):.1f}")
short = [x for x in h2 if x[3] < x[1]]
print(f"  簇装不下 N 的 {len(short)} 例(count-annotated cluster 在这些题上不可能给出正确计数):")
for x in short[:10]:
    print(f"    N={x[1]:3d} distinct={x[3]:3d} judge={'Y' if x[4] else 'N'} | {x[6][:70]}")
