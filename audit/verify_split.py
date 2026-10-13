import json, re, sys, socket, statistics as st, random
class _B(RuntimeError): pass
def _nn(*a, **k): raise _B("blocked")
sys.path.insert(0, "/root/agent_memory_iclr/code")
import common
socket.socket.connect = _nn; socket.create_connection = _nn
common.chat = _nn
import lifecycle; lifecycle.chat = _nn
from lifecycle import build_lifecycle, render_view, _group_by_session
from run_baseline import bm25_rank
_al = re.compile(r"[a-z0-9]+")
CAP = 20000
def cap(t):
    if len(t) <= CAP: return t
    k, tot = [], 0
    for l in t.splitlines():
        if tot + len(l) + 1 > CAP: break
        k.append(l); tot += len(l) + 1
    return "\n".join(k)

with open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json") as f:
    data = json.load(f)
byid = {q["question_id"]: q for q in data}
rows = json.load(open("/root/agent_memory_iclr/audit/verify_viewdiff.json"))
routed = [r for r in rows if r["routed_diff"]]

# --- 1. prefix-restricted cosmetic/real split (a6_cap.py's own diff definition) ----
tot_p = tot_pc = tot_pr = 0
per_q_real = []
for r in routed:
    pass
# recompute properly per question
res = []
random.seed(0)
samp = set(x["qid"] for x in random.sample(routed, 40))
raw_lens = []
for r in routed:
    q = byid[r["qid"]]
    store = build_lifecycle(q, None)
    cur = cap(render_view(store, "current")).splitlines()
    hist = cap(render_view(store, "historical")).splitlines()
    keys = sorted(store)
    pc = pr = 0
    for j in range(min(len(cur), len(hist))):
        if cur[j] == hist[j]: continue
        if sorted(_al.findall(cur[j].lower())) == sorted(_al.findall(hist[j].lower())):
            pc += 1
        else:
            pr += 1
    tot_p += pc + pr; tot_pc += pc; tot_pr += pr
    per_q_real.append(pr)
    res.append((r["qid"], pc, pr, len(cur), len(hist)))
    if r["qid"] in samp:  # cost of the identical raw-turn half of the prompt
        units = []
        for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
            for t in sess:
                if t.get("content"):
                    units.append((date, t["role"], t["content"]))
        order = sorted(bm25_rank(q["question"], [u[2] for u in units])[:20])
        raw_lens.append(len("\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in order)))

n = len(routed)
print("routed n=%d  prefix-counted differing lines total=%d (mean %.2f/q)" % (n, tot_p, tot_p / n))
print("  of those: SEPARATOR-ONLY (identical tokens) = %d (%.1f%%)   content-differing = %d (%.1f%%)"
      % (tot_pc, 100 * tot_pc / tot_p, tot_pr, 100 * tot_pr / tot_p))
v = sorted(per_q_real)
print("  content-differing lines per question: min=%d p25=%d median=%.1f p75=%d max=%d mean=%.2f"
      % (v[0], v[n // 4], st.median(v), v[3 * n // 4], v[-1], sum(v) / n))
print("  questions with 0 content-differing lines in the prefix: %d" % sum(1 for x in v if x == 0))
mean_kept = sum(r["kept_cur"] for r in routed) / n
print("  mean fraction of view lines differing            : %.4f" % (tot_p / n / mean_kept))
print("  mean fraction with a CONTENT difference          : %.4f" % (tot_pr / n / mean_kept))

# --- 2. how big is the identical half of the prompt -------------------------------
print("\nBM25 raw-turn block (identical across arms), sample n=%d: mean %.0f chars, median %.0f, min %d max %d"
      % (len(raw_lens), sum(raw_lens) / len(raw_lens), st.median(raw_lens), min(raw_lens), max(raw_lens)))
mean_view = sum(r["len_cur"] for r in routed) / n
print("view block mean %.0f chars -> view is %.1f%% of the memory context"
      % (mean_view, 100 * mean_view / (mean_view + sum(raw_lens) / len(raw_lens))))
