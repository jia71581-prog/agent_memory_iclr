import json, os, sys, socket, statistics as st
class _B(RuntimeError): pass
def _nn(*a, **k): raise _B("blocked")
sys.path.insert(0, "/root/agent_memory_iclr/code")
import common
socket.socket.connect = _nn; socket.create_connection = _nn
common.chat = _nn
import lifecycle; lifecycle.chat = _nn
from lifecycle import build_lifecycle, render_view, _group_by_session
import re
_al = re.compile(r"[a-z0-9]+")

with open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json") as f:
    data = json.load(f)
rows = json.load(open("/root/agent_memory_iclr/audit/verify_viewdiff.json"))
routed = [r for r in rows if r["routed_diff"]]
# question with the FEWEST real differing lines = worst case for the claim
routed.sort(key=lambda r: r["n_diff_real"])
worst = routed[0]
byid = {q["question_id"]: q for q in data}
CAP = 20000
def cap(t):
    if len(t) <= CAP: return t
    k, tot = [], 0
    for l in t.splitlines():
        if tot + len(l) + 1 > CAP: break
        k.append(l); tot += len(l) + 1
    return "\n".join(k)

for tag, r in [("WORST CASE (fewest real diffs)", worst), ("MEDIAN CASE", routed[len(routed)//2])]:
    q = byid[r["qid"]]
    store = build_lifecycle(q, None)
    cur = cap(render_view(store, "current")).splitlines()
    hist = cap(render_view(store, "historical")).splitlines()
    keys = sorted(store)
    print("=" * 78)
    print("%s qid=%s type=%s  chains=%d kept cur/hist=%d/%d  diff(prefix)=%d real=%d cosmetic=%d"
          % (tag, r["qid"], r["type"], r["n_chains"], len(cur), len(hist),
             r["n_diff_prefix"], r["n_diff_real"], r["n_diff_cosmetic"]))
    shown_c = shown_r = 0
    for j in range(min(len(cur), len(hist))):
        if cur[j] == hist[j]: continue
        cosmetic = sorted(_al.findall(cur[j].lower())) == sorted(_al.findall(hist[j].lower()))
        if cosmetic and shown_c >= 3: continue
        if not cosmetic and shown_r >= 3: continue
        if cosmetic: shown_c += 1
        else: shown_r += 1
        ng = len(_group_by_session(store[keys[j]]))
        print("  [%s] line %d chain=%s groups=%d" % ("COSMETIC" if cosmetic else "REAL   ", j, keys[j], ng))
        print("      cur : " + cur[j][:200])
        print("      hist: " + hist[j][:200])
    print("  tail-only lines (present in one view, cut by cap in the other): %d" % abs(len(cur) - len(hist)))
