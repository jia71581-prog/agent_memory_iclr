"""H3 part 5: does 'view carries less gold vocabulary than raw' generalize beyond the SSP-30?
Stratified sample of the other 470 questions, same measurement."""
import json, re, sys, pickle, random
from statistics import mean
sys.path.insert(0, "/root/agent_memory_iclr/code")
import common
_c = {"n": 0}
def _no(*a, **k):
    _c["n"] += 1
    raise RuntimeError("cache miss")
common.chat = _no
import lifecycle; lifecycle.chat = _no
from lifecycle import build_lifecycle, render_view
from common import load_dataset
from query_classifier import classify_query
from run_baseline import bm25_rank

R = "/root/agent_memory_iclr/results/"
def load(t): return {r["qid"]: r for r in (json.loads(l) for l in open(R + f"full500_{t}.jsonl"))}
qc, full = load("qc"), load("full")
data = load_dataset("s_cleaned")
random.seed(0)
by_type = {}
for q in data:
    if q["question_type"] == "single-session-preference": continue
    by_type.setdefault(q["question_type"], []).append(q)
sample = []
PER = 20
for t, qs in sorted(by_type.items()):
    sample += random.sample(qs, min(PER, len(qs)))
print("sample n =", len(sample), {t: min(PER, len(v)) for t, v in by_type.items()})

W = re.compile(r"[a-z0-9]+")
def tok(s): return W.findall(str(s).lower())
STOP = set("""a an the and or but if then than that this these those of in on at to for with from by as is are was were be been being am do does did done have has had having i me my mine you your yours he him his she her hers it its we us our ours they them their theirs what which who whom when where why how all any both each few more most other some such no nor not only own same so too very can will just should now about into over after before again further once here there also would could may might must shall""".split())
CAP = 20000
def cap(txt):
    if len(txt) <= CAP: return txt
    kept, tot = [], 0
    for l in txt.splitlines():
        if tot + len(l) + 1 > CAP: break
        kept.append(l); tot += len(l) + 1
    return "\n".join(kept)

per = {"view": [], "raw20": [], "raw30": [], "view+raw20": []}
loose = {"view": [], "raw30": []}
bytype = {}
trunc = 0
for i, q in enumerate(sample):
    units = []
    for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        for t in sess:
            if t.get("content"): units.append((date, t["role"], t["content"]))
    order = bm25_rank(q["question"], [u[2] for u in units])
    def raw(k):
        return "\n".join(f"[{units[j][0]}] [{units[j][1]}] {units[j][2]}" for j in sorted(order[:k]))
    store = build_lifecycle(q, None)
    lab, _ = classify_query(q["question"])
    mode = "historical" if lab == "historical" else "current"
    vfull = render_view(store, mode); v = cap(vfull)
    if len(v) < len(vfull): trunc += 1
    df = {}
    for u in units:
        for w in set(tok(u[2])): df[w] = df.get(w, 0) + 1
    n = len(units)
    g = {w for w in set(tok(q["answer"])) - set(tok(q["question"])) if w not in STOP}
    D = {w for w in g if df.get(w, 0) / max(1, n) <= 0.02}
    L = {w for w in set(tok(q["answer"])) if w not in STOP}
    ctx = {"view": v, "raw20": raw(20), "raw30": raw(30), "view+raw20": v + "\n" + raw(20)}
    if D:
        vals = {}
        for k, c in ctx.items():
            val = len(D & set(tok(c))) / len(D)
            per[k].append(val); vals[k] = val
        bytype.setdefault(q["question_type"], []).append(vals)
    if L:
        for k in ("view", "raw30"):
            loose[k].append(len(L & set(tok(ctx[k]))) / len(L))
    if (i + 1) % 20 == 0: print("  ...", i + 1, flush=True)

print("api attempts:", _c["n"], "| views truncated by the 20k cap:", trunc, "/", len(sample))
print("== non-SSP sample, distinctive gold-term coverage ==")
for k in per: print(f"  {k:11s} {mean(per[k]):.4f} (n={len(per[k])})")
print("== loose content-word recall == view %.4f  raw30 %.4f" % (mean(loose["view"]), mean(loose["raw30"])))
print("== by type ==")
for t, V in sorted(bytype.items()):
    print(f"  {t:26s} n={len(V):3d} view={mean([x['view'] for x in V]):.3f} raw20={mean([x['raw20'] for x in V]):.3f} "
          f"raw30={mean([x['raw30'] for x in V]):.3f} view+raw20={mean([x['view+raw20'] for x in V]):.3f}")
pickle.dump({"per": per, "bytype": bytype, "loose": loose}, open("/root/agent_memory_iclr/audit/v_h3_nonssp.pkl", "wb"))
