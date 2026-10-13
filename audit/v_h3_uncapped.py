"""Does an UNCAPPED view (or historical mode) reproduce the stated 0.278 / 0.485?
Also: how much of the view is thrown away by the 20k char cap?"""
import json, re, sys, pickle
from statistics import mean
sys.path.insert(0, "/root/agent_memory_iclr/code")
import common
common.chat = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no api"))
import lifecycle; lifecycle.chat = common.chat
from lifecycle import build_lifecycle, render_view
from common import load_dataset
from query_classifier import classify_query

rows = pickle.load(open("/root/agent_memory_iclr/audit/v_h3_ssp.pkl", "rb"))
byid = {q["question_id"]: q for q in load_dataset("s_cleaned")}
W = re.compile(r"[a-z0-9]+")
def tok(s): return W.findall(str(s).lower())
STOP = set("""a an the and or but if then than that this these those of in on at to for with from by as is are was were be been being am do does did done have has had having i me my mine you your yours he him his she her hers it its we us our ours they them their theirs what which who whom when where why how all any both each few more most other some such no nor not only own same so too very can will just should now about into over after before again further once here there also would could may might must shall""".split())

full_views = {}
for r in rows:
    q = byid[r["qid"]]
    store = build_lifecycle(q, None)
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    full_views[r["qid"]] = render_view(store, mode)

kept = [len(r["view"]) / len(full_views[r["qid"]]) for r in rows]
print(f"view cap: kept fraction of rendered view chars: mean={mean(kept):.3f} min={min(kept):.3f} "
      f"| truncated questions: {sum(1 for k in kept if k < 0.999)}/30")
print(f"uncapped view chars mean={mean([len(v) for v in full_views.values()]):.0f}")

def cov_set(r):
    turns = r["turns"]; n = len(turns)
    df = {}
    for t in turns:
        for w in set(tok(t)): df[w] = df.get(w, 0) + 1
    g = {w for w in set(tok(r["answer"])) - set(tok(r["question"])) if w not in STOP}
    return {w for w in g if df.get(w, 0) / max(1, n) <= 0.02}

per = {"view_capped": [], "view_uncapped": [], "raw20": [], "raw30": [],
       "uncapped+raw20": [], "capped+raw20": []}
for r in rows:
    D = cov_set(r)
    if not D: continue
    fv = full_views[r["qid"]]
    ctx = {"view_capped": r["view"], "view_uncapped": fv, "raw20": r["raw20"], "raw30": r["raw30"],
           "uncapped+raw20": fv + "\n" + r["raw20"], "capped+raw20": r["view"] + "\n" + r["raw20"]}
    for k, c in ctx.items():
        per[k].append(len(D & set(tok(c))) / len(D))
for k, v in per.items():
    print(f"  {k:15s} {mean(v):.4f}")
