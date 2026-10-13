import pickle, sys, re, json
sys.path.insert(0, "/root/agent_memory_iclr/code")
from v5_temporal import resolve_question_anchors, anchor_block, ANCHOR_WINDOW, _stem
sys.path.insert(0, "/root/agent_memory_iclr/vfy2")
exec(open("/root/agent_memory_iclr/vfy2/p10_hatch.py").read().split("fixed=buggy=0")[0])

# 1) hatch precision on gpt4_4929293b: how many lines differ, and does Feb leak in?
q=tr["gpt4_4929293b"]
a0=[a for a in resolve_question_anchors(q["question"], q["question_date"]) if a["conf"]=="high"][0]
w0,w1=ANCHOR_WINDOW.get(a0["kind"],(1,3))
bf=block(q,a0["date"],w0,buggy=False); bb=block(q,a0["date"],w0,buggy=True)
print("anchor", a0["date"].date(), "fixed n=",len(bf), "buggy n=",len(bb))
print("lines only in buggy:")
for x in bb:
    if x not in bf: print("   -", x[:150])
print("lines only in fixed:")
for x in bf:
    if x not in bb: print("   +", x[:150])

# 2) truncation: deduped candidate count vs cap 10 ('Everything ...' header truthfulness)
print()
rows=json.load(open("/root/agent_memory_iclr/vfy2/anchor_rows.json"))
tot=[]
for r in rows:
    qq=tr[r["qid"]]
    an=[a for a in resolve_question_anchors(qq["question"], qq["question_date"]) if a["conf"]=="high"]
    if not an:
        an=[a for a in resolve_question_anchors(qq["question"], qq["question_date"])
            if a["kind"] in ("offset-coarse","month-anchor")][:1]
    a=an[0]; W0,W1=ANCHOR_WINDOW.get(a["kind"],(1,3))
    w = r["window"]
    full=block(qq,a["date"],w,cap=10**6)
    tot.append((r["r" if False else "qid"], len(full)))
    print(f'{r["qid"]:16s} window=+/-{w} deduped_candidates={len(full):4d} shown=10 -> shown_frac={min(10,len(full))/max(len(full),1):.2f}')
import statistics as st
print("median deduped candidates on the anchor day-window:", st.median([t[1] for t in tot]),
      " blocks truncated (>10):", sum(t[1]>10 for t in tot), "/", len(tot))
