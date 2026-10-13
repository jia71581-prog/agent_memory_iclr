"""Null test: does the anchor-day dump still contain gold when the anchor is deliberately wrong?"""
import pickle, sys, re, json, random
from datetime import timedelta
sys.path.insert(0, "/root/agent_memory_iclr/code")
from v5_temporal import (detect_family, resolve_question_anchors, anchor_block, _stem)

tr = {q["question_id"]: q for q in pickle.load(open("/root/agent_memory_iclr/vfy2/tr.pkl","rb"))}
rows = json.load(open("/root/agent_memory_iclr/vfy2/anchor_rows.json"))
def norm(s): return re.sub(r"[^a-z0-9]+"," ",str(s).lower()).strip()
STOPW=("the","and","for","was","were","you","your","not","did","with")
def cstem(hay,needle):
    h={_stem(w) for w in norm(hay).split()}
    toks=[_stem(t) for t in norm(needle).split() if len(t)>2 and t not in STOPW]
    return bool(toks) and sum(t in h for t in toks)>=max(1,int(round(0.8*len(toks))))

shifts=[-60,-30,-14,-7,7,14,30,60]
agg={s:0 for s in shifts}; agg["true"]=0
per=[]
for r in rows:
    q=tr[r["qid"]]
    anchors=[a for a in resolve_question_anchors(q["question"], q["question_date"]) if a["conf"]=="high"]
    if not anchors:
        anchors=[a for a in resolve_question_anchors(q["question"], q["question_date"])
                 if a["kind"] in ("offset-coarse","month-anchor")][:1]
    a0=dict(anchors[0])
    blk,_=anchor_block(q,a0)
    t=int(any(cstem(x,q["answer"]) for x in blk)); agg["true"]+=t
    line={"qid":r["qid"],"true":t}
    for s in shifts:
        a=dict(a0); a["date"]=a0["date"]+timedelta(days=s)
        b,_=anchor_block(q,a)
        v=int(any(cstem(x,q["answer"]) for x in b)); agg[s]+=v; line[str(s)]=v
    per.append(line)
n=len(rows)
print("true anchor      :", agg["true"], "/", n)
for s in shifts:
    print(f"anchor shifted {s:+4d}d: {agg[s]} / {n}   ({agg[s]/n:.3f})")
print()
print("mean over 8 wrong anchors:", sum(agg[s] for s in shifts)/(8*n))
json.dump(per, open("/root/agent_memory_iclr/vfy2/null_rows.json","w"), indent=1)
