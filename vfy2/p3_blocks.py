"""Dump the actual anchor blocks for inspection, and flag which line matched."""
import pickle, sys, re, json
sys.path.insert(0, "/root/agent_memory_iclr/code")
from v5_temporal import (detect_family, resolve_question_anchors, anchor_block, _stem, day0, parse_date)

tr = {q["question_id"]: q for q in pickle.load(open("/root/agent_memory_iclr/vfy2/tr.pkl","rb"))}
def norm(s): return re.sub(r"[^a-z0-9]+"," ",str(s).lower()).strip()
STOPW=("the","and","for","was","were","you","your","not","did","with")
def cstem(hay,needle):
    h={_stem(w) for w in norm(hay).split()}
    toks=[_stem(t) for t in norm(needle).split() if len(t)>2 and t not in STOPW]
    return bool(toks) and sum(t in h for t in toks)>=max(1,int(round(0.8*len(toks))))

want = sys.argv[1:] or ['71017277','gpt4_1e4a8aec','gpt4_f420262d','gpt4_4929293b','eac54add']
for qid in want:
    q = tr[qid]
    fam,_ = detect_family(q["question"])
    anchors=[a for a in resolve_question_anchors(q["question"], q["question_date"]) if a["conf"]=="high"]
    if not anchors:
        anchors=[a for a in resolve_question_anchors(q["question"], q["question_date"])
                 if a["kind"] in ("offset-coarse","month-anchor")][:1]
    blk,w = anchor_block(q, anchors[0])
    print("="*100)
    print(qid, "| qdate", q["question_date"], "| fam", fam, "| anchor", anchors[0]["date"].date(),
          anchors[0]["kind"], repr(anchors[0]["expr"]), "| window", w)
    print("Q:", q["question"])
    print("GOLD:", q["answer"])
    for i,x in enumerate(blk):
        print(f"  {i+1:2d}{'*' if cstem(x,q['answer']) else ' '} {x}")
