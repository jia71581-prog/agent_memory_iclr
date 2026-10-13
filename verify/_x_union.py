import json, os, sys, re, hashlib, collections
sys.path.insert(0,"/root/agent_memory_iclr/code")
import lifecycle as LC
from run_baseline import bm25_rank
from query_classifier import classify_query
R="/root/agent_memory_iclr/results"
def norm(s): return re.sub(r"[^a-z0-9]+"," ",str(s).lower()).strip()
def contains(hay,needle):
    h,n=norm(hay),norm(needle)
    if not n: return False
    if n in h: return True
    toks=[t for t in n.split() if len(t)>2 and t not in ("the","and","for","was","were","you","your")]
    return bool(toks) and all(t in h for t in toks)
data={q["question_id"]:q for q in json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))}
rows={r["qid"]:r for r in json.load(open("/root/agent_memory_iclr/verify/_x_rows.json"))}
out=[]
for qid,r in rows.items():
    q=data[qid]; g=q["answer"]
    units=[]
    for date,sess in zip(q["haystack_dates"],q["haystack_sessions"]):
        for t in sess:
            if t.get("content"): units.append((date,t["role"],t["content"]))
    order=sorted(bm25_rank(q["question"],[u[2] for u in units])[:20])
    raw="\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in order)
    raw_ch=len(raw)
    r=dict(r); r["in_raw20"]=contains(raw,g); r["raw_chars"]=raw_ch
    r["gold_norm_len"]=len(norm(g)); r["gold_is_int"]=str(g).strip().isdigit()
    out.append(r)
json.dump(out,open("/root/agent_memory_iclr/verify/_x_union.json","w"))
print("done",len(out))
