import json, re, sys, itertools, collections
from datetime import datetime
sys.path.insert(0,"/root/agent_memory_iclr/code")
from v5_shape import classify_shape
D=json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
R="/root/agent_memory_iclr/results"
def pd(s): return datetime.strptime(s.split(" (")[0],"%Y/%m/%d")
NUM=re.compile(r"(\d+(?:\.\d+)?)")
rows={}
for q in D:
    if classify_shape(q["question"])[0]!="TEMP.offset" or q["question_id"].endswith("_abs"): continue
    sid2d=dict(zip(q["haystack_session_ids"],q["haystack_dates"]))
    evd=sorted({pd(sid2d[s]) for s in q["answer_session_ids"] if s in sid2d})
    dg={float(x) for x in NUM.findall(str(q["answer"]))}
    rows[q["question_id"]]=dict(dg=dg,evd=evd,qd=pd(q["question_date"]))
def deriv(r,tol=1):
    A={abs((r["qd"]-e).days) for e in r["evd"]}
    B={abs((a-b).days) for a,b in itertools.combinations(r["evd"],2)}
    def ok(v): return (any(abs(v-x)<=tol for x in r["dg"]) or any(abs(v/7-x)<=0.5 for x in r["dg"])
                       or any(abs(v/30-x)<=0.5 for x in r["dg"]))
    return any(ok(v) for v in A|B)
elig={k for k,r in rows.items() if r["dg"] and r["evd"]}
print("non-abs total in dataset:",sum(1 for q in D if not q["question_id"].endswith("_abs")))
print("eligible:",len(elig),"derivable:",sum(deriv(rows[k]) for k in elig))
for p in ["full500_qc.jsonl","full500_full_p2.jsonl","full500_full.jsonl","full500_qc_no_qc.jsonl",
          "full500_qc_no_lifecycle.jsonl","full500_qc_view_only.jsonl","full500_fifo.jsonl"]:
    J={json.loads(l)["qid"]:json.loads(l)["judge"] for l in open(f"{R}/{p}")}
    err=[k for k in rows if not J.get(k,True)]
    ed=[k for k in err if k in elig and deriv(rows[k])]
    print(f"  {p:34s} offset errors={len(err):2d}  eligible&derivable errors={len(ed):2d} {ed}")
