import json, re, sys, itertools, collections
from datetime import datetime
sys.path.insert(0,"/root/agent_memory_iclr/code")
from v5_shape import classify_shape
D=json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
R="/root/agent_memory_iclr/results"
def load(p):
    o={}
    for l in open(f"{R}/{p}"):
        d=json.loads(l); o[d["qid"]]=d
    return o
arms={p.split("full500_")[1].split(".jsonl")[0]:load(p) for p in
      ["full500_qc.jsonl","full500_full.jsonl","full500_fifo.jsonl","full500_qc_no_lifecycle.jsonl",
       "full500_qc_no_qc.jsonl","full500_qc_view_only.jsonl","full500_full_p2.jsonl"]}
OFF=[q for q in D if classify_shape(q["question"])[0]=="TEMP.offset" and not q["question_id"].endswith("_abs")]
print("non-abs TEMP.offset:",len(OFF))
for a,rows in arms.items():
    err=[q["question_id"] for q in OFF if q["question_id"] in rows and not rows[q["question_id"]]["judge"]]
    print(f"  arm {a:20s} n_rows_covering={sum(1 for q in OFF if q['question_id'] in rows):3d} errors={len(err):2d} {err}")
