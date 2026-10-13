import json,sys
sys.path.insert(0,"/root/agent_memory_iclr/code"); sys.path.insert(0,"/root/agent_memory_iclr/results")
import _recon_ctx as RC
from common import load_dataset
from run_baseline import bm25_rank
data={q['question_id']:q for q in load_dataset("s_cleaned")}
PROBE={'e3038f8c':['25 rare coins','5 rare books'],'67e0d0f2':['8 edX','previous 8'],
       'gpt4_1e4a8aec':['12 new tomato','tomato saplings'],'6cb6f249':['10-day'],
       '80ec1f4f':['Modern Art Museum'],'f35224e0':['My Favorite Murder']}
for qid,pr in PROBE.items():
    q=data[qid]; units=RC.units_of(q); docs=[u[2] for u in units]
    o=bm25_rank(q['question'],docs)
    t20=" ".join(docs[i] for i in o[:20]); t50=" ".join(docs[i] for i in o[:50]); t100=" ".join(docs[i] for i in o[:100])
    print(qid,'gold=',str(q['answer'])[:40])
    for p in pr:
        print('    "%s": raw20=%s raw50=%s raw100=%s  (chars 20/50=%d/%d)'%(p,p.lower() in t20.lower(),p.lower() in t50.lower(),p.lower() in t100.lower(),len(t20),len(t50)))
