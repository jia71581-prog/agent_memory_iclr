import json,sys,re
sys.path.insert(0,'/root/agent_memory_iclr/code')
from v5_guards import is_decline
R='/root/agent_memory_iclr/results/'
ARMS=['qc','qc_no_qc','qc_no_lifecycle','full','decay','qc_view_only','fifo']
D={a:{r['qid']:r for r in map(json.loads,open(R+'full500_%s.jsonl'%a))} for a in ARMS}
Q={q['question_id']:q for q in json.load(open('/root/agent_memory_iclr/data/longmemeval_s_cleaned.json'))}
# the graded-CORRECT false declines: are they "partial declines that answered part"?
for a in ['qc','qc_no_qc','qc_no_lifecycle','full','decay']:
    rows=[r for r in D[a].values() if not r['qid'].endswith('_abs') and is_decline(r['pred']) and r['judge']]
    print("="*100); print(a, len(rows))
    for r in rows:
        q=Q[r['qid']]
        print(f"--- {r['qid']} [{r['type']}] f1={r['f1']}")
        print("   Q:", q['question'][:160].replace('\n',' '))
        print("   GOLD:", str(q['answer'])[:120].replace('\n',' '))
        print("   PRED:", r['pred'][:400].replace('\n',' '))
