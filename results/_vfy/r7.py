import json,sys,re,random
sys.path.insert(0,'/root/agent_memory_iclr/code')
from v5_guards import is_decline, DECLINE_RE, nz
R='/root/agent_memory_iclr/results/'
D={a:{r['qid']:r for r in map(json.loads,open(R+'full500_%s.jsonl'%a))} for a in ['qc','full','fifo','qc_view_only']}
Q={q['question_id']:q for q in json.load(open('/root/agent_memory_iclr/data/longmemeval_s_cleaned.json'))}
random.seed(7)
for arm,n in [('full',18),('fifo',18),('qc_view_only',14)]:
    fp=[r for r in D[arm].values() if not r['qid'].endswith('_abs') and is_decline(r['pred'])]
    S=random.sample(fp,n)
    print("#"*100); print(arm, "sample", n, "of", len(fp))
    for r in S:
        m=DECLINE_RE.search(nz(r['pred']))
        print(f"-- judge={int(r['judge'])} {r['qid']} match={m.group(0)!r} pos={m.start()}/{len(r['pred'])}")
        print("   GOLD:", str(Q[r['qid']]['answer'])[:90].replace('\n',' '))
        print("   PRED:", r['pred'][:300].replace('\n',' '))
