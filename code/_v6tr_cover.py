import os, sys, json, re; sys.path.insert(0,'/root/agent_memory_iclr/code')
import importlib, v6_tr; importlib.reload(v6_tr)
from v6_tr import tr_scaffold, tr_gate
def norm(s): return re.sub(r'[^a-z0-9]','',s.lower())
CACHE='/root/agent_memory_iclr/results/_v6tr_bad12.json'
if not os.path.exists(CACHE):
    from common import load_dataset
    rows_=[json.loads(l) for l in open('/root/agent_memory_iclr/results/full500_qc_v5.jsonl')]
    ids={r['qid'] for r in rows_ if r['type']=='temporal-reasoning' and not r['judge']}
    json.dump([q for q in load_dataset('s_cleaned') if q['question_id'] in ids], open(CACHE,'w'))
bad=json.load(open(CACHE))
show = sys.argv[1:] if len(sys.argv)>1 else None
tot=cov=0
for q in bad:
    blk=tr_scaffold(q)
    golds=[]
    for si,(d,s) in enumerate(zip(q['haystack_dates'],q['haystack_sessions'])):
        for t in s:
            if t.get('has_answer') and t.get('role')!='assistant': golds.append((si,d,t['content']))
    nb=norm(blk); res=[]
    for si,d,c in golds:
        sents=[x for x in re.split(r'(?<=[.!?])\s+', c) if len(x)>20]
        ok=any(norm(x)[:45] in nb for x in sents)
        res.append((si,d.split()[0],ok)); tot+=1; cov+=ok
    print(f"{q['question_id']:16s} gate={','.join(sorted(tr_gate(q['question']))):22s} chars={len(blk):5d} "
          f"gold_in_block={sum(r[2] for r in res)}/{len(res)} " + " ".join(f"s{a}:{b}:{'Y' if c else 'N'}" for a,b,c in res))
    if show and q['question_id'] in show:
        print(blk); print('-'*100)
print(f'TOTAL gold evidence turns in block: {cov}/{tot}')
