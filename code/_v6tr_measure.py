import sys, json, time; sys.path.insert(0,'/root/agent_memory_iclr/code')
from common import load_dataset
from _v6tr_v5ctx import v5_context
from v6_tr import tr_scaffold, tr_gate, tr_trim_raw
data=load_dataset('s_cleaned')
rows={r['qid']:r for r in (json.loads(l) for l in open('/root/agent_memory_iclr/results/full500_qc_v5.jsonl'))}
out=[]; t0=time.time()
for i,q in enumerate(data):
    qid=q['question_id']
    try:
        ctx,store,units,sel,mode=v5_context(q)
    except Exception as e:
        print(i,qid,'CTXERR',e,flush=True); continue
    blk=tr_scaffold(q)
    g=sorted(tr_gate(q['question']))
    raw_start=ctx.find('### Retrieved raw conversation turns')
    raw=ctx[raw_start+len('### Retrieved raw conversation turns')+1:] if raw_start>=0 else ''
    trimmed=tr_trim_raw(q['question'], raw, len(blk)+2) if blk else raw
    out.append({'qid':qid,'type':q['question_type'],'judge':rows.get(qid,{}).get('judge'),
                'gate':g,'v5':len(ctx),'blk':len(blk),'raw':len(raw),'raw_trim':len(trimmed)})
    if (i+1)%50==0: print(f'{i+1}/500 {time.time()-t0:.0f}s',flush=True)
json.dump(out, open('/tmp/measure500.json','w'))
print('done',len(out),time.time()-t0)
