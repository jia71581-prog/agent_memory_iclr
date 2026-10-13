import json,sys,re
sys.path.insert(0,'/root/agent_memory_iclr/code')
from v5_guards import is_decline, DECLINE_RE, nz
R='/root/agent_memory_iclr/results/'
D={a:{r['qid']:r for r in map(json.loads,open(R+'full500_%s.jsonl'%a))} for a in ['qc','qc_no_qc']}
Q={q['question_id']:q for q in json.load(open('/root/agent_memory_iclr/data/longmemeval_s_cleaned.json'))}
def goldhit(qid,pred):
    g=str(Q[qid]['answer']).lower()
    toks=[t for t in re.findall(r"[a-z0-9$.,'-]{3,}",g) if t not in ('the','user','would','prefer','responses','that','their')]
    if not toks: return False
    p=pred.lower().replace(',','')
    return sum(1 for t in toks if t.replace(',','') in p)/len(toks)>=0.6
for a in ['qc','qc_no_qc']:
    fp=[r for r in D[a].values() if not r['qid'].endswith('_abs') and is_decline(r['pred'])]
    print("="*96); print(a,"answerable declines:",len(fp))
    hard=lead=late=0
    for r in sorted(fp,key=lambda x:-x['judge']):
        m=DECLINE_RE.search(nz(r['pred'])); pos=m.start(); L=len(r['pred'])
        kind='LATE-CAVEAT' if pos>0.4*L else 'LEADING'
        if r['judge']:
            print(f"  judge=1 {kind:12s} pos={pos:4d}/{L:<5d} goldInPred={goldhit(r['qid'],r['pred'])} {r['qid']:16s} | {r['pred'][:110].replace(chr(10),' ')}")
        if not r['judge']: hard+=1
    print(f"  -> hard (decline & graded wrong) = {hard}; scored-anyway = {len(fp)-hard}")
