import json,sys,re,os
from collections import defaultdict, Counter
sys.path.insert(0,'/root/agent_memory_iclr/code')
from common import load_dataset
from run_baseline import bm25_rank, tokenize
from v5_shape import classify_shape
from multiprocessing import Pool

R='/root/agent_memory_iclr/results'
TOP_K=20

def norm(s): return re.sub(r"[^a-z0-9]+"," ",str(s).lower()).strip()
STOP=("the","and","for","was","were","you","your")
def loose(hay,needle):
    h,n=norm(hay),norm(needle)
    if not n: return False
    if n in h: return True
    toks=[t for t in n.split() if len(t)>2 and t not in STOP]
    if not toks: return False
    return all(t in h for t in toks)
def strict(hay,needle):
    h,n=norm(hay),norm(needle)
    return bool(n) and n in h

# oracle: turn-level has_answer
orc={}
for q in json.load(open('/root/agent_memory_iclr/data/longmemeval_oracle.json')):
    ev=set()
    for sid,sess in zip(q['haystack_session_ids'],q['haystack_sessions']):
        for t in sess:
            if t.get('has_answer') and t.get('content'):
                ev.add((sid, t['role'], norm(t['content'])))
    orc[q['question_id']]=ev

judge={}
for line in open(f'{R}/full500_qc.jsonl'):
    r=json.loads(line)
    if 'judge' in r: judge[r['qid']]=r['judge']

def cov_select(order, units, top_k=TOP_K, per_pass=2):
    best=defaultdict(list)
    for i in order: best[units[i][0]].append(i)
    pos={s:order.index(best[s][0]) for s in best}
    sess_rank=sorted(best,key=lambda s:pos[s])
    cov,ptr=[],0
    while len(cov)<top_k and ptr<per_pass:
        for s in sess_rank:
            if len(cov)>=top_k: break
            if len(best[s])>ptr: cov.append(best[s][ptr])
        ptr+=1
    return cov

def work(q):
    qid=q['question_id']
    shape,fired=classify_shape(q['question'])
    sid2idx={s:i for i,s in enumerate(q['haystack_session_ids'])}
    ev_sess={sid2idx[s] for s in q['answer_session_ids'] if s in sid2idx}
    units=[]
    for si,(date,sess) in enumerate(zip(q['haystack_dates'],q['haystack_sessions'])):
        sid=q['haystack_session_ids'][si]
        for t in sess:
            if t.get('content'): units.append((si,date,t['role'],t['content'],sid))
    order=bm25_rank(q['question'],[u[3] for u in units])
    flat=order[:TOP_K]; cov=cov_select(order,units)
    fs={units[i][0] for i in flat}; cs={units[i][0] for i in cov}
    gold=str(q['answer'])
    ftxt=' '.join(units[i][3] for i in flat); ctxt=' '.join(units[i][3] for i in cov)
    hay=' '.join(u[3] for u in units)
    # oracle evidence turns present in units
    oev=orc.get(qid,set())
    ev_turn_idx={i for i,u in enumerate(units) if (u[4],u[2],norm(u[3])) in oev}
    return dict(qid=qid,shape=shape,type=q['question_type'],fired=fired,
        judge=judge.get(qid), n_sess=len(q['haystack_sessions']), n_units=len(units),
        n_ev_sess=len(ev_sess), ev_sess_ok=(len(ev_sess)==len(q['answer_session_ids'])),
        flat_all=ev_sess<=fs, cov_all=ev_sess<=cs,
        flat_hit=len(ev_sess&fs), cov_hit=len(ev_sess&cs),
        n_flat_sess=len(fs), n_cov_sess=len(cs), flat_sub_cov=fs<=cs,
        flat_chars=sum(len(units[i][3]) for i in flat), cov_chars=sum(len(units[i][3]) for i in cov),
        flat_toks=sum(len(tokenize(units[i][3])) for i in flat),
        cov_toks=sum(len(tokenize(units[i][3])) for i in cov),
        n_oracle_ev_turns=len(oev), n_ev_turns_matched=len(ev_turn_idx),
        ev_turns_flat=len(ev_turn_idx&set(flat)), ev_turns_cov=len(ev_turn_idx&set(cov)),
        evturn_all_flat=bool(ev_turn_idx) and ev_turn_idx<=set(flat),
        evturn_all_cov=bool(ev_turn_idx) and ev_turn_idx<=set(cov),
        evturn_any_flat=bool(ev_turn_idx&set(flat)), evturn_any_cov=bool(ev_turn_idx&set(cov)),
        gold_loose_flat=loose(ftxt,gold), gold_loose_cov=loose(ctxt,gold), gold_loose_hay=loose(hay,gold),
        gold_strict_hay=strict(hay,gold),
        n_goldturn_strict=sum(1 for u in units if strict(u[3],gold)),
        n_goldturn_loose=sum(1 for u in units if loose(u[3],gold)),
        gold=gold, question=q['question'])

if __name__=='__main__':
    data=load_dataset('s_cleaned')
    with Pool(16) as p: rows=p.map(work,data,chunksize=4)
    json.dump(rows,open('/tmp/v2_rows.json','w'),ensure_ascii=False)
    print('saved',len(rows))
