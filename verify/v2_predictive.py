"""Adversarial test 2: does the coverage metric predict correctness? Is the session/turn
gap an artifact of has_answer being flagged on BOTH members of a user/assistant pair?"""
import json, math, re
from collections import Counter, defaultdict
_w = re.compile(r"[a-z0-9']+")
def tok(s): return _w.findall(s.lower())
def bm25_order(query, docs, k1=1.5, b=0.75):
    qt=tok(query); dt=[tok(d) for d in docs]
    avgdl=sum(len(d) for d in dt)/max(1,len(dt)); df=Counter()
    for d in dt: df.update(set(d))
    n=len(dt); sc=[]
    for d in dt:
        tf=Counter(d); s=0.0
        for w in qt:
            if w not in tf: continue
            s += math.log(1+(n-df[w]+0.5)/(df[w]+0.5))*tf[w]*(k1+1)/(tf[w]+k1*(1-b+b*len(d)/avgdl))
        sc.append(s)
    return sorted(range(n), key=lambda i:-sc[i])

data=json.load(open('data/longmemeval_s_cleaned.json'))
out=[]
for qi,q in enumerate(data):
    if q['question_id'].endswith('_abs'): continue
    sid2i={s:i for i,s in enumerate(q['haystack_session_ids'])}
    gold=set(sid2i[s] for s in q['answer_session_ids'] if s in sid2i)
    units=[]
    for si,sess in enumerate(q['haystack_sessions']):
        for t in sess:
            if t.get('content'):
                units.append((si,t['role'],t['content'],bool(t.get('has_answer'))))
    order=bm25_order(q['question'],[u[2] for u in units])
    row={'qid':q['question_id'],'type':q['question_type']}
    ev=[i for i,u in enumerate(units) if u[3]]
    row['ev_roles']=Counter(units[i][1] for i in ev)
    # per gold session: how many ev turns, roles
    per_sess={}
    for si in gold:
        e=[i for i in ev if units[i][0]==si]
        per_sess[si]={'n_ev':len(e),'roles':[units[i][1] for i in e]}
    row['per_sess']=per_sess
    for K in (20,30,50):
        t=set(order[:K])
        sess_hit={units[i][0] for i in order[:K]}
        # gold sessions hit AT ALL vs hit BY an evidence turn
        hit_any=gold & sess_hit
        hit_by_ev={si for si in gold if any(i in t for i in ev if units[i][0]==si)}
        hit_all_ev={si for si in gold if all(i in t for i in ev if units[i][0]==si)}
        row[f'K{K}']={'n_gold':len(gold),'hit_any':len(hit_any),'hit_by_ev':len(hit_by_ev),
                      'hit_all_ev':len(hit_all_ev),
                      'n_ev':len(ev),'ev_in':sum(1 for i in ev if i in t),
                      'full_ev': len(ev)>0 and all(i in t for i in ev),
                      'any_ev_per_gold_all': len(hit_by_ev)==len(gold),
                      # user-side only variant
                      'full_ev_user': (lambda eu: len(eu)>0 and all(i in t for i in eu))([i for i in ev if units[i][1]=='user']),
                      'any_ev': any(i in t for i in ev)}
    out.append(row)
    if (qi+1)%100==0: print(' ..',qi+1,flush=True)
json.dump(out, open('verify/v2_rows.json','w'))
print('saved',len(out))
