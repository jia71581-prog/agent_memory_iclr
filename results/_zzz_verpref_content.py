import json,sys,os,re,hashlib
sys.path.insert(0,'/root/agent_memory_iclr/code')
from common import load_dataset
from lifecycle import assemble_chains, render_view, _session_text, SESSION_CACHE_DIR
from v5_shape import classify_shape
from query_classifier import classify_query
from run_baseline import bm25_rank, tokenize
from collections import defaultdict, Counter
VIEW_CAP=20000; TOP_K=20
STOP=set('the and for was were you your that this with have has had are is a an of to in on it my me i they their them would prefer prefers not might may also such as would be their they''s about some more like when what which how do does did can could should on at or if from by than then there'.split())

def cached(sess):
    t=_session_text(sess)
    if not t: return []
    p=os.path.join(SESSION_CACHE_DIR, hashlib.md5(t.encode()).hexdigest()[:16]+'.json')
    return json.load(open(p)) if os.path.exists(p) else None

orc={}
for q in json.load(open('/root/agent_memory_iclr/data/longmemeval_oracle.json')):
    orc[q['question_id']]=[t['content'] for sess in q['haystack_sessions'] for t in sess if t.get('has_answer') and t.get('content')]

judge={}
for line in open('/root/agent_memory_iclr/results/full500_qc.jsonl'):
    r=json.loads(line)
    if 'judge' in r: judge[r['qid']]=r['judge']

def cov_select(order,units,top_k=TOP_K,per_pass=2):
    best=defaultdict(list)
    for i in order: best[units[i][0]].append(i)
    pos={s:order.index(best[s][0]) for s in best}
    cov,ptr=[],0
    for _ in range(per_pass):
        for s in sorted(best,key=lambda s:pos[s]):
            if len(cov)>=top_k: break
            if len(best[s])>ptr: cov.append(best[s][ptr])
        ptr+=1
    return cov[:top_k]

data={q['question_id']:q for q in load_dataset('s_cleaned')}
rows=[]
for qid,q in data.items():
    if classify_shape(q['question'])[0]!='PREF.advice': continue
    label,_=classify_query(q['question']); mode='historical' if label=='historical' else 'current'
    facts=[]
    for si,(date,sess) in enumerate(zip(q['haystack_dates'],q['haystack_sessions'])):
        for e,a,v in (cached(sess) or []): facts.append({'entity':e,'attribute':a,'value':v,'date':date,'sidx':si})
    view=render_view(assemble_chains(facts),mode)
    if len(view)>VIEW_CAP:
        kept,tot=[],0
        for ln in view.splitlines():
            if tot+len(ln)+1>VIEW_CAP: break
            kept.append(ln); tot+=len(ln)+1
        view='\n'.join(kept)
    units=[]
    for si,(date,sess) in enumerate(zip(q['haystack_dates'],q['haystack_sessions'])):
        for t in sess:
            if t.get('content'): units.append((si,date,t['role'],t['content']))
    order=bm25_rank(q['question'],[u[3] for u in units])
    flat=sorted(order[:TOP_K]); covi=cov_select(order,units)
    raw_flat='\n'.join(f"[{units[i][1]}] [{units[i][2]}] {units[i][3]}" for i in flat)
    raw_cov='\n'.join(f"[{units[i][1]}] [{units[i][2]}] {units[i][3]}" for i in sorted(covi))
    ctx_flat=view+'\n'+raw_flat; ctx_cov=view+'\n'+raw_cov
    # key tokens = rubric tokens grounded in the oracle evidence turn(s), minus question tokens
    evtxt=' '.join(orc[qid])
    qtok=set(tokenize(q['question']))
    rt=set(t for t in tokenize(str(q['answer'])) if len(t)>2 and t not in STOP)
    et=set(tokenize(evtxt))
    key=sorted((rt & et) - qtok - STOP)
    # document frequency of each key token across sessions (rarity)
    sess_tok=[set(tokenize(' '.join(t['content'] for t in s if t.get('content')))) for s in q['haystack_sessions']]
    df={t:sum(1 for st in sess_tok if t in st) for t in key}
    rare=[t for t in key if df[t]<=2]
    def cvg(ctx,toks):
        c=set(tokenize(ctx))
        return (sum(1 for t in toks if t in c)/len(toks)) if toks else None
    rows.append(dict(qid=qid,judge=judge[qid],n_key=len(key),n_rare=len(rare),
        key_cov_view=cvg(view,key), key_cov_rawflat=cvg(raw_flat,key),
        key_cov_flat=cvg(ctx_flat,key), key_cov_cov=cvg(ctx_cov,key),
        rare_cov_view=cvg(view,rare), rare_cov_rawflat=cvg(raw_flat,rare),
        rare_cov_flat=cvg(ctx_flat,rare), rare_cov_cov=cvg(ctx_cov,rare),
        view_chars=len(view), raw_flat_chars=len(raw_flat), raw_cov_chars=len(raw_cov),
        ctx_flat_toks=len(tokenize(ctx_flat)), ctx_cov_toks=len(tokenize(ctx_cov)),
        key=key, rare=rare))
json.dump(rows,open('/tmp/v4_content.json','w'))
import statistics as st
def m(g,k): 
    v=[r[k] for r in g if r[k] is not None]
    return round(st.mean(v),3) if v else None
ok=[r for r in rows if r['judge']]; bad=[r for r in rows if not r['judge']]
print('n=',len(rows),'correct',len(ok),'wrong',len(bad))
print('mean n_key',round(st.mean(r['n_key'] for r in rows),1),'mean n_rare',round(st.mean(r['n_rare'] for r in rows),1))
for k in ['key_cov_view','key_cov_rawflat','key_cov_flat','key_cov_cov','rare_cov_view','rare_cov_rawflat','rare_cov_flat','rare_cov_cov']:
    print(f'{k:16s} ALL {m(rows,k)}  correct {m(ok,k)}  wrong {m(bad,k)}')
print()
print('per-question rare-token coverage (grounded rubric specifics):')
for r in sorted(rows,key=lambda r:(r['judge'],r['qid'])):
    print(f"{r['qid']} judge={str(r['judge'])[:5]:5s} n_rare={r['n_rare']:2d} rare_in_view={r['rare_cov_view']} rare_in_raw20={r['rare_cov_rawflat']} rare_in_ctx={r['rare_cov_flat']} rare_in_cov={r['rare_cov_cov']}")
print()
import statistics as st
print('ctx tokens flat mean',int(st.mean(r['ctx_flat_toks'] for r in rows)),'cov mean',int(st.mean(r['ctx_cov_toks'] for r in rows)))
