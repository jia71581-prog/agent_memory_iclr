import json,sys,os,re,hashlib
sys.path.insert(0,'/root/agent_memory_iclr/code')
from common import load_dataset
from lifecycle import assemble_chains, _group_by_session, _norm, _session_text, SESSION_CACHE_DIR
from v5_shape import classify_shape
from query_classifier import classify_query
from run_baseline import bm25_rank
VIEW_CAP=20000

def cached(sess):
    t=_session_text(sess)
    if not t: return []
    p=os.path.join(SESSION_CACHE_DIR, hashlib.md5(t.encode()).hexdigest()[:16]+'.json')
    if not os.path.exists(p): return None
    return json.load(open(p))

def render_with_sidx(store, mode):
    """same as render_view but returns (line, set_of_sidx_rendered)"""
    out=[]
    for k in sorted(store):
        chain=store[k]
        ent,attr=chain[0]['entity'],chain[0]['attribute']
        accum=chain[0]['relation']=='accumulative'
        conflict=any(c['relation']=='contradiction' for c in chain)
        if accum:
            vals='; '.join(f"{c['value']} ({c['date'].split()[0]})" for c in chain)
            out.append((f"{ent} | {attr} = {vals}", {c.get('sidx') for c in chain}))
        elif mode=='current':
            g=_group_by_session(chain); ld,lf=g[-1][0][0],g[-1][1]
            vals='; '.join(c['value'] for c in lf)
            tag=' [CONFLICT: multiple values reported at the same time]' if conflict else ''
            out.append((f"[{ld}] {ent} | {attr} = {vals}{tag}", {c.get('sidx') for c in lf}))
        else:
            g=_group_by_session(chain)
            if len(g)==1:
                vals=' / '.join(c['value'] for c in g[0][1])
                out.append((f"[{g[0][0][0]}] {ent} | {attr} = {vals}", {c.get('sidx') for c in g[0][1]}))
            else:
                steps=' -> '.join(' / '.join(c['value'] for c in gg[1])+f" ({gg[0][0].split()[0]}{'/conflict' if gg[1][0]['relation']=='contradiction' else ''})" for gg in g)
                out.append((f"{ent} | {attr}: {steps}", {c.get('sidx') for c in chain}))
    return out

data={q['question_id']:q for q in load_dataset('s_cleaned')}
res=[]
for qid,q in data.items():
    shape,_=classify_shape(q['question'])
    if shape!='PREF.advice': continue
    label,_=classify_query(q['question'])
    mode='historical' if label=='historical' else 'current'
    sid2idx={s:i for i,s in enumerate(q['haystack_session_ids'])}
    ev={sid2idx[s] for s in q['answer_session_ids'] if s in sid2idx}
    facts=[];miss=False
    for si,(date,sess) in enumerate(zip(q['haystack_dates'],q['haystack_sessions'])):
        tri=cached(sess)
        if tri is None: miss=True;continue
        for e,a,v in tri: facts.append({'entity':e,'attribute':a,'value':v,'date':date,'sidx':si})
    lines=render_with_sidx(assemble_chains(facts),mode)
    # apply deployed 20k line-boundary cap
    kept=[];tot=0
    for ln,sx in lines:
        if tot+len(ln)+1>VIEW_CAP: break
        kept.append((ln,sx)); tot+=len(ln)+1
    n_facts_ev=sum(1 for f in facts if f['sidx'] in ev)
    uncapped_ev_lines=sum(1 for ln,sx in lines if sx & ev)
    capped_ev_lines=sum(1 for ln,sx in kept if sx & ev)
    res.append(dict(qid=qid,mode=mode,miss=miss,n_lines=len(lines),n_kept=len(kept),
        view_chars_uncapped=sum(len(l)+1 for l,_ in lines), view_chars=tot,
        n_facts_total=len(facts), n_facts_ev=n_facts_ev,
        uncapped_ev_lines=uncapped_ev_lines, capped_ev_lines=capped_ev_lines))
json.dump(res,open('/tmp/v3_view.json','w'))
import statistics as st
print('n=',len(res),'cache_missing:',sum(r['miss'] for r in res))
print('mean lines',st.mean(r['n_lines'] for r in res),'kept',st.mean(r['n_kept'] for r in res))
print('view chars uncapped mean',int(st.mean(r['view_chars_uncapped'] for r in res)),'capped',int(st.mean(r['view_chars'] for r in res)))
print('questions with >=1 evidence-session line in UNCAPPED view:',sum(1 for r in res if r['uncapped_ev_lines']>0),'/',len(res))
print('questions with >=1 evidence-session line in DEPLOYED capped view:',sum(1 for r in res if r['capped_ev_lines']>0),'/',len(res))
print('mean ev facts extracted per q:',round(st.mean(r['n_facts_ev'] for r in res),2))
for r in res: print(r['qid'],'kept',r['n_kept'],'/',r['n_lines'],'evlines cap/uncap',r['capped_ev_lines'],'/',r['uncapped_ev_lines'],'evfacts',r['n_facts_ev'])
