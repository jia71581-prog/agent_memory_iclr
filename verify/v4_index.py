"""Adversarial test 4: is the fact store actually a DISCRIMINATIVE session-level index,
or does '0.9944 have >=1 fact' just mean extraction is near-universal? Score chains against
the question and measure gold-session recall at top-3/5/10 sessions. Offline, no API."""
import json, os, sys, hashlib, pickle
sys.path.insert(0,'/root/agent_memory_iclr/code')
from lifecycle import assemble_chains, _session_text, SESSION_CACHE_DIR
import v5_retrieval as V

data=json.load(open('/root/agent_memory_iclr/data/longmemeval_s_cleaned.json'))
def facts_of(sess):
    t=_session_text(sess)
    if not t: return []
    p=os.path.join(SESSION_CACHE_DIR, hashlib.md5(t.encode()).hexdigest()[:16]+'.json')
    return json.load(open(p)) if os.path.exists(p) else None

rec={3:[0,0],5:[0,0],10:[0,0],20:[0,0]}
allsess=0; nq=0
for q in data:
    if q['question_id'].endswith('_abs'): continue
    sid2i={s:i for i,s in enumerate(q['haystack_session_ids'])}
    gold=set(sid2i[s] for s in q['answer_session_ids'] if s in sid2i)
    facts=[]
    for si,(date,sess) in enumerate(zip(q['haystack_dates'],q['haystack_sessions'])):
        tri=facts_of(sess)
        if tri is None: continue
        for e,a,v in tri: facts.append({'entity':e,'attribute':a,'value':v,'date':date,'sidx':si})
    store=assemble_chains(facts)
    scored=V.score_chains(store,q['question'])
    ss,_=V.session_scores(store,scored)
    rank=[si for si,_ in sorted(ss.items(),key=lambda kv:-kv[1])]
    for K in rec:
        rec[K][0]+=len(gold & set(rank[:K])); rec[K][1]+=len(gold)
    allsess+=len(q['haystack_sessions']); nq+=1
print("fact-store session index (chain-score -> provenance session ranking), non-abs n=",nq)
for K in sorted(rec):
    print(f"  gold-session recall @ top-{K} sessions = {rec[K][0]}/{rec[K][1]} = {rec[K][0]/rec[K][1]:.4f}"
          f"   (random baseline {K/(allsess/nq):.4f})")
print("  mean sessions/question", round(allsess/nq,2))
