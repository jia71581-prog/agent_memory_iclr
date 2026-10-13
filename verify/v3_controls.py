"""Adversarial test 3: (a) base rate of 'session has >=1 extracted fact' over ALL sessions
(is 0.9944 discriminative or just 'extraction rarely empty'?); (b) random-ranking control for
session coverage; (c) char budget at k=20/30/50; (d) type-level rank correspondence."""
import json, math, re, os, hashlib, random
from collections import Counter
import sys
sys.path.insert(0,'/root/agent_memory_iclr/code')
from lifecycle import _session_text, SESSION_CACHE_DIR

_w=re.compile(r"[a-z0-9']+")
def tok(s): return _w.findall(s.lower())
def bm25_order(query,docs,k1=1.5,b=0.75):
    qt=tok(query); dt=[tok(d) for d in docs]
    avgdl=sum(len(d) for d in dt)/max(1,len(dt)); df=Counter()
    for d in dt: df.update(set(d))
    n=len(dt); sc=[]
    for d in dt:
        tf=Counter(d); s=0.0
        for w in qt:
            if w not in tf: continue
            s+=math.log(1+(n-df[w]+0.5)/(df[w]+0.5))*tf[w]*(k1+1)/(tf[w]+k1*(1-b+b*len(d)/avgdl))
        sc.append(s)
    return sorted(range(n),key=lambda i:-sc[i])

def facts_of(sess):
    t=_session_text(sess)
    if not t: return []
    p=os.path.join(SESSION_CACHE_DIR, hashlib.md5(t.encode()).hexdigest()[:16]+'.json')
    return json.load(open(p)) if os.path.exists(p) else None

data=json.load(open('/root/agent_memory_iclr/data/longmemeval_s_cleaned.json'))
# (a) base rate over ALL unique cached sessions
allf=os.listdir(SESSION_CACHE_DIR)
nonempty=0; nfacts=[]
for fn in allf:
    v=json.load(open(os.path.join(SESSION_CACHE_DIR,fn)))
    nfacts.append(len(v)); nonempty += len(v)>0
print(f"(a) ALL {len(allf)} cached unique sessions: >=1 fact {nonempty}/{len(allf)} = {nonempty/len(allf):.4f}")
print(f"    mean facts/session {sum(nfacts)/len(nfacts):.2f}  median {sorted(nfacts)[len(nfacts)//2]}")

# gold-session fact counts vs non-gold, per question (non-abs)
rng=random.Random(0)
gf=[]; ngf=[]; rand_cov=[]; chars={20:[],30:[],50:[]}; span={20:[],30:[],50:[]}
sess_cov_rand={20:[0,0]}
for q in data:
    if q['question_id'].endswith('_abs'): continue
    sid2i={s:i for i,s in enumerate(q['haystack_session_ids'])}
    gold=set(sid2i[s] for s in q['answer_session_ids'] if s in sid2i)
    units=[]
    for si,sess in enumerate(q['haystack_sessions']):
        for t in sess:
            if t.get('content'): units.append((si,t['content']))
    order=bm25_order(q['question'],[u[1] for u in units])
    for K in (20,30,50):
        top=order[:K]
        chars[K].append(sum(len(units[i][1]) for i in top))
        span[K].append(len({units[i][0] for i in top}))
    # random control: sample 20 turns uniformly
    r=rng.sample(range(len(units)),20)
    sh={units[i][0] for i in r}
    sess_cov_rand[20][0]+=len(gold&sh); sess_cov_rand[20][1]+=len(gold)
    for si in range(len(q['haystack_sessions'])):
        f=facts_of(q['haystack_sessions'][si])
        if f is None: continue
        (gf if si in gold else ngf).append(len(f))
print(f"(b) random-20-turn control: gold session coverage {sess_cov_rand[20][0]/sess_cov_rand[20][1]:.4f} (BM25 top-20 = 0.9303)")
print(f"(c) mean context chars: k=20 {sum(chars[20])/len(chars[20]):.0f}  k=30 {sum(chars[30])/len(chars[30]):.0f}  k=50 {sum(chars[50])/len(chars[50]):.0f}")
print(f"    mean sessions spanned: k=20 {sum(span[20])/len(span[20]):.2f}  k=30 {sum(span[30])/len(span[30]):.2f}  k=50 {sum(span[50])/len(span[50]):.2f}")
print(f"(d) facts per session: GOLD mean {sum(gf)/len(gf):.2f} (n={len(gf)}, zero {sum(1 for x in gf if x==0)}) | NON-GOLD mean {sum(ngf)/len(ngf):.2f} (n={len(ngf)}, zero-rate {sum(1 for x in ngf if x==0)/len(ngf):.4f})")
