"""Offline context-size reconstruction for all 7 arms (no API)."""
import json,sys,os,math
from datetime import datetime
sys.path.insert(0,'/root/agent_memory_iclr/code')
from common import load_dataset
from lifecycle import render_view
from query_classifier import classify_query
from run_baseline import bm25_rank
from _adv_build import build_store_offline
CAP=20000
def units_of(q):
    u=[]
    for d,s in zip(q['haystack_dates'],q['haystack_sessions']):
        for t in s:
            if t.get('content'): u.append((d,t['role'],t['content']))
    return u
def raw_txt(q,units,k):
    if not units: return ""
    o=sorted(bm25_rank(q['question'],[x[2] for x in units])[:k])
    return "\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in o)
def cap(t):
    if len(t)<=CAP: return t
    kept,tot=[],0
    for L in t.splitlines():
        if tot+len(L)+1>CAP: break
        kept.append(L); tot+=len(L)+1
    return "\n".join(kept)
data=load_dataset('s_cleaned'); out=[]; miss_tot=0
for i,q in enumerate(data):
    U=units_of(q)
    dates=sorted(set(q['haystack_dates']))[-15:]
    Ufifo=[u for u in U if u[0] in set(dates)]
    qd=datetime.strptime(q['question_date'].split()[0],'%Y/%m/%d')
    Udec=[u for u in U if math.exp(-max(0,(qd-datetime.strptime(u[0].split()[0],'%Y/%m/%d')).days)*math.log(2)/30.0)>=0.05]
    st,miss=build_store_offline(q); miss_tot+=miss
    lab,_=classify_query(q['question']); mode='historical' if lab=='historical' else 'current'
    v_cur=cap(render_view(st,mode)); v_hist=cap(render_view(st,'historical'))
    r20=raw_txt(q,U,20)
    out.append({'qid':q['question_id'],'miss':miss,
        'full':len(raw_txt(q,U,30)),'fifo':len(raw_txt(q,Ufifo,30)),'decay':len(raw_txt(q,Udec,30)),
        'qc':len(v_cur)+len(r20),'qc_no_qc':len(v_hist)+len(r20),
        'qc_no_lifecycle':len(r20),'qc_view_only':len(v_cur),
        'n_units':len(U),'n_fifo':len(Ufifo),'n_dec':len(Udec)})
    if (i+1)%100==0: print(' ..',i+1,flush=True)
json.dump(out,open('/root/agent_memory_iclr/results/_vfy/ctxsize.json','w'))
print("cache misses (sessions):",miss_tot)
ARMS=['qc','qc_no_qc','qc_no_lifecycle','full','decay','qc_view_only','fifo']
print(f"\n{'arm':18s} {'mean chars':>11s} {'median':>8s} {'~tokens':>8s}   rel-to-full")
mf=sum(r['full'] for r in out)/len(out)
for a in ARMS:
    v=sorted(r[a] for r in out); m=sum(v)/len(v)
    print(f"{a:18s} {m:11.0f} {v[len(v)//2]:8.0f} {m/4:8.0f}   {m/mf:5.2f}x")
print(f"\nmem units kept: full {sum(r['n_units'] for r in out)/len(out):.0f}  fifo {sum(r['n_fifo'] for r in out)/len(out):.0f}  decay {sum(r['n_dec'] for r in out)/len(out):.0f}")
