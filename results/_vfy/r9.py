import json,sys,math,re
sys.path.insert(0,'/root/agent_memory_iclr/code')
from v5_guards import is_decline, DECLINE_RE, nz
R='/root/agent_memory_iclr/results/'
ARMS=['qc','qc_no_qc','qc_no_lifecycle','full','decay','qc_view_only','fifo']
D={a:{r['qid']:r for r in map(json.loads,open(R+'full500_%s.jsonl'%a))} for a in ARMS}
QIDS=list(D['qc']); AN=[q for q in QIDS if not q.endswith('_abs')]; AB=[q for q in QIDS if q.endswith('_abs')]
SENT=re.compile(r'(?<=[.!?])\s+')
def lead(p):
    t=nz(p).strip(); return bool(DECLINE_RE.search(SENT.split(t)[0])) if t else False
print("=== detector summary statistics (labels = answerable vs unanswerable) ===")
print(f"{'arm':16s} {'prec':>6s} {'rec':>6s} {'F1':>6s} {'BalAcc':>7s} {'MCC':>7s} | {'BalAcc*':>8s} {'MCC*':>7s} | {'leadBA':>7s}")
for a in ARMS:
    ab=[D[a][q] for q in AB]; an=[D[a][q] for q in AN]
    def stats(tp,fp,fn,tn):
        p=tp/max(1,tp+fp); r=tp/max(1,tp+fn)
        f=2*p*r/max(1e-9,p+r); ba=0.5*(r+tn/max(1,tn+fp))
        num=tp*tn-fp*fn; den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) or 1
        return p,r,f,ba,num/den
    tp=sum(1 for r in ab if is_decline(r['pred'])); fn=30-tp
    fp=sum(1 for r in an if is_decline(r['pred'])); tn=470-fp
    p,r,f,ba,mcc=stats(tp,fp,fn,tn)
    tp2=sum(1 for x in ab if is_decline(x['pred']) and x['judge']); fn2=30-tp2
    fp2=sum(1 for x in an if is_decline(x['pred']) and not x['judge']); tn2=470-fp2
    _,_,_,ba2,mcc2=stats(tp2,fp2,fn2,tn2)
    tpl=sum(1 for x in ab if lead(x['pred'])); fpl=sum(1 for x in an if lead(x['pred']))
    _,_,_,bal,_=stats(tpl,fpl,30-tpl,470-fpl)
    print(f"{a:16s} {p:6.3f} {r:6.3f} {f:6.3f} {ba:7.3f} {mcc:7.3f} | {ba2:8.3f} {mcc2:7.3f} | {bal:7.3f}")

print("\n=== is_decline as a proxy for correctness on the 470 answerable ===")
print(f"{'arm':16s} {'P(correct|decline)':>19s} {'P(correct|no decline)':>22s}")
for a in ARMS:
    an=[D[a][q] for q in AN]
    d=[r for r in an if is_decline(r['pred'])]; nd=[r for r in an if not is_decline(r['pred'])]
    print(f"{a:16s} {sum(r['judge'] for r in d)/max(1,len(d)):19.3f} {sum(r['judge'] for r in nd)/max(1,len(nd)):22.3f}")
