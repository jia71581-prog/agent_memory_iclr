import json,sys,re,math,random
sys.path.insert(0,'/root/agent_memory_iclr/code')
from v5_guards import is_decline, DECLINE_RE, nz
R='/root/agent_memory_iclr/results/'
ARMS=['qc','qc_no_qc','qc_no_lifecycle','full','decay','qc_view_only','fifo']
D={a:{r['qid']:r for r in map(json.loads,open(R+'full500_%s.jsonl'%a))} for a in ARMS}
QIDS=list(D['qc'])
AN=[q for q in QIDS if not q.endswith('_abs')]; AB=[q for q in QIDS if q.endswith('_abs')]

def mcnemar(b,c):
    n=b+c
    if n==0: return 1.0
    p=sum(math.comb(n,k) for k in range(0,min(b,c)+1))/2**n*2
    return min(1.0,p)

print("=== paired tests, qc vs qc_no_qc, on the 470 answerable ===")
for lab,fn in [('decline (theirs)', lambda r: is_decline(r['pred'])),
               ('harmful decline (decline & wrong)', lambda r: is_decline(r['pred']) and not r['judge']),
               ('judge correct', lambda r: bool(r['judge']))]:
    b=sum(1 for q in AN if fn(D['qc'][q]) and not fn(D['qc_no_qc'][q]))
    c=sum(1 for q in AN if fn(D['qc_no_qc'][q]) and not fn(D['qc'][q]))
    print(f"  {lab:36s} qc-only={b:3d} noqc-only={c:3d}  exact McNemar p={mcnemar(b,c):.3f}")
print("=== abs subset, 30 questions ===")
for a in ARMS:
    ab=[D[a][q] for q in AB]
    print(f"  {a:16s} judge {sum(r['judge'] for r in ab):2d}/30")
b=sum(1 for q in AB if D['qc'][q]['judge'] and not D['full'][q]['judge'])
c=sum(1 for q in AB if D['full'][q]['judge'] and not D['qc'][q]['judge'])
print(f"  qc vs full on abs: qc-only={b} full-only={c} p={mcnemar(b,c):.3f}")
b=sum(1 for q in AB if D['qc'][q]['judge'] and not D['fifo'][q]['judge'])
c=sum(1 for q in AB if D['fifo'][q]['judge'] and not D['qc'][q]['judge'])
print(f"  qc vs fifo on abs: qc-only={b} fifo-only={c} p={mcnemar(b,c):.3f}")

print("\n=== monotonicity: decline rate on answerable vs abs accuracy (7 arms) ===")
xs=[];ys=[]
for a in ARMS:
    an=[D[a][q] for q in AN]; ab=[D[a][q] for q in AB]
    x=sum(1 for r in an if is_decline(r['pred']))/470; y=sum(r['judge'] for r in ab)/30
    xs.append(x);ys.append(y); print(f"  {a:16s} rate={x:.3f} absAcc={y:.3f}")
def spear(a,b):
    def rk(v):
        s=sorted(range(len(v)),key=lambda i:v[i]); r=[0]*len(v)
        for j,i in enumerate(s): r[i]=j+1
        return r
    ra,rb=rk(a),rk(b); n=len(a)
    ma=sum(ra)/n; mb=sum(rb)/n
    num=sum((x-ma)*(y-mb) for x,y in zip(ra,rb))
    den=math.sqrt(sum((x-ma)**2 for x in ra)*sum((y-mb)**2 for y in rb))
    return num/den
print(f"  Spearman rho = {spear(xs,ys):.3f} (n=7 arms)")

print("\n=== share of each arm's ANSWERABLE errors that are declines ===")
for a in ARMS:
    an=[D[a][q] for q in AN]
    err=[r for r in an if not r['judge']]
    dec=[r for r in err if is_decline(r['pred'])]
    print(f"  {a:16s} answerable errors {len(err):3d}, of which declines {len(dec):3d} ({len(dec)/max(1,len(err)):.2f})")
