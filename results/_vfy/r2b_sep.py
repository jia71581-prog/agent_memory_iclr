"""numpy-only supervised upper bound on surface separability of abstention."""
import json, re, numpy as np, collections, random
D=json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
qs=[r["question"] for r in D]; y=np.array([1.0 if r["question_id"].endswith("_abs") else 0.0 for r in D])
W=re.compile(r"[a-z0-9']+")
def feats(s):
    t=W.findall(s.lower()); f=list(t)+[a+"_"+b for a,b in zip(t,t[1:])]; return f
DIM=2**14
def vec(s):
    v=np.zeros(DIM)
    for f in feats(s): v[hash(f)%DIM]+=1.0
    n=np.linalg.norm(v); return v/n if n else v
X=np.stack([vec(q) for q in qs])
def train(Xtr,ytr,epochs=400,lr=0.5,l2=1e-3):
    w=np.zeros(Xtr.shape[1]); b=0.0
    pos=ytr.sum(); neg=len(ytr)-pos
    sw=np.where(ytr==1, neg/pos, 1.0)
    for _ in range(epochs):
        z=Xtr@w+b; p=1/(1+np.exp(-z)); g=(p-ytr)*sw
        w-=lr*(Xtr.T@g/len(ytr)+l2*w); b-=lr*g.mean()
    return w,b
rng=np.random.RandomState(0); idx=rng.permutation(len(y)); folds=np.array_split(idx,5)
pred=np.zeros(len(y))
for k in range(5):
    te=folds[k]; tr=np.concatenate([folds[j] for j in range(5) if j!=k])
    w,b=train(X[tr],y[tr]); pred[te]=1/(1+np.exp(-(X[te]@w+b)))
def ap(y,p):
    o=np.argsort(-p); yy=y[o]; c=np.cumsum(yy); prec=c/np.arange(1,len(yy)+1)
    return (prec*yy).sum()/yy.sum()
def auc(y,p):
    o=np.argsort(p); r=np.empty(len(p)); r[o]=np.arange(1,len(p)+1)
    npos=y.sum(); nneg=len(y)-npos
    return (r[y==1].sum()-npos*(npos+1)/2)/(npos*nneg)
o=np.argsort(-pred)
print(f"[E] CV logreg(word+bigram,hashed): AP={ap(y,pred):.3f} (chance {y.mean():.3f}) "
      f"AUC={auc(y,pred):.3f} P@30={y[o[:30]].mean():.3f} P@60={y[o[:60]].mean():.3f} "
      f"recall@60={int(y[o[:60]].sum())}/30")
# permutation null for AUC
null=[]
for s in range(200):
    yp=y.copy(); np.random.RandomState(s).shuffle(yp); null.append(auc(yp,pred))
null=np.array(null)
print(f"[E] permutation null AUC mean={null.mean():.3f} p95={np.percentile(null,95):.3f} "
      f"p(one-sided)={(null>=auc(y,pred)).mean():.3f}")
