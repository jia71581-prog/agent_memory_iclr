import json, math, re, difflib
R="/root/agent_memory_iclr/results/"
def load(t): return {r["qid"]: r for r in (json.loads(l) for l in open(R+f"full500_{t}.jsonl"))}
qc,nq=load("qc"),load("qc_no_qc")
qids=list(qc)
W=re.compile(r"[a-z0-9]+")
def toks(s): return [t for t in W.findall(s.lower()) if t not in ("a","an","the")]
def f1(p,g):
    p,g=toks(p),toks(g)
    if not p or not g: return 0.0
    pc,gc={},{}
    for t in p: pc[t]=pc.get(t,0)+1
    for t in g: gc[t]=gc.get(t,0)+1
    ov=sum(min(c,gc.get(t,0)) for t,c in pc.items())
    if not ov: return 0.0
    pr,rc=ov/len(p),ov/len(g)
    return 2*pr*rc/(pr+rc)

diff=[q for q in qids if qc[q]['view_mode']!=nq[q]['view_mode']]
same=[q for q in qids if qc[q]['view_mode']==nq[q]['view_mode']]
for name,ids in [("IDENTICAL-MODE (context byte-identical by construction)",same),
                 ("ROUTED-DIFFERENT (view mode differs)",diff)]:
    exact=sum(1 for q in ids if qc[q]['pred'].strip()==nq[q]['pred'].strip())
    sims=[f1(qc[q]['pred'],nq[q]['pred']) for q in ids]
    ratios=[difflib.SequenceMatcher(None,qc[q]['pred'],nq[q]['pred']).ratio() for q in ids]
    jflip=sum(1 for q in ids if qc[q]['judge']!=nq[q]['judge'])
    f1flip=sum(1 for q in ids if abs(qc[q]['f1']-nq[q]['f1'])>1e-9)
    hi=sum(1 for s in sims if s>0.9); lo=sum(1 for s in sims if s<0.5)
    print(f"\n[{name}] n={len(ids)}")
    print(f"  exact pred match      : {exact}/{len(ids)} = {exact/len(ids):.3f}")
    print(f"  mean pred-pred tokenF1: {sum(sims)/len(sims):.3f}   median={sorted(sims)[len(sims)//2]:.3f}")
    print(f"  mean char SequenceMatcher ratio: {sum(ratios)/len(ratios):.3f}")
    print(f"  pred-pair F1>0.9      : {hi}/{len(ids)} = {hi/len(ids):.3f}")
    print(f"  pred-pair F1<0.5      : {lo}/{len(ids)} = {lo/len(ids):.3f}")
    print(f"  gold-F1 differs       : {f1flip}/{len(ids)} = {f1flip/len(ids):.3f}")
    print(f"  JUDGE FLIP RATE       : {jflip}/{len(ids)} = {jflip/len(ids):.3f}")

# what the overall delta decomposes into
print("\n== decomposition of overall +0.004 (2 questions) ==")
for name,ids in [("identical-mode",same),("routed-different",diff)]:
    n01=sum(1 for q in ids if not nq[q]['judge'] and qc[q]['judge'])
    n10=sum(1 for q in ids if nq[q]['judge'] and not qc[q]['judge'])
    print(f"  {name:18s} net qc-nq = {n01-n10:+d} questions (n01={n01}, n10={n10})")
print("\n== the 2 identical-mode flips (both should be nondeterminism) ==")
for q in same:
    if qc[q]['judge']!=nq[q]['judge']:
        print(f"  qid={q} type={qc[q]['type']} qlabel={qc[q]['qlabel']} qc.judge={qc[q]['judge']} nq.judge={nq[q]['judge']} "
              f"predF1sim={f1(qc[q]['pred'],nq[q]['pred']):.3f} f1 qc={qc[q]['f1']} nq={nq[q]['f1']}")
