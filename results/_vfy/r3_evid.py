"""Is an evidence-side 'missing operand/entity' rule derivable ONLY for AGG.arith,
or generically? Test: content words of the question that never appear in the haystack.
Also: does such a rule over-abstain on answerable questions?"""
import json, re, sys, collections, statistics
sys.path.insert(0,"/root/agent_memory_iclr/code")
from v5_shape import classify_shape
W=re.compile(r"[a-z0-9']+")
STOP=set("""i me my myself we our ours you your yours the a an and or but if of at by for with about
into to from in on off over under again then once here there all any both each few more most other
some such no nor not only own same so than too very can will just don should now do does did is are
was were be been being have has had having it its this that these those what which who whom when
where why how much many long ago last past total number average combined take took get got go going
me's s t""".split())
D=json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
out=[]
for r in D:
    hay=r["haystack_sessions"]
    if isinstance(hay,str):
        toks=set(W.findall(hay.lower()))
    else:
        buf=[]
        for s in hay:
            for t in s: buf.append(t.get("content",""))
        toks=set(W.findall(" ".join(buf).lower()))
    q=[t for t in W.findall(r["question"].lower()) if t not in STOP and not t.isdigit()]
    miss=[t for t in q if t not in toks]
    sh,_=classify_shape(r["question"])
    out.append({"qid":r["question_id"],"abs":r["question_id"].endswith("_abs"),"shape":sh,
                "n_content":len(q),"n_miss":len(miss),"miss":miss})
json.dump(out, open("/root/agent_memory_iclr/results/_vfy/evid_miss.json","w"))
A=[r for r in out if r["abs"]]; N=[r for r in out if not r["abs"]]
print(f"[G] abs: mean missing content words={statistics.mean(r['n_miss'] for r in A):.2f}  "
      f">=1 missing: {sum(r['n_miss']>0 for r in A)}/{len(A)}")
print(f"[G] ans: mean missing content words={statistics.mean(r['n_miss'] for r in N):.2f}  "
      f">=1 missing: {sum(r['n_miss']>0 for r in N)}/{len(N)}")
for k in (1,2,3):
    tp=sum(r['n_miss']>=k for r in A); fp=sum(r['n_miss']>=k for r in N)
    prec=tp/max(1,tp+fp)
    print(f"[G] rule 'n_miss>={k}' -> recall {tp}/30={tp/30:.2f}  false-fire on answerable "
          f"{fp}/470={fp/470:.3f}  precision={prec:.3f}")
# per-shape, AGG.arith specifically
print()
for sh in ("AGG.arith","AGG.enum","LOOKUP.plain","TEMP.offset","TEMP.order","AGG.latest_count","LOOKUP.rate"):
    a=[r for r in A if r["shape"]==sh]; n=[r for r in N if r["shape"]==sh]
    if not a: continue
    tp=sum(r['n_miss']>=1 for r in a); fp=sum(r['n_miss']>=1 for r in n)
    print(f"[H] {sh:18s} abs n={len(a)} fires {tp}  | answerable n={len(n)} false-fires {fp} "
          f"({fp/max(1,len(n)):.3f})  precision={tp/max(1,tp+fp):.3f}")
print()
print("[I] the 3 AGG.arith abstention questions, missing tokens:")
for r in A:
    if r["shape"]=="AGG.arith": print("   ", r["qid"], r["miss"])
