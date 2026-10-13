import json, collections, statistics as st, math
R="/root/agent_memory_iclr/results"
rows={r["qid"]: r for r in json.load(open(f"{R}/_verify/b5_repro.json"))}
qc={}
for line in open(f"{R}/full500_qc.jsonl"):
    d=json.loads(line)
    if "judge" in d: qc[d["qid"]]=d
print("qc rows", len(qc))

# 1. does base20 evidence-turn recall predict judge?
print("\n=== base20 per-question evidence recall vs judge ===")
print(f"{'type':28s} {'n':>4s} {'acc':>6s} {'rec|correct':>12s} {'rec|wrong':>10s} {'full|corr':>10s} {'full|wrong':>11s}")
def rec(r,a): return r["arms"][a]["hits"]/max(1,r["ev_n"])
allc=[];allw=[]
for t in sorted({r["type"] for r in rows.values()}):
    sub=[r for r in rows.values() if r["type"]==t and r["qid"] in qc]
    c=[r for r in sub if qc[r["qid"]]["judge"]]; w=[r for r in sub if not qc[r["qid"]]["judge"]]
    allc+=c; allw+=w
    fc=sum(1 for r in c if rec(r,"base20")>=1.0)/max(1,len(c))
    fw=sum(1 for r in w if rec(r,"base20")>=1.0)/max(1,len(w))
    print(f"{t:28s} {len(sub):4d} {len(c)/max(1,len(sub)):6.3f} "
          f"{st.mean([rec(r,'base20') for r in c]) if c else float('nan'):12.3f} "
          f"{st.mean([rec(r,'base20') for r in w]) if w else float('nan'):10.3f} {fc:10.3f} {fw:11.3f}")
print(f"{'ALL':28s} {len(allc)+len(allw):4d} {len(allc)/(len(allc)+len(allw)):6.3f} "
      f"{st.mean([rec(r,'base20') for r in allc]):12.3f} {st.mean([rec(r,'base20') for r in allw]):10.3f} "
      f"{sum(1 for r in allc if rec(r,'base20')>=1)/len(allc):10.3f} {sum(1 for r in allw if rec(r,'base20')>=1)/len(allw):11.3f}")

# 2x2 on full-evidence-present vs judge
a=sum(1 for r in allc if rec(r,"base20")>=1); b=sum(1 for r in allw if rec(r,"base20")>=1)
c_=len(allc)-a; d_=len(allw)-b
print(f"\nfull-evidence-in-top20: correct {a} wrong {b} -> acc {a/(a+b):.3f}")
print(f"partial/none         : correct {c_} wrong {d_} -> acc {c_/(c_+d_):.3f}")
try:
    from scipy.stats import fisher_exact
    print("fisher p =", fisher_exact([[a,b],[c_,d_]])[1])
except Exception as e: print("(scipy unavailable)", e)

# 2. SSP failures: is missing evidence the cause?
print("\n=== SSP questions: judge vs base20/s1/s2 recall ===")
ssp=[r for r in rows.values() if r["type"]=="single-session-preference"]
F={"0edc2aef","35a27287","32260d93","195a1a1b","09d032c9","75f70248","d6233ab6","a89d7624","1d4e3b97"}
actual_fail={r["qid"] for r in ssp if r["qid"] in qc and not qc[r["qid"]]["judge"]}
print("claimed failure set == actual qc failures:", F==actual_fail, "| actual:", sorted(actual_fail))
print("claimed-not-actual:", sorted(F-actual_fail), " actual-not-claimed:", sorted(actual_fail-F))
print(f"{'qid':10s} {'judge':>5s} {'ev':>3s} {'b20':>5s} {'q14':>5s} {'s1':>5s} {'s2':>5s} {'s3':>5s} {'goldSessRank':>12s}")
for r in sorted(ssp, key=lambda x: (x["qid"] not in actual_fail, x["qid"])):
    j = qc[r["qid"]]["judge"] if r["qid"] in qc else None
    print(f"{r['qid']:10s} {str(j):>5s} {r['ev_n']:3d} " + " ".join(f"{rec(r,a):5.2f}" for a in ("base20","q14u6a","s1","s2","s3")) + f" {str(r['gold_sess_rank']):>12s}")
cc=[r for r in ssp if r["qid"] in qc and qc[r["qid"]]["judge"]]
ww=[r for r in ssp if r["qid"] in qc and not qc[r["qid"]]["judge"]]
for a in ("base20","q14u6a","s1","s2","s3"):
    print(f"  {a:8s} mean recall correct={st.mean([rec(r,a) for r in cc]):.3f} wrong={st.mean([rec(r,a) for r in ww]):.3f}")

# 3. gain concentration: which questions gain evidence s1 vs base20, and are they already correct?
print("\n=== who gains, and were they already right? ===")
gain=[r for r in rows.values() if r["arms"]["s1"]["hits"]>r["arms"]["base20"]["hits"] and r["qid"] in qc]
loss=[r for r in rows.values() if r["arms"]["s1"]["hits"]<r["arms"]["base20"]["hits"] and r["qid"] in qc]
print(f"questions gaining >=1 ev turn: {len(gain)} of 470; already correct: {sum(1 for r in gain if qc[r['qid']]['judge'])} ({sum(1 for r in gain if qc[r['qid']]['judge'])/max(1,len(gain)):.2%})")
print(f"questions losing  >=1 ev turn: {len(loss)}; of which correct now: {sum(1 for r in loss if qc[r['qid']]['judge'])}")
print("  losers:", [(r["qid"], r["type"], r["arms"]["base20"]["hits"], r["arms"]["s1"]["hits"], qc[r["qid"]]["judge"]) for r in loss][:20])
# questions currently WRONG that gain evidence -> the only ones that could flip
flip=[r for r in gain if not qc[r["qid"]]["judge"]]
print(f"\ncurrently-wrong questions that gain evidence (s1): {len(flip)} -> max possible accuracy gain {len(flip)/470:.3%}")
cnt=collections.Counter(r["type"] for r in flip); print("  by type:", dict(cnt))
flip2=[r for r in rows.values() if r["qid"] in qc and not qc[r["qid"]]["judge"] and r["arms"]["s2"]["hits"]>r["arms"]["base20"]["hits"]]
print(f"s2: {len(flip2)} -> max {len(flip2)/470:.3%}; by type: {dict(collections.Counter(r['type'] for r in flip2))}")
# and full-evidence transitions
f1=[r for r in rows.values() if r["qid"] in qc and not qc[r["qid"]]["judge"] and rec(r,"base20")<1 and rec(r,"s2")>=1]
print(f"wrong & partial->FULL evidence under s2: {len(f1)} ({len(f1)/470:.2%}) by type {dict(collections.Counter(r['type'] for r in f1))}")
