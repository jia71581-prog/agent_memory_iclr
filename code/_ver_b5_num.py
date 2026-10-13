import json, collections, statistics as st, sys
sys.path.insert(0,"/root/agent_memory_iclr/code")
data=json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
def units_of(q):
    return [(d,t["role"],t["content"]) for d,s in zip(q["haystack_dates"],q["haystack_sessions"]) for t in s if t.get("content")]
# where could 1086 / user 887 come from?
c=collections.Counter()
for q in data:
    abs_ = q["question_id"].endswith("_abs")
    sid2=dict(zip(q["haystack_session_ids"],q["haystack_sessions"]))
    # (a) gold-session-restricted has_answer
    for sid in q["answer_session_ids"]:
        for t in sid2.get(sid,[]):
            if t.get("has_answer") and t.get("content"):
                c[("goldsess",abs_,t["role"])]+=1
    # (b) all has_answer turns anywhere in haystack
    for s in q["haystack_sessions"]:
        for t in s:
            if t.get("has_answer") and t.get("content"):
                c[("any",abs_,t["role"])]+=1
print("has_answer counts (scope, is_abs, role) -> n")
for k in sorted(c): print("  ",k,c[k])
g_non=sum(v for k,v in c.items() if k[0]=="goldsess" and not k[1])
g_all=sum(v for k,v in c.items() if k[0]=="goldsess")
a_non=sum(v for k,v in c.items() if k[0]=="any" and not k[1])
a_all=sum(v for k,v in c.items() if k[0]=="any")
print(f"totals: goldsess non-abs {g_non}, goldsess all500 {g_all}, any non-abs {a_non}, any all500 {a_all}")

# assistant share of top-20 chars: ratio-of-means vs mean-of-ratios
rows=json.load(open("/root/agent_memory_iclr/results/_verify/b5_repro.json"))
tot=sum(r["arms"]["base20"]["chars"] for r in rows); ach=sum(r["arms"]["base20"]["achars"] for r in rows)
print(f"\nassistant share top-20: ratio-of-means {ach/tot:.4f}  mean-of-per-question-ratios {st.mean([r['arms']['base20']['achars']/max(1,r['arms']['base20']['chars']) for r in rows]):.4f}")
print(f"assistant chars kept: base20 {st.mean([r['arms']['base20']['achars'] for r in rows]):.0f} -> s1 {st.mean([r['arms']['s1']['achars'] for r in rows]):.0f} ({st.mean([r['arms']['s1']['achars'] for r in rows])/st.mean([r['arms']['base20']['achars'] for r in rows]):.2f}x)")

# per-question losses vs base20
for a in ("q14u6a","s1","s2"):
    lose=[r for r in rows if r["arms"][a]["hits"]<r["arms"]["base20"]["hits"]]
    print(f"{a}: questions losing >=1 gold ev turn vs base20: {len(lose)} {[ (r['qid'],r['type']) for r in lose]}")
# rendered-char accounting (each turn gets a [date] [role] header)
hdr=len("[2023/05/30 (Tue) 23:40] [assistant] ")
for a in ("base20","q14u6a","s1","s2","s3"):
    ch=st.mean([r["arms"][a]["chars"] for r in rows]); nt=st.mean([r["arms"][a]["nturn"] for r in rows])
    print(f"  {a:8s} raw {ch:7.0f}  turns {nt:5.1f}  rendered~{ch+nt*hdr:7.0f}")

# conversion rate among k30 gainers that were wrong
R="/root/agent_memory_iclr/results"
def load(p):
    o={}
    for l in open(f"{R}/{p}"):
        d=json.loads(l)
        if "judge" in d: o[d["qid"]]=d
    return o
full=load("full500_full.jsonl"); nolc=load("full500_qc_no_lifecycle.jsonl"); qc=load("full500_qc.jsonl")
idx={r["qid"]:r for r in rows}
g=[q for q in idx if idx[q]["arms"]["base30"]["hits"]>idx[q]["arms"]["base20"]["hits"]]
gw=[q for q in g if not nolc[q]["judge"]]
print(f"\nk30 gainers {len(g)}; wrong@k20 among them {len(gw)}; flipped to right {sum(1 for q in gw if full[q]['judge'])} -> conversion {sum(1 for q in gw if full[q]['judge'])/max(1,len(gw)):.2f}")
try:
    from scipy.stats import fisher_exact
    ng=[q for q in idx if q not in g]
    a=sum(1 for q in g if not nolc[q]["judge"] and full[q]["judge"]); b=sum(1 for q in g if nolc[q]["judge"] and not full[q]["judge"])
    c2=sum(1 for q in ng if not nolc[q]["judge"] and full[q]["judge"]); d2=sum(1 for q in ng if nolc[q]["judge"] and not full[q]["judge"])
    print("flip asymmetry gainers vs non-gainers", [[a,b],[c2,d2]], "fisher p=", fisher_exact([[a,b],[c2,d2]])[1])
except Exception as e: print("scipy?", e)
# SSP flips between the two arms
ssp=[q for q in idx if idx[q]["type"]=="single-session-preference"]
print("SSP k20->k30 flips: w->r", sum(1 for q in ssp if not nolc[q]["judge"] and full[q]["judge"]),
      " r->w", sum(1 for q in ssp if nolc[q]["judge"] and not full[q]["judge"]))
# projected gain for candidate arms using observed conversion among wrong gainers
for a in ("s1","s2","s3"):
    wg=[q for q in idx if idx[q]["arms"][a]["hits"]>idx[q]["arms"]["base20"]["hits"] and not qc[q]["judge"]]
    print(f"candidate {a}: wrong-and-gaining {len(wg)} -> projected +{len(wg)*0.72/470*100:.1f} pp at 72% conversion, ceiling +{len(wg)/470*100:.1f} pp")
