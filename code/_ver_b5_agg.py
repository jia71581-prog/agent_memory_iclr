import json, statistics as st, collections
R="/root/agent_memory_iclr/results"
rows=json.load(open(f"{R}/_verify/b5_repro.json"))
print("n =", len(rows), " ev turns =", sum(r["ev_n"] for r in rows))
roles=collections.Counter()
ssa_asst=0
for r in rows:
    for ro in r["ev_roles"]:
        roles[ro]+=1
        if ro=="assistant" and r["type"]=="single-session-assistant": ssa_asst+=1
print("ev roles:", dict(roles), " assistant-in-SSA:", ssa_asst)
byt=collections.Counter()
for r in rows: byt[r["type"]]+=1
print("types:", dict(byt))
ARMS=["base20","q14u6a","s1","s2","s3","base30","base50",
      "q18u6a","q20u6a","q24u6a","q28u6a","q34u6a","q40u6a","q50u6a","q60u6a",
      "u14only","u20only","u26only","u34only","u44only",
      "s1_random","s2_random","s1_topturn","s1_oracle"]
def agg(sub, arms=ARMS):
    ev=sum(r["ev_n"] for r in sub); n=len(sub)
    out={}
    for a in arms:
        out[a]=(sum(r["arms"][a]["hits"] for r in sub)/max(1,ev),
                st.mean([r["arms"][a]["chars"] for r in sub]),
                st.mean([r["arms"][a]["nturn"] for r in sub]))
    return n, ev, out
SHOW=["base20","q14u6a","s1","s2"]
print("\n=== reproduce reported table (recall | mean chars) ===")
print(f"{'type':28s} {'n':>4s} {'ev':>5s} " + " ".join(f"{a:>9s}" for a in SHOW))
for t in sorted(byt):
    sub=[r for r in rows if r["type"]==t]
    n,ev,o=agg(sub)
    print(f"{t:28s} {n:4d} {ev:5d} " + " ".join(f"{o[a][0]:9.3f}" for a in SHOW))
n,ev,o=agg(rows)
print(f"{'ALL':28s} {n:4d} {ev:5d} " + " ".join(f"{o[a][0]:9.3f}" for a in SHOW))
print(f"{'mean chars':28s} {'':4s} {'':5s} " + " ".join(f"{o[a][1]:9.0f}" for a in SHOW))
print(f"{'mean turns':28s} {'':4s} {'':5s} " + " ".join(f"{o[a][2]:9.1f}" for a in SHOW))
ach=st.mean([r["arms"]["base20"]["achars"] for r in rows]); tch=o["base20"][1]
print(f"assistant share of top-20 chars: {ach/tch:.3f}")

print("\n=== budget-matched controls (ALL, n=470) ===")
print(f"{'arm':12s} {'recall':>7s} {'chars':>8s} {'turns':>6s}")
for a in ARMS:
    print(f"{a:12s} {o[a][0]:7.3f} {o[a][1]:8.0f} {o[a][2]:6.1f}")

print("\n=== SSP-only arms ===")
ssp=[r for r in rows if r["type"]=="single-session-preference"]
n,ev,o2=agg(ssp)
print(f"n={n} ev={ev}")
for a in ARMS:
    print(f"{a:12s} {o2[a][0]:7.3f} {o2[a][1]:8.0f} {o2[a][2]:6.1f}")
print("\ndup ev strings (gold content also occurring outside gold sessions):",
      sum(r["dup_ev_strings"] for r in rows), "questions affected:",
      sum(1 for r in rows if r["dup_ev_strings"]))
print("gold session rank0 (session BM25):", sum(1 for r in rows if r["gold_sess_rank"]==0), "/", len(rows))
print("SSP gold session rank0:", sum(1 for r in ssp if r["gold_sess_rank"]==0), "/", len(ssp),
      " top3:", sum(1 for r in ssp if r["gold_sess_rank"] is not None and r["gold_sess_rank"]<3))
