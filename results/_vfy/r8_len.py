import json, re, statistics
D=json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
W=re.compile(r"[a-z0-9']+")
rows=[(len(W.findall(r["question"].lower())), r["question_id"].endswith("_abs")) for r in D]
print("[R] best length-threshold rule 'len<=k => abstain':")
best=None
for k in range(4,26):
    tp=sum(1 for L,a in rows if L<=k and a); fp=sum(1 for L,a in rows if L<=k and not a)
    if tp==0: continue
    prec=tp/(tp+fp); rec=tp/30
    if best is None or prec>best[1]: best=(k,prec,rec,tp,fp)
    if k in (8,10,12,14): print(f"    len<={k:2d}: recall {tp}/30={rec:.2f} false-fire {fp}/470={fp/470:.3f} precision={prec:.3f}")
print(f"    best precision over all k: k={best[0]} precision={best[1]:.3f} recall={best[2]:.2f} (tp={best[3]}, fp={best[4]})")
