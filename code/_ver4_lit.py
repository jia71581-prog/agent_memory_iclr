"""Strict check of the artifact claim: for each term the note would declare absent,
is the SPACE-JOINED normalisation of the original question token literally a substring
of the normalised memory?  (Only the 54 firing questions -> fast.)"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_guards as G

RES = "/root/agent_memory_iclr/results"
data = {q["question_id"]: q for q in json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))}
rows = json.load(open(f"{RES}/_ver4_a4.json"))
fire = [r for r in rows if r["d12"] and not r["advice"] and not r["meta"]]
print("firing questions:", len(fire), "abs:", sum(1 for r in fire if r["is_abs"]))

nlit = nnot = 0
for r in sorted(fire, key=lambda x: (not x["is_abs"], x["qid"])):
    q = data[r["qid"]]
    hay = " ".join(G.norm(u[2]) for u in G.units_of(q))
    verdict = []
    for t in r["terms"]:
        raws = [w for w in re.findall(r"[A-Za-z0-9'\-]+", q["question"])
                if re.sub(r"[^a-z0-9]", "", w.lower()) == t]
        joined = sorted({G.norm(w) for w in raws}, key=len, reverse=True)
        lit = next((j for j in joined if j and j in hay), None)
        verdict.append((t, joined[:1], lit))
    any_lit = any(v[2] for v in verdict)
    nlit += any_lit
    nnot += not any_lit
    tag = "LITERALLY PRESENT" if any_lit else "absent"
    print(f"  [{'abs' if r['is_abs'] else 'ans'}] {r['qid']:20s} judge={str(r['judge']):5s} "
          f"{tag:18s} " + "; ".join(f"note:'{t}' <- question token {j} -> memory has {l!r}"
                                     for t, j, l in verdict))
print(f"\nfirings where the note's asserted-absent term is literally in memory as a phrase: "
      f"{nlit}/{len(fire)}")
for lab in (True, False):
    sub = [r for r in fire if r["is_abs"] == lab]
    c = 0
    for r in sub:
        q = data[r["qid"]]
        hay = " ".join(G.norm(u[2]) for u in G.units_of(q))
        ok = False
        for t in r["terms"]:
            raws = [w for w in re.findall(r"[A-Za-z0-9'\-]+", q["question"])
                    if re.sub(r"[^a-z0-9]", "", w.lower()) == t]
            if any(G.norm(w) and G.norm(w) in hay for w in raws):
                ok = True
        c += ok
    print(f"  {'abs' if lab else 'answerable'}: {c}/{len(sub)}")
