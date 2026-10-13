import json, re, sys, difflib, collections
sys.path.insert(0,"/root/agent_memory_iclr/code")
from v5_shape import classify_shape
D=json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
byid={r["question_id"]:r for r in D}
W=re.compile(r"[A-Za-z0-9']+")
AB=[r for r in D if r["question_id"].endswith("_abs")]
print("[M] token-level edit between abstention question and its qid-paired source:")
ndiff=collections.Counter()
for a in AB:
    b=byid.get(a["question_id"][:-4])
    if not b: print("   NO SOURCE:", a["question_id"], a["question"]); continue
    ta=W.findall(a["question"].lower()); tb=W.findall(b["question"].lower())
    sm=difflib.SequenceMatcher(None, ta, tb)
    ops=[o for o in sm.get_opcodes() if o[0]!="equal"]
    changed=sum(max(o[2]-o[1], o[4]-o[3]) for o in ops)
    ndiff[changed]+=1
print("   distribution of #changed tokens:", dict(sorted(ndiff.items())))
print(f"   <=2 changed tokens: {sum(v for k,v in ndiff.items() if k<=2)}/29 ; <=3: {sum(v for k,v in ndiff.items() if k<=3)}/29")
print()
print("[N] the 3 AGG.arith abstention questions vs their sources:")
for a in AB:
    sh,_=classify_shape(a["question"])
    if sh!="AGG.arith": continue
    b=byid.get(a["question_id"][:-4])
    print(f"   ABS  {a['question_id']}: {a['question']}")
    print(f"        gold: {a['answer'][:150]}")
    if b: print(f"   SRC  {b['question_id']}: {b['question']}")
    if b: print(f"        gold: {b['answer'][:100]}")
    print()
print("[O] 4 shape-disagreement pairs, full text:")
for qid in ("eeda8a6d_abs","ba358f49_abs","982b5123_abs","031748ae_abs"):
    a=byid[qid]; b=byid.get(qid[:-4])
    print("   ABS:", a["question"], "->", classify_shape(a["question"])[0])
    print("   SRC:", b["question"] if b else None, "->", classify_shape(b["question"])[0] if b else None)
    print("   absgold:", a["answer"][:120]); print()
