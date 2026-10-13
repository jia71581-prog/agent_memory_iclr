import json, re, sys, collections
sys.path.insert(0,"/root/agent_memory_iclr/code")
from v5_shape import classify_shape
D=json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
NUMW={"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
      "eleven":11,"twelve":12,"fifteen":15,"twenty":20,"a":1,"an":1,"half":0.5}
G=re.compile(r"(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)\s*(?:and a half\s*)?(second|minute|hour|day|night|week|month|year)s?",re.I)
BARE=re.compile(r"(\d+(?:\.\d+)?)")
QUNIT=re.compile(r"\b(seconds?|minutes?|hours?|days?|nights?|weeks?|months?|years?)\b",re.I)
n=0
for q in D:
    sh,f=classify_shape(q["question"])
    if sh!="TEMP.offset": continue
    if q["question_id"].endswith("_abs"): continue
    n+=1
    a=str(q["answer"])
    uv=[(x.group(1).lower(),x.group(2).lower()) for x in G.finditer(a)]
    qu=[u.lower().rstrip('s') for u in QUNIT.findall(q["question"].lower())]
    print(f"{n:3d} {q['question_id'][:14]:15s} {q['question_type'][:20]:20s} qunit={qu} goldparse={uv} bare={BARE.findall(a)[:4]}")
    print(f"      Q: {q['question'][:135]}")
    print(f"      A: {a[:135]}")
print("non-abs TEMP.offset n =",n)
