import json, re, sys, collections
sys.path.insert(0,"/root/agent_memory_iclr/code")
from v5_shape import classify_shape
D=json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
print("dataset n =",len(D))
c=collections.Counter()
off=[]
for q in D:
    sh,f=classify_shape(q["question"])
    c[sh]+=1
    if sh=="TEMP.offset": off.append((q,f))
for k,v in c.most_common(): print(f"  {k:20s} {v}")
print("TEMP.offset n =",len(off))
print("abs among them:",sum(q["question_id"].endswith("_abs") for q,_ in off))
print("types:",collections.Counter(q["question_type"] for q,_ in off))
# gold numeric?
NUMW={"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
      "eleven":11,"twelve":12,"fifteen":15,"twenty":20,"a":1,"an":1}
G=re.compile(r"(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)\s*(?:and a half\s*)?(day|night|week|month|year)s?",re.I)
BARE=re.compile(r"\b(\d+(?:\.\d+)?)\b")
rows=[]
for q,f in off:
    a=str(q["answer"])
    m=[(x.group(1).lower(),x.group(2).lower()) for x in G.finditer(a)]
    bare=BARE.findall(a)
    sid2d=dict(zip(q["haystack_session_ids"],q["haystack_dates"]))
    evd=[sid2d.get(s) for s in (q.get("answer_session_ids") or [])]
    rows.append(dict(qid=q["question_id"],ans=a,unitvals=m,bare=bare,nev=len(evd),
                     nev_found=sum(x is not None for x in evd),fired=f,qtype=q["question_type"]))
print("\n-- gold shape breakdown --")
print("with unit+value match :",sum(1 for r in rows if r["unitvals"]))
print("bare number only     :",sum(1 for r in rows if not r["unitvals"] and r["bare"]))
print("no number at all     :",sum(1 for r in rows if not r["unitvals"] and not r["bare"]))
print("ev sessions all found:",sum(1 for r in rows if r["nev"]==r["nev_found"] and r["nev"]>0))
print("ev sessions missing  :",[ (r["qid"],r["nev"],r["nev_found"]) for r in rows if r["nev"]!=r["nev_found"] or r["nev"]==0])
print("\n-- no-number golds --")
for r in rows:
    if not r["unitvals"] and not r["bare"]: print("  ",r["qid"],r["qtype"],"|",r["ans"][:110])
print("\n-- bare-number golds --")
for r in rows:
    if not r["unitvals"] and r["bare"]: print("  ",r["qid"],r["qtype"],"|",r["ans"][:110])
json.dump([r for r in rows],open("/root/agent_memory_iclr/verify/off_rows.json","w"))
