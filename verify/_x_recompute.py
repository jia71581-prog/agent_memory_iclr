import json, os, sys, re, hashlib, random, collections
sys.path.insert(0, "/root/agent_memory_iclr/code")
import lifecycle as LC
from run_baseline import bm25_rank
from query_classifier import classify_query

R="/root/agent_memory_iclr/results"
DATA="/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"
_WORD=re.compile(r"[a-z0-9]+")
def norm(s): return re.sub(r"[^a-z0-9]+"," ",str(s).lower()).strip()
STOP=("the","and","for","was","were","you","your")
def contains(hay,needle):
    h,n=norm(hay),norm(needle)
    if not n: return False
    if n in h: return True
    toks=[t for t in n.split() if len(t)>2 and t not in STOP]
    return bool(toks) and all(t in h for t in toks)
def contains_line(lines,needle):
    return any(contains(l,needle) for l in lines)

def cached_store(q):
    facts=[]
    for si,(date,sess) in enumerate(zip(q["haystack_dates"],q["haystack_sessions"])):
        text=LC._session_text(sess)
        if not text: continue
        key=hashlib.md5(text.encode()).hexdigest()[:16]
        p=os.path.join(LC.SESSION_CACHE_DIR,f"{key}.json")
        if not os.path.exists(p): continue
        for e,a,v in json.load(open(p)):
            facts.append({"entity":e,"attribute":a,"value":v,"date":date,"sidx":si})
    return LC.assemble_chains(facts)

def cut_break(lines,budget=20000):
    keep,tot=[],0
    for l in lines:
        if tot+len(l)+1>budget: break
        keep.append(l); tot+=len(l)+1
    return keep,tot
def cut_fill(lines,order,budget=20000):
    keep,tot=[],0
    for i in order:
        if tot+len(lines[i])+1>budget: continue
        keep.append(i); tot+=len(lines[i])+1
    return [lines[i] for i in sorted(keep)],tot

data=json.load(open(DATA))
arms={}
for a in ["qc","qc_view_only","qc_no_lifecycle","full"]:
    p=f"{R}/full500_{a}.jsonl"
    if os.path.exists(p): arms[a]={r["qid"]:r for r in (json.loads(l) for l in open(p))}

rows=[]
for qi,q in enumerate(data):
    if q["question_id"].endswith("_abs"): continue
    store=cached_store(q)
    label,_=classify_query(q["question"])
    mode="historical" if label=="historical" else "current"
    lines=LC.render_view(store,mode).splitlines()
    g=q["answer"]
    full_txt="\n".join(lines); full_chars=sum(len(l)+1 for l in lines)
    a_lines,a_ch=cut_break(lines)
    rel_order=bm25_rank(q["question"],lines)
    r_lines,r_ch=cut_fill(lines,rel_order)
    # CONTROL 1: alphabetical order but same greedy skip-and-continue packing
    f_lines,f_ch=cut_fill(lines,list(range(len(lines))))
    # CONTROL 2: random orders, greedy packing
    rnd=[]
    for seed in range(5):
        idx=list(range(len(lines))); random.Random(1000+seed).shuffle(idx)
        rl,rc=cut_fill(lines,idx); rnd.append((contains("\n".join(rl),g),contains_line(rl,g),len(rl),rc))
    # CONTROL 3: shortest-first greedy (max lines, no relevance)
    s_order=sorted(range(len(lines)),key=lambda i:len(lines[i]))
    s_lines,s_ch=cut_fill(lines,s_order)
    rows.append(dict(
        qid=q["question_id"],type=q["question_type"],mode=mode,
        n_lines=len(lines),full_chars=full_chars,
        capped=full_chars>20000,
        full=contains(full_txt,g), alpha=contains("\n".join(a_lines),g), rel=contains("\n".join(r_lines),g),
        fill=contains("\n".join(f_lines),g), short=contains("\n".join(s_lines),g),
        rnd_blob=sum(x[0] for x in rnd)/5.0, rnd_line=sum(x[1] for x in rnd)/5.0,
        rnd_lines=sum(x[2] for x in rnd)/5.0, rnd_ch=sum(x[3] for x in rnd)/5.0,
        full_L=contains_line(lines,g), alpha_L=contains_line(a_lines,g), rel_L=contains_line(r_lines,g),
        fill_L=contains_line(f_lines,g), short_L=contains_line(s_lines,g),
        a_ch=a_ch,r_ch=r_ch,f_ch=f_ch,s_ch=s_ch,
        a_n=len(a_lines),r_n=len(r_lines),f_n=len(f_lines),s_n=len(s_lines),
        judge_qc=arms.get("qc",{}).get(q["question_id"],{}).get("judge"),
        judge_vo=arms.get("qc_view_only",{}).get(q["question_id"],{}).get("judge"),
    ))
    if qi%100==0: print("...",qi,flush=True)
json.dump(rows,open("/root/agent_memory_iclr/verify/_x_rows.json","w"))
print("n",len(rows))
