import json, sys, statistics as st
from collections import defaultdict
from multiprocessing import Pool
sys.path.insert(0,"/root/agent_memory_iclr/code")
from common import load_dataset
from run_baseline import bm25_rank
rows={r["qid"]:r for r in json.load(open("/root/agent_memory_iclr/verify/adv_cov1.json"))}
LOSE={q for q,r in rows.items() if r["ansturn_any_flat"] and not r["ansturn_any_cov"]}
def work(q):
    qid=q["question_id"]
    if qid not in LOSE: return None
    units,ans=[],[]
    for si,(d,sess) in enumerate(zip(q["haystack_dates"],q["haystack_sessions"])):
        for t in sess:
            if t.get("content"):
                if t.get("has_answer"): ans.append(len(units))
                units.append((si,t["content"]))
    order=bm25_rank(q["question"],[u[1] for u in units])
    rank={i:r for r,i in enumerate(order)}
    out=[]
    for i in ans:
        sib=sorted([j for j,u in enumerate(units) if u[0]==units[i][0]],key=lambda j:rank[j])
        out.append((sib.index(i)+1,len(sib),rank[i]+1))
    return out
if __name__=="__main__":
    data=load_dataset("s_cleaned")
    with Pool(24) as p: res=[x for x in p.map(work,data,chunksize=4) if x]
    flat=[t for r in res for t in r]
    print(f"questions losing their annotated answer turn under coverage: {len(res)}")
    print(f"annotated answer turns involved: {len(flat)}")
    print(f"  within-session BM25 rank of that turn: median {st.median([t[0] for t in flat]):.0f} "
          f"mean {st.mean([t[0] for t in flat]):.2f}  (coverage keeps only rank 1)")
    print(f"  rank==1 (would survive): {sum(1 for t in flat if t[0]==1)}/{len(flat)}")
    print(f"  global BM25 rank of that turn: median {st.median([t[2] for t in flat]):.0f} "
          f"(<=20 => flat top-20 kept it): {sum(1 for t in flat if t[2]<=20)}/{len(flat)}")
    print(f"  turns per evidence session: median {st.median([t[1] for t in flat]):.0f}")
