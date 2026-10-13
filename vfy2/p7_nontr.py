"""Does the anchor-day block fire in production render on non-TR questions, and is it misleading?"""
import json, sys, re, pickle
sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from v5_temporal import detect_family, render_timeline, resolve_question_anchors, _stem
from v5_temporal_val import build_store_cached

data = load_dataset("s_cleaned")
qc={json.loads(l)["qid"]: json.loads(l) for l in open("/root/agent_memory_iclr/results/full500_qc.jsonl")}
def norm(s): return re.sub(r"[^a-z0-9]+"," ",str(s).lower()).strip()
STOPW=("the","and","for","was","were","you","your","not","did","with")
def cstem(hay,needle):
    h={_stem(w) for w in norm(hay).split()}
    toks=[_stem(t) for t in norm(needle).split() if len(t)>2 and t not in STOPW]
    return bool(toks) and sum(t in h for t in toks)>=max(1,int(round(0.8*len(toks))))

n_fire=0; rows=[]
for q in data:
    fam,_=detect_family(q["question"])
    if fam!="anchor_lookup" or q["question_type"]=="temporal-reasoning":
        continue
    store,_m=build_store_cached(q)
    tl=render_timeline(q, store, q["question"])
    fired = "listed regardless of wording overlap" in tl
    n_fire += fired
    # extract the block
    blk=[]
    if fired:
        keep=False
        for line in tl.splitlines():
            if "listed regardless of wording overlap" in line: keep=True; continue
            if keep:
                if line.startswith("  - "): blk.append(line[4:])
                else: break
    rows.append(dict(qid=q["question_id"], type=q["question_type"], question=q["question"],
                     gold=q["answer"], fired=bool(fired), nblk=len(blk),
                     has=int(any(cstem(x,q["answer"]) for x in blk)),
                     judge=bool(qc[q["question_id"]]["judge"]), tl_chars=len(tl)))
    print(f'{q["question_id"]:16s} {q["question_type"]:16s} fired={int(bool(fired))} nblk={len(blk):2d} '
          f'goldInBlk={rows[-1]["has"]} judge={int(rows[-1]["judge"])} tl={len(tl)}c | {q["question"][:78]}')
print()
print("non-TR anchor_lookup:", len(rows), " block fires:", n_fire,
      " gold in block:", sum(r["has"] for r in rows), " currently correct:", sum(r["judge"] for r in rows))
fired=[r for r in rows if r["fired"]]
print("among firings:", len(fired), " gold in block:", sum(r["has"] for r in fired))
json.dump(rows, open("/root/agent_memory_iclr/vfy2/nontr_rows.json","w"), indent=1)
