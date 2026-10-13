"""Decisive proxy test: was the gold already in the qc context for the 5 failing anchor questions?"""
import pickle, sys, re, json
sys.path.insert(0, "/root/agent_memory_iclr/code")
sys.path.insert(0, "/root/agent_memory_iclr/results")
from _recon_ctx import qc_context, units_of, evidence_turns, coverage
from v5_temporal import _stem

tr = {q["question_id"]: q for q in pickle.load(open("/root/agent_memory_iclr/vfy2/tr.pkl","rb"))}
rows = json.load(open("/root/agent_memory_iclr/vfy2/anchor_rows.json"))
qc = {json.loads(l)["qid"]: json.loads(l) for l in open("/root/agent_memory_iclr/results/full500_qc.jsonl")}

def norm(s): return re.sub(r"[^a-z0-9]+"," ",str(s).lower()).strip()
STOPW=("the","and","for","was","were","you","your","not","did","with")
def cstrict(hay,needle):
    h,n=norm(hay),norm(needle)
    if not n: return False
    if n in h: return True
    toks=[t for t in n.split() if len(t)>2 and t not in STOPW]
    return bool(toks) and all(t in h for t in toks)
def cstem(hay,needle):
    h={_stem(w) for w in norm(hay).split()}
    toks=[_stem(t) for t in norm(needle).split() if len(t)>2 and t not in STOPW]
    return bool(toks) and sum(t in h for t in toks)>=max(1,int(round(0.8*len(toks))))

out=[]
for r in rows:
    qid=r["qid"]; q=tr[qid]
    label, mode, view, raw_idx = qc_context(q)
    units = units_of(q)
    raw_txt = "\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in raw_idx)
    ctx = view + "\n" + raw_txt
    ev = evidence_turns(q)
    cov, hit, nev = coverage(q, raw_idx)
    rec=dict(qid=qid, judge=bool(qc[qid]["judge"]),
             gold_in_raw20=cstrict(raw_txt, q["answer"]), gold_in_raw20_stem=cstem(raw_txt, q["answer"]),
             gold_in_view=cstrict(view, q["answer"]), gold_in_ctx=cstrict(ctx, q["answer"]),
             gold_in_ctx_stem=cstem(ctx, q["answer"]),
             ev_turn_in_raw20=(hit if nev else None), n_ev=nev,
             blk_has=r["has"], blk_has_stem=r["has_stem"], blk_rank=r["rank"],
             ctx_chars=len(ctx), view_chars=len(view), raw_chars=len(raw_txt))
    out.append(rec)
    print(f'{qid:16s} judge={int(rec["judge"])} goldInRaw20={int(rec["gold_in_raw20"])}/{int(rec["gold_in_raw20_stem"])} '
          f'goldInView={int(rec["gold_in_view"])} goldInCtx={int(rec["gold_in_ctx"])}/{int(rec["gold_in_ctx_stem"])} '
          f'evTurnsInRaw20={hit}/{nev} blkHas={int(rec["blk_has"])} rank={rec["blk_rank"]} ctx={rec["ctx_chars"]}c')
json.dump(out, open("/root/agent_memory_iclr/vfy2/ctx_rows.json","w"), indent=1)
print()
errs=[r for r in out if not r["judge"]]
print("ERRORS n=",len(errs))
print("  gold already in qc context (strict):", sum(r["gold_in_ctx"] for r in errs))
print("  gold already in qc context (stem)  :", sum(r["gold_in_ctx_stem"] for r in errs))
print("  gold evidence turn already in BM25 top-20:", sum((r["ev_turn_in_raw20"] or 0)>0 for r in errs))
ok=[r for r in out if r["judge"]]
print("CORRECT n=",len(ok), " gold in ctx:", sum(r["gold_in_ctx"] for r in ok), " stem:", sum(r["gold_in_ctx_stem"] for r in ok))
print("mean ctx chars over 21:", sum(r["ctx_chars"] for r in out)/len(out))
