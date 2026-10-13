"""Fair, same-granularity comparison + provenance audit of the matched block line."""
import pickle, sys, re, json
sys.path.insert(0, "/root/agent_memory_iclr/code"); sys.path.insert(0, "/root/agent_memory_iclr/results")
from _recon_ctx import qc_context, units_of
from v5_temporal import (detect_family, resolve_question_anchors, anchor_block, _stem, _sentences)

tr={q["question_id"]:q for q in pickle.load(open("/root/agent_memory_iclr/vfy2/tr.pkl","rb"))}
rows=json.load(open("/root/agent_memory_iclr/vfy2/anchor_rows.json"))
qc={json.loads(l)["qid"]: json.loads(l) for l in open("/root/agent_memory_iclr/results/full500_qc.jsonl")}
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
    qid=r["qid"]; q=tr[qid]; gold=q["answer"]
    label,mode,view,raw_idx = qc_context(q)
    units=units_of(q)
    view_lines=[l for l in view.splitlines() if l.strip()]
    raw_turns=[units[i][2] for i in raw_idx]
    # per-unit (same granularity as a block line): a single view line, a single retrieved turn,
    # or a single sentence of a retrieved turn
    raw_sents=[s for t in raw_turns for s in _sentences(t)]
    unit_hit_strict = any(cstrict(x,gold) for x in view_lines+raw_turns)
    unit_hit_stem   = any(cstem(x,gold)   for x in view_lines+raw_turns)
    vline = any(cstrict(x,gold) for x in view_lines)
    rturn = any(cstrict(x,gold) for x in raw_turns)
    # block provenance: is the matched line really a gold-evidence sentence?
    anchors=[a for a in resolve_question_anchors(q["question"], q["question_date"]) if a["conf"]=="high"]
    if not anchors:
        anchors=[a for a in resolve_question_anchors(q["question"], q["question_date"])
                 if a["kind"] in ("offset-coarse","month-anchor")][:1]
    blk,w = anchor_block(q, anchors[0])
    matched=[x for x in blk if cstem(x,gold)]
    ans_ids=set(q.get("answer_session_ids",[]))
    gold_sents=set(); ev_sents=set()
    for sid,sess in zip(q["haystack_session_ids"], q["haystack_sessions"]):
        for t in sess:
            if t.get("role")!="user" or not t.get("content"): continue
            for s in _sentences(t["content"]):
                key=re.sub(r"\s+"," ",s).strip()[:120]
                if sid in ans_ids:
                    gold_sents.add(key)
                    if t.get("has_answer"): ev_sents.add(key)
    def prov(x):
        k=re.sub(r"\.\.\.$","",x)[:120].strip()
        insess = any(k[:80] and k[:80] in g for g in gold_sents)
        inev   = any(k[:80] and k[:80] in g for g in ev_sents)
        return insess, inev
    pv=[prov(x) for x in matched]
    from_gold_sess = any(a for a,b in pv); from_ev_turn = any(b for a,b in pv)
    out.append(dict(qid=qid, judge=bool(qc[qid]["judge"]), blk_stem=r["has_stem"], blk_strict=r["has"],
                    ctx_unit_strict=unit_hit_strict, ctx_unit_stem=unit_hit_stem,
                    in_view_line=vline, in_raw_turn=rturn,
                    matched_from_gold_session=from_gold_sess, matched_from_has_answer_turn=from_ev_turn,
                    n_matched=len(matched)))
    print(f'{qid:16s} judge={int(out[-1]["judge"])} blk={r["has"]}/{r["has_stem"]} '
          f'ctxUnit={int(unit_hit_strict)}/{int(unit_hit_stem)} viewLine={int(vline)} rawTurn={int(rturn)} '
          f'matchedFromGoldSess={int(from_gold_sess)} fromHasAnswerTurn={int(from_ev_turn)} nMatched={len(matched)}')
n=len(out)
print()
print("BLOCK gold-line          strict %d/%d  stem %d/%d" % (sum(r["blk_strict"] for r in out), n, sum(r["blk_stem"] for r in out), n))
print("PRODUCTION CONTEXT, same granularity (one view line or one retrieved turn):")
print("   strict %d/%d  stem %d/%d" % (sum(r["ctx_unit_strict"] for r in out), n, sum(r["ctx_unit_stem"] for r in out), n))
print("   of which via lifecycle view line: %d ; via BM25 turn: %d" %
      (sum(r["in_view_line"] for r in out), sum(r["in_raw_turn"] for r in out)))
print("PROVENANCE of the matched block line: from a gold answer session %d/%d ; from a has_answer turn %d/%d"
      % (sum(r["matched_from_gold_session"] for r in out), n, sum(r["matched_from_has_answer_turn"] for r in out), n))
errs=[r for r in out if not r["judge"]]
print()
print("Among the 5 current qc errors:")
for r in errs:
    print(f'   {r["qid"]:16s} blockHasGold={r["blk_strict"]}/{r["blk_stem"]}  alreadyInCtxUnit={int(r["ctx_unit_strict"])}/{int(r["ctx_unit_stem"])}')
print("   NEW coverage the block adds (gold not already in any context unit, strict):",
      sum(r["blk_strict"] and not r["ctx_unit_strict"] for r in errs), "/", len(errs))
json.dump(out, open("/root/agent_memory_iclr/vfy2/fair_rows.json","w"), indent=1)
