"""Adversarial recompute of the ordering-family claim. Zero API."""
import json, os, re, sys
sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from v5_temporal import (render_timeline, extract_events, detect_family, parse_date, day0,
                         _content_tokens, question_alternatives, match_item_date)
from v5_temporal_val import build_store_cached
from v5_temporal_measure import ordering_items, _match_event, contains, norm

data = load_dataset("s_cleaned")
out = []
for q in data:
    if q["question_type"] != "temporal-reasoning":
        continue
    fam, unit = detect_family(q["question"])
    if fam != "ordering":
        continue
    store, miss = build_store_cached(q)
    events = extract_events(q, store, q["question"])
    gseq, cands = ordering_items(q["question"], q["answer"])
    dmap = {c: _match_event(c, events) for c in cands}
    s2d = dict(zip(q["haystack_session_ids"], q["haystack_dates"]))
    ans_dates = sorted({day0(parse_date(s2d[s])) for s in (q.get("answer_session_ids") or [])
                        if s in s2d and parse_date(s2d[s])})
    ev_dates = {e.ev_dt for e in events}
    sess_dates = {day0(e.sess_dt) for e in events}
    shown = ev_dates | sess_dates
    verdict = None; got = None; tie = None
    if len(cands) >= 2 and all(v is not None for v in dmap.values()):
        got = sorted(cands, key=lambda c: dmap[c])
        tie = len({dmap[c] for c in cands}) < len(cands)
        if len(gseq) >= 3:
            verdict = [norm(x) for x in got] == [norm(x) for x in gseq]
        elif gseq:
            verdict = contains(got[0], gseq[0]) or contains(gseq[0], got[0])
    # trivial baselines on the SAME computable subset
    first_mention = cands[0] if cands else None
    last_mention = cands[-1] if cands else None
    b_first = b_last = None
    if verdict is not None and len(gseq) < 3 and gseq:
        b_first = contains(first_mention, gseq[0]) or contains(gseq[0], first_mention)
        b_last = contains(last_mention, gseq[0]) or contains(gseq[0], last_mention)
    out.append(dict(qid=q["question_id"], question=q["question"], gold=str(q["answer"]),
                    n_items=len(cands), cands=cands, gseq=gseq,
                    dates={c: (dmap[c].strftime("%Y/%m/%d") if dmap[c] else None) for c in cands},
                    got=got, tie=tie, verdict=verdict,
                    b_first=b_first, b_last=b_last,
                    ans_dates=[d.strftime("%Y/%m/%d") for d in ans_dates],
                    uniq_ans=len(ans_dates), n_ans_sessions=len(q.get("answer_session_ids") or []),
                    dates_present=sum(d in shown for d in ans_dates)/max(1, len(ans_dates)),
                    n_events=len(events), miss=miss,
                    chars=len(render_timeline(q, store, q["question"]))))
json.dump(out, open("/root/agent_memory_iclr/verify/adv_ord.json", "w"), ensure_ascii=False, indent=1)
print("n", len(out))
