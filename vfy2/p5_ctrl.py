"""Controls: is 'ignoring lexical relevance' what buys the anchor-block containment?"""
import pickle, sys, re, json
sys.path.insert(0, "/root/agent_memory_iclr/code")
import v5_temporal as V
from v5_temporal import (detect_family, resolve_question_anchors, anchor_block, anchor_day_events,
                         extract_events, _stem, _content_tokens, _sentences, best_event_date,
                         day0, parse_date, ANCHOR_RE, EVENT_CUE, INTENT_RE, ANCHOR_WINDOW,
                         FAMILY_BUDGET)
from v5_temporal_val import build_store_cached

tr = {q["question_id"]: q for q in pickle.load(open("/root/agent_memory_iclr/vfy2/tr.pkl","rb"))}
rows = json.load(open("/root/agent_memory_iclr/vfy2/anchor_rows.json"))
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

def day_candidates(q, anchor_dt, window, question):
    """Same candidate generation as anchor_day_events, but return (pri, rel, date, txt) unsorted."""
    out=[]
    qtok=set(_content_tokens(question, qside=True))
    for date_s, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        sd=parse_date(date_s)
        if sd is None: continue
        same_session = abs((day0(sd)-anchor_dt).days)<=window
        for t in sess:
            if t.get("role")!="user": continue
            for sent in _sentences(t.get("content") or ""):
                ev,prov = best_event_date(sent, date_s)
                if ev is None: continue
                default_dated = prov in ("","today")
                if abs((ev-anchor_dt).days)>window and not (same_session and default_dated and ANCHOR_RE.search(sent)):
                    continue
                cued = bool(ANCHOR_RE.search(sent)) or bool(EVENT_CUE.search(sent))
                if not (cued or re.search(r"\bi\b|\bi'|\bmy\b|\bwe\b|\bour\b", sent, re.I)): continue
                if not cued and sent.rstrip().endswith("?"): continue
                if INTENT_RE.search(sent): continue
                txt=re.sub(r"\s+"," ",sent).strip()
                if len(txt)>175: txt=txt[:175].rsplit(" ",1)[0]+"..."
                pri = 0 if ANCHOR_RE.search(sent) else (1 if cued else 2)
                rel = len(qtok & set(_content_tokens(sent)))
                out.append((pri, rel, day0(sd), txt))
    return out

def dedup_cap(seq, cap=10):
    seen,res=set(),[]
    for txt in seq:
        k=txt.lower()[:60]
        if k in seen: continue
        seen.add(k); res.append(txt)
        if len(res)>=cap: break
    return res

res=[]
for r in rows:
    qid=r["qid"]; q=tr[qid]
    store,_ = build_store_cached(q)
    fam,_u = detect_family(q["question"])
    anchors=[a for a in resolve_question_anchors(q["question"], q["question_date"]) if a["conf"]=="high"]
    if not anchors:
        anchors=[a for a in resolve_question_anchors(q["question"], q["question_date"])
                 if a["kind"] in ("offset-coarse","month-anchor")][:1]
    a0=anchors[0]
    prod, w = anchor_block(q, a0)                 # production block
    cands = day_candidates(q, a0["date"], w, q["question"])
    # C1: relevance-only timeline (no anchor block at all) at the family budget
    b = FAMILY_BUDGET.get(fam, {})
    evs = extract_events(q, store, q["question"], max_events=b.get("max_events",18))
    tl_lines=[e.text for e in evs]
    # C2: anchor-day candidates but keep only lexically-relevant ones (rel>=1), cap 10
    relonly = dedup_cap([t for _,rel,_,t in sorted(cands, key=lambda x:(x[0],-x[1],x[2])) if rel>=1])
    # C3: anchor-day dump ordered by date only (no relevance tie-break), cap 10
    bydate = dedup_cap([t for _,_,_,t in sorted(cands, key=lambda x:(x[2],x[0]))])
    # C4: anchor-day dump, priority only, relevance tie-break REMOVED (stable original order)
    bypri  = dedup_cap([t for _,_,_,t in sorted(cands, key=lambda x:(x[0],x[2]))])
    row=dict(qid=qid, gold=q["answer"], n_cands=len(cands), w=w,
             prod=int(any(cstem(x,q["answer"]) for x in prod)),
             prod_strict=int(any(cstrict(x,q["answer"]) for x in prod)),
             tl=int(any(cstem(x,q["answer"]) for x in tl_lines)),
             tl_strict=int(any(cstrict(x,q["answer"]) for x in tl_lines)),
             tl_n=len(tl_lines), tl_chars=sum(len(x) for x in tl_lines),
             relonly=int(any(cstem(x,q["answer"]) for x in relonly)), relonly_n=len(relonly),
             bydate=int(any(cstem(x,q["answer"]) for x in bydate)),
             bypri=int(any(cstem(x,q["answer"]) for x in bypri)),
             prod_chars=sum(len(x) for x in prod))
    res.append(row)
    print(f'{qid:16s} nCands={row["n_cands"]:4d} w={w} prod={row["prod"]}/{row["prod_strict"]} '
          f'timeline({row["tl_n"]}L,{row["tl_chars"]}c)={row["tl"]}/{row["tl_strict"]} '
          f'relFiltDay({row["relonly_n"]}L)={row["relonly"]} dayByDate={row["bydate"]} dayByPri={row["bypri"]}')
n=len(res)
print()
print("gold-in-block (stem) :  production anchor block", sum(r["prod"] for r in res), "/", n)
print("                        relevance-only timeline", sum(r["tl"] for r in res), "/", n)
print("                        anchor-day + relevance filter (rel>=1)", sum(r["relonly"] for r in res), "/", n)
print("                        anchor-day sorted by date (no rel tie-break)", sum(r["bydate"] for r in res), "/", n)
print("                        anchor-day sorted by pri only (no rel tie-break)", sum(r["bypri"] for r in res), "/", n)
print("gold-in-block (strict): production", sum(r["prod_strict"] for r in res), "  timeline", sum(r["tl_strict"] for r in res))
print("mean chars: anchor block", sum(r["prod_chars"] for r in res)/n, " timeline", sum(r["tl_chars"] for r in res)/n)
json.dump(res, open("/root/agent_memory_iclr/vfy2/ctrl_rows.json","w"), indent=1)
