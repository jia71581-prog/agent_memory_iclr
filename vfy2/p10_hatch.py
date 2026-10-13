"""Counterfactual: the escape-hatch guard (default_dated) and CI on 20/21."""
import pickle, sys, re, json, random
sys.path.insert(0, "/root/agent_memory_iclr/code")
import v5_temporal as V
from v5_temporal import (resolve_question_anchors, _stem, _sentences, _content_tokens,
                         best_event_date, day0, parse_date, ANCHOR_RE, EVENT_CUE, INTENT_RE, ANCHOR_WINDOW)
tr={q["question_id"]:q for q in pickle.load(open("/root/agent_memory_iclr/vfy2/tr.pkl","rb"))}
rows=json.load(open("/root/agent_memory_iclr/vfy2/anchor_rows.json"))
def norm(s): return re.sub(r"[^a-z0-9]+"," ",str(s).lower()).strip()
STOPW=("the","and","for","was","were","you","your","not","did","with")
def cstem(hay,needle):
    h={_stem(w) for w in norm(hay).split()}
    toks=[_stem(t) for t in norm(needle).split() if len(t)>2 and t not in STOPW]
    return bool(toks) and sum(t in h for t in toks)>=max(1,int(round(0.8*len(toks))))

def block(q, anchor_dt, window, buggy=False, cap=10):
    out=[]; qtok=set(_content_tokens(q["question"], qside=True))
    for date_s, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        sd=parse_date(date_s)
        if sd is None: continue
        same=abs((day0(sd)-anchor_dt).days)<=window
        for t in sess:
            if t.get("role")!="user": continue
            for sent in _sentences(t.get("content") or ""):
                ev,prov=best_event_date(sent, date_s)
                if ev is None: continue
                dd = prov in ("","today")
                hatch = (same and ANCHOR_RE.search(sent)) if buggy else (same and dd and ANCHOR_RE.search(sent))
                if abs((ev-anchor_dt).days)>window and not hatch: continue
                cued=bool(ANCHOR_RE.search(sent)) or bool(EVENT_CUE.search(sent))
                if not (cued or re.search(r"\bi\b|\bi'|\bmy\b|\bwe\b|\bour\b", sent, re.I)): continue
                if not cued and sent.rstrip().endswith("?"): continue
                if INTENT_RE.search(sent): continue
                txt=re.sub(r"\s+"," ",sent).strip()
                if len(txt)>175: txt=txt[:175].rsplit(" ",1)[0]+"..."
                pri=0 if ANCHOR_RE.search(sent) else (1 if cued else 2)
                out.append((pri,-len(qtok&set(_content_tokens(sent))),day0(sd),txt))
    seen,res=set(),[]
    for p,nr,sd,txt in sorted(out,key=lambda x:(x[0],x[1],x[2])):
        k=txt.lower()[:60]
        if k in seen: continue
        seen.add(k); res.append(txt)
        if len(res)>=cap: break
    return res

fixed=buggy=0; per=[]
for r in rows:
    q=tr[r["qid"]]
    anchors=[a for a in resolve_question_anchors(q["question"], q["question_date"]) if a["conf"]=="high"]
    if not anchors:
        anchors=[a for a in resolve_question_anchors(q["question"], q["question_date"])
                 if a["kind"] in ("offset-coarse","month-anchor")][:1]
    a0=anchors[0]; w0,w1=ANCHOR_WINDOW.get(a0["kind"],(1,3))
    def esc(buggy_flag):
        for w in (w0,w1):
            b=block(q,a0["date"],w,buggy=buggy_flag)
            if b: return b
        return []
    bf=esc(False); bb=esc(True)
    f=int(any(cstem(x,q["answer"]) for x in bf)); b_=int(any(cstem(x,q["answer"]) for x in bb))
    fixed+=f; buggy+=b_
    per.append((r["qid"],f,b_))
    if f!=b_: print("DIFF", r["qid"], "fixed",f,"buggy",b_)
n=len(rows)
print(f"with default_dated guard (production): {fixed}/{n}")
print(f"without the guard (the bug):           {buggy}/{n}")
# bootstrap CI on the 20/21 containment and 20/21 anchor hit
hits=[r["has_stem"] for r in rows]; anc=[r["hit"] for r in rows]
random.seed(0)
def boot(v, B=20000):
    m=[]
    for _ in range(B):
        s=[v[random.randrange(len(v))] for _ in range(len(v))]
        m.append(sum(s)/len(s))
    m.sort(); return m[int(.025*B)], m[int(.975*B)]
print("gold-in-block stem 20/21 = %.3f, bootstrap 95%% CI %s" % (sum(hits)/n, boot(hits)))
print("anchor within 3d 20/21 = %.3f, bootstrap 95%% CI %s" % (sum(anc)/n, boot(anc)))
