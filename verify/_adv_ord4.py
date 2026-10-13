"""Does the 'non-computable' bucket come from retrieval, or from the harness matcher
not stemming? Re-run item matching with stemming (everything else identical)."""
import json, re, sys
sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from v5_temporal import extract_events, detect_family, _content_tokens, _stem
from v5_temporal_val import build_store_cached
from v5_temporal_measure import ordering_items, _match_event, contains, norm

def match_stem(item, events):
    it = [_stem(w) for w in _content_tokens(item) if len(w) > 2]
    if not it:
        return None
    best = None
    for e in events:
        et = {_stem(w) for w in _content_tokens(e.text)}
        hit = sum(w in et for w in it)
        need = len(it) if len(it) <= 2 else max(2, int(round(0.6 * len(it))))
        if hit >= need and (best is None or e.ev_dt < best):
            best = e.ev_dt
    return best

data = load_dataset("s_cleaned")
rows = []
for q in data:
    if q["question_type"] != "temporal-reasoning" or detect_family(q["question"])[0] != "ordering":
        continue
    store, _ = build_store_cached(q)
    ev = extract_events(q, store, q["question"])
    gseq, cands = ordering_items(q["question"], q["answer"])
    for tag, fn in (("orig", _match_event), ("stem", match_stem)):
        dm = {c: fn(c, ev) for c in cands}
        v = None
        if len(cands) >= 2 and all(x is not None for x in dm.values()):
            got = sorted(cands, key=lambda c: dm[c])
            v = ([norm(x) for x in got] == [norm(x) for x in gseq]) if len(gseq) >= 3 else \
                (bool(gseq) and (contains(got[0], gseq[0]) or contains(gseq[0], got[0])))
        rows.append(dict(qid=q["question_id"], tag=tag, n_items=len(cands), verdict=v,
                         abst="not enough" in str(q["answer"]).lower()))
json.dump(rows, open("/root/agent_memory_iclr/verify/adv_ord4.json", "w"), indent=1)
for tag in ("orig", "stem"):
    S = [r for r in rows if r["tag"] == tag]
    c = [r for r in S if r["verdict"] is not None]
    b = [r for r in c if r["n_items"] == 2]
    print("%s: computable %d/%d  correct %d/%d=%.3f  binary %d/%d=%.3f" % (
        tag, len(c), len(S), sum(r["verdict"] for r in c), len(c),
        sum(r["verdict"] for r in c)/len(c), sum(r["verdict"] for r in b), len(b),
        sum(r["verdict"] for r in b)/max(1, len(b))))
o = {r["qid"]: r for r in rows if r["tag"] == "orig"}
s = {r["qid"]: r for r in rows if r["tag"] == "stem"}
print("newly computable with stemming:",
      [(k, s[k]["verdict"], s[k]["n_items"], "ABSTENTION" if s[k]["abst"] else "")
       for k in o if o[k]["verdict"] is None and s[k]["verdict"] is not None])
print("lost:", [k for k in o if o[k]["verdict"] is not None and s[k]["verdict"] is None])
print("flipped:", [(k, o[k]["verdict"], s[k]["verdict"]) for k in o
                   if o[k]["verdict"] is not None and s[k]["verdict"] is not None
                   and o[k]["verdict"] != s[k]["verdict"]])
