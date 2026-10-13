"""For every non-computable ordering case: is the missing item absent from the timeline
(retrieval failure) or present but rejected by the harness matcher (metric artifact)?"""
import json, re, sys
sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from v5_temporal import extract_events, detect_family, _content_tokens, _stem, render_timeline
from v5_temporal_val import build_store_cached

adv = {r["qid"]: r for r in json.load(open("/root/agent_memory_iclr/verify/adv_ord.json"))}
data = {q["question_id"]: q for q in load_dataset("s_cleaned")}
nc = [r for r in adv.values() if r["verdict"] is None]
for r in sorted(nc, key=lambda x: x["qid"]):
    q = data[r["qid"]]
    store, _ = build_store_cached(q)
    events = extract_events(q, store, q["question"])
    tl = render_timeline(q, store, q["question"])
    tlstems = {_stem(w) for w in re.sub(r"[^a-z0-9]+", " ", tl.lower()).split()}
    ev_sess = " ".join(" ".join(t.get("content") or "" for t in sess)
                       for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"])
                       if sid in (q.get("answer_session_ids") or []))
    evstems = {_stem(w) for w in re.sub(r"[^a-z0-9]+", " ", ev_sess.lower()).split()}
    hay = " ".join(" ".join(t.get("content") or "" for t in sess) for sess in q["haystack_sessions"])
    haystems = {_stem(w) for w in re.sub(r"[^a-z0-9]+", " ", hay.lower()).split()}
    miss = [c for c, d in r["dates"].items() if d is None]
    print(r["qid"], "| n_items", r["n_items"], "| abstention gold:", "not enough" in r["gold"].lower())
    for c in miss:
        toks = [_stem(t) for t in _content_tokens(c) if len(t) > 2]
        if not toks:
            print("   -", repr(c), "-> no content tokens"); continue
        f_tl = sum(t in tlstems for t in toks) / len(toks)
        f_ev = sum(t in evstems for t in toks) / len(toks)
        f_hay = sum(t in haystems for t in toks) / len(toks)
        print("   - %-58r tok=%d in_timeline=%.2f in_evidence_sessions=%.2f in_haystack=%.2f"
              % (c[:56], len(toks), f_tl, f_ev, f_hay))
