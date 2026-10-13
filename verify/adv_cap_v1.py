"""ADVERSARIAL recompute #1: cap firing rate, view size, and line drops. Offline, no API."""
import json, os, hashlib, statistics as st, pickle, sys
sys.path.insert(0, "/root/agent_memory_iclr/code")
import lifecycle as LC
from query_classifier import classify_query

DATA = "/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"
CAP = 20000
data = json.load(open(DATA))

def cached_store(q):
    facts, miss, tot = [], 0, 0
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        text = LC._session_text(sess)
        if not text:
            continue
        tot += 1
        key = hashlib.md5(text.encode()).hexdigest()[:16]
        p = os.path.join(LC.SESSION_CACHE_DIR, f"{key}.json")
        if not os.path.exists(p):
            miss += 1
            continue
        for e, a, v in json.load(open(p)):
            facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
    return LC.assemble_chains(facts), miss, tot

def truncate(txt, cap=CAP):
    """exactly run_qc.py's truncation"""
    if len(txt) <= cap:
        return txt, False
    kept, total = [], 0
    for line in txt.splitlines():
        if total + len(line) + 1 > cap:
            break
        kept.append(line)
        total += len(line) + 1
    return "\n".join(kept), True

rows = []
blob = {}
for i, q in enumerate(data):
    store, miss, tot = cached_store(q)
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    full = LC.render_view(store, mode)
    cap, fired = truncate(full)
    nlines_full = len(full.splitlines())
    nlines_cap = len(cap.splitlines()) if cap else 0
    rows.append(dict(qid=q["question_id"], type=q["question_type"], is_abs=q["question_id"].endswith("_abs"),
                     mode=mode, chars_full=len(full), chars_cap=len(cap), fired=fired,
                     lines_full=nlines_full, lines_cap=nlines_cap, dropped=nlines_full - nlines_cap,
                     n_chains=len(store), cache_miss=miss, n_sess=tot))
    blob[q["question_id"]] = {"full": full, "cap": cap, "mode": mode}
    if (i + 1) % 100 == 0:
        print("...", i + 1, flush=True)

pickle.dump(blob, open("/root/agent_memory_iclr/verify/_views.pkl", "wb"))
json.dump(rows, open("/root/agent_memory_iclr/verify/_cap_rows.json", "w"))

ans = [r for r in rows if not r["is_abs"]]
abs_ = [r for r in rows if r["is_abs"]]
print(f"n={len(rows)} answerable={len(ans)} abs={len(abs_)}")
print(f"cache misses: {sum(r['cache_miss'] for r in rows)} sessions over {sum(r['n_sess'] for r in rows)}")
print(f"cap fires (full view > {CAP} chars): {sum(r['fired'] for r in rows)}/500  "
      f"(answerable {sum(r['fired'] for r in ans)}/{len(ans)}, abs {sum(r['fired'] for r in abs_)}/{len(abs_)})")
print(f"median full-view chars: ALL {st.median([r['chars_full'] for r in rows]):.0f}  "
      f"answerable {st.median([r['chars_full'] for r in ans]):.0f}  abs {st.median([r['chars_full'] for r in abs_]):.0f}")
print(f"median lines full: ALL {st.median([r['lines_full'] for r in rows]):.0f} "
      f"ans {st.median([r['lines_full'] for r in ans]):.0f} abs {st.median([r['lines_full'] for r in abs_]):.0f}")
print(f"median lines dropped: ALL {st.median([r['dropped'] for r in rows]):.0f} "
      f"ans {st.median([r['dropped'] for r in ans]):.0f} abs {st.median([r['dropped'] for r in abs_]):.0f}")
print(f"median lines kept: ALL {st.median([r['lines_cap'] for r in rows]):.0f}")
print(f"median capped chars: {st.median([r['chars_cap'] for r in rows]):.0f}  "
      f"min {min(r['chars_cap'] for r in rows)}  max {max(r['chars_cap'] for r in rows)}")
ssp = [r for r in rows if r["type"] == "single-session-preference"]
print(f"SSP: n={len(ssp)} cap fires {sum(r['fired'] for r in ssp)}  median chars_full "
      f"{st.median([r['chars_full'] for r in ssp]):.0f}  median dropped {st.median([r['dropped'] for r in ssp]):.0f}")
print("mode split:", sum(1 for r in rows if r["mode"] == "current"), "current /",
      sum(1 for r in rows if r["mode"] == "historical"), "historical")
