"""Probe (no API): baseline behaviour that motivates provenance-guided retrieval.

Measures on all 500 questions:
  - real deployed view size vs the 20k char cap (run_qc truncates; coverage_diag did NOT
    -> published in_view=0.634 may be an overestimate of what the answerer saw)
  - BM25 top-20 turn budget: how many distinct sessions it spans
  - oracle evidence recall: has_answer turns / answer_session_ids covered by top-20
"""
import hashlib
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from coverage_diag import contains
from lifecycle import assemble_chains, render_view, _session_text, SESSION_CACHE_DIR
from query_classifier import classify_query
from run_baseline import bm25_rank

R = "/root/agent_memory_iclr/results"
VIEW_CAP = 20000
DATA = None


def cached_facts(sess):
    text = _session_text(sess)
    if not text:
        return []
    key = hashlib.md5(text.encode()).hexdigest()[:16]
    p = os.path.join(SESSION_CACHE_DIR, f"{key}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def cap_view(view_txt, cap=VIEW_CAP):
    """run_qc.py 的行边界截断,逐字复制。"""
    if len(view_txt) <= cap:
        return view_txt
    kept, total = [], 0
    for line in view_txt.splitlines():
        if total + len(line) + 1 > cap:
            break
        kept.append(line)
        total += len(line) + 1
    return "\n".join(kept)


def one(qi):
    q = DATA[qi]
    gold = q["answer"]
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"

    all_facts = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        tri = cached_facts(sess) or []
        for e, a, v in tri:
            all_facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
    store = assemble_chains(all_facts)
    view_full = render_view(store, mode)
    view_cap = cap_view(view_full)

    units = []  # (sidx, date, role, content, has_answer)
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for t in sess:
            if t.get("content"):
                units.append((si, date, t["role"], t["content"], bool(t.get("has_answer"))))
    order = bm25_rank(q["question"], [u[3] for u in units])
    top20 = order[:20]
    raw20 = "\n".join(units[i][3] for i in top20)

    ev_sidx = {i for i, sid in enumerate(q["haystack_session_ids"]) if sid in set(q["answer_session_ids"])}
    ev_turns = [i for i, u in enumerate(units) if u[4]]
    return {
        "qid": q["question_id"], "type": q["question_type"], "mode": mode,
        "n_chains": len(store), "n_facts": len(all_facts),
        "view_full_chars": len(view_full), "view_cap_chars": len(view_cap),
        "view_full_lines": len(view_full.splitlines()), "view_cap_lines": len(view_cap.splitlines()),
        "gold_in_view_full": contains(view_full, gold), "gold_in_view_cap": contains(view_cap, gold),
        "raw20_chars": len(raw20), "gold_in_raw20": contains(raw20, gold),
        "n_units": len(units), "n_sessions": len(q["haystack_sessions"]),
        "top20_sessions": len({units[i][0] for i in top20}),
        "n_ev_sessions": len(ev_sidx), "n_ev_turns": len(ev_turns),
        "ev_sessions_in_top20": len({units[i][0] for i in top20} & ev_sidx),
        "ev_turns_in_top20": len(set(top20) & set(ev_turns)),
        "ev_sessions_with_facts": len(ev_sidx & {f["sidx"] for f in all_facts}),
    }


def main():
    global DATA
    DATA = load_dataset("s_cleaned")
    with Pool(30) as p:
        rows = p.map(one, range(len(DATA)), chunksize=4)
    with open(f"{R}/prov_v5_probe.json", "w") as f:
        json.dump(rows, f)
    print("saved", len(rows))


if __name__ == "__main__":
    main()
