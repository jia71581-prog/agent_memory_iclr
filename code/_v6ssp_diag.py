"""SSP 回归诊断:重建 v5 / full 的上下文,定位机制。无 API(lifecycle 全缓存命中)。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_baseline import bm25_rank
import run_qc as R
from lifecycle import build_lifecycle, render_view
from query_classifier import classify_query

D = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
SSP = [q for q in D if q["question_type"] == "single-session-preference"]
LOSS = ["32260d93", "6b7dfb22", "57f827a0", "1da05512", "b6025781", "1c0ddc50"]
BOTH = ["0edc2aef", "195a1a1b", "09d032c9", "95228167", "75f70248", "d6233ab6"]


def units_of(q):
    units = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for ti, t in enumerate(sess):
            if t.get("content"):
                units.append({"date": date, "role": t["role"], "content": t["content"],
                              "si": si, "ti": ti, "has_answer": bool(t.get("has_answer"))})
    return units


def gold_sessions(q):
    ids = q.get("answer_session_ids") or []
    hs = q.get("haystack_session_ids") or []
    return {hs.index(i) for i in ids if i in hs}


def v5_ctx(q, store):
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    view = R._rank_view_lines(store, mode, q["question"], R.VIEW_CHAR_CAP)
    units, sel = R._select_turns_v5(q, q["question"], R.TOP_K)
    return mode, view, units, sel


def full_sel(q, units):
    order = bm25_rank(q["question"], [u["content"] for u in units])[:30]
    return sorted(order)


def main():
    rows = []
    for q in SSP:
        qid = q["question_id"]
        u = units_of(q)
        gs = gold_sessions(q)
        store = build_lifecycle(q, None)
        mode, view, v5units, v5sel = v5_ctx(q, store)
        fsel = full_sel(q, u)
        gold_turn_idx = {i for i, x in enumerate(u) if x["has_answer"]}
        gold_sess_idx = {i for i, x in enumerate(u) if x["si"] in gs}
        v5s, fs = set(v5sel), set(fsel)
        raw_v5 = "\n".join(f"[{v5units[i][0]}] [{v5units[i][1]}] {v5units[i][2]}" for i in v5sel)
        raw_f = "\n".join(f"[{u[i]['date']}] [{u[i]['role']}] {u[i]['content']}" for i in fsel)
        rows.append({
            "qid": qid, "mode": mode,
            "n_units": len(u), "gold_sess": sorted(gs),
            "gold_turns": len(gold_turn_idx),
            "v5_gold_turns": len(v5s & gold_turn_idx), "full_gold_turns": len(fs & gold_turn_idx),
            "v5_goldsess_turns": len(v5s & gold_sess_idx), "full_goldsess_turns": len(fs & gold_sess_idx),
            "v5_asst": sum(1 for i in v5sel if v5units[i][1] == "assistant"),
            "full_asst": sum(1 for i in fsel if u[i]["role"] == "assistant"),
            "v5_goldsess_asst": sum(1 for i in v5sel if i in gold_sess_idx and v5units[i][1] == "assistant"),
            "full_goldsess_asst": sum(1 for i in fsel if i in gold_sess_idx and u[i]["role"] == "assistant"),
            "view_chars": len(view), "v5_raw_chars": len(raw_v5), "full_raw_chars": len(raw_f),
            "v5_total": len(view) + len(raw_v5), "full_total": len(raw_f),
            "n_chains": len(store), "n_view_lines": view.count("\n") + 1 if view else 0,
        })
        print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open("/tmp/ssp_diag.json", "w"), indent=1)

    def agg(sub, name):
        s = [r for r in rows if r["qid"] in sub] if sub else rows
        k = lambda f: sum(r[f] for r in s) / len(s)
        print(f"\n[{name}] n={len(s)}  gold_turns_in_ctx v5={k('v5_gold_turns'):.2f} full={k('full_gold_turns'):.2f}"
              f" | goldsess_turns v5={k('v5_goldsess_turns'):.2f} full={k('full_goldsess_turns'):.2f}"
              f" | goldsess_ASST v5={k('v5_goldsess_asst'):.2f} full={k('full_goldsess_asst'):.2f}"
              f" | asst v5={k('v5_asst'):.2f} full={k('full_asst'):.2f}"
              f" | chars v5={k('v5_total'):.0f} full={k('full_total'):.0f} (view {k('view_chars'):.0f})")
    agg(None, "ALL SSP")
    agg(LOSS, "LOSS (full right, v5 wrong)")
    agg(BOTH, "BOTH WRONG")
    won = [r["qid"] for r in rows if r["qid"] not in LOSS + BOTH]
    agg(won, "v5 CORRECT")


if __name__ == "__main__":
    main()
