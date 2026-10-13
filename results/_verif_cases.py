"""Independent adversarial recomputation of the 4 case-level claims.
No API. Recomputes evidence-turn hits from the dataset + cached extractions,
and decomposes which selection channel (provenance vs BM25 rank) produced each hit.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from lifecycle import assemble_chains, _session_text, SESSION_CACHE_DIR
from query_classifier import classify_query
from run_baseline import bm25_rank
import v5_retrieval as V5

QIDS = ["3a704032", "6cb6f249", "0a995998", "46a3abf7"]
CFGS = ["baseline", "v5_drop_in", "v5_realloc", "realloc", "bm25_k30", "bm25_k50"]


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


def main():
    D = {q["question_id"]: q for q in load_dataset("s_cleaned")}
    report = {}
    for qid in QIDS:
        q = D[qid]
        # --- independent unit construction + has_answer flags (single pass, same order) ---
        units, ev_turns, ev_units = [], set(), []
        for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
            for t in sess:
                if not t.get("content"):
                    continue
                idx = len(units)
                units.append((si, date, t["role"], t["content"]))
                if t.get("has_answer"):
                    ev_turns.add(idx)
                    ev_units.append((idx, si, t["role"], t["content"][:120]))
        assert units == V5.build_units(q), "unit construction mismatch vs V5.build_units"
        ans_ids = set(q["answer_session_ids"])
        ev_sess = {i for i, sid in enumerate(q["haystack_session_ids"]) if sid in ans_ids}

        # --- chains from cache ---
        facts, missing = [], 0
        for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
            tri = cached_facts(sess)
            if tri is None:
                missing += 1
                continue
            for e, a, v in tri:
                facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
        store = assemble_chains(facts)
        label, _ = classify_query(q["question"])
        mode = "historical" if label == "historical" else "current"

        # --- deployed baseline, recomputed directly from run_qc's own recipe ---
        dep_order = sorted(bm25_rank(q["question"], [u[3] for u in units])[:20])
        dep_hits = sorted(set(dep_order) & ev_turns)

        rows = {}
        bm = V5.bm25_scores(q["question"], [u[3] for u in units])
        bm_rank_of = {i: r for r, i in enumerate(sorted(range(len(units)), key=lambda i: (-bm[i], i)))}
        for cfg in CFGS:
            vt, rt, dbg = V5.build_context(q, store, q["question"], config=cfg, mode=mode,
                                           units=units, bm_scores=bm)
            idx = dbg["turn_idx"]
            hits = sorted(set(idx) & ev_turns)
            prov = set(dbg["prov_idx"])
            rows[cfg] = {
                "n_turns": len(idx), "n_turn_sessions": len(dbg["turn_sessions"]),
                "view_chars": len(vt), "raw_chars": len(rt), "total_chars": len(vt) + len(rt),
                "ev_turn_hits": len(hits), "n_ev_turns": len(ev_turns),
                "ev_sess_hits": len({units[i][0] for i in idx} & ev_sess), "n_ev_sess": len(ev_sess),
                "hit_ids": hits,
                "hit_channel": {i: ("prov" if i in prov else "bm25") for i in hits},
                "hit_bm25_rank": {i: bm_rank_of[i] for i in hits},
                "count_mode": dbg["count_mode"],
                "n_prov": len(prov),
            }
        report[qid] = {
            "question": q["question"], "gold": q["answer"], "type": q["question_type"],
            "cache_missing_sessions": missing, "n_chains": len(store),
            "n_units": len(units), "n_sessions": len(q["haystack_sessions"]),
            "ev_turns": ev_units, "ev_sess": sorted(ev_sess), "n_ev_sess_annotated": len(ans_ids),
            "deployed_baseline_hits": dep_hits,
            "cfgs": rows,
        }
    with open("/root/agent_memory_iclr/results/_verif_cases.json", "w") as f:
        json.dump(report, f, indent=1)

    for qid, r in report.items():
        print("=" * 100)
        print(f"{qid}  gold={r['gold']!r}  ev_turns={[e[0] for e in r['ev_turns']]} "
              f"ev_sess={r['ev_sess']} (annotated {r['n_ev_sess_annotated']}) "
              f"cache_missing={r['cache_missing_sessions']} chains={r['n_chains']}")
        print(f"  Q: {r['question']}")
        print(f"  deployed run_qc BM25-top20 evidence hits: {r['deployed_baseline_hits']}")
        for c, d in r["cfgs"].items():
            print(f"  {c:12s} evT {d['ev_turn_hits']}/{d['n_ev_turns']}  evS {d['ev_sess_hits']}/{d['n_ev_sess']}"
                  f"  turns {d['n_turns']:2d}/{d['n_turn_sessions']:2d}sess"
                  f"  view {d['view_chars']:6d} raw {d['raw_chars']:6d} tot {d['total_chars']:6d}"
                  f"  cm={int(d['count_mode'])} hits={d['hit_ids']} via={d['hit_channel']} bmrank={d['hit_bm25_rank']}")


if __name__ == "__main__":
    main()
