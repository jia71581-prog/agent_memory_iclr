"""Item-level recall proxy (grounded in has_answer evidence turns) + evidence-complete
AGG.enum error signature audit. Offline."""
import json, re, sys, hashlib, os, statistics as st
from collections import Counter
from multiprocessing import Pool
sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains
from v5_chains import merge_chains, aggregation_printed_groups, _value_key, _stem_tokens
from _adv6_build import facts_from_cache, gold_number, is_enum_count
from v5_shape import classify_shape

R = "/root/agent_memory_iclr/results"
LIGHTISH = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "my", "i",
            "is", "was", "are", "were", "at", "it", "that", "this", "user"}


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def work(q):
    facts, _h, _m = facts_from_cache(q)
    store5 = merge_chains(assemble_chains(facts))
    kept, _ = aggregation_printed_groups(store5, q["question"])
    sid2idx = {s: i for i, s in enumerate(q["haystack_session_ids"])}
    gold_sidx = {sid2idx[s] for s in q["answer_session_ids"] if s in sid2idx}
    # evidence-turn text: has_answer turns inside gold sessions
    ev_txt = []
    for si, sess in enumerate(q["haystack_sessions"]):
        if si not in gold_sidx:
            continue
        for t in sess:
            if t.get("has_answer") and t.get("content"):
                ev_txt.append(t["content"])
    ev_norm = norm(" ".join(ev_txt))
    ev_sess_norm = norm(" ".join(t.get("content", "") for si, s in enumerate(q["haystack_sessions"])
                                 if si in gold_sidx for t in s))

    def grounded(val, hay):
        toks = [t for t in norm(val).split() if t not in LIGHTISH and len(t) > 2]
        if not toks:
            return False
        return all(t in hay for t in toks)

    def count_grounded(items, hay, restrict_gold=False):
        seen = set()
        for score, c, attr, nterms in items:
            if restrict_gold and c.get("sidx") not in gold_sidx:
                continue
            v = norm(c["value"])
            if v in seen:
                continue
            if grounded(c["value"], hay):
                seen.add(v)
        return len(seen)

    union_items = [it for g in kept for it in g["items"]]
    top_items = kept[0]["items"] if kept else []
    max_g = max(kept, key=lambda g: len(g["items"])) if kept else {"items": []}
    return {
        "qid": q["question_id"],
        "n_ev_turns": len(ev_txt),
        "u_gold_sess_distinct": len({norm(it[1]["value"]) for it in union_items if it[1].get("sidx") in gold_sidx}),
        "t_gold_sess_distinct": len({norm(it[1]["value"]) for it in top_items if it[1].get("sidx") in gold_sidx}),
        "m_gold_sess_distinct": len({norm(it[1]["value"]) for it in max_g["items"] if it[1].get("sidx") in gold_sidx}),
        "u_grounded_evturn": count_grounded(union_items, ev_norm),
        "u_grounded_evturn_gold": count_grounded(union_items, ev_norm, True),
        "u_grounded_evsess": count_grounded(union_items, ev_sess_norm, True),
        "t_grounded_evturn": count_grounded(top_items, ev_norm),
        "m_grounded_evturn": count_grounded(max_g["items"], ev_norm),
    }


if __name__ == "__main__":
    data = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
    with Pool(16) as p:
        extra = {r["qid"]: r for r in p.map(work, data, chunksize=2)}
    rows = json.load(open(f"{R}/_adv6_rows2.json"))
    for r in rows:
        r.update(extra[r["qid"]])
    json.dump(rows, open(f"{R}/_adv6_rows3.json", "w"), ensure_ascii=False)
    QC = {}
    for l in open(f"{R}/full500_qc.jsonl"):
        d = json.loads(l)
        if "judge" in d:
            QC[d["qid"]] = d
    for r in rows:
        r["judge"] = QC[r["qid"]]["judge"]
        r["pred"] = QC[r["qid"]].get("pred", "")

    sub = [r for r in rows if r["enum"]]
    print(f"enum subset n={len(sub)}; questions with >=1 has_answer evidence turn: "
          f"{sum(1 for r in sub if r['n_ev_turns'] > 0)}")
    print("\n=== 'cluster has >= N candidates' vs item-level recall proxies (enum60) ===")
    for f, lab in [("union_dv", "union cardinality >= N (claim's metric, union)"),
                   ("max_dv", "largest cluster cardinality >= N"),
                   ("top_dv", "top-scored cluster cardinality >= N"),
                   ("u_gold_sess_distinct", "union items FROM gold sessions >= N"),
                   ("m_gold_sess_distinct", "largest cluster items from gold sess >= N"),
                   ("u_grounded_evturn", "union items grounded in has_answer turns >= N"),
                   ("u_grounded_evturn_gold", "union items from gold sess AND grounded >= N"),
                   ("t_grounded_evturn", "top cluster items grounded >= N"),
                   ("m_grounded_evturn", "largest cluster items grounded >= N")]:
        ok = sum(1 for r in sub if r[f] >= r["gold_num"])
        print(f"  {lab:48s} {ok:3d}/{len(sub)} = {ok/len(sub):.3f}   mean={st.mean(r[f] for r in sub):5.2f}")
