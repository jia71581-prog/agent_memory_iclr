"""Independent adversarial recomputation of the anchor_lookup claim. Zero API."""
import json, os, re, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import timedelta
from common import load_dataset
import v5_temporal as V
from v5_temporal import (detect_family, resolve_question_anchors, parse_date, day0,
                         anchor_block, anchor_day_events, extract_events, render_timeline,
                         ANCHOR_WINDOW, _content_tokens, _sentences, best_event_date,
                         ANCHOR_RE, EVENT_CUE, INTENT_RE)
from v5_temporal_measure import contains, contains_stem, norm
from v5_temporal_val import build_store_cached

R = "/root/agent_memory_iclr/results"
data = load_dataset("s_cleaned")
al = []
for q in data:
    fam, unit = detect_family(q["question"])
    if fam == "anchor_lookup":
        al.append(q)
print("anchor_lookup total", len(al),
      "TR", sum(1 for q in al if q["question_type"] == "temporal-reasoning"))
tr = [q for q in al if q["question_type"] == "temporal-reasoning"]

rows = []
for q in tr:
    store, miss = build_store_cached(q)
    qd = q["question_date"]
    anchors = [a for a in resolve_question_anchors(q["question"], qd) if a["conf"] == "high"]
    fallback = False
    if not anchors:
        anchors = [a for a in resolve_question_anchors(q["question"], qd)
                   if a["kind"] in ("offset-coarse", "month-anchor")][:1]
        fallback = bool(anchors)
    s2d = dict(zip(q["haystack_session_ids"], q["haystack_dates"]))
    s2t = {sid: " ".join((t.get("content") or "") for t in sess)
           for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"])}
    ans_pairs = []
    for sid in q.get("answer_session_ids") or []:
        d = parse_date(s2d.get(sid, ""))
        if d:
            ans_pairs.append((sid, day0(d)))
    gold_sid = [sid for sid, d in ans_pairs if contains(s2t.get(sid, ""), q["answer"])]
    tgt = [d for sid, d in ans_pairs if sid in gold_sid]
    tgt_from_gold = bool(tgt)
    if not tgt:
        tgt = [d for _, d in ans_pairs]
    rec = {"qid": q["question_id"], "gold": str(q["answer"])[:110],
           "question": q["question"], "qdate": qd,
           "n_ans_sessions": len(ans_pairs), "n_gold_sessions": len(gold_sid),
           "tgt_from_gold": tgt_from_gold, "fallback_anchor": fallback,
           "anchor_kind": anchors[0]["kind"] if anchors else None,
           "anchor_expr": anchors[0]["expr"] if anchors else None,
           "anchor_date": anchors[0]["date"].strftime("%Y/%m/%d") if anchors else None,
           "n_high_anchors": len([a for a in resolve_question_anchors(q["question"], qd)
                                  if a["conf"] == "high"])}
    if anchors and tgt:
        a = anchors[0]["date"]
        errs = [abs((a - d).days) for d in tgt]
        others = [abs((a - d).days) for _, d in ans_pairs if d not in tgt]
        rec["err_days"] = min(errs)
        rec["n_others"] = len(others)
        rec["others_empty"] = not others
        rec["disc"] = bool(min(errs) <= 3 and (not others or min(errs) < min(others)))
        for tol in (0, 1, 2, 3, 5, 7):
            rec[f"hit_tol{tol}"] = bool(min(errs) <= tol)
    # deployed block
    if anchors:
        day, w = anchor_block(q, anchors[0])
        rec["w_used"] = w
        rec["w_cfg"] = ANCHOR_WINDOW.get(anchors[0]["kind"])
        rec["block_n"] = len(day)
        rec["block_chars"] = sum(len(x) for x in day)
        rec["has_gold"] = any(contains(x, q["answer"]) for x in day)
        rec["has_gold_stem"] = any(contains_stem(x, q["answer"]) for x in day)
        rec["gold_rank"] = next((i + 1 for i, x in enumerate(day)
                                 if contains_stem(x, q["answer"])), None)
        rec["gold_rank_strict"] = next((i + 1 for i, x in enumerate(day)
                                        if contains(x, q["answer"])), None)
        rec["block"] = day
    # relevance-only baseline (timeline events, no anchor block) and matched-budget version
    for n in (18, 28, 40):
        ev = extract_events(q, store, q["question"], max_events=n)
        rec[f"rel{n}_has_gold"] = any(contains(e.text, q["answer"]) for e in ev)
        rec[f"rel{n}_has_gold_stem"] = any(contains_stem(e.text, q["answer"]) for e in ev)
        rec[f"rel{n}_n"] = len(ev)
        rec[f"rel{n}_chars"] = sum(len(e.text) for e in ev)
    rec["timeline_chars"] = len(render_timeline(q, store, q["question"]))
    rows.append(rec)
    print(".", end="", flush=True)
print()
json.dump(rows, open(f"{R}/_vf_anchor_repro.json", "w"), ensure_ascii=False, indent=1)


def rate(key, sub=None):
    rs = rows if sub is None else sub
    n = sum(1 for r in rs if r.get(key))
    return f"{n}/{len(rs)} = {n/len(rs):.3f}"


print("n =", len(rows))
for k in ["hit_tol0", "hit_tol1", "hit_tol2", "hit_tol3", "disc", "has_gold", "has_gold_stem",
          "rel18_has_gold", "rel18_has_gold_stem", "rel28_has_gold_stem", "rel40_has_gold_stem",
          "others_empty", "tgt_from_gold", "fallback_anchor"]:
    print(f"  {k:24s} {rate(k)}")
from collections import Counter
print("  anchor_kind", Counter(r["anchor_kind"] for r in rows))
print("  w_used", Counter(r.get("w_used") for r in rows))
print("  block_n", Counter(r.get("block_n") for r in rows))
print("  gold_rank(stem)", Counter(r.get("gold_rank") for r in rows))
print("  gold_rank(strict)", Counter(r.get("gold_rank_strict") for r in rows))
hits = [r for r in rows if r.get("has_gold_stem")]
print("  rank1 among stem hits:", sum(1 for r in hits if r.get("gold_rank") == 1), "/", len(hits))
hs = [r for r in rows if r.get("has_gold")]
print("  rank1 among strict hits:", sum(1 for r in hs if r.get("gold_rank_strict") == 1), "/", len(hs))
