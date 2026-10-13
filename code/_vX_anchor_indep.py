"""Independent adversarial recomputation of the anchor_lookup claim. Offline, zero API."""
import json
import os
import random
import re
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from v5_temporal import (detect_family, parse_date, day0, resolve_question_anchors,
                         anchor_day_events, anchor_block, extract_events, _sentences,
                         _content_tokens, _stem, ANCHOR_WINDOW)
from v5_temporal_val import build_store_cached

R = "/root/agent_memory_iclr/results"


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def gold_toks(gold):
    n = norm(gold)
    return [t for t in n.split() if len(t) > 2 and t not in
            ("the", "and", "for", "was", "were", "you", "your", "not", "did", "with")]


def contains_all(hay, gold):
    """same convention as v5_temporal_measure.contains"""
    h, n = norm(hay), norm(gold)
    if not n:
        return False
    if n in h:
        return True
    t = gold_toks(gold)
    return bool(t) and all(x in h.split() or x in h for x in t)


def contains_substr(hay, gold):
    """strict: full normalized gold substring only"""
    n = norm(gold)
    return bool(n) and n in norm(hay)


def contains_stem(hay, gold):
    h = {_stem(w) for w in norm(hay).split()}
    t = [_stem(x) for x in gold_toks(gold)]
    return bool(t) and sum(x in h for x in t) >= max(1, int(round(0.8 * len(t))))


def main():
    data = load_dataset("s_cleaned")
    tr = [q for q in data if q["question_type"] == "temporal-reasoning"]
    out = []
    rnd = random.Random(20260917)
    for q in tr:
        fam, unit = detect_family(q["question"])
        if fam != "anchor_lookup":
            continue
        qid = q["question_id"]
        gold = q["answer"]
        qd = q["question_date"]
        q_dt = parse_date(qd)
        store, miss = build_store_cached(q)

        anchors = [a for a in resolve_question_anchors(q["question"], qd) if a["conf"] == "high"]
        if not anchors:
            anchors = [a for a in resolve_question_anchors(q["question"], qd)
                       if a["kind"] in ("offset-coarse", "month-anchor")][:1]
        rec = {"qid": qid, "gold": str(gold), "n_gold_toks": len(gold_toks(gold)),
               "question": q["question"], "qdate": qd, "cache_miss": miss,
               "n_high_anchors": len(anchors)}

        # --- evidence sessions
        sid2date = dict(zip(q["haystack_session_ids"], q["haystack_dates"]))
        sid2txt = {sid: " ".join((t.get("content") or "") for t in sess)
                   for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"])}
        ans = [(sid, day0(parse_date(sid2date[sid]))) for sid in (q.get("answer_session_ids") or [])
               if sid in sid2date and parse_date(sid2date[sid])]
        gold_sids = [sid for sid, d in ans if contains_all(sid2txt.get(sid, ""), gold)]
        rec["n_ans_sessions"] = len(ans)
        rec["n_gold_ans_sessions"] = len(gold_sids)
        rec["gold_fallback_used"] = (len(gold_sids) == 0)
        tgt = [d for sid, d in ans if sid in gold_sids] or [d for _, d in ans]
        rec["n_tgt_dates"] = len(set(tgt))
        others = sorted({d for _, d in ans} - set(tgt))
        rec["n_other_ans_dates"] = len(others)
        rec["disc_vacuous"] = (len(others) == 0)

        # --- anchor error
        if anchors:
            a = anchors[0]
            rec["anchor_kind"] = a["kind"]
            rec["anchor_expr"] = a["expr"]
            rec["anchor_date"] = a["date"].strftime("%Y/%m/%d")
            errs = [abs((a["date"] - d).days) for d in tgt]
            rec["err_days"] = min(errs)
            rec["hit3"] = min(errs) <= 3
            rec["hit0"] = min(errs) == 0
            rec["disc_vs_other_ans"] = bool(rec["hit3"] and
                                            (not others or min(errs) < min(abs((a["date"] - d).days) for d in others)))
            # discriminativeness vs ALL haystack sessions, not just evidence sessions
            all_days = sorted({day0(parse_date(ds)) for ds in q["haystack_dates"] if parse_date(ds)})
            w0, w1 = ANCHOR_WINDOW.get(a["kind"], (1, 3))
            rec["n_hay_sessions"] = len(q["haystack_dates"])
            rec["n_hay_days"] = len(all_days)
            for w in (0, 1, 3):
                rec[f"n_hay_days_within_{w}"] = sum(1 for d in all_days
                                                    if abs((d - a["date"]).days) <= w)
                rec[f"n_hay_sess_within_{w}"] = sum(1 for ds in q["haystack_dates"]
                                                    if parse_date(ds) and
                                                    abs((day0(parse_date(ds)) - a["date"]).days) <= w)
            # --- real block
            blk, used_w = anchor_block(q, a)
            rec["used_window"] = used_w
            rec["block_n"] = len(blk)
            rec["block_chars"] = sum(len(x) for x in blk)
            rec["gold_all"] = any(contains_all(x, gold) for x in blk)
            rec["gold_substr"] = any(contains_substr(x, gold) for x in blk)
            rec["gold_stem"] = any(contains_stem(x, gold) for x in blk)
            rec["gold_rank_all"] = next((i + 1 for i, x in enumerate(blk) if contains_all(x, gold)), None)
            rec["gold_rank_stem"] = next((i + 1 for i, x in enumerate(blk) if contains_stem(x, gold)), None)
            # how many block lines come from the gold session's day
            rec["block_uniq_lines_hit"] = sum(1 for x in blk if contains_all(x, gold))

            # --- SHAM anchors: same machinery, wrong day
            sham = {}
            for off in (-30, -14, -7, 7, 14, 30):
                d2 = a["date"] + timedelta(days=off)
                b2 = anchor_day_events(q, d2, cap=10, window=used_w, question=q["question"])
                if not b2:
                    b2 = anchor_day_events(q, d2, cap=10, window=max(used_w, 3), question=q["question"])
                sham[f"off{off:+d}"] = {"n": len(b2), "gold_all": any(contains_all(x, gold) for x in b2),
                                        "gold_stem": any(contains_stem(x, gold) for x in b2)}
            # random day inside haystack range
            lo, hi = min(all_days), max(all_days)
            span = (hi - lo).days
            rg = []
            for _ in range(5):
                d2 = lo + timedelta(days=rnd.randint(0, max(1, span)))
                b2 = anchor_day_events(q, d2, cap=10, window=3, question=q["question"])
                rg.append({"n": len(b2), "gold_all": any(contains_all(x, gold) for x in b2),
                           "gold_stem": any(contains_stem(x, gold) for x in b2)})
            sham["rand3"] = rg
            # question-day dump
            b2 = anchor_day_events(q, day0(q_dt), cap=10, window=3, question=q["question"])
            sham["qday"] = {"n": len(b2), "gold_all": any(contains_all(x, gold) for x in b2),
                            "gold_stem": any(contains_stem(x, gold) for x in b2)}
            rec["sham"] = sham
        # --- size-matched lexical-only baseline (no dates at all): top-k by same IDF score
        evs = extract_events(q, store, q["question"])
        rec["n_events"] = len(evs)
        rec["events_chars"] = sum(len(e.text) for e in evs)
        by_rank = sorted(evs, key=lambda e: e.rank)
        for k in (10, 20):
            sub = [e.text for e in by_rank[:k]]
            rec[f"rel{k}_gold_all"] = any(contains_all(x, gold) for x in sub)
            rec[f"rel{k}_gold_stem"] = any(contains_stem(x, gold) for x in sub)
            rec[f"rel{k}_chars"] = sum(len(x) for x in sub)
        out.append(rec)
        print(f"  {qid} done", flush=True)
    with open(f"{R}/_vX_anchor_indep.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved", len(out))


if __name__ == "__main__":
    main()
