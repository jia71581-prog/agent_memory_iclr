"""Adversarial variants: sham anchors, relevance-filtered window, generous relevance budget."""
import json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import timedelta
from common import load_dataset
import v5_temporal as V
from v5_temporal import (detect_family, resolve_question_anchors, parse_date, day0,
                         anchor_block, anchor_day_events, extract_events, ANCHOR_WINDOW,
                         _content_tokens, _sentences, best_event_date, ANCHOR_RE, EVENT_CUE,
                         INTENT_RE)
from v5_temporal_measure import contains, contains_stem
from v5_temporal_val import build_store_cached
import re

R = "/root/agent_memory_iclr/results"
data = load_dataset("s_cleaned")
tr = [q for q in data
      if detect_family(q["question"])[0] == "anchor_lookup" and q["question_type"] == "temporal-reasoning"]


def rel_filtered_day(q, anchor_dt, cap=10, window=0, min_rel=1):
    """Same as anchor_day_events but ALSO requires >=min_rel query-token overlap."""
    qtok = set(_content_tokens(q.get("question", ""), qside=True))
    out = []
    for date_s, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        sd = parse_date(date_s)
        if sd is None:
            continue
        same_session = abs((day0(sd) - anchor_dt).days) <= window
        for t in sess:
            if t.get("role") != "user":
                continue
            for sent in _sentences(t.get("content") or ""):
                ev, prov = best_event_date(sent, date_s)
                if ev is None:
                    continue
                default_dated = prov in ("", "today")
                if abs((ev - anchor_dt).days) > window and not (
                        same_session and default_dated and ANCHOR_RE.search(sent)):
                    continue
                cued = bool(ANCHOR_RE.search(sent)) or bool(EVENT_CUE.search(sent))
                if not (cued or re.search(r"\bi\b|\bi'|\bmy\b|\bwe\b|\bour\b", sent, re.I)):
                    continue
                if not cued and sent.rstrip().endswith("?"):
                    continue
                if INTENT_RE.search(sent):
                    continue
                rel = len(qtok & set(_content_tokens(sent)))
                if rel < min_rel:
                    continue
                txt = re.sub(r"\s+", " ", sent).strip()
                if len(txt) > 175:
                    txt = txt[:175].rsplit(" ", 1)[0] + "..."
                pri = 0 if ANCHOR_RE.search(sent) else (1 if cued else 2)
                out.append((pri, -rel, day0(sd), txt))
    seen, res = set(), []
    for pri, nrel, sd, txt in sorted(out, key=lambda x: (x[0], x[1], x[2])):
        k = txt.lower()[:60]
        if k in seen:
            continue
        seen.add(k)
        res.append(txt)
        if len(res) >= cap:
            break
    return res


def block_at(q, anchor_dt, kind, cap=10, fn=anchor_day_events, **kw):
    w0, w1 = ANCHOR_WINDOW.get(kind, (1, 3))
    for w in (w0, w1):
        got = fn(q, anchor_dt, cap=cap, window=w, **({"question": q.get("question")} if fn is anchor_day_events else {}), **kw)
        if got:
            return got, w
    return [], w1


rng = random.Random(20260917)
rows = []
for q in tr:
    store, _ = build_store_cached(q)
    qd = q["question_date"]
    q_dt = parse_date(qd)
    anchors = [a for a in resolve_question_anchors(q["question"], qd) if a["conf"] == "high"]
    if not anchors:
        anchors = [a for a in resolve_question_anchors(q["question"], qd)
                   if a["kind"] in ("offset-coarse", "month-anchor")][:1]
    a = anchors[0]
    kind, adt = a["kind"], a["date"]
    sess_dates = sorted({day0(parse_date(d)) for d in q["haystack_dates"] if parse_date(d)})
    rec = {"qid": q["question_id"], "kind": kind, "gold": str(q["answer"])[:90]}
    G = lambda blk: (any(contains(x, q["answer"]) for x in blk),
                     any(contains_stem(x, q["answer"]) for x in blk), len(blk),
                     sum(len(x) for x in blk))

    # real
    blk, w = block_at(q, adt, kind)
    rec["real"] = G(blk); rec["w"] = w
    # sham anchors of the same machinery
    for nm, d in [("qdate", day0(q_dt)),
                  ("plus30", adt + timedelta(days=30)),
                  ("minus14", adt - timedelta(days=14)),
                  ("rand_sess", rng.choice(sess_dates)),
                  ("rand_sess2", rng.choice(sess_dates)),
                  ("rand_sess3", rng.choice(sess_dates))]:
        b2, _ = block_at(q, d, kind)
        rec[nm] = G(b2)
    # relevance-filtered version of the SAME window/cap (>=1 overlapping content token)
    b3, _ = block_at(q, adt, kind, fn=rel_filtered_day, min_rel=1)
    rec["rel_in_window"] = G(b3)
    b4, _ = block_at(q, adt, kind, fn=rel_filtered_day, min_rel=2)
    rec["rel2_in_window"] = G(b4)
    # cap sweep on the real anchor
    for cap in (3, 5, 10, 20):
        bc, _ = block_at(q, adt, kind, cap=cap)
        rec[f"cap{cap}"] = G(bc)
    # generous relevance-only budget (bypass family budget override)
    for nm, kw in [("rel_gen", dict(family="__x__", max_events=40, rel_floor=0.0,
                                    per_date_turns=6, per_date_facts=3)),
                   ("rel_huge", dict(family="__x__", max_events=100, rel_floor=0.0,
                                     per_date_turns=99, per_date_facts=99))]:
        ev = extract_events(q, store, q["question"], **kw)
        rec[nm] = (any(contains(e.text, q["answer"]) for e in ev),
                   any(contains_stem(e.text, q["answer"]) for e in ev),
                   len(ev), sum(len(e.text) for e in ev))
    rows.append(rec)
    print(".", end="", flush=True)
print()
json.dump(rows, open(f"{R}/_vf_anchor_sham.json", "w"), ensure_ascii=False, indent=1)
keys = ["real", "qdate", "plus30", "minus14", "rand_sess", "rand_sess2", "rand_sess3",
        "rel_in_window", "rel2_in_window", "cap3", "cap5", "cap10", "cap20", "rel_gen", "rel_huge"]
import statistics as stt
n = len(rows)
print(f"{'variant':16s} {'strict':>10s} {'stem':>10s} {'med_lines':>9s} {'med_chars':>9s}")
for k in keys:
    s = sum(1 for r in rows if r[k][0]); m = sum(1 for r in rows if r[k][1])
    print(f"{k:16s} {s}/{n}={s/n:.3f} {m}/{n}={m/n:.3f} "
          f"{stt.median([r[k][2] for r in rows]):9.0f} {stt.median([r[k][3] for r in rows]):9.0f}")
