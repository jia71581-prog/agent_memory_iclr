"""Independent recomputation 2: anchor_lookup metrics, with vacuity + window + crowding checks."""
import pickle, sys, json, re
sys.path.insert(0, "/root/agent_memory_iclr/code")
from v5_temporal import (detect_family, resolve_question_anchors, parse_date, day0,
                         anchor_day_events, anchor_block, ANCHOR_WINDOW, _stem)

tr = pickle.load(open("/root/agent_memory_iclr/vfy2/tr.pkl", "rb"))
gold = pickle.load(open("/root/agent_memory_iclr/vfy2/tr_gold.pkl", "rb"))

def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

STOPW = ("the","and","for","was","were","you","your","not","did","with")

def contains_strict(hay, needle):
    h, n = norm(hay), norm(needle)
    if not n: return False
    if n in h: return True
    toks = [t for t in n.split() if len(t) > 2 and t not in STOPW]
    return bool(toks) and all(t in h for t in toks)

def contains_stem(hay, needle):
    h = {_stem(w) for w in norm(hay).split()}
    toks = [_stem(t) for t in norm(needle).split() if len(t) > 2 and t not in STOPW]
    return bool(toks) and sum(t in h for t in toks) >= max(1, int(round(0.8*len(toks))))

def sess_text(sess):
    return "\n".join(f"[{t.get('role')}] {t.get('content','')}" for t in sess)

rows = []
for q in tr:
    fam, unit = detect_family(q["question"])
    if fam != "anchor_lookup":
        continue
    qid = q["question_id"]
    qd = q["question_date"]
    ans_ids = set(q.get("answer_session_ids", []))
    hay_ids = q.get("haystack_session_ids", [])
    # evidence sessions (from the full haystack, so text == what the pipeline sees)
    ans_pairs = []   # (sid, date, text)
    for sid, dt, sess in zip(hay_ids, q["haystack_dates"], q["haystack_sessions"]):
        if sid in ans_ids:
            ans_pairs.append((sid, day0(parse_date(dt)), sess_text(sess)))
    gold_str = q["answer"]
    goldbearing = [(sid, d) for sid, d, t in ans_pairs if contains_strict(t, gold_str)]
    goldbearing_stem = [(sid, d) for sid, d, t in ans_pairs if contains_stem(t, gold_str)]
    used_fallback = not goldbearing
    tgt = [d for _, d in goldbearing] or [d for _, d, _ in ans_pairs]
    others = [d for _, d, _ in ans_pairs if d not in tgt]

    anchors = [a for a in resolve_question_anchors(q["question"], qd) if a["conf"] == "high"]
    hatch = False
    if not anchors:
        anchors = [a for a in resolve_question_anchors(q["question"], qd)
                   if a["kind"] in ("offset-coarse", "month-anchor")][:1]
        hatch = bool(anchors)
    a0 = anchors[0] if anchors else None
    errs = [abs((a0["date"] - d).days) for d in tgt] if a0 else []
    err = min(errs) if errs else None
    hit = (err is not None and err <= 3)
    oerrs = [abs((a0["date"] - d).days) for d in others] if a0 else []
    disc = bool(hit and (not oerrs or err < min(oerrs)))

    blk, w = anchor_block(q, a0) if a0 else ([], None)
    has = any(contains_strict(x, gold_str) for x in blk)
    has_stem = any(contains_stem(x, gold_str) for x in blk)
    rank = next((i+1 for i, x in enumerate(blk) if contains_stem(x, gold_str)), None)
    rank_strict = next((i+1 for i, x in enumerate(blk) if contains_strict(x, gold_str)), None)
    # crowding: how many haystack sessions lie inside the used window of the anchor
    nsess_in_win = 0
    if a0 is not None and w is not None:
        for dt in q["haystack_dates"]:
            d = day0(parse_date(dt))
            if d is not None and abs((d - a0["date"]).days) <= w:
                nsess_in_win += 1
    rows.append(dict(qid=qid, question=q["question"], gold=gold_str, qdate=qd,
                     n_ans=len(ans_pairs), n_goldbearing=len(goldbearing),
                     n_goldbearing_stem=len(goldbearing_stem),
                     fallback_tgt=used_fallback, escape_hatch=hatch,
                     anchor=str(a0["date"].date()) if a0 else None,
                     anchor_kind=a0["kind"] if a0 else None,
                     anchor_expr=a0["expr"] if a0 else None,
                     n_high=len([a for a in resolve_question_anchors(q["question"], qd) if a["conf"]=="high"]),
                     err=err, hit=hit, disc=disc, others_empty=(not oerrs),
                     window=w, blk_n=len(blk), blk_chars=sum(len(x) for x in blk),
                     has=has, has_stem=has_stem, rank=rank, rank_strict=rank_strict,
                     n_sess_in_window=nsess_in_win,
                     ans_dates=[str(d.date()) for _,d,_ in ans_pairs],
                     gold_dates=[str(d.date()) for _,d in goldbearing]))

json.dump(rows, open("/root/agent_memory_iclr/vfy2/anchor_rows.json","w"), ensure_ascii=False, indent=1)
n = len(rows)
print("n =", n)
print("hit<=3d      :", sum(r["hit"] for r in rows), "/", n)
print("discriminative:", sum(r["disc"] for r in rows), "/", n,
      " (of which vacuous, no other evidence session:", sum(r["disc"] and r["others_empty"] for r in rows), ")")
from collections import Counter
print("err day dist :", dict(Counter(r["err"] for r in rows)))
print("has gold strict:", sum(r["has"] for r in rows), "/", n)
print("has gold stem  :", sum(r["has_stem"] for r in rows), "/", n)
hits = [r for r in rows if r["has_stem"]]
print("rank==1 among stem hits:", sum(r["rank"]==1 for r in hits), "/", len(hits))
print("rank dist:", dict(Counter(r["rank"] for r in rows)))
print("fallback tgt (gold in no evidence session):", sum(r["fallback_tgt"] for r in rows))
print("escape hatch used:", sum(r["escape_hatch"] for r in rows))
print("window used dist:", dict(Counter(r["window"] for r in rows)))
print("blk_n dist:", dict(Counter(r["blk_n"] for r in rows)))
print("sessions inside window: median", sorted(r["n_sess_in_window"] for r in rows)[n//2],
      "max", max(r["n_sess_in_window"] for r in rows))
print()
for r in rows:
    print(f'{r["qid"]:16s} anch={r["anchor"]} kind={r["anchor_kind"]:18s} w={r["window"]} err={r["err"]} '
          f'hit={int(r["hit"])} disc={int(r["disc"])}{"(vac)" if r["disc"] and r["others_empty"] else ""} '
          f'blk={r["blk_n"]}/{r["blk_chars"]}c has={int(r["has"])} stem={int(r["has_stem"])} rank={r["rank"]} '
          f'sessInWin={r["n_sess_in_window"]} nans={r["n_ans"]} ngb={r["n_goldbearing"]}')
