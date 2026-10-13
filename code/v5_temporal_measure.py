"""v5_temporal 离线验证:全 500 题跑时间脚手架,按 TR 题族检查"gold 是否可由脚手架机械推导"。

零 API。真值来自数据集自带的 answer_session_ids(证据会话)→ 其会话日期。
分族判定标准(每族一条明确的可判定条件):
  delta_between / delta_since : gold 数值能否由脚手架给出的日期做减法得到
        A. dates_present  证据会话日期是否都出现在时间线里(必要条件,不可被"堆日期"刷分)
        B. gold_derivable 证据日期(∪ 问题日期)两两/对问题日的差 == gold(单位取问题单位)
        C. gold_in_arith  预计算块里是否**字面**给出了 gold 数值(= 直接把答案递到模型面前)
        D. narrowing      预计算块给出的候选数值个数(越小越有区分度;C 若无 D 无意义)
  ordering                    : 证据会话日期是否全在时间线且构成严格全序(同日证据须靠文内相对
                                表达拆开,单独统计)
  anchor_lookup               : 问句相对/命名日期解析出的锚点,是否落在含 gold 串的证据会话日期
                                ±tol 天内,且比其他证据会话更近(可区分)
  count_before / other        : 只报 dates_present
"""
import json
import os
import re
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from v5_temporal import (render_timeline, extract_events, detect_family, parse_date, day0,
                         days_between, months_between, resolve_question_anchors,
                         precompute_arithmetic, consecutive_pairs, top_dates_by_relevance,
                         resolve_expressions, sentence_dates, _content_tokens, _sentences,
                         anchor_day_events, anchor_block, question_alternatives, match_item_date)
from v5_temporal_val import build_store_cached

R = "/root/agent_memory_iclr/results"
NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
        "a": 1, "an": 1, "half": 0.5}
_G = re.compile(r"(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)"
                r"(?:\s*(?:and a half)?)?\s*(day|week|month|year)s?", re.I)
_GNUM = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$")


def parse_gold_values(gold, unit_hint):
    """gold 串 → [(value, unit)] 全部可接受值(含 '... is also acceptable' 的备选)。"""
    g = str(gold)
    out = []
    for m in _G.finditer(g):
        v = m.group(1).lower()
        v = float(v) if re.match(r"^\d", v) else NUMW.get(v)
        if v is None:
            continue
        if "and a half" in g[max(0, m.start() - 14):m.start() + 14].lower():
            v += 0.5
        out.append((v, m.group(2).lower()))
    if not out:
        m = _GNUM.match(g)
        if m and unit_hint:
            out.append((float(m.group(1)), unit_hint.rstrip("s")))
        else:
            for w, v in NUMW.items():
                if re.match(r"^\s*" + w + r"\s*$", g, re.I) and unit_hint:
                    out.append((v, unit_hint.rstrip("s")))
    # 'ranging from 7 days to 10 days' 已由 finditer 收进两端;补中间整数
    rng = re.search(r"ranging from (\d+) (day|week|month)s? to (\d+)", g, re.I)
    if rng:
        for k in range(int(rng.group(1)), int(rng.group(3)) + 1):
            out.append((float(k), rng.group(2).lower()))
    return sorted(set(out))


def delta_in_unit(d1, d2, unit):
    """返回该单位下的若干合法读法(不同人类约定)。"""
    nd = abs(days_between(d1, d2))
    u = (unit or "day").rstrip("s")
    if u == "day":
        return {float(nd), float(nd + 1)}          # 含端点读法
    if u == "week":
        return {round(nd / 7.0, 1), float(nd // 7), float(round(nd / 7.0))}
    if u == "month":
        m = abs(months_between(min(d1, d2), max(d1, d2)))
        return {float(m), float(round(nd / 30.44)), float(m + 1)}
    if u == "year":
        return {float(round(nd / 365.25)), float(abs(d2.year - d1.year))}
    return {float(nd)}


def gold_match(vals, cand_set, unit):
    for v, u in vals:
        if u != (unit or "day").rstrip("s"):
            continue
        for c in cand_set:
            if abs(c - v) <= 0.05:
                return True
    return False


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def contains_stem(hay, needle):
    """词干版 gold 命中:coverage_diag 的严格版把 'planting 12 tomato saplings' vs
    'planted 12 tomato saplings' 判成不命中,那是度量假阴性而不是脚手架失败。"""
    from v5_temporal import _stem
    h = {_stem(w) for w in norm(hay).split()}
    toks = [_stem(t) for t in norm(needle).split() if len(t) > 2 and t not in
            ("the", "and", "for", "was", "were", "you", "your", "not", "did", "with")]
    return bool(toks) and sum(t in h for t in toks) >= max(1, int(round(0.8 * len(toks))))


def contains(hay, needle):
    h, n = norm(hay), norm(needle)
    if not n:
        return False
    if n in h:
        return True
    toks = [t for t in n.split() if len(t) > 2 and t not in
            ("the", "and", "for", "was", "were", "you", "your", "not", "did", "with")]
    return bool(toks) and all(t in h for t in toks)


def evidence_dates(q):
    """真值上限:只看证据会话(answer_session_ids),把它们的会话日期 + 句内可解析的
    相对/命名日期全部收起来。回答不用到检索,因此这是"时间解析器能不能把
    gold 所需日期做出来"的纯能力测量，与我的相关性排序无关。"""
    sid2 = dict(zip(q["haystack_session_ids"], zip(q["haystack_dates"], q["haystack_sessions"])))
    dates, low = set(), set()
    for sid in q.get("answer_session_ids") or []:
        if sid not in sid2:
            continue
        date_s, sess = sid2[sid]
        sd = parse_date(date_s)
        if sd is None:
            continue
        dates.add(day0(sd))
        for t in sess:
            if t.get("role") != "user":
                continue
            for sent in _sentences(t.get("content") or ""):
                for h in resolve_expressions(sent, date_s):
                    if h["conf"] == "high" and h["kind"] != "explicit-future":
                        dates.add(h["date"])
                    elif h["conf"] == "med":
                        low.add(h["date"])
    return sorted(dates), sorted(low)


def _split_items(text):
    parts = re.split(r",\s*(?:and\s+)?(?:then\s+)?|\s+then\s+|\s+and\s+lastly\s+|\s+or\s+|;\s*", text)
    out = []
    for p in parts:
        p = re.sub(r"^\s*(first|second|third|lastly|finally|next)[,:]?\s*", "", p.strip(), flags=re.I)
        p = re.sub(r"^(i|the day i|the)\s+", "", p.strip(), flags=re.I).strip(" .'\"")
        if len(p) > 2:
            out.append(p)
    return out


def ordering_items(question, gold):
    """返回 (gold 预期顶序, 候选项全集)。二选一型从问句取两个候选，列表型从 gold 取序。"""
    g = str(gold)
    gi = _split_items(re.sub(r"^(the order .*?is:?|first,?)\s*", "", g, flags=re.I))
    gi = [re.sub(r"^\d+\.\s*", "", x) for x in gi]
    if len(gi) >= 3:
        return gi, gi
    m = re.search(r"(?:first|earliest|recently|latest)[,:]?\s*(.+)\?", question, re.I)
    cands = _split_items(m.group(1)) if m else []
    return gi[:1], (cands if len(cands) >= 2 else gi[:1])


def _match_event(item, events):
    """item 文本 → 时间线上最早匹配事件日期(None = 没匹配上)。"""
    it = [w for w in _content_tokens(item) if len(w) > 2]
    if not it:
        return None
    best = None
    for e in events:
        et = set(_content_tokens(e.text))
        hit = sum(w in et for w in it)
        need = len(it) if len(it) <= 2 else max(2, int(round(0.6 * len(it))))
        if hit >= need:
            if best is None or e.ev_dt < best:
                best = e.ev_dt
    return best


def analyse(q, store):
    question, qd = q["question"], q["question_date"]
    q_dt = parse_date(qd)
    family, unit = detect_family(question)
    events = extract_events(q, store, question)
    arith = precompute_arithmetic(question, qd, events, family, unit, q_for_anchor=q)
    pairs, _ = consecutive_pairs(events)
    timeline = render_timeline(q, store, question)

    s2d = dict(zip(q["haystack_session_ids"], q["haystack_dates"]))
    s2t = {sid: " ".join((t.get("content") or "") for t in sess)
           for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"])}
    ans_dates, ans_pairs = [], []
    for sid in q.get("answer_session_ids") or []:
        d = parse_date(s2d.get(sid, ""))
        if d:
            ans_dates.append(day0(d))
            ans_pairs.append((sid, day0(d)))
    ev_dates = {e.ev_dt for e in events}
    sess_dates = {day0(e.sess_dt) for e in events}
    shown = ev_dates | sess_dates

    rec = {"qid": q["question_id"], "family": family, "unit": unit,
           "gold": str(q["answer"])[:90], "n_ans_sessions": len(ans_dates),
           "n_events": len(events), "chars": len(timeline),
           "ans_dates": [d.strftime("%Y/%m/%d") for d in sorted(set(ans_dates))],
           "dates_present": (sum(d in shown for d in set(ans_dates)) / max(1, len(set(ans_dates)))),
           "consec_pairs": len(pairs)}
    vals = parse_gold_values(q["answer"], unit)
    rec["gold_vals"] = vals

    # ---- A. gold 能否由证据日期做减法得到(真值锚定,与我的检索无关)
    derivable, how = False, ""
    cand_dates = sorted(set(ans_dates))
    if vals and family in ("delta_between", "delta_since", "count_before", "other"):
        for i in range(len(cand_dates)):
            for j in range(i + 1, len(cand_dates)):
                if gold_match(vals, delta_in_unit(cand_dates[i], cand_dates[j], unit), unit):
                    derivable, how = True, f"{cand_dates[i]:%m/%d}->{cand_dates[j]:%m/%d}"
        for d in cand_dates:
            if gold_match(vals, delta_in_unit(d, q_dt, unit), unit):
                derivable, how = True, f"{d:%m/%d}->question"
    rec["gold_derivable_from_ans_dates"] = derivable
    rec["derivation"] = how

    # ---- B. 脚手架给出的日期是否也够(用时间线里的日期,而不是真值日期)
    tl_ok, tl_how = False, ""
    tl = sorted(ev_dates)
    if vals:
        for i in range(len(tl)):
            for j in range(i + 1, len(tl)):
                if gold_match(vals, delta_in_unit(tl[i], tl[j], unit), unit):
                    tl_ok, tl_how = True, f"{tl[i]:%m/%d}->{tl[j]:%m/%d}"
                    break
            if tl_ok:
                break
        if not tl_ok:
            for d in tl:
                if gold_match(vals, delta_in_unit(d, q_dt, unit), unit):
                    tl_ok, tl_how = True, f"{d:%m/%d}->question"
                    break
    rec["gold_derivable_from_timeline"] = tl_ok
    rec["tl_derivation"] = tl_how

    # ---- C. 预计算块里是否字面出现 gold 数值 + D. 候选数值个数(区分度)
    atxt = "\n".join(arith)
    lit, ncand = False, 0
    if vals and unit:
        u = unit.rstrip("s")
        got = set()
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:full\s+)?" + u + r"s?\b", atxt):
            got.add(float(m.group(1)))
        ncand = len(got)
        lit = gold_match(vals, got, unit)
    rec["gold_literal_in_arith"] = lit
    rec["n_arith_candidates"] = ncand

    # ---- A2. 真值上限:只用证据会话(会话日期 + 句内解析日期)能否推出 gold
    ev_d, ev_low = evidence_dates(q)
    ok2, how2 = False, ""
    if vals:
        pool = sorted(set(ev_d))
        poolL = sorted(set(ev_d) | set(ev_low))
        for name, P in (("high", pool), ("high+med", poolL)):
            for i in range(len(P)):
                for j in range(i + 1, len(P)):
                    if gold_match(vals, delta_in_unit(P[i], P[j], unit), unit):
                        ok2, how2 = True, f"{name}:{P[i]:%m/%d}->{P[j]:%m/%d}"
                        break
                if ok2:
                    break
            if not ok2:
                for d in P:
                    if gold_match(vals, delta_in_unit(d, q_dt, unit), unit):
                        ok2, how2 = True, f"{name}:{d:%m/%d}->question"
                        break
            if ok2:
                break
    rec["gold_derivable_from_evidence"] = ok2
    rec["evidence_derivation"] = how2
    rec["n_evidence_dates"] = len(ev_d)

    # ---- ordering:证据日期全在 + 严格全序
    if family == "ordering":
        uniq = sorted(set(ans_dates))
        rec["ans_same_day_collision"] = len(uniq) < len(ans_dates)
        rec["ordering_all_present"] = all(d in shown for d in uniq)
        # 同日证据 → 需靠文内表达拆开:检查这些会话的事件日期是否被拆成不同天
        if rec["ans_same_day_collision"]:
            split = {day0(e.sess_dt): set() for e in events}
            for e in events:
                split[day0(e.sess_dt)].add(e.ev_dt)
            rec["same_day_split"] = any(len(v) > 1 for k, v in split.items() if k in uniq)
        rec["ordering_strict"] = bool(rec["ordering_all_present"] and not rec["ans_same_day_collision"])
        # 条目级可判定性:把 gold 里(或问句候选里)的每个条目对到时间线上最早的匹配事件,
        # 看时间线给出的先后是否与 gold 一致。二选一型的随机基线 = 0.5,列表型 = 1/n!。
        gseq, cands = ordering_items(question, q["answer"])
        dates_i = {c: _match_event(c, events) for c in cands}
        rec["ordering_items"] = cands
        rec["ordering_items_matched"] = sum(v is not None for v in dates_i.values())
        rec["ordering_n_items"] = len(cands)
        verdict = None
        if len(cands) >= 2 and all(v is not None for v in dates_i.values()):
            got = sorted(cands, key=lambda c: dates_i[c])
            if len(gseq) >= 3:
                verdict = [norm(x) for x in got] == [norm(x) for x in gseq]
            elif gseq:
                verdict = contains(got[0], gseq[0]) or contains(gseq[0], got[0])
        rec["ordering_item_correct"] = verdict

    # ---- anchor_lookup:锚点是否指向含 gold 的那个证据会话
    if family == "ordering":
        alts = question_alternatives(question)
        rec["n_alts"] = len(alts)
        rec["missing_alts"] = [a for a in alts if match_item_date(a, events) is None]
        rec["missing_note_fires"] = bool(alts and rec["missing_alts"])
    if family == "anchor_lookup":
        anchors = [a for a in resolve_question_anchors(question, qd) if a["conf"] == "high"]
        rec["n_high_anchors"] = len(anchors)
        gold_sid = [sid for sid, d in ans_pairs if contains(s2t.get(sid, ""), q["answer"])]
        tgt = [d for sid, d in ans_pairs if sid in gold_sid]
        if not tgt:                                  # gold 串不在任何证据会话 → 用最近的证据会话兜底
            tgt = [d for _, d in ans_pairs]
        # 锚日直取是否真的把 gold 摊出来了(anchor_lookup 族最强的可判定指标)
        rec["anchor_day_has_gold"] = False
        rec["anchor_day_n"] = 0
        if not anchors:
            anchors = [a for a in resolve_question_anchors(question, qd)
                       if a["kind"] in ("offset-coarse", "month-anchor")][:1]
        if anchors:
            day = anchor_block(q, anchors[0])[0]
            rec["anchor_day_n"] = len(day)
            rec["anchor_day_has_gold"] = any(contains(x, q["answer"]) for x in day)
            rec["anchor_day_has_gold_stem"] = any(contains_stem(x, q["answer"]) for x in day)
            rec["anchor_day_gold_rank"] = next(
                (i + 1 for i, x in enumerate(day) if contains_stem(x, q["answer"])), None)
        rec["anchor_hit"] = False
        rec["anchor_discriminative"] = False
        rec["anchor_err_days"] = None
        if anchors and tgt:
            a = anchors[0]["date"]
            errs = [abs((a - d).days) for d in tgt]
            rec["anchor_err_days"] = min(errs)
            rec["anchor_hit"] = min(errs) <= 3
            others = [abs((a - d).days) for _, d in ans_pairs if d not in tgt]
            rec["anchor_discriminative"] = bool(rec["anchor_hit"] and
                                                (not others or min(errs) < min(others)))
    return rec, timeline


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    data = load_dataset("s_cleaned")
    if only == "tr":
        data = [q for q in data if q["question_type"] == "temporal-reasoning"]
    out, tls = [], {}
    for i, q in enumerate(data):
        store, miss = build_store_cached(q)
        rec, tl = analyse(q, store)
        rec["type"] = q["question_type"]
        rec["cache_miss"] = miss
        out.append(rec)
        tls[q["question_id"]] = tl
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(data)}", flush=True)
    tag = "tr" if only == "tr" else "all"
    with open(f"{R}/v5_temporal_{tag}.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(f"{R}/v5_timelines_{tag}.json", "w") as f:
        json.dump(tls, f, ensure_ascii=False)
    print("saved", f"{R}/v5_temporal_{tag}.json", len(out))


if __name__ == "__main__":
    main()
