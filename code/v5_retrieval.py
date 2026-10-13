"""QC-Lifecycle v5: provenance-guided evidence retrieval (training-free, no API at query time).

Motivation (measured, see v5_probe.py / v5_eval.py):
  1. run_qc.py caps the rendered view at 20,000 chars but render_view() emits a mean of
     30,629 chars sorted ALPHABETICALLY by chain key, so 34.5% of chain lines are dropped by
     an ordering that is independent of the question. 50/470 questions lose the gold string
     to that truncation alone (real deployed view coverage 0.528, not the 0.634 that
     coverage_diag.py reports on the un-capped view).
  2. Oracle evidence-turn recall of BM25 top-20 is 0.767, and only 0.529 of multi-session
     questions have their FULL has_answer turn set inside the top-20 - while 0.930 of gold
     evidence *sessions* are already touched by top-20 and 0.994 of them carry at least one
     extracted fact. So the fact store is a usable index into the right sessions; the failure
     is turn-level, inside sessions we already reach. Raising k does not fix it (measured:
     k=20 -> 50 recovers 0 of the 24 evidence-missing errors), because lexical turn ranking
     concentrates its budget on the same few sessions.

Design (all deterministic, no model calls):
  score_chains()    - IDF-weighted question/chain lexical match, attribute slot boosted.
  rank_view()       - render the view in relevance order so the char cap keeps the chains the
                      question is about (drop-in replacement for render_view + tail truncation).
  session_scores()  - project chain scores onto the sessions their facts came from (provenance).
  select_turns()    - round-robin over provenance sessions, value-anchored inside each session.
  build_context()   - assemble (view_text, raw_text, debug).

Public entry point:
    build_context(q, store, question, budget_chars=None, config="v5_drop_in")
      q        : LongMemEval question dict (haystack_dates / haystack_sessions[/ _session_ids])
      store    : lifecycle.assemble_chains() output for that question
      question : question string (kept explicit so callers can pass a rewritten query)
      budget_chars : (view_cap, raw_cap) or a single int total; None -> config defaults
      returns  : (view_text, raw_text, debug_dict)
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifecycle import _group_by_session

# ---------------------------------------------------------------- budgets
VIEW_CAP = 20000          # identical to run_qc.VIEW_CHAR_CAP
RAW_TURNS = 20            # identical to run_qc.TOP_K
COUNT_RE = re.compile(r"how many|how much|total|number of", re.I)

_word = re.compile(r"[a-z0-9']+")
STOP = set("""a an the and or but if then than that this these those of in on at to for from by with
about into over after before is are was were be been being do does did doing have has had having
i me my mine we our you your he she it they them his her its their as so not no nor too very can
will would should could may might must just also more most other some such only own same s t don
now what which who whom when where why how many much any all each both few there here get got""".split())


def toks(s):
    return _word.findall(str(s).lower())


def qterms(question):
    """Content tokens of the question, deduped, order preserved."""
    out, seen = [], set()
    for t in toks(question):
        if t in STOP or len(t) < 3:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _idf(docs_tokens):
    """Smoothed IDF over a per-question document collection."""
    n = len(docs_tokens)
    df = {}
    for d in docs_tokens:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    return {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}, n


def bm25_scores(query, docs, k1=1.5, b=0.75):
    """BM25 scores (same formula/params as run_baseline.bm25_rank, which returns only the order).
    Returned as a list aligned with docs so callers can rank inside a subset of documents."""
    qt = toks(query)
    dt = [toks(d) for d in docs]
    avgdl = sum(len(d) for d in dt) / max(1, len(dt))
    df = {}
    for d in dt:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    ndocs = len(dt)
    out = []
    for d in dt:
        tf = {}
        for w in d:
            tf[w] = tf.get(w, 0) + 1
        s = 0.0
        for w in qt:
            if w not in tf:
                continue
            idf = math.log(1 + (ndocs - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avgdl))
        out.append(s)
    return out


# ---------------------------------------------------------------- 1. chain scoring
def chain_text(chain):
    ent, attr = chain[0]["entity"], chain[0]["attribute"]
    return ent, attr, [c["value"] for c in chain]


def score_chains(store, question, attr_boost=2.0, value_weight=1.0, entity_weight=1.0):
    """Score every lifecycle chain against the question.

    score = sum over question content terms t of idf(t) * max(
                attr_boost   * 1[t in attribute],
                entity_weight* 1[t in entity],
                value_weight * 1[t in any value]) ,
    normalised by sqrt(1 + log(#value tokens)) so verbose chains do not win on length alone.
    The attribute is boosted because it is the slot the question names ("job", "rent", "jogs"),
    while values are the answer side.
    Returns list of (key, score, n_sessions) sorted by (-score, key) -> fully deterministic.
    """
    keys = sorted(store)
    docs = []
    for k in keys:
        ent, attr, vals = chain_text(store[k])
        docs.append(toks(ent) + toks(attr) + toks(" ".join(vals)))
    idf, _ = _idf(docs)
    qt = qterms(question)
    scored = []
    for k in keys:
        chain = store[k]
        ent, attr, vals = chain_text(chain)
        a_set, e_set = set(toks(attr)), set(toks(ent))
        v_set = set(toks(" ".join(vals)))
        s = 0.0
        for t in qt:
            w = idf.get(t, 0.0)
            if w <= 0:
                continue
            hit = 0.0
            if t in a_set:
                hit = attr_boost
            if t in e_set:
                hit = max(hit, entity_weight)
            if t in v_set:
                hit = max(hit, value_weight)
            s += w * hit
        nv = sum(len(toks(v)) for v in vals)
        s /= math.sqrt(1.0 + math.log(1.0 + max(1, nv)))
        scored.append((k, s, len({(c["date"], c.get("sidx")) for c in chain})))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored


# ---------------------------------------------------------------- 2. relevance-ordered view
def _render_chain_line(chain, mode):
    """One view line for one chain - byte-identical to lifecycle.render_view()'s per-chain branch."""
    ent, attr = chain[0]["entity"], chain[0]["attribute"]
    is_accum = chain[0]["relation"] == "accumulative"
    has_conflict = any(c["relation"] == "contradiction" for c in chain)
    if is_accum:
        vals = "; ".join(f"{c['value']} ({c['date'].split()[0]})" for c in chain)
        return f"{ent} | {attr} = {vals}"
    if mode == "current":
        groups = _group_by_session(chain)
        last_date, last_facts = groups[-1][0][0], groups[-1][1]
        vals = "; ".join(c["value"] for c in last_facts)
        tag = " [CONFLICT: multiple values reported at the same time]" if has_conflict else ""
        return f"[{last_date}] {ent} | {attr} = {vals}{tag}"
    groups = _group_by_session(chain)
    if len(groups) == 1:
        vals = " / ".join(c["value"] for c in groups[0][1])
        return f"[{groups[0][0][0]}] {ent} | {attr} = {vals}"
    steps = " -> ".join(
        " / ".join(c["value"] for c in g[1])
        + f" ({g[0][0].split()[0]}{'/conflict' if g[1][0]['relation'] == 'contradiction' else ''})"
        for g in groups)
    return f"{ent} | {attr}: {steps}"


def rank_view(store, question, mode, cap=VIEW_CAP, scored=None, focus_head=0):
    """Render the view keeping the chains the question is about when the cap bites.

    Same lines and same cap as run_qc (line-boundary truncation); only the ORDER in which lines
    compete for the budget changes: relevance-first instead of alphabetical-by-chain-key.
    Kept lines are re-emitted in the original alphabetical order so the rendered text is
    order-identical to render_view() restricted to the kept subset (no new format to the LLM).
    focus_head>0 additionally front-loads the top-n scoring chains (position effects on the
    answerer are NOT offline-measurable; only the char position of the gold string is).
    """
    if scored is None:
        scored = score_chains(store, question)
    kept, total = [], 0
    for k, _s, _n in scored:
        line = _render_chain_line(store[k], mode)
        if total + len(line) + 1 > cap:
            continue          # skip, keep filling with shorter relevant lines
        kept.append(k)
        total += len(line) + 1
    if focus_head:
        head = kept[:focus_head]
        tail = sorted(kept[focus_head:])
        order = head + tail
    else:
        order = sorted(kept)
    return "\n".join(_render_chain_line(store[k], mode) for k in order), order


# ---------------------------------------------------------------- 3. chains -> sessions
def session_scores(store, scored, top_chains=40, rank_decay=0.9):
    """Project chain relevance onto provenance sessions.

    Every fact carries (date, sidx) from the session it was extracted from. A session's score is
    the decayed sum of the scores of the top-`top_chains` chains that have a fact in it, so a
    session vouched for by several matching chains outranks one vouched for by a single chain.
    Returns (sess_score: {sidx: float}, sess_values: {sidx: [value strings of matching facts]}).
    """
    sess, vals = {}, {}
    for rank, (k, s, _n) in enumerate(scored[:top_chains]):
        if s <= 0:
            break
        w = s * (rank_decay ** rank)
        for c in store[k]:
            si = c.get("sidx")
            if si is None:
                continue
            sess[si] = sess.get(si, 0.0) + w
            vals.setdefault(si, []).append(c["value"])
    return sess, vals


def chain_session_sets(store, scored, top_chains=40, min_sessions=2):
    """Provenance session sets of the best-matching MULTI-session chains, best chain first.
    Used by count mode: for an aggregation question the complete evidence set is (approximately)
    every session the matching chain touches, not the k most lexically similar turns."""
    out = []
    for k, s, n in scored[:top_chains]:
        if s <= 0:
            break
        if n >= min_sessions:
            sidxs = sorted({c.get("sidx") for c in store[k] if c.get("sidx") is not None})
            out.append((k, s, sidxs))
    return out


# ---------------------------------------------------------------- 4. turn selection
def build_units(q):
    """[(sidx, date, role, content)] in haystack order - same unit definition as run_qc/run_baseline."""
    units = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for t in sess:
            if t.get("content"):
                units.append((si, date, t["role"], t["content"]))
    return units


def _value_anchor(content, values, qt):
    """How strongly a turn is the source of the matched fact values: fraction of value content
    tokens present in the turn, over the values whose tokens also matter to the question."""
    if not values:
        return 0.0
    ct = set(toks(content))
    best = 0.0
    for v in values:
        vt = [t for t in toks(v) if t not in STOP and len(t) > 2]
        if not vt:
            continue
        frac = sum(t in ct for t in vt) / len(vt)
        if any(t in qt for t in vt):
            frac *= 1.5
        best = max(best, frac)
    return min(best, 1.5)


def select_turns(q, store, question, scored, n_turns, bm_scores=None, units=None,
                 top_sessions=8, per_session=3, anchor_weight=1.0, count_mode=False,
                 top_chains=40):
    """Provenance turn selection: round-robin over the highest-scoring provenance sessions.

    Round-robin is the point. A flat BM25 top-20 spends its budget wherever the lexical mass is
    (measured: 10.2 distinct sessions, but the evidence turn of a multi-session question is
    routinely outside it). Taking the best turn of session 1, then of session 2, ... guarantees
    breadth over the sessions the fact store says are relevant.
    count_mode: order the sessions by the provenance set of the best multi-session chain first,
    so an aggregation question gets one turn from EVERY session of the chain being counted.
    """
    if units is None:
        units = build_units(q)
    if bm_scores is None:
        bm_scores = bm25_scores(question, [u[3] for u in units])
    qt = set(qterms(question))
    sess, svals = session_scores(store, scored, top_chains=top_chains)

    order = sorted(sess, key=lambda si: (-sess[si], si))
    if count_mode:
        head = []
        for _k, _s, sidxs in chain_session_sets(store, scored, top_chains=top_chains):
            for si in sorted(sidxs, key=lambda x: (-sess.get(x, 0.0), x)):
                if si not in head:
                    head.append(si)
            if len(head) >= max(top_sessions, n_turns):
                break
        order = head + [si for si in order if si not in head]
    order = order[:max(top_sessions, 1)]

    by_sess = {}
    for i, u in enumerate(units):
        by_sess.setdefault(u[0], []).append(i)
    ranked = {}
    for si in order:
        idxs = by_sess.get(si, [])
        if not idxs:
            continue
        mx = max((bm_scores[i] for i in idxs), default=0.0) or 1.0
        ranked[si] = sorted(
            idxs,
            key=lambda i: (-(bm_scores[i] / mx + anchor_weight * _value_anchor(units[i][3], svals.get(si, []), qt)), i)
        )[:per_session]

    picked, slot = [], 0
    while len(picked) < n_turns and slot < per_session:
        added = False
        for si in order:
            if len(picked) >= n_turns:
                break
            lst = ranked.get(si, [])
            if slot < len(lst):
                picked.append(lst[slot])
                added = True
        if not added:
            break
        slot += 1
    return picked, order, sess


def bm25_turns(question, units, n_turns, bm_scores=None, session_cap=None):
    """Global BM25 turn budget, optionally with a max-turns-per-session diversity cap.
    session_cap=None reproduces run_qc's selection exactly."""
    if bm_scores is None:
        bm_scores = bm25_scores(question, [u[3] for u in units])
    order = sorted(range(len(units)), key=lambda i: (-bm_scores[i], i))
    if session_cap is None:
        return order[:n_turns]
    out, used = [], {}
    for i in order:
        si = units[i][0]
        if used.get(si, 0) >= session_cap:
            continue
        used[si] = used.get(si, 0) + 1
        out.append(i)
        if len(out) >= n_turns:
            break
    return out


# ---------------------------------------------------------------- 5. assembly
CONFIGS = {
    # exact reproduction of the deployed arm: alphabetical view + tail cut, flat BM25 top-20
    "baseline":     dict(view="alpha", n_bm25=20, n_prov=0, session_cap=None),
    # only change: the view budget is spent on question-relevant chains
    "rankview":     dict(view="rank",  n_bm25=20, n_prov=0, session_cap=None),
    # diversity-only control: no chain information, just spread BM25 over sessions
    "divcap2":      dict(view="rank",  n_bm25=20, n_prov=0, session_cap=2),
    # provenance-guided: part of the turn budget is chain-selected, session-round-robin
    "prov_hybrid":  dict(view="rank",  n_bm25=12, n_prov=8,  session_cap=None),
    "prov_hybrid6": dict(view="rank",  n_bm25=14, n_prov=6,  session_cap=None),
    "prov_only":    dict(view="rank",  n_bm25=0,  n_prov=20, session_cap=None),
    # full method: relevance view + provenance turns + count-mode session expansion
    "v5":           dict(view="rank",  n_bm25=12, n_prov=8,  session_cap=None, count_mode=True),
    "v5_divcap":    dict(view="rank",  n_bm25=12, n_prov=8,  session_cap=3,    count_mode=True),
    # provenance-selection ablations (all still 20 turns, matched budget)
    "v5_deep":      dict(view="rank",  n_bm25=12, n_prov=8,  session_cap=None, count_mode=True,
                         top_sessions=4, per_session=2),
    "v5_wide":      dict(view="rank",  n_bm25=10, n_prov=10, session_cap=None, count_mode=True,
                         top_sessions=10),
    "v5_noanchor":  dict(view="rank",  n_bm25=12, n_prov=8,  session_cap=None, count_mode=True,
                         anchor_weight=0.0),
    "v5_nocount":   dict(view="rank",  n_bm25=12, n_prov=8,  session_cap=None, count_mode=False),
    "v5_focus":     dict(view="rank",  n_bm25=12, n_prov=8,  session_cap=None, count_mode=True,
                         focus_head=10),
    # larger-budget references (NOT budget matched; they answer "is this just more turns?")
    "bm25_k30":     dict(view="rank",  n_bm25=30, n_prov=0, session_cap=None),
    "bm25_k50":     dict(view="rank",  n_bm25=50, n_prov=0, session_cap=None),
    # budget reallocation: shrink the view, spend the chars on turns (total chars ~matched)
    "realloc":      dict(view="rank",  n_bm25=17, n_prov=11, session_cap=None, count_mode=True,
                         top_sessions=11, view_cap=12000),
    # reallocation CONTROL: same shrunken view, same 28 turns, but no provenance at all.
    # Isolates "spend chars on turns instead of view" from "select turns by provenance".
    "realloc_bm":   dict(view="rank",  n_bm25=28, n_prov=0,  session_cap=None, view_cap=12000),
    "realloc_bm_a": dict(view="alpha", n_bm25=28, n_prov=0,  session_cap=None, view_cap=12000),
    "realloc8k":    dict(view="rank",  n_bm25=21, n_prov=13, session_cap=None, count_mode=True,
                         top_sessions=13, view_cap=8000),
    "realloc8k_bm": dict(view="rank",  n_bm25=34, n_prov=0,  session_cap=None, view_cap=8000),
    # ==================== RECOMMENDED CONFIGS (measured, see v5_report.py) ====================
    # A) pure drop-in for run_qc.py: same 20,000-char view cap, same 20 raw turns.
    #    6 of the 20 turns come from the top-4 provenance sessions (<=2 each), value-anchored.
    #    allEvTurns 0.745 -> 0.770 (+12/-0, exact McNemar p=0.0005), MS 0.529 -> 0.570,
    #    gold-in-view 0.528 -> 0.611 (+39/-0, p<1e-11), 1.003x baseline chars.
    "v5_drop_in":  dict(view="rank", n_bm25=14, n_prov=6, session_cap=None,
                        top_sessions=4, per_session=2),
    # B) budget-reallocating: the ranked view is cut to 8,000 chars (still HIGHER gold coverage
    #    than the deployed 20,000-char alphabetical view) and the freed chars buy 11 more turns.
    #    allEvTurns 0.745 -> 0.796 (+24/-0, p=1.2e-7), MS 0.529 -> 0.620, at 0.95x baseline chars.
    #    CAVEAT: validated on gold-string / evidence-turn coverage only. Cutting the view to 8k
    #    keeps 81 of 317 chain lines; the view's OTHER function (global current-state summary)
    #    is not measured by any offline metric and could cost accuracy. A/B this one.
    "v5_realloc":  dict(view="rank", n_bm25=25, n_prov=6, session_cap=None,
                        top_sessions=4, per_session=2, view_cap=8000),
    # combining both wins: compressed ranked view + a SMALL provenance allocation (6 turns from
    # the top-4 provenance sessions, <=2 each) on top of a larger BM25 budget.
    "final12k":    dict(view="rank", n_bm25=22, n_prov=6, session_cap=None, top_sessions=4, per_session=2, view_cap=12000),
    "final12k_bm": dict(view="rank", n_bm25=28, n_prov=0, session_cap=None, view_cap=12000),
    "final10k":    dict(view="rank", n_bm25=24, n_prov=6, session_cap=None, top_sessions=4, per_session=2, view_cap=10000),
    "final10k_bm": dict(view="rank", n_bm25=30, n_prov=0, session_cap=None, view_cap=10000),
    "final8k":     dict(view="rank", n_bm25=25, n_prov=6, session_cap=None, top_sessions=4, per_session=2, view_cap=8000),
    "final8k_bm":  dict(view="rank", n_bm25=31, n_prov=0, session_cap=None, view_cap=8000),
    # k=20 provenance sweep (the regime where provenance-guided selection actually wins)
    "p4":     dict(view="rank", n_bm25=16, n_prov=4,  session_cap=None),
    "p6":     dict(view="rank", n_bm25=14, n_prov=6,  session_cap=None),
    "p8":     dict(view="rank", n_bm25=12, n_prov=8,  session_cap=None),
    "p10":    dict(view="rank", n_bm25=10, n_prov=10, session_cap=None),
    "p6_ts4": dict(view="rank", n_bm25=14, n_prov=6,  session_cap=None, top_sessions=4, per_session=2),
    "p6_ts6": dict(view="rank", n_bm25=14, n_prov=6,  session_cap=None, top_sessions=6),
    "p6_a0":  dict(view="rank", n_bm25=14, n_prov=6,  session_cap=None, anchor_weight=0.0),
    "p6_a2":  dict(view="rank", n_bm25=14, n_prov=6,  session_cap=None, anchor_weight=2.0),
    "p6_c15": dict(view="rank", n_bm25=14, n_prov=6,  session_cap=None, top_chains=15),
    "p6_c80": dict(view="rank", n_bm25=14, n_prov=6,  session_cap=None, top_chains=80),
    # budget-split sweep: shrink the ranked view, spend the freed chars on more turns
    # (~1,060 chars/turn measured). Each has a provenance-free BM25 control.
    "split4k":      dict(view="rank", n_bm25=21, n_prov=14, session_cap=None, count_mode=True,
                         top_sessions=14, view_cap=4000),
    "split4k_bm":   dict(view="rank", n_bm25=35, n_prov=0,  session_cap=None, view_cap=4000),
    "split6k":      dict(view="rank", n_bm25=20, n_prov=13, session_cap=None, count_mode=True,
                         top_sessions=13, view_cap=6000),
    "split6k_bm":   dict(view="rank", n_bm25=33, n_prov=0,  session_cap=None, view_cap=6000),
    "split8k":      dict(view="rank", n_bm25=19, n_prov=12, session_cap=None, count_mode=True,
                         top_sessions=12, view_cap=8000),
    "split8k_bm":   dict(view="rank", n_bm25=31, n_prov=0,  session_cap=None, view_cap=8000),
    "split10k":     dict(view="rank", n_bm25=18, n_prov=12, session_cap=None, count_mode=True,
                         top_sessions=12, view_cap=10000),
    "split10k_bm":  dict(view="rank", n_bm25=30, n_prov=0,  session_cap=None, view_cap=10000),
    # RECOMMENDED: relevance-ranked 12k view (still better gold coverage than the deployed 20k
    # alphabetical view) + the 8k chars it frees spent on 8 more turns, 11 of the 28 provenance-
    # selected. Evaluated both free-running and strictly char-matched to the baseline (v5f_bm).
    "v5_final":     dict(view="rank",  n_bm25=17, n_prov=11, session_cap=None, count_mode=True,
                         top_sessions=11, view_cap=12000),
}


def build_context(q, store, question, budget_chars=None, config="v5_drop_in", mode="current",
                  units=None, bm_scores=None, raw_char_cap=None):
    """Assemble (view_text, raw_text, debug).

    budget_chars: None -> (VIEW_CAP, unbounded turn chars, RAW_TURNS turns)
                  int  -> total char budget, split 50/50 between view and raw
                  (view_cap, raw_char_cap) tuple -> explicit
    raw_char_cap: hard char cap on the raw block (used for strictly budget-matched comparisons);
                  turns are dropped from the lowest-priority end, then re-sorted chronologically.
    """
    cfg = dict(CONFIGS[config]) if isinstance(config, str) else dict(config)
    view_cap = cfg.get("view_cap", VIEW_CAP)
    if isinstance(budget_chars, (tuple, list)):
        view_cap, raw_char_cap = budget_chars
    elif isinstance(budget_chars, int):
        view_cap, raw_char_cap = budget_chars // 2, budget_chars // 2

    if units is None:
        units = build_units(q)
    if bm_scores is None:
        bm_scores = bm25_scores(question, [u[3] for u in units])

    scored = score_chains(store, question)

    if cfg["view"] == "rank":
        view_txt, kept = rank_view(store, question, mode, cap=view_cap, scored=scored,
                                   focus_head=cfg.get("focus_head", 0))
    else:                                   # alphabetical + tail truncation (deployed behaviour)
        from lifecycle import render_view
        full = render_view(store, mode)
        kept_lines, total = [], 0
        for line in full.splitlines():
            if total + len(line) + 1 > view_cap:
                break
            kept_lines.append(line)
            total += len(line) + 1
        view_txt, kept = "\n".join(kept_lines), None

    n_prov = cfg.get("n_prov", 0)
    prov, sess_order, sess_sc = [], [], {}
    if n_prov:
        prov, sess_order, sess_sc = select_turns(
            q, store, question, scored, n_prov, bm_scores=bm_scores, units=units,
            top_sessions=cfg.get("top_sessions", 8), per_session=cfg.get("per_session", 3),
            anchor_weight=cfg.get("anchor_weight", 1.0), top_chains=cfg.get("top_chains", 40),
            count_mode=cfg.get("count_mode", False) and bool(COUNT_RE.search(question)))
    n_bm = cfg.get("n_bm25", 0)
    bm = bm25_turns(question, units, n_bm + n_prov, bm_scores=bm_scores,
                    session_cap=cfg.get("session_cap")) if n_bm else []

    # priority order: provenance picks first (they are the added axis), then BM25, dedup,
    # then truncate to the turn budget and finally to raw_char_cap if one is given.
    picked, seen = [], set()
    for i in list(prov) + list(bm):
        if i in seen:
            continue
        seen.add(i)
        picked.append(i)
        if len(picked) >= n_bm + n_prov:
            break
    if raw_char_cap is not None:
        acc, total = [], 0
        for i in picked:
            ln = len(f"[{units[i][1]}] [{units[i][2]}] {units[i][3]}") + 1
            if total + ln > raw_char_cap:
                continue
            acc.append(i)
            total += ln
        picked = acc
    picked_sorted = sorted(picked)
    raw_txt = "\n".join(f"[{units[i][1]}] [{units[i][2]}] {units[i][3]}" for i in picked_sorted)

    debug = {
        "config": config if isinstance(config, str) else "custom",
        "mode": mode,
        "n_chains": len(store),
        "view_chars": len(view_txt),
        "view_lines": len(view_txt.splitlines()),
        "n_chains_kept": len(kept) if kept is not None else len(view_txt.splitlines()),
        "kept_keys": list(kept) if kept is not None else None,
        "raw_chars": len(raw_txt),
        "n_turns": len(picked_sorted),
        "turn_idx": picked_sorted,
        "prov_idx": list(prov),
        "bm25_idx": list(bm)[:n_bm + n_prov],
        "turn_sessions": sorted({units[i][0] for i in picked_sorted}),
        "prov_sessions": list(sess_order),
        "scored": scored,
        "top_chains": [(k, round(s, 3)) for k, s, _ in scored[:8]],
        "count_mode": bool(cfg.get("count_mode") and COUNT_RE.search(question)),
    }
    return view_txt, raw_txt, debug


# ---------------------------------------------------------------- self-test (no API)
if __name__ == "__main__":
    from lifecycle import assemble_chains, render_view
    F = lambda e, a, v, d, s: {"entity": e, "attribute": a, "value": v, "date": d, "sidx": s}
    facts = [
        F("user", "job", "teacher", "2023/05/20 (Sat) 02:21", 0),
        F("user", "job", "engineer", "2023/05/25 (Thu) 10:00", 3),
        F("user", "zebra fact", "irrelevant filler about zebras", "2023/05/21 (Sun) 09:00", 1),
        F("user", "plants acquired", "a monstera", "2023/05/20 (Sat) 02:21", 0),
        F("user", "plants acquired", "a fern", "2023/05/22 (Mon) 09:00", 2),
        F("user", "plants acquired", "a cactus", "2023/05/24 (Wed) 09:00", 4),
    ]
    store = assemble_chains(facts)
    sc = score_chains(store, "What is my current job?")
    assert sc[0][0] == "user::job", sc
    sc2 = score_chains(store, "How many plants did I acquire?")
    assert sc2[0][0] == "user::plants acquired", sc2

    # rank_view keeps the relevant chain when the cap only fits one line
    line_job = _render_chain_line(store["user::job"], "current")
    v, kept = rank_view(store, "What is my current job?", "current", cap=len(line_job) + 1)
    assert kept == ["user::job"] and v == line_job, (kept, v)
    # and its rendering is identical to render_view restricted to the kept chains
    full = render_view(store, "current")
    v2, kept2 = rank_view(store, "What is my current job?", "current", cap=10 ** 9)
    assert v2 == full, "uncapped rank_view must reproduce render_view byte-for-byte"
    hist2, _ = rank_view(store, "x", "historical", cap=10 ** 9)
    assert hist2 == render_view(store, "historical")

    q = {"haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 09:00",
                            "2023/05/22 (Mon) 09:00", "2023/05/25 (Thu) 10:00",
                            "2023/05/24 (Wed) 09:00"],
         "haystack_sessions": [
             [{"role": "user", "content": "I bought a monstera today for the living room."}],
             [{"role": "user", "content": "Zebras are striped."}],
             [{"role": "user", "content": "Picked up a fern at the market."}],
             [{"role": "user", "content": "I started work as an engineer."}],
             [{"role": "user", "content": "Got a cactus from my neighbour."}]]}
    vt, rt, dbg = build_context(q, store, "How many plants did I acquire?", config="v5")
    assert dbg["count_mode"] and len(dbg["turn_sessions"]) >= 3, dbg
    assert "monstera" in rt and "fern" in rt and "cactus" in rt, rt
    vt0, rt0, dbg0 = build_context(q, store, "How many plants did I acquire?", config="baseline")
    assert dbg0["count_mode"] is False
    # exact-budget mode
    vt3, rt3, dbg3 = build_context(q, store, "What is my current job?", config="v5",
                                   budget_chars=(500, 120))
    assert dbg3["view_chars"] <= 500 and dbg3["raw_chars"] <= 120, dbg3
    # bm25_scores must reproduce run_baseline.bm25_rank's ordering
    from run_baseline import bm25_rank
    docs = [u[3] for u in build_units(q)]
    s = bm25_scores("How many plants did I acquire?", docs)
    assert sorted(range(len(docs)), key=lambda i: -s[i]) == bm25_rank("How many plants did I acquire?", docs)
    print("v5_retrieval self-test OK")
    print("--- v5 raw ---\n" + rt)
