"""B6 part 2: index size, global-BM25 recovery, shipped scoped delivery, budget matching,
and duplication against the raw-turn channel already in context.
"""
import json, os, re, statistics as st, sys
from collections import defaultdict

sys.path.insert(0, "/root/agent_memory_iclr/code")
import v5_guards as G
from run_baseline import bm25_rank

data = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
non_abs = [q for q in data if not q["question_id"].endswith("_abs")]
SENT, PREF = G.SENT_SPLIT, G.PREF_RE


def mined_sents(turns, min_len=25):
    out = []
    for t in turns:
        if t.get("role") != "user" or not t.get("content"):
            continue
        for s in SENT.split(G.nz(t["content"])):
            s = s.strip()
            if len(s) > min_len and any(p.search(s.lower()) for p in PREF):
                out.append(s)
    return out


def full_index(q):
    return mined_sents([t for sess in q["haystack_sessions"] for t in sess])


def gold_turns(q):
    sid2 = dict(zip(q["haystack_session_ids"], q["haystack_sessions"]))
    return [t for sid in q["answer_session_ids"] for t in sid2.get(sid, [])
            if t.get("has_answer") and t.get("content")]


def gold_sess_idx(q):
    gs = set(q["answer_session_ids"])
    return {i for i, s in enumerate(q["haystack_session_ids"]) if s in gs}


# ------------- index size -------------
rows = []
for q in non_abs:
    idx = full_index(q)
    rows.append((len(idx), sum(len(s) + 1 for s in idx)))
print("=== index size over the whole haystack (n=%d answerable questions) ===" % len(rows))
ns = sorted(r[0] for r in rows); cs = sorted(r[1] for r in rows)
print(f"sentences: median {st.median(ns):.0f} mean {st.mean(ns):.1f} p10 {ns[len(ns)//10]} p90 {ns[9*len(ns)//10]}")
print(f"chars    : median {st.median(cs):.0f} mean {st.mean(cs):.0f} p10 {cs[len(cs)//10]} p90 {cs[9*len(cs)//10]}")

# also over all 500 (incl abs) for completeness
rows500 = [(len(full_index(q)), sum(len(s) + 1 for s in full_index(q))) for q in data]
print(f"all 500: median sents {st.median([r[0] for r in rows500]):.0f} "
      f"median chars {st.median([r[1] for r in rows500]):.0f}")

# ------------- global BM25 over the index vs scoped delivery -------------
# turn-level capture, matching the metric that reproduced 0.659/0.783/0.036
def turn_capture(gts, avail_set):
    """hit if >=1 mined sentence of a user gold turn is present in avail_set"""
    hit = tot = 0
    for t in gts:
        tot += 1
        if t.get("role") != "user":
            continue
        ss = [s.strip() for s in SENT.split(G.nz(t["content"]))]
        ss = [s for s in ss if len(s) > 25 and any(p.search(s.lower()) for p in PREF)]
        if ss and any(s in avail_set for s in ss):
            hit += 1
    return hit, tot


agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
chars = defaultdict(lambda: defaultdict(list))
nq = defaultdict(list)
dup_frac, quote_counts, quote_chars = [], defaultdict(list), defaultdict(list)
sessrec = defaultdict(lambda: [0, 0])

for q in non_abs:
    t = q["question_type"]
    idx = full_index(q)
    gts = gold_turns(q)
    gsi = gold_sess_idx(q)
    order = bm25_rank(q["question"], idx) if idx else []

    # A. whole index (upper bound of the miner)
    for k, avail in [("whole_index", set(idx))]:
        h, n = turn_capture(gts, avail)
        agg[k][t][0] += h; agg[k][t][1] += n
        chars[k][t].append(sum(len(s) + 1 for s in idx))

    # B. global BM25 top-k over the index
    for k in (15, 30, 45):
        sel = [idx[i] for i in order[:k]]
        h, n = turn_capture(gts, set(sel))
        agg[f"bm25_top{k}"][t][0] += h; agg[f"bm25_top{k}"][t][1] += n
        chars[f"bm25_top{k}"][t].append(sum(len(s) + 1 for s in sel))

    # C. shipped scoped block: n sessions by session-level BM25 over user text
    srank = bm25_rank(q["question"], G.session_docs(q, user_only=True))
    for ns_ in (1, 2, 3):
        sel_sess = srank[:ns_]
        qs = []
        for si in sel_sess:
            qs += G.preference_sentences(q["haystack_sessions"][si])
        h, n = turn_capture(gts, set(qs))
        agg[f"scoped_n{ns_}"][t][0] += h; agg[f"scoped_n{ns_}"][t][1] += n
        chars[f"scoped_n{ns_}"][t].append(sum(len(s) + 1 for s in qs))
        quote_counts[ns_].append(len(qs))
        quote_chars[ns_].append(sum(len(s) + 1 for s in qs))
        if ns_ == 3:
            sessrec[t][0] += int(bool(gsi & set(sel_sess))); sessrec[t][1] += 1

    # D. duplication of the scoped block against the raw-turn channel actually in context
    advice = G.advice_shaped(q["question"])
    n_sess = 3 if advice else 1
    units = G.units_of(q)
    keep, sess_idx = G.retrieve_v5(q, n_sessions=n_sess)
    raw_txt = "\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in keep)
    raw_n = G.nz(raw_txt)
    qs = []
    for si in sess_idx:
        qs += G.preference_sentences(q["haystack_sessions"][si])
    if qs:
        dup = sum(1 for s in qs if s in raw_n)
        dup_frac.append((q["question_id"], t, advice, len(qs), dup,
                         sum(len(s) + 1 for s in qs), len(raw_txt)))

print("\n=== turn-level capture of gold evidence (denominator = all gold evidence turns) ===")
order_names = ["whole_index", "bm25_top15", "bm25_top30", "bm25_top45",
               "scoped_n1", "scoped_n2", "scoped_n3"]
types = sorted({t for k in agg for t in agg[k]})
hdr = f"{'variant':14s}" + "".join(f"{t[:12]:>13s}" for t in types) + f"{'ALL':>8s}{'chars':>8s}"
print(hdr)
for k in order_names:
    line = f"{k:14s}"
    H = N = 0
    for t in types:
        h, n = agg[k][t]
        H += h; N += n
        line += f"{h/max(1,n):13.3f}"
    allc = st.mean([c for t in types for c in chars[k][t]])
    line += f"{H/max(1,N):8.3f}{allc:8.0f}"
    print(line)

print("\n=== SSP only: capture vs mean block chars ===")
for k in order_names:
    h, n = agg[k]["single-session-preference"]
    c = st.mean(chars[k]["single-session-preference"])
    print(f"  {k:14s} SSP capture {h/max(1,n):.3f} ({h}/{n})  mean block chars {c:.0f}")

print("\n=== quotes per question delivered by the scoped block (all 470 answerable) ===")
for ns_ in (1, 2, 3):
    qc = sorted(quote_counts[ns_]); cc = sorted(quote_chars[ns_])
    print(f"  n_sessions={ns_}: quotes median {st.median(qc):.0f} mean {st.mean(qc):.1f} "
          f"p10 {qc[len(qc)//10]} p90 {qc[9*len(qc)//10]} max {qc[-1]}   "
          f"chars median {st.median(cc):.0f} mean {st.mean(cc):.0f} p90 {cc[9*len(cc)//10]}")

print("\n=== gold session inside top-3 session BM25 (the scoped block's precondition) ===")
for t in sorted(sessrec):
    h, n = sessrec[t]
    print(f"  {t:28s} {h}/{n} = {h/n:.3f}")

print("\n=== duplication: scoped quotes already present verbatim in the raw-turn channel ===")
adv = [r for r in dup_frac if r[2]]
alln = dup_frac
def summ(rows, name):
    if not rows:
        print(f"  {name}: none"); return
    tq = sum(r[3] for r in rows); td = sum(r[4] for r in rows)
    print(f"  {name}: n_questions {len(rows)}  quotes {tq}  already in raw {td} ({td/tq:.4f})  "
          f"mean quote chars {st.mean([r[5] for r in rows]):.0f}  mean raw chars {st.mean([r[6] for r in rows]):.0f}")
summ(alln, "all answerable with a block")
summ(adv, "advice-shaped only")
print("  advice-shaped quote counts:", sorted(r[3] for r in adv))
