"""B6 part 2 (efficient): index size, global-BM25 recovery, shipped scoped delivery,
budget-matched comparison, duplication vs the raw-turn channel.
One BM25 pass per question per granularity.
"""
import json, statistics as st, sys
from collections import defaultdict

sys.path.insert(0, "/root/agent_memory_iclr/code")
import v5_guards as G
from run_baseline import bm25_rank

data = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
non_abs = [q for q in data if not q["question_id"].endswith("_abs")]
SENT, PREF = G.SENT_SPLIT, G.PREF_RE


def sess_pref(sess):
    return G.preference_sentences(sess)


def gold_turns(q):
    sid2 = dict(zip(q["haystack_session_ids"], q["haystack_sessions"]))
    return [t for sid in q["answer_session_ids"] for t in sid2.get(sid, [])
            if t.get("has_answer") and t.get("content")]


def gold_mined_sents(t):
    if t.get("role") != "user":
        return []
    return [s.strip() for s in SENT.split(G.nz(t["content"]))
            if len(s.strip()) > 25 and any(p.search(s.strip().lower()) for p in PREF)]


agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
chars = defaultdict(lambda: defaultdict(list))
idxsize = []
sessrec = defaultdict(lambda: defaultdict(lambda: [0, 0]))
dup_rows = []
matched_k = []

for qi, q in enumerate(non_abs):
    t = q["question_type"]
    units = G.units_of(q)
    docs = [u[2] for u in units]
    torder = bm25_rank(q["question"], docs)
    sdocs = G.session_docs(q, user_only=True)
    sorder = bm25_rank(q["question"], sdocs)
    per_sess_pref = {si: sess_pref(sess) for si, sess in enumerate(q["haystack_sessions"])}
    index = [s for si in range(len(q["haystack_sessions"])) for s in per_sess_pref[si]]
    iorder = bm25_rank(q["question"], index) if index else []
    idxsize.append((len(index), sum(len(s) + 1 for s in index)))
    gts = gold_turns(q)
    gms = {id(x): gold_mined_sents(x) for x in gts}
    gsi = {i for i, s in enumerate(q["haystack_session_ids"]) if s in set(q["answer_session_ids"])}

    def cap(avail):
        h = 0
        for x in gts:
            ss = gms[id(x)]
            if ss and any(s in avail for s in ss):
                h += 1
        return h, len(gts)

    def rec(name, sel_sents):
        h, n = cap(set(sel_sents))
        agg[name][t][0] += h; agg[name][t][1] += n
        chars[name][t].append(sum(len(s) + 1 for s in sel_sents))

    rec("whole_index", index)
    for k in (10, 15, 20, 30, 45, 60):
        rec(f"bm25_top{k}", [index[i] for i in iorder[:k]])
    scoped = {}
    for ns_ in (1, 2, 3):
        sel = [s for si in sorder[:ns_] for s in per_sess_pref[si]]
        scoped[ns_] = sel
        rec(f"scoped_n{ns_}", sel)
        sessrec[ns_][t][0] += int(bool(gsi & set(sorder[:ns_])))
        sessrec[ns_][t][1] += 1
    # budget-matched global BM25: take sentences until chars >= scoped_n3 chars
    budget = sum(len(s) + 1 for s in scoped[3])
    sel, tot = [], 0
    for i in iorder:
        if tot + len(index[i]) + 1 > budget:
            break
        sel.append(index[i]); tot += len(index[i]) + 1
    rec("bm25_matched_n3", sel)
    matched_k.append(len(sel))

    # duplication vs the raw channel that build_context_v5 emits for this question
    advice = G.advice_shaped(q["question"])
    ns_used = 3 if advice else 1
    uidx = [i for i in torder if units[i][1] == "user"][:14]
    aidx = [i for i in torder if units[i][1] == "assistant"][:6]
    keep = set(uidx) | set(aidx)
    pos, per_sess = 0, {}
    for si, sess in enumerate(q["haystack_sessions"]):
        for tt in sess:
            if tt.get("content"):
                per_sess.setdefault(si, []).append((pos, tt["role"]))
                pos += 1
    for si in sorder[:ns_used]:
        for i, role in per_sess.get(si, []):
            if role == "user":
                keep.add(i)
    raw = G.nz("\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in sorted(keep)))
    qs = scoped[ns_used]
    if qs:
        dup_rows.append((q["question_id"], t, advice, len(qs),
                         sum(1 for s in qs if s in raw),
                         sum(len(s) + 1 for s in qs), len(raw)))

print("=== index size, n=%d answerable ===" % len(idxsize))
ns = sorted(r[0] for r in idxsize); cs = sorted(r[1] for r in idxsize)
print(f"sentences median {st.median(ns):.0f} mean {st.mean(ns):.1f} p10 {ns[len(ns)//10]} p90 {ns[9*len(ns)//10]}")
print(f"chars     median {st.median(cs):.0f} mean {st.mean(cs):.0f} p10 {cs[len(cs)//10]} p90 {cs[9*len(cs)//10]}")

names = ["whole_index", "bm25_top10", "bm25_top15", "bm25_top20", "bm25_top30", "bm25_top45",
         "bm25_top60", "bm25_matched_n3", "scoped_n1", "scoped_n2", "scoped_n3"]
types = sorted({t for k in agg for t in agg[k]})
print("\n=== turn-level capture of gold evidence turns (same metric that yields 0.659/0.783) ===")
print(f"{'variant':17s}{'ALL':>7s}{'chars':>7s}" + "".join(f"{x[:11]:>12s}" for x in types))
for k in names:
    H = N = 0
    line = ""
    for x in types:
        h, n = agg[k][x]; H += h; N += n
        line += f"{h/max(1,n):12.3f}"
    print(f"{k:17s}{H/max(1,N):7.3f}{st.mean([c for x in types for c in chars[k][x]]):7.0f}" + line)

print("\n=== SSP only: capture vs mean block chars ===")
for k in names:
    h, n = agg[k]["single-session-preference"]
    print(f"  {k:17s} capture {h/max(1,n):.3f} ({h}/{n})  mean chars {st.mean(chars[k]['single-session-preference']):.0f}")
print(f"  budget-matched global BM25 took mean {st.mean(matched_k):.1f} sentences")

print("\n=== gold session inside top-n session BM25 (scoped block precondition) ===")
for ns_ in (1, 2, 3):
    H = N = 0
    parts = []
    for x in types:
        h, n = sessrec[ns_][x]; H += h; N += n
        parts.append(f"{x[:11]} {h}/{n}")
    print(f"  n={ns_}: ALL {H}/{N} = {H/N:.3f}   " + "  ".join(parts))

print("\n=== duplication of scoped quotes inside the raw-turn channel ===")
def summ(rows, name):
    if not rows:
        print(f"  {name}: none"); return
    tq = sum(r[3] for r in rows); td = sum(r[4] for r in rows)
    print(f"  {name}: questions {len(rows)} quotes {tq} already-in-raw {td} ({td/tq:.4f}) "
          f"mean quote chars {st.mean([r[5] for r in rows]):.0f} mean raw chars {st.mean([r[6] for r in rows]):.0f} "
          f"quote/raw {st.mean([r[5]/max(1,r[6]) for r in rows]):.4f}")
summ(dup_rows, "all answerable with a block")
summ([r for r in dup_rows if r[2]], "advice-shaped")
summ([r for r in dup_rows if not r[2]], "non-advice (block off in shipped config)")
print("  advice-shaped quote counts:", sorted(r[3] for r in dup_rows if r[2]))
json.dump({"idxsize": idxsize,
           "agg": {k: {x: agg[k][x] for x in agg[k]} for k in agg},
           "chars": {k: {x: st.mean(chars[k][x]) for x in chars[k]} for k in chars},
           "dup_rows": dup_rows},
          open("/root/agent_memory_iclr/results/_ver5_b6_index.json", "w"))
