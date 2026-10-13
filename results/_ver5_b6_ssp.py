"""B6 part 3 (SSP only, fast): does turn-level 'capture' actually deliver the
evidence-bearing sentence? And what does the shipped block look like per SSP question?
"""
import json, re, statistics as st, sys
sys.path.insert(0, "/root/agent_memory_iclr/code")
import v5_guards as G
from run_baseline import bm25_rank, tokenize

data = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
ssp = [q for q in data if q["question_type"] == "single-session-preference"]
SENT, PREF = G.SENT_SPLIT, G.PREF_RE

STOPW = set("the a an and or of to in on for with my i you your me it is are was were be that this "
            "would prefer responses user they might not general related other about some more".split())


def content_toks(s):
    return {w for w in tokenize(s) if w not in STOPW and len(w) > 2}


rows = []
for q in ssp:
    sid2 = dict(zip(q["haystack_session_ids"], q["haystack_sessions"]))
    gts = [t for sid in q["answer_session_ids"] for t in sid2.get(sid, [])
           if t.get("has_answer") and t.get("content")]
    rub = content_toks(q["answer"])
    for t in gts:
        if t.get("role") != "user":
            rows.append((q["question_id"], "assistant_gold", 0, 0, 0.0, 0.0, ""))
            continue
        sents = [s.strip() for s in SENT.split(G.nz(t["content"])) if s.strip()]
        keep = [s for s in sents if len(s) > 25 and any(p.search(s.lower()) for p in PREF)]
        drop = [s for s in sents if s not in keep]
        # rubric-term coverage of the gold turn vs of the mined subset
        cov_all = len(rub & set().union(*[content_toks(s) for s in sents])) / max(1, len(rub)) if sents else 0
        cov_keep = len(rub & set().union(*[content_toks(s) for s in keep])) / max(1, len(rub)) if keep else 0
        rows.append((q["question_id"], "user_gold", len(sents), len(keep), cov_all, cov_keep,
                     "; ".join(d[:60] for d in drop[:2])))

user_rows = [r for r in rows if r[1] == "user_gold"]
print("=== SSP gold user turns: rubric-term coverage kept by the miner vs whole turn ===")
print(f"n gold user turns {len(user_rows)}  captured(>=1 sent) {sum(1 for r in user_rows if r[3]>0)}")
ca = st.mean([r[4] for r in user_rows]); ck = st.mean([r[5] for r in user_rows])
print(f"mean rubric-token coverage: whole gold turn {ca:.3f}   miner subset {ck:.3f}   "
      f"retained fraction {ck/max(1e-9,ca):.3f}")
cap = [r for r in user_rows if r[3] > 0]
print(f"among captured turns only: whole {st.mean([r[4] for r in cap]):.3f} "
      f"miner {st.mean([r[5] for r in cap]):.3f}")
loss = [r for r in cap if r[5] < r[4] - 1e-9]
print(f"captured turns where the miner drops rubric tokens present in the turn: {len(loss)}/{len(cap)}")
for r in loss[:8]:
    print(f"   {r[0]:16s} sents {r[2]} kept {r[3]} cov {r[4]:.2f}->{r[5]:.2f}  dropped: {r[6][:90]}")

print("\n=== per-SSP-question: shipped scoped block (n=3) vs raw channel ===")
tot_q = tot_dup = 0
qchars, rchars, counts = [], [], []
for q in ssp:
    units = G.units_of(q)
    keep, sess_idx = G.retrieve_v5(q, n_sessions=3)
    raw = G.nz("\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in keep))
    qs = []
    for si in sess_idx:
        qs += G.preference_sentences(q["haystack_sessions"][si])
    dup = sum(1 for s in qs if s in raw)
    tot_q += len(qs); tot_dup += dup
    counts.append(len(qs))
    qchars.append(sum(len(s) + 1 for s in qs)); rchars.append(len(raw))
    print(f"  {q['question_id']:16s} advice={int(G.advice_shaped(q['question']))} quotes {len(qs):3d} "
          f"dup_in_raw {dup:3d} quote_chars {sum(len(s)+1 for s in qs):5d} raw_chars {len(raw):6d}")
print(f"SSP totals: quotes {tot_q} duplicated in raw {tot_dup} ({tot_dup/max(1,tot_q):.4f})")
print(f"quotes/question: median {st.median(counts):.0f} min {min(counts)} max {max(counts)}  "
      f"quote chars mean {st.mean(qchars):.0f} raw chars mean {st.mean(rchars):.0f}")
