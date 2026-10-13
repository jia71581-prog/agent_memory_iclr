"""B6 part 4: does either variant add evidence the raw-turn channel does not already carry?
Scoped block draws only from sessions whose user turns retrieve_v5 already injects in full,
so its marginal contribution is measured here against the global index alternative.
SSP + the 1 advice-shaped KU question (the shipped firing set), plus all 470 for the
global-index side.
"""
import json, statistics as st, sys
from collections import defaultdict
sys.path.insert(0, "/root/agent_memory_iclr/code")
import v5_guards as G
from run_baseline import bm25_rank

data = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
SENT, PREF = G.SENT_SPLIT, G.PREF_RE
fire = [q for q in data if not q["question_id"].endswith("_abs") and G.advice_shaped(q["question"])]
print("shipped firing set:", len(fire), "questions")


def mined_of(turn):
    if turn.get("role") != "user" or not turn.get("content"):
        return []
    return [s.strip() for s in SENT.split(G.nz(turn["content"]))
            if len(s.strip()) > 25 and any(p.search(s.strip().lower()) for p in PREF)]


tot = defaultdict(int)
per_q = []
for q in fire:
    units = G.units_of(q)
    docs = [u[2] for u in units]
    torder = bm25_rank(q["question"], docs)
    sorder = bm25_rank(q["question"], G.session_docs(q, user_only=True))
    per_sess = {si: G.preference_sentences(s) for si, s in enumerate(q["haystack_sessions"])}
    index = [s for si in range(len(q["haystack_sessions"])) for s in per_sess[si]]
    iorder = bm25_rank(q["question"], index) if index else []

    # raw channel exactly as build_context_v5 renders it (advice -> n_sessions=3)
    uidx = [i for i in torder if units[i][1] == "user"][:14]
    aidx = [i for i in torder if units[i][1] == "assistant"][:6]
    keep = set(uidx) | set(aidx)
    pos, sess_pos = 0, {}
    for si, sess in enumerate(q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                sess_pos.setdefault(si, []).append((pos, t["role"]))
                pos += 1
    for si in sorder[:3]:
        for i, role in sess_pos.get(si, []):
            if role == "user":
                keep.add(i)
    raw = G.nz("\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in sorted(keep)))

    scoped = [s for si in sorder[:3] for s in per_sess[si]]
    b15 = [index[i] for i in iorder[:15]]
    budget = sum(len(s) + 1 for s in scoped)
    matched, c = [], 0
    for i in iorder:
        if c + len(index[i]) + 1 > budget:
            break
        matched.append(index[i]); c += len(index[i]) + 1

    def novel(sel):
        return [s for s in sel if s not in raw]

    # gold evidence turns the raw channel misses, and whether either block recovers them
    sid2 = dict(zip(q["haystack_session_ids"], q["haystack_sessions"]))
    gts = [t for sid in q["answer_session_ids"] for t in sid2.get(sid, [])
           if t.get("has_answer") and t.get("content")]
    miss = [t for t in gts if G.nz(t["content"]) not in raw]
    rec_scoped = sum(1 for t in miss if any(s in scoped for s in mined_of(t)))
    rec_b15 = sum(1 for t in miss if any(s in b15 for s in mined_of(t)))
    rec_matched = sum(1 for t in miss if any(s in matched for s in mined_of(t)))

    tot["q"] += 1
    tot["scoped_sents"] += len(scoped); tot["scoped_novel"] += len(novel(scoped))
    tot["b15_sents"] += len(b15); tot["b15_novel"] += len(novel(b15))
    tot["matched_sents"] += len(matched); tot["matched_novel"] += len(novel(matched))
    tot["scoped_chars"] += budget
    tot["b15_chars"] += sum(len(s) + 1 for s in b15)
    tot["matched_chars"] += c
    tot["gold"] += len(gts); tot["gold_missing_from_raw"] += len(miss)
    tot["rec_scoped"] += rec_scoped; tot["rec_b15"] += rec_b15; tot["rec_matched"] += rec_matched
    per_q.append((q["question_id"], q["question_type"], len(scoped), len(novel(scoped)),
                  len(b15), len(novel(b15)), len(gts), len(miss), rec_scoped, rec_b15, rec_matched))

print("\n=== marginal contribution over the raw-turn channel already in context ===")
print(f"questions {tot['q']}")
print(f"scoped block : {tot['scoped_sents']} sentences, {tot['scoped_novel']} not already in raw "
      f"({tot['scoped_novel']/max(1,tot['scoped_sents']):.4f}), {tot['scoped_chars']/tot['q']:.0f} chars/q")
print(f"global top15 : {tot['b15_sents']} sentences, {tot['b15_novel']} not already in raw "
      f"({tot['b15_novel']/max(1,tot['b15_sents']):.4f}), {tot['b15_chars']/tot['q']:.0f} chars/q")
print(f"global matched-budget: {tot['matched_sents']} sentences, {tot['matched_novel']} novel "
      f"({tot['matched_novel']/max(1,tot['matched_sents']):.4f}), {tot['matched_chars']/tot['q']:.0f} chars/q")
print(f"\ngold evidence turns {tot['gold']}, missing from the raw channel {tot['gold_missing_from_raw']}")
print(f"  recovered by scoped block          {tot['rec_scoped']}")
print(f"  recovered by global BM25 top-15    {tot['rec_b15']}")
print(f"  recovered by matched-budget global {tot['rec_matched']}")
print("\nper-question (qid, type, scoped_n, scoped_novel, b15_n, b15_novel, gold, gold_missing, rec_scoped, rec_b15, rec_matched):")
for r in per_q:
    print("  ", r)
