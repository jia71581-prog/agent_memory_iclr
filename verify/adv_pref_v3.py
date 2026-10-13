"""ADVERSARIAL recompute #4: per-value fate of the 45, the raw channel, duplicate survival,
and whether cap survival predicts SSP judge correctness."""
import json, os, hashlib, pickle, sys, re, statistics as st
sys.path.insert(0, "/root/agent_memory_iclr/code")
import lifecycle as LC
from run_baseline import bm25_rank

data = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
byid = {q["question_id"]: q for q in data}
views = pickle.load(open("/root/agent_memory_iclr/verify/_views.pkl", "rb"))
qc = {r["qid"]: r for r in (json.loads(l) for l in open("/root/agent_memory_iclr/results/full500_qc.jsonl"))}
ssp = [q for q in data if q["question_type"] == "single-session-preference"]

def nrm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

STOP = set("the and for was were you your a an of in on to with my i is are that this it as at".split())

def sess_facts(q):
    out = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        text = LC._session_text(sess)
        if not text:
            continue
        key = hashlib.md5(text.encode()).hexdigest()[:16]
        p = os.path.join(LC.SESSION_CACHE_DIR, f"{key}.json")
        if os.path.exists(p):
            for e, a, v in json.load(open(p)):
                out.append((si, e, a, v))
    return out

def raw20(q):
    units = []
    for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                units.append((date, t["role"], t["content"]))
    order = sorted(bm25_rank(q["question"], [u[2] for u in units])[:20])
    return "\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in order)

rows = []
perq = {}
for q in ssp:
    qid = q["question_id"]
    gs = {i for i, sid in enumerate(q["haystack_session_ids"]) if sid in set(q["answer_session_ids"])}
    sid2 = dict(zip(q["haystack_session_ids"], q["haystack_sessions"]))
    ev = " ".join(t["content"] for sid in q["answer_session_ids"] for t in sid2.get(sid, [])
                  if t.get("has_answer") and t.get("content"))
    V, R = views[qid], raw20(q)
    seen, vals = set(), []
    for si, e, a, v in sess_facts(q):
        if si in gs and v.lower() in ev.lower() and nrm(v) not in seen:
            seen.add(nrm(v)); vals.append((e, a, v))
    for e, a, v in vals:
        toks = [t for t in nrm(v).split() if t not in STOP and len(t) > 2]
        # does the content survive ANYWHERE in the capped view (possibly on a different line)?
        elsewhere = bool(toks) and all(t in nrm(V["cap"]) for t in toks)
        rows.append(dict(qid=qid, ent=e, attr=a, val=v,
                         full=v.lower() in V["full"].lower(),
                         cap=v.lower() in V["cap"].lower(),
                         cap_tokens=elsewhere,
                         raw20=v.lower() in R.lower(),
                         raw20_tokens=bool(toks) and all(t in nrm(R) for t in toks),
                         judge=qc[qid]["judge"]))
    perq[qid] = dict(n=len(vals), full=sum(1 for r in rows if r["qid"] == qid and r["full"]),
                     cap=sum(1 for r in rows if r["qid"] == qid and r["cap"]),
                     judge=qc[qid]["judge"])

n = len(rows)
print(f"n values = {n}; in full view {sum(r['full'] for r in rows)}; in capped view {sum(r['cap'] for r in rows)}")
lost_render = [r for r in rows if not r["full"]]
lost_cap = [r for r in rows if r["full"] and not r["cap"]]
print(f"lost to rendering {len(lost_render)}; lost to the cap {len(lost_cap)}")
print(f"\nof the {len(lost_cap)} cap-lost values:")
print(f"  content still present elsewhere in the CAPPED view (all content tokens): "
      f"{sum(r['cap_tokens'] for r in lost_cap)}")
print(f"  verbatim present in the production raw top-20 turns: {sum(r['raw20'] for r in lost_cap)}")
print(f"  content tokens present in the raw top-20 turns:      {sum(r['raw20_tokens'] for r in lost_cap)}")
print(f"  => absent from the WHOLE production context: "
      f"{sum(1 for r in lost_cap if not r['cap_tokens'] and not r['raw20_tokens'])}")
print("\ncap-lost values, per question:")
for r in lost_cap:
    print(f"  {r['qid']:10s} judge={str(r['judge']):5s} capTok={str(r['cap_tokens']):5s} "
          f"rawTok={str(r['raw20_tokens']):5s}  {r['ent']} | {r['attr']} = {r['val'][:70]}")
print("\nrender-lost values:")
for r in lost_render:
    print(f"  {r['qid']:10s} judge={str(r['judge']):5s} {r['ent']} | {r['attr']} = {r['val'][:70]}")

# ---- does cap survival predict correctness? (30 SSP questions)
print("\n=== predictiveness of the proxy on the 30 SSP questions (qc arm judge) ===")
qs = [(k, v) for k, v in perq.items() if v["n"] > 0]
print(f"questions with >=1 traced value: {len(qs)} of 30")
for lab, sel in [("judge=True", True), ("judge=False", False)]:
    S = [v for k, v in qs if v["judge"] == sel]
    if S:
        tot = sum(v["n"] for v in S); cap = sum(v["cap"] for v in S)
        print(f"  {lab}: n_q={len(S)} values={tot} in_cap={cap} ({cap/tot:.3f}) "
              f"mean per-q survival {st.mean([v['cap']/v['n'] for v in S]):.3f}")
anyloss = [(k, v) for k, v in qs if v["cap"] < v["n"]]
noloss = [(k, v) for k, v in qs if v["cap"] == v["n"]]
print(f"  any value lost to cap: {len(anyloss)} questions, judge_acc "
      f"{sum(v['judge'] for k, v in anyloss)/max(1,len(anyloss)):.3f}")
print(f"  no value lost:         {len(noloss)} questions, judge_acc "
      f"{sum(v['judge'] for k, v in noloss)/max(1,len(noloss)):.3f}")
print("  SSP questions with NO traced value at all:",
      [q['question_id'] for q in ssp if perq[q['question_id']]['n'] == 0],
      "judge:", [qc[q['question_id']]['judge'] for q in ssp if perq[q['question_id']]['n'] == 0])
json.dump(rows, open("/root/agent_memory_iclr/verify/_pref45.json", "w"), indent=1)
