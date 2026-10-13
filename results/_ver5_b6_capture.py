"""Independent recomputation of the v5_guards B6 (verbatim self-disclosure miner) claims.
Offline only. No API.
"""
import json, os, re, statistics as st, sys
from collections import defaultdict

sys.path.insert(0, "/root/agent_memory_iclr/code")
import v5_guards as G
from run_baseline import bm25_rank

DATA = "/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"
data = json.load(open(DATA))
non_abs = [q for q in data if not q["question_id"].endswith("_abs")]

SENT = G.SENT_SPLIT
PREF = G.PREF_RE


def mined(sent, min_len=25):
    s = sent.strip()
    return len(s) > min_len and any(p.search(s.lower()) for p in PREF)


def gold_turns(q):
    sid2 = dict(zip(q["haystack_session_ids"], q["haystack_sessions"]))
    out = []
    for sid in q["answer_session_ids"]:
        for t in sid2.get(sid, []):
            if t.get("has_answer") and t.get("content"):
                out.append(t)
    return out


# ---------------- variant definitions of "capture of gold evidence sentences" -------------
variants = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # variant -> type -> [hit, tot]

for q in non_abs:
    t = q["question_type"]
    gts = gold_turns(q)
    # V1: all sentences of all gold turns (any role); hit if miner would emit it (needs role==user)
    for tt in gts:
        for s in SENT.split(G.nz(tt["content"])):
            s = s.strip()
            if not s:
                continue
            variants["V1_all_sents_anyrole"][t][1] += 1
            variants["V1_all_sents_anyrole"][t][0] += int(tt["role"] == "user" and mined(s))
    # V2: sentences with len>25 only (any role)
    for tt in gts:
        for s in SENT.split(G.nz(tt["content"])):
            s = s.strip()
            if len(s) <= 25:
                continue
            variants["V2_len25_sents_anyrole"][t][1] += 1
            variants["V2_len25_sents_anyrole"][t][0] += int(tt["role"] == "user" and mined(s))
    # V3: gold turn level: >=1 mined sentence in the turn
    for tt in gts:
        variants["V3_turnlevel_anyrole"][t][1] += 1
        variants["V3_turnlevel_anyrole"][t][0] += int(
            tt["role"] == "user" and any(mined(s) for s in SENT.split(G.nz(tt["content"]))))
    # V4: user gold turns only, sentence level
    for tt in gts:
        if tt["role"] != "user":
            continue
        for s in SENT.split(G.nz(tt["content"])):
            s = s.strip()
            if not s:
                continue
            variants["V4_usergold_sents"][t][1] += 1
            variants["V4_usergold_sents"][t][0] += int(mined(s))
    # V5: question-level: any mined sentence from any gold turn
    variants["V5_questionlevel"][t][1] += 1
    variants["V5_questionlevel"][t][0] += int(any(
        tt["role"] == "user" and any(mined(s) for s in SENT.split(G.nz(tt["content"]))) for tt in gts))

print("=== capture-metric variants (which reproduces SSP 0.659 / ALL 0.783 / SSA 0.036?) ===")
for vn in sorted(variants):
    per = variants[vn]
    tot = [sum(x[0] for x in per.values()), sum(x[1] for x in per.values())]
    ssp = per["single-session-preference"]
    ssa = per["single-session-assistant"]
    print(f"{vn:26s} ALL {tot[0]/tot[1]:.3f} ({tot[0]}/{tot[1]})  "
          f"SSP {ssp[0]/max(1,ssp[1]):.3f} ({ssp[0]}/{ssp[1]})  "
          f"SSA {ssa[0]/max(1,ssa[1]):.3f} ({ssa[0]}/{ssa[1]})")
    for t in sorted(per):
        h, n = per[t]
        print(f"    {t:28s} {h/max(1,n):.3f} ({h}/{n})")
