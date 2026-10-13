"""ADVERSARIAL follow-ups:
 (a) does the cited 'flight departs on August 15th' / session 04/19 evidence exist?
 (b) explicit-future census: past-tense (year-rollback would have been right) vs genuine plan
 (c) residual precision of the numeric-date rule AFTER the guards (full accepted set)
 (d) blast radius of bug1 (month alias) = how many corpus sentences resolve a month name
Offline, no API.
"""
import json
import re
import sys
from collections import Counter

sys.path.insert(0, "/root/agent_memory_iclr/code")
import v5_temporal as V

D = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
res = {}

# ---- (a) the cited example --------------------------------------------------
hits_flight, hits_0419 = [], []
seen = {}
for q in D:
    for ds, sid, sess in zip(q["haystack_dates"], q["haystack_session_ids"], q["haystack_sessions"]):
        if sid in seen:
            continue
        seen[sid] = (ds, [t.get("content") or "" for t in sess if t.get("role") == "user"])
for sid, (ds, turns) in seen.items():
    for c in turns:
        cl = c.lower()
        if re.search(r"flight[^.]{0,40}\bdepart", cl):
            hits_flight.append((ds, sid, re.sub(r"\s+", " ", c)[:200]))
        if ds.startswith("2023/04/19") and re.search(r"august\s+15", cl):
            hits_0419.append((ds, sid, re.sub(r"\s+", " ", c)[:200]))
res["a_flight_departs_matches"] = hits_flight[:10]
res["a_n_flight_departs"] = len(hits_flight)
res["a_0419_august15"] = hits_0419
# any session whose date is 04/19 mentioning any explicit later-in-year date?
apr19 = [(ds, sid, c) for sid, (ds, turns) in seen.items() if ds.startswith("2023/04/19")
         for c in turns]
res["a_n_sessions_dated_0419"] = len({sid for sid, (ds, t) in seen.items() if ds.startswith("2023/04/19")})

# ---- (b) explicit-future census -------------------------------------------
PAST = re.compile(r"\b(was|were|had|did|went|got|bought|attended|visited|saw|watched|ate|"
                  r"finished|completed|made|met|took|flew|stopped by|scored|set|sold|"
                  r"participated|joined|received|wore|celebrated|did|ran|hosted|"
                  r"i'?ve (been|had|attended|visited|done)|just got back|last (week|month|year|weekend))\b")
FUT = re.compile(r"\b(will|i'?ll|planning|plan to|upcoming|booked|scheduled|departs?|leaves?|"
                 r"starts? on|starting|next (week|month|year|semester)|looking forward|"
                 r"thinking of|hoping|going to|i'?m attending|due)\b")
fut = []
for sid, (ds, turns) in seen.items():
    for c in turns:
        for sent in V._sentences(c):
            for h in V.resolve_expressions(sent, ds):
                if h["kind"] != "explicit-future":
                    continue
                s = sent.lower()
                fut.append({"expr": h["expr"], "date": h["date"].strftime("%Y/%m/%d"),
                            "sess": ds[:10], "sid": sid,
                            "past": bool(PAST.search(s)), "fut": bool(FUT.search(s)),
                            "sent": re.sub(r"\s+", " ", sent)[:200]})
cnt = Counter((h["past"], h["fut"]) for h in fut)
res["b_total"] = len(fut)
res["b_past_only"] = cnt[(True, False)]
res["b_fut_only"] = cnt[(False, True)]
res["b_both"] = cnt[(True, True)]
res["b_neither"] = cnt[(False, False)]
res["b_past_only_samples"] = [h["sent"] for h in fut if h["past"] and not h["fut"]][:25]
# decision-relevant subset: explicit-future hits inside GOLD sessions of TR questions
gold_sids = set()
for q in D:
    if q["question_type"] == "temporal-reasoning":
        gold_sids.update(q["answer_session_ids"])
res["b_in_TR_gold_sessions"] = sum(1 for h in fut if h["sid"] in gold_sids)
res["b_TR_gold_samples"] = [(h["sess"], h["expr"], h["date"], h["sent"])
                            for h in fut if h["sid"] in gold_sids][:25]

# ---- (c) residual precision of numeric-date rule after guards -------------
NUMDATE = re.compile(r"(?<![\d/\-])(\d{1,2})[/\-](\d{1,2})(?:[/\-](\d{2,4}))?(?![\d/\-])")
NOUN_G = re.compile(r"^\s*(?:more\s+)?(episodes?|times?|days?|hours?|minutes?|weeks?|months?|years?|"
                    r"people|cups?|tsp|tbsp|miles?|km|kg|lbs?|oz|sets?|reps?|pages?|chapters?|"
                    r"servings?|glasses|bottles?|nights?|sessions?)\b")
QUANT_G = re.compile(r"\b(at least|about|around|roughly|maybe|between|some|every|per)\s*$")
UNIT_G = re.compile(r"\b(cup|tsp|tbsp|teaspoon|tablespoon|inch|mile|hour|ratio|odds|split|scale|"
                    r"out of|24/7|lb|oz|km|cm|mm)\b")
DATEY = re.compile(r"\b(on|since|until|till|dated|from|starting|beginning|by)\s*(the\s+)?$")
acc = []
for sid, (ds, turns) in seen.items():
    for c in turns:
        for sent in V._sentences(c):
            t = sent.lower()
            for m in NUMDATE.finditer(t):
                a, b = int(m.group(1)), int(m.group(2))
                if not (1 <= a <= 12 and 1 <= b <= 31):
                    continue
                after = t[m.end():m.end() + 24]
                before = t[max(0, m.start() - 18):m.start()]
                if NOUN_G.search(after) or QUANT_G.search(before) or UNIT_G.search(before + after):
                    continue
                acc.append({"tok": m.group(0), "yr": bool(m.group(3)),
                            "datey_before": bool(DATEY.search(before)),
                            "ctx": t[max(0, m.start() - 45):m.end() + 45], "sess": ds[:10]})
res["c_accepted"] = len(acc)
res["c_with_year_group"] = sum(1 for x in acc if x["yr"])
res["c_datey_preposition"] = sum(1 for x in acc if x["datey_before"])
res["c_neither"] = sum(1 for x in acc if not x["yr"] and not x["datey_before"])
res["c_neither_samples"] = [x["ctx"] for x in acc if not x["yr"] and not x["datey_before"]][:60]
res["c_datey_samples"] = [x["ctx"] for x in acc if x["datey_before"] and not x["yr"]][:30]

# ---- (d) blast radius of the month-alias bug -----------------------------
MO = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|"
                r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b")
n_sent_mo = 0
n_sent = 0
for sid, (ds, turns) in seen.items():
    for c in turns:
        for sent in V._sentences(c):
            n_sent += 1
            if MO.search(sent.lower()):
                n_sent_mo += 1
res["d_sentences"] = n_sent
res["d_sentences_with_month_name"] = n_sent_mo

json.dump(res, open("/root/agent_memory_iclr/results/_vf_v5t_follow.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=1))
