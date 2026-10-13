"""ADVERSARIAL corpus-wide audit of the four v5_temporal.py parser-fix claims.
Offline, no API. Dumps /root/agent_memory_iclr/results/_vf_v5t_audit.json
"""
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timedelta

sys.path.insert(0, "/root/agent_memory_iclr/code")
import v5_temporal as V

D = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
out = {}

# ---------------------------------------------------------------- M1: MONTHS
m1 = {}
names = ["january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"]
m1["full_ok"] = all(V.MONTHS[n] == i + 1 for i, n in enumerate(names))
m1["abbr_ok"] = all(V.MONTHS[n[:3]] == i + 1 for i, n in enumerate(names))
m1["abbr_map"] = {n[:3]: V.MONTHS[n[:3]] for n in names}
# reachability: does the production path always go through the [:3] alias?
m1["slice_uses"] = len(re.findall(r"MONTHS\[[^\]]*\[:3\]\]", open(V.__file__).read()))
m1["nonslice_uses"] = len(re.findall(r"MONTHS\[(?!.*\[:3\])", open(V.__file__).read()))
# off-by-one injection: what would every abbreviated month resolve to?
V.MONTHS.update({n[:3]: (i + 2 if i < 11 else 1) for i, n in enumerate(names)})
bad = {h["expr"]: h["date"].strftime("%Y/%m/%d") for h in
       V.resolve_expressions("I visited on March 15th and on Dec 3rd.", "2023/05/25")}
V.MONTHS.update({n[:3]: i + 1 for i, n in enumerate(names)})
good = {h["expr"]: h["date"].strftime("%Y/%m/%d") for h in
        V.resolve_expressions("I visited on March 15th and on Dec 3rd.", "2023/05/25")}
m1["injected_offby1"] = bad
m1["current"] = good
out["M1_months"] = m1

# --------------------------------------------- gather unique user sentences
seen_sess = {}
for q in D:
    for ds, sid, sess in zip(q["haystack_dates"], q["haystack_session_ids"], q["haystack_sessions"]):
        if sid in seen_sess:
            continue
        seen_sess[sid] = (ds, [t.get("content") or "" for t in sess if t.get("role") == "user"])
print("unique sessions", len(seen_sess), file=sys.stderr)

NUMDATE = re.compile(r"(?<![\d/\-])(\d{1,2})[/\-](\d{1,2})(?:[/\-](\d{2,4}))?(?![\d/\-])")
NOUN_G = re.compile(r"^\s*(?:more\s+)?(episodes?|times?|days?|hours?|minutes?|weeks?|months?|years?|"
                    r"people|cups?|tsp|tbsp|miles?|km|kg|lbs?|oz|sets?|reps?|pages?|chapters?|"
                    r"servings?|glasses|bottles?|nights?|sessions?)\b")
QUANT_G = re.compile(r"\b(at least|about|around|roughly|maybe|between|some|every|per)\s*$")
UNIT_G = re.compile(r"\b(cup|tsp|tbsp|teaspoon|tablespoon|inch|mile|hour|ratio|odds|split|scale|"
                    r"out of|24/7|lb|oz|km|cm|mm)\b")

acc, blk_noun, blk_quant, blk_unit, bad_range = [], [], [], [], []
fut_hits = []
sent_n = 0
for sid, (ds, turns) in seen_sess.items():
    ref = V.parse_date(ds)
    if ref is None:
        continue
    for c in turns:
        for sent in V._sentences(c):
            sent_n += 1
            t = sent.lower()
            for m in NUMDATE.finditer(t):
                a, b = int(m.group(1)), int(m.group(2))
                if not (1 <= a <= 12 and 1 <= b <= 31):
                    bad_range.append(m.group(0))
                    continue
                after = t[m.end():m.end() + 24]
                before = t[max(0, m.start() - 18):m.start()]
                ctx = t[max(0, m.start() - 40):m.end() + 40]
                if NOUN_G.search(after):
                    blk_noun.append((m.group(0), ctx, sid, ds))
                elif QUANT_G.search(before):
                    blk_quant.append((m.group(0), ctx, sid, ds))
                elif UNIT_G.search(before + after):
                    blk_unit.append((m.group(0), ctx, sid, ds))
                else:
                    acc.append((m.group(0), ctx, sid, ds))
            # explicit-future census (both explicit month-day rules + numeric)
            for h in V.resolve_expressions(sent, ds):
                if h["kind"] == "explicit-future":
                    fut_hits.append({"expr": h["expr"], "date": h["date"].strftime("%Y/%m/%d"),
                                     "sess": ds, "sid": sid,
                                     "sent": re.sub(r"\s+", " ", sent)[:220]})

out["M2_numdate"] = {
    "sentences_scanned": sent_n,
    "regex_matches_total": len(acc) + len(blk_noun) + len(blk_quant) + len(blk_unit) + len(bad_range),
    "out_of_range_dropped": len(bad_range),
    "accepted_as_date": len(acc),
    "blocked_by_noun_guard": len(blk_noun),
    "blocked_by_quant_guard": len(blk_quant),
    "blocked_by_unit_guard": len(blk_unit),
    "noun_guard_samples": blk_noun[:40],
    "quant_guard_samples": blk_quant[:40],
    "accepted_samples": acc[:60],
    "accepted_token_hist": Counter(x[0] for x in acc).most_common(25),
}
out["M3_future"] = {
    "n_hits": len(fut_hits),
    "n_unique_sent": len({h["sent"] for h in fut_hits}),
    "samples": fut_hits[:80],
}
json.dump(out, open("/root/agent_memory_iclr/results/_vf_v5t_audit.json", "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)}
                  for k, v in out.items()}, indent=1))
