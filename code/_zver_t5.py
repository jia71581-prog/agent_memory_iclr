"""Test 5: does extraction actually 'keep user assertions and drop assistant advice'?
For each view line, decide whether its value is traceable to user turns or only to assistant turns."""
import json, re, sys, collections
sys.path.insert(0, "/root/agent_memory_iclr/code")
import lifecycle as LC
from _zver_core import load, cached_extract, norm
from query_classifier import classify_query

LC.extract_session_facts = cached_extract
D = {q["question_id"]: q for q in load()}
absq = [qid for qid in D if qid.endswith("_abs")]
print("abs questions:", len(absq))

STOP = set("the a an is are was were be to of in on at for with and or user yes no it this that they".split())


def toks(s):
    return [t for t in norm(s).split() if len(t) > 3 and t not in STOP]


tot = collections.Counter()
per_q = []
for qid in sorted(absq):
    q = D[qid]
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    # per-session extraction, so each fact's source session is known
    utext, atext = collections.defaultdict(list), collections.defaultdict(list)
    for si, sess in enumerate(q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                (utext if t["role"] == "user" else atext)[si].append(norm(t["content"]))
    ub = {si: " ".join(v) for si, v in utext.items()}
    ab = {si: " ".join(v) for si, v in atext.items()}
    c = collections.Counter()
    for si, sess in enumerate(q["haystack_sessions"]):
        for ent, attr, val in cached_extract(sess):
            tk = toks(val)
            if not tk:
                c["notoken"] += 1
                continue
            inu = sum(1 for t in tk if t in ub.get(si, "")) / len(tk)
            ina = sum(1 for t in tk if t in ab.get(si, "")) / len(tk)
            if inu >= 0.6 and ina < 0.6:
                c["user_only"] += 1
            elif ina >= 0.6 and inu < 0.6:
                c["assistant_only"] += 1
            elif inu >= 0.6 and ina >= 0.6:
                c["both"] += 1
            else:
                c["neither"] += 1
    per_q.append((qid, c))
    tot.update(c)

n = sum(tot[k] for k in ("user_only", "assistant_only", "both", "neither"))
print("\n=== speaker traceability of extracted facts, 30 abstention questions (n=%d facts) ===" % n)
for k in ("user_only", "assistant_only", "both", "neither", "notoken"):
    print("  %-15s %6d  %.3f" % (k, tot[k], tot[k] / n))
print("\nassistant-only extracted facts as a share of speaker-attributable facts: %.3f"
      % (tot["assistant_only"] / (tot["user_only"] + tot["assistant_only"] + tot["both"])))
print("\nper-question assistant_only counts (top 8):")
for qid, c in sorted(per_q, key=lambda x: -x[1]["assistant_only"])[:8]:
    print("  %-24s assistant_only=%3d user_only=%3d both=%3d" % (qid, c["assistant_only"], c["user_only"], c["both"]))
