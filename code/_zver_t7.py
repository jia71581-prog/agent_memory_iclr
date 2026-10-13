"""Test 7: precise version of the mechanism claim. A number in a view line is 'assistant-sourced'
if that numeral appears in the session's assistant turns but in none of its user turns."""
import json, re, sys, collections
sys.path.insert(0, "/root/agent_memory_iclr/code")
import lifecycle as LC
from _zver_core import load, cached_extract
from query_classifier import classify_query

LC.extract_session_facts = cached_extract
D = {q["question_id"]: q for q in load()}
absq = sorted(qid for qid in D if qid.endswith("_abs"))
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")

tot = collections.Counter()
per_q = {}
examples = []
for qid in absq:
    q = D[qid]
    c = collections.Counter()
    for si, sess in enumerate(q["haystack_sessions"]):
        ut = " ".join(t["content"] for t in sess if t.get("content") and t["role"] == "user")
        at = " ".join(t["content"] for t in sess if t.get("content") and t["role"] == "assistant")
        for ent, attr, val in cached_extract(sess):
            nums = set(NUM.findall(val))
            if not nums:
                continue
            for nstr in nums:
                if len(nstr) < 2:      # single digits are too common to attribute
                    continue
                inu, ina = nstr in ut, nstr in at
                if inu:
                    c["user_num"] += 1
                elif ina:
                    c["asst_only_num"] += 1
                    if len(examples) < 14:
                        examples.append((qid, ent, attr, val, nstr))
                else:
                    c["neither_num"] += 1
    per_q[qid] = c
    tot.update(c)

n = tot["user_num"] + tot["asst_only_num"] + tot["neither_num"]
print("=== numerals inside extracted view facts, 30 abstention questions (n=%d numeral instances) ===" % n)
for k in ("user_num", "asst_only_num", "neither_num"):
    print("  %-14s %6d  %.3f" % (k, tot[k], tot[k] / n))
print("\nnumerals in view lines that appear ONLY in assistant turns: %d (%.1f%% of numerals)"
      % (tot["asst_only_num"], 100 * tot["asst_only_num"] / n))
print("questions with >=1 assistant-only numeral promoted into the view: %d/30"
      % sum(1 for c in per_q.values() if c["asst_only_num"]))
print("\nexamples (view fact | numeral present only in assistant turns of that session):")
for qid, e, a, v, nstr in examples:
    print("   %-22s %s | %s = %s      [num %s]" % (qid, e[:16], a[:38], v[:52], nstr))
print("\nper-question assistant-only-numeral counts, top 8:")
for qid, c in sorted(per_q.items(), key=lambda x: -x[1]["asst_only_num"])[:8]:
    print("   %-24s asst_only=%3d user=%3d" % (qid, c["asst_only_num"], c["user_num"]))
