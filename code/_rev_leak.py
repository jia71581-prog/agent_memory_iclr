"""Reviewer leakage test: blank every gold-metadata field, then check the flag-on
prompts are byte-identical. If the method reads gold anywhere, a prompt must change.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
import lifecycle


def _no_api(*a, **k):
    raise RuntimeError("API call attempted")


common.chat = _no_api
lifecycle.chat = _no_api
import run_qc

CAP = {}
run_qc.answer_chat = lambda s, u, l, op, max_tokens=6144: (CAP.update(sys=s, user=u), "STUB")[1]

data = common.load_dataset("s_cleaned")
gated = set(json.load(open("/tmp/rev_gated.json")))
ledger = common.CostLedger()


def run(qs):
    out = {}
    for q in qs:
        CAP.clear()
        run_qc.answer_question(q, ledger)
        out[q["question_id"]] = hashlib.sha256(CAP["user"].encode()).hexdigest()[:16]
    return out


base = run(data)

# scramble gold: wrong type label, blanked answer, blanked answer_session_ids,
# and every has_answer flag flipped to False (i.e. the oracle signal destroyed)
POISON = "ZZZ_POISONED_ZZZ"
for q in data:
    q["question_type"] = POISON
    q["answer"] = POISON
    q["answer_session_ids"] = []
    for s in q["haystack_sessions"]:
        for t in s:
            if "has_answer" in t:
                t["has_answer"] = False
after = run(data)

diff = [k for k in base if base[k] != after[k]]
print(f"gold-poisoned vs clean: {len(base)-len(diff)}/{len(base)} prompts identical, {len(diff)} changed")
print("  changed inside gate:", len([k for k in diff if k in gated]))
for k in diff[:10]:
    print("  CHANGED", k)
print("VERDICT:", "no gold-metadata dependence" if not diff else "LEAK - prompts depend on gold fields")
