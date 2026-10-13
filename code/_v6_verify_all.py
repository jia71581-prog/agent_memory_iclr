"""Full-500 verification: v5 vs v6-AGG end-to-end context (real run_qc path, API hard-blocked).
Asserts every non-firing question is byte-identical; reports the char delta on fired ones."""
import json, os, sys, statistics as st
os.environ["QC_V5"] = "1"; os.environ["QC_V6_AGG"] = "1"
sys.path.insert(0, "/root/agent_memory_iclr/code")
import common, lifecycle, run_qc, v6_agg

def _blocked(*a, **k):
    raise RuntimeError("API CALL ATTEMPTED")
common.chat = _blocked; lifecycle.chat = _blocked
CAP = {}
run_qc.answer_chat = lambda s, u, l, op, max_tokens=6144: (CAP.update(ctx=u), "STUB")[1]

def ctx_for(q, v6):
    run_qc.V6_AGG = v6
    run_qc.answer_question(q, None)
    return CAP["ctx"]

data = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
rows = {json.loads(l)["qid"]: json.loads(l) for l in open("/root/agent_memory_iclr/results/full500_qc_v5.jsonl")}
deltas, ledger_sizes, nonfire_ident, fired = [], [], 0, 0
for i, q in enumerate(data):
    f = v6_agg.is_agg_question(q["question"])
    c5, c6 = ctx_for(q, False), ctx_for(q, True)
    if not f:
        assert c5 == c6, f"NON-FIRING CHANGED: {q['question_id']}"
        nonfire_ident += 1
        continue
    fired += 1
    deltas.append(len(c6) / len(c5) - 1)
    ledger_sizes.append(c6.count("\n- [") )
    if (i + 1) % 100 == 0:
        print(f"...{i+1}/500", flush=True)
print(f"non-firing byte-identical: {nonfire_ident}/{nonfire_ident} | fired: {fired}")
print("fired char delta: mean %+.3f%%  median %+.3f%%  min %+.2f%%  max %+.2f%%  |  over all 500: %+.3f%%"
      % (100*st.mean(deltas), 100*st.median(deltas), 100*min(deltas), 100*max(deltas),
         100*sum(deltas)/500))
print("ledger lines: mean %.1f median %.0f max %d" % (st.mean(ledger_sizes), st.median(ledger_sizes), max(ledger_sizes)))
