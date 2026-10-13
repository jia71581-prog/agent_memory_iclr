"""v6-INF/VER offline measurement (no API calls).

Exercises the real run_qc code path with build_lifecycle swapped for the on-disk
extraction cache and answer_chat stubbed, so the captured (system, user) pair is
byte-exact what the online arm would send.

Reports:
 1. byte-identity of the v6 first-pass user content with v5,
 2. measured char delta of the QC_V6_INF scaffold (and of the budget-neutral variant),
 3. QC_V6_VER gate firing rate over all 500 questions, using the stored v5
    predictions as the first-pass drafts,
 4. per-error-case: which scaffold blocks attach and which gate fires.
"""
import json
import os
import sys

os.environ["QC_V5"] = "1"
os.environ.pop("QC_V6_INF", None)
os.environ.pop("QC_V6_VER", None)
os.environ.pop("QC_V6_SSP", None)
sys.path.insert(0, "/root/agent_memory_iclr/code")
sys.path.insert(0, "/root/agent_memory_iclr/results")

import run_qc as RQ
from _recon_ctx import build_store, CACHE_MISS
from common import load_dataset
from v5_shape import classify_shape

CAP = []
RQ.build_lifecycle = lambda q, ledger: build_store(q)
RQ.answer_chat = lambda s, u, ledger, op, max_tokens=6144: (CAP.append((s, u)), "STUB")[1]

rows = {r["qid"]: r for r in (json.loads(l) for l in
        open("/root/agent_memory_iclr/results/full500_qc_v5.jsonl"))}
data = load_dataset("s_cleaned")
out = []
for i, q in enumerate(data):
    qid = q["question_id"]
    CAP.clear()
    RQ.answer_question(q, None)
    sysp, user = CAP[-1]
    ctx = user.rsplit("\n\nQuestion (asked on ", 1)[0]
    scaf = RQ.v6_scaffold_for(q["question"])
    shape, _ = classify_shape(q["question"])
    v5_chars = len(sysp) + len(user)
    inf_chars = len(sysp) + (len(scaf) + 2 if scaf else 0) + len(user)
    neutral_ctx = RQ._v6_trim_view_tail(ctx, len(scaf)) if scaf else ctx
    neu_chars = inf_chars - (len(ctx) - len(neutral_ctx))
    draft = rows[qid]["pred"] if qid in rows else ""
    out.append({"qid": qid, "type": q["question_type"], "shape": shape,
                "judge": rows.get(qid, {}).get("judge"),
                "v5_chars": v5_chars, "inf_chars": inf_chars, "neu_chars": neu_chars,
                "scaf_chars": len(scaf), "ctx_chars": len(ctx),
                "gate": RQ._v6_verify_gate(shape, draft, ctx),
                "gate_neutral_ctx_delta": len(neutral_ctx) - len(ctx)})
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(data)}", flush=True)

with open("/root/agent_memory_iclr/results/_v6inf_measure.json", "w") as f:
    json.dump(out, f, indent=1)
print("cache misses:", CACHE_MISS[0], "n =", len(out))
