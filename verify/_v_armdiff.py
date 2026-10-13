"""Arm-differenced token calibration: measure chars-per-token separately for the
structured lifecycle view vs the prose raw turns, using the three real arms.
qc          = system + view + raw20 + question
qc_view_only= system + view          + question
qc_no_lifecycle = system +     raw20 + question
Differences cancel the fixed overhead exactly.
"""
import json
import statistics as st
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
sys.path.insert(0, "/root/agent_memory_iclr/results")
from common import load_dataset, SHARED_ANSWER_PROMPT
from _recon_ctx import qc_context, units_of

HEAD_CUR = ("Current-state fact view (latest value per fact; [CONFLICT] marks values reported "
            "at the same time that could not be ordered)")
HEAD_HIST = "Fact evolution view (full value history per fact, oldest -> newest)"
HEAD_RAW = "Retrieved raw conversation turns"

data = load_dataset("s_cleaned")
rows = []
for qi, q in enumerate(data):
    label, mode, view_txt, raw_idx = qc_context(q)
    units = units_of(q)
    raw_txt = "\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in raw_idx)
    head = HEAD_CUR if mode == "current" else HEAD_HIST
    view_block = f"### {head}\n{view_txt}"
    raw_block = f"### {HEAD_RAW}\n{raw_txt}"
    tail = f"\n\nQuestion (asked on {q['question_date']}): {q['question']}"
    qc_user = view_block + "\n\n" + raw_block + tail
    vo_user = view_block + tail
    nl_user = raw_block + tail
    rows.append({"qid": q["question_id"], "type": q["question_type"],
                 "qc": len(qc_user), "vo": len(vo_user), "nl": len(nl_user),
                 "view_block": len(view_block), "raw_block": len(raw_block),
                 "view_txt": len(view_txt), "raw_txt": len(raw_txt)})
    if (qi + 1) % 150 == 0:
        print("recon", qi + 1, flush=True)

json.dump(rows, open("/root/agent_memory_iclr/verify/_v_armchars.json", "w"))
m = lambda k: st.mean([r[k] for r in rows])
print("\nmean reconstructed user chars: qc %.0f  view_only %.0f  no_lifecycle %.0f"
      % (m("qc"), m("vo"), m("nl")))
print("mean view block %.0f  raw block %.0f" % (m("view_block"), m("raw_block")))

TOK = {"qc": 11343.6, "vo": 6632.0, "nl": 4832.0}
import json as _j
for tag, f in [("qc", "full500_qc_ledger_1789633940.json"),
               ("vo", "full500_qc_view_only_ledger_1789635396.json"),
               ("nl", "full500_qc_no_lifecycle_ledger_1789630196.json")]:
    d = _j.load(open("/root/agent_memory_iclr/results/" + f))
    pt = [r["prompt_tokens"] for r in d["rows"] if r["op"] == "answer/qc"]
    TOK[tag] = st.mean(pt)
    print("%s ledger: n=%d mean prompt tokens %.1f" % (tag, len(pt), st.mean(pt)))

d_view_tok = TOK["qc"] - TOK["nl"]
d_view_chr = m("qc") - m("nl")
d_raw_tok = TOK["qc"] - TOK["vo"]
d_raw_chr = m("qc") - m("vo")
print("\n--- marginal chars per token (arm differencing) ---")
print("VIEW (structured, date-stamped lines): %.0f chars -> %.0f tokens = %.3f chars/tok"
      % (d_view_chr, d_view_tok, d_view_chr / d_view_tok))
print("RAW  (prose conversation turns)      : %.0f chars -> %.0f tokens = %.3f chars/tok"
      % (d_raw_chr, d_raw_tok, d_raw_chr / d_raw_tok))
print("whole qc prompt (incl. fixed overhead): %.0f chars / %.1f tok = %.3f chars/tok"
      % (m("qc") + len(SHARED_ANSWER_PROMPT), TOK["qc"],
         (m("qc") + len(SHARED_ANSWER_PROMPT)) / TOK["qc"]))
# consistency check: predicted qc tokens from the two marginals + overhead
ovh_vo = TOK["vo"] - m("view_block") / (d_view_chr / d_view_tok)
ovh_nl = TOK["nl"] - m("raw_block") / (d_raw_chr / d_raw_tok)
print("implied fixed overhead tokens: from view_only %.0f, from no_lifecycle %.0f" % (ovh_vo, ovh_nl))
