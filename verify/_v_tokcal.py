"""Reconstruct the exact answer/qc prompt per question, map to ledger prompt_tokens,
and derive chars-per-token (total ratio + marginal slope, overall and per component)."""
import json
import os
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
sys.path.insert(0, "/root/agent_memory_iclr/results")
from common import load_dataset, SHARED_ANSWER_PROMPT
from _recon_ctx import qc_context, units_of

data = load_dataset("s_cleaned")
led = json.load(open("/root/agent_memory_iclr/results/full500_qc_ledger_1789633940.json"))
rows = led["rows"]
# walk: answer/qc [answer/qc-retry] judge [judge-retry] per question, in dataset order
ans_tokens = []
i = 0
while i < len(rows):
    r = rows[i]
    if r["op"] == "answer/qc":
        ans_tokens.append(r["prompt_tokens"])
    i += 1
print("answer/qc rows:", len(ans_tokens))
assert len(ans_tokens) == 500

out = []
for qi, q in enumerate(data):
    label, mode, view_txt, raw_idx = qc_context(q)
    units = units_of(q)
    raw_txt = "\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in raw_idx)
    parts = []
    if view_txt:
        head = ("Current-state fact view (latest value per fact; [CONFLICT] marks values reported "
                "at the same time that could not be ordered)"
                if mode == "current" else
                "Fact evolution view (full value history per fact, oldest -> newest)")
        parts.append(f"### {head}\n{view_txt}")
    if raw_txt:
        parts.append(f"### Retrieved raw conversation turns\n{raw_txt}")
    ctx = "\n\n".join(parts)
    user = f"{ctx}\n\nQuestion (asked on {q['question_date']}): {q['question']}"
    out.append({"qid": q["question_id"], "type": q["question_type"],
                "view_chars": len(view_txt), "raw_chars": len(raw_txt),
                "user_chars": len(user), "sys_chars": len(SHARED_ANSWER_PROMPT),
                "tok": ans_tokens[qi]})
    if (qi + 1) % 100 == 0:
        print("recon", qi + 1, flush=True)

json.dump(out, open("/root/agent_memory_iclr/verify/_v_prompt_chars.json", "w"))
import statistics as st
tot_c = [r["user_chars"] + r["sys_chars"] for r in out]
tot_t = [r["tok"] for r in out]
print("mean prompt chars %.0f  mean tokens %.1f  overall chars/tok %.3f"
      % (st.mean(tot_c), st.mean(tot_t), sum(tot_c) / sum(tot_t)))
print("per-question chars/tok: median %.3f  min %.3f  max %.3f  p10 %.3f"
      % (st.median([c / t for c, t in zip(tot_c, tot_t)]),
         min(c / t for c, t in zip(tot_c, tot_t)),
         max(c / t for c, t in zip(tot_c, tot_t)),
         sorted(c / t for c, t in zip(tot_c, tot_t))[50]))

# OLS: tok = a + b*chars  -> marginal chars/tok = 1/b
n = len(out)
mx, my = st.mean(tot_c), st.mean(tot_t)
b = sum((x - mx) * (y - my) for x, y in zip(tot_c, tot_t)) / sum((x - mx) ** 2 for x in tot_c)
a = my - b * mx
print("OLS tok = %.4f + %.6f*chars  -> marginal chars/tok = %.3f  (intercept %.1f tok)"
      % (a, b, 1 / b, a))

# two-component OLS: tok = a + b1*view_chars + b2*raw_chars
import itertools
X = [[1.0, float(r["view_chars"]), float(r["raw_chars"])] for r in out]
Y = [float(r["tok"]) for r in out]
p = 3
XtX = [[sum(X[k][i] * X[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
XtY = [sum(X[k][i] * Y[k] for k in range(n)) for i in range(p)]
# gaussian elim
M = [row[:] + [XtY[i]] for i, row in enumerate(XtX)]
for c in range(p):
    piv = max(range(c, p), key=lambda r: abs(M[r][c]))
    M[c], M[piv] = M[piv], M[c]
    for r in range(p):
        if r != c:
            f = M[r][c] / M[c][c]
            for k in range(c, p + 1):
                M[r][k] -= f * M[c][k]
beta = [M[i][p] / M[i][i] for i in range(p)]
print("tok = %.2f + %.6f*view_chars + %.6f*raw_chars" % tuple(beta))
print("  view marginal chars/tok = %.3f ; raw marginal chars/tok = %.3f"
      % (1 / beta[1], 1 / beta[2]))
