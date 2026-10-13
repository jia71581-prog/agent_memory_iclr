"""Two-class BPE token model calibrated on the two exactly-measured marginals:
  view block  : 20056 chars -> 6511 tokens   (structured, date-stamped lines)
  raw  block  : 21172 chars -> 4711 tokens   (prose conversation turns)
Model: tokens = a*(alpha+space) + b*(digit+punct).  Solve 2x2, apply to the scaffold.
"""
import json
import statistics as st
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
sys.path.insert(0, "/root/agent_memory_iclr/results")
from common import load_dataset
from _recon_ctx import qc_context, units_of

HEAD_CUR = ("Current-state fact view (latest value per fact; [CONFLICT] marks values reported "
            "at the same time that could not be ordered)")
HEAD_HIST = "Fact evolution view (full value history per fact, oldest -> newest)"
HEAD_RAW = "Retrieved raw conversation turns"


def cls2(s):
    """(alpha+space, digit+punct)"""
    a = d = 0
    for ch in s:
        if ch.isalpha() or ch.isspace():
            a += 1
        else:
            d += 1
    return a, d


data = load_dataset("s_cleaned")
TL = json.load(open("/root/agent_memory_iclr/results/v5_timelines_all.json"))
V = [0, 0]
R = [0, 0]
S = [0, 0]
mism = 0
sizes = {r["qid"]: r["chars"] for r in json.load(open("/root/agent_memory_iclr/verify/_v_sizes.json"))}
per = []
for qi, q in enumerate(data):
    label, mode, view_txt, raw_idx = qc_context(q)
    units = units_of(q)
    raw_txt = "\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in raw_idx)
    vb = f"### {HEAD_CUR if mode == 'current' else HEAD_HIST}\n{view_txt}"
    rb = f"### {HEAD_RAW}\n{raw_txt}"
    a, d = cls2(vb); V[0] += a; V[1] += d
    a, d = cls2(rb); R[0] += a; R[1] += d
    t = TL[q["question_id"]]
    if len(t) != sizes[q["question_id"]]:
        mism += 1
    a, d = cls2(t); S[0] += a; S[1] += d
    per.append((q["question_id"], q["question_type"], a, d, len(t)))
    if (qi + 1) % 150 == 0:
        print("recon", qi + 1, flush=True)

print("stored-timeline length mismatches vs my recompute:", mism, "/500")
V = [x / 500.0 for x in V]; R = [x / 500.0 for x in R]; S = [x / 500.0 for x in S]
print("mean per question  view(alpha+sp %.0f, dig+punct %.0f)  raw(%.0f, %.0f)  scaffold(%.0f, %.0f)"
      % (V[0], V[1], R[0], R[1], S[0], S[1]))
print("dig+punct fraction: view %.3f  raw %.3f  scaffold %.3f"
      % (V[1] / sum(V), R[1] / sum(R), S[1] / sum(S)))
TV, TR_ = 6511.4, 4711.1
det = V[0] * R[1] - V[1] * R[0]
a = (TV * R[1] - TR_ * V[1]) / det
b = (V[0] * TR_ - R[0] * TV) / det
print("solved: a(alpha+space)=%.5f tok/char (%.2f chars/tok)  b(digit+punct)=%.5f (%.2f chars/tok)"
      % (a, 1 / a, b, 1 / b))
sc_tok = a * S[0] + b * S[1]
print("SCAFFOLD mean chars %.0f -> predicted %.0f tokens = %.3f chars/tok" % (sum(S), sc_tok, sum(S) / sc_tok))
print("check: view %.0f (actual 6511), raw %.0f (actual 4711)" % (a * V[0] + b * V[1], a * R[0] + b * R[1]))

# per-subset predictions
def pred(rows):
    A = sum(r[2] for r in rows) / 500.0
    D = sum(r[3] for r in rows) / 500.0
    return a * A + b * D, sum(r[4] for r in rows) / 500.0

all_t, all_c = pred(per)
gated = [r for r in per if r[0] in {x["qid"] for x in json.load(open("/root/agent_memory_iclr/verify/_v_sizes.json")) if x["fam"] != "other"}]
g_t, g_c = pred(gated)
tr = [r for r in per if r[1] == "temporal-reasoning"]
tr_t = (a * sum(r[2] for r in tr) + b * sum(r[3] for r in tr)) / len(tr)
tr_c = sum(r[4] for r in tr) / len(tr)
BASE = 11343.6
print("\n--- cost vs the real qc answer prompt (mean %.0f tokens) ---" % BASE)
print("ungated all-500 : %.0f chars -> %.0f tok = +%.2f%%" % (all_c, all_t, 100 * all_t / BASE))
print("gated  amortized: %.0f chars -> %.0f tok = +%.2f%%" % (g_c, g_t, 100 * g_t / BASE))
print("TR-only per-question: %.0f chars -> %.0f tok = +%.2f%% prompt growth" % (tr_c, tr_t, 100 * tr_t / BASE))
json.dump({"a": a, "b": b, "V": V, "R": R, "S": S}, open("/root/agent_memory_iclr/verify/_v_tok2.json", "w"))
