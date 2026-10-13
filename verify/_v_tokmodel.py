"""Character-class token model fitted on 7 real arms (arm-level aggregates, exact prompt
reconstruction), then applied to the v5_temporal scaffold text.
tokens = c0 + a*alpha + b*digit + c*space + d*punct
"""
import json
import statistics as st
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
sys.path.insert(0, "/root/agent_memory_iclr/results")
from common import load_dataset, SHARED_ANSWER_PROMPT
from _recon_ctx import qc_context, units_of
from run_baseline import build_memory, bm25_rank
from v5_temporal import render_timeline
from v5_temporal_val import build_store_cached

HEAD_CUR = ("Current-state fact view (latest value per fact; [CONFLICT] marks values reported "
            "at the same time that could not be ordered)")
HEAD_HIST = "Fact evolution view (full value history per fact, oldest -> newest)"
HEAD_RAW = "Retrieved raw conversation turns"


def cls(s):
    a = d = sp = p = 0
    for ch in s:
        if ch.isalpha():
            a += 1
        elif ch.isdigit():
            d += 1
        elif ch.isspace():
            sp += 1
        else:
            p += 1
    return [a, d, sp, p]


data = load_dataset("s_cleaned")
acc = {k: [0, 0, 0, 0] for k in ["qc", "vo", "nl", "noqc", "full", "fifo", "decay", "scaf"]}
scaf_rows = []
for qi, q in enumerate(data):
    label, mode, view_txt, raw_idx = qc_context(q)
    _, hmode, vhist, _ = qc_context(q, ablate="no_qc")
    units = units_of(q)
    raw_txt = "\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in raw_idx)
    vb_cur = f"### {HEAD_CUR if mode == 'current' else HEAD_HIST}\n{view_txt}"
    vb_hist = f"### {HEAD_HIST}\n{vhist}"
    rb = f"### {HEAD_RAW}\n{raw_txt}"
    tail = f"\n\nQuestion (asked on {q['question_date']}): {q['question']}"
    sysp = SHARED_ANSWER_PROMPT
    texts = {"qc": sysp + vb_cur + "\n\n" + rb + tail,
             "vo": sysp + vb_cur + tail,
             "nl": sysp + rb + tail,
             "noqc": sysp + vb_hist + "\n\n" + rb + tail}
    for sysname in ["full", "fifo", "decay"]:
        u = build_memory(q, sysname)
        docs = [c for (_, _, c) in u]
        order = sorted(bm25_rank(q["question"], docs)[:30])
        ctx = "\n".join(f"[{u[i][0]}] [{u[i][1]}] {u[i][2]}" for i in order)
        texts[sysname] = (sysp + "Retrieved memory (each line: [session date] [speaker] text):\n"
                          + ctx + tail)
    store, miss = build_store_cached(q)
    scaf = render_timeline(q, store, q["question"])
    texts["scaf"] = scaf
    scaf_rows.append({"qid": q["question_id"], "type": q["question_type"],
                      "chars": len(scaf), "cls": cls(scaf)})
    for k, t in texts.items():
        c = cls(t)
        for j in range(4):
            acc[k][j] += c[j]
    if (qi + 1) % 100 == 0:
        print("recon", qi + 1, flush=True)

TOKFILE = {"qc": "full500_qc_ledger_1789633940.json",
           "vo": "full500_qc_view_only_ledger_1789635396.json",
           "nl": "full500_qc_no_lifecycle_ledger_1789630196.json",
           "noqc": "full500_qc_no_qc_ledger_1789634556.json",
           "full": "full500_full_ledger_1789630177.json",
           "fifo": "full500_fifo_ledger_1789630113.json",
           "decay": "full500_decay_ledger_1789630149.json"}
TOK = {}
for k, f in TOKFILE.items():
    d = json.load(open("/root/agent_memory_iclr/results/" + f))
    pt = [r["prompt_tokens"] for r in d["rows"] if r["op"].startswith("answer/") and not r["op"].endswith("-retry")]
    TOK[k] = st.mean(pt)
    print("%-6s n=%d mean tok %.1f  mean chars %.0f  ratio %.3f"
          % (k, len(pt), TOK[k], sum(acc[k]) / 500.0, sum(acc[k]) / 500.0 / TOK[k]))

keys = list(TOKFILE)
X = [[1.0] + [acc[k][j] / 500.0 for j in range(4)] for k in keys]
Y = [TOK[k] for k in keys]
p = 5
XtX = [[sum(X[r][i] * X[r][j] for r in range(len(keys))) for j in range(p)] for i in range(p)]
XtY = [sum(X[r][i] * Y[r] for r in range(len(keys))) for i in range(p)]
M = [XtX[i][:] + [XtY[i]] for i in range(p)]
for c in range(p):
    piv = max(range(c, p), key=lambda r: abs(M[r][c]))
    M[c], M[piv] = M[piv], M[c]
    for r in range(p):
        if r != c and abs(M[c][c]) > 1e-12:
            f = M[r][c] / M[c][c]
            for kk in range(c, p + 1):
                M[r][kk] -= f * M[c][kk]
beta = [M[i][p] / M[i][i] for i in range(p)]
print("\nfit: tok = %.2f + %.5f*alpha + %.5f*digit + %.5f*space + %.5f*punct" % tuple(beta))
for k, x, y in zip(keys, X, Y):
    pred = sum(b * xi for b, xi in zip(beta, x))
    print("  %-6s actual %.0f  pred %.0f  err %+.1f%%" % (k, y, pred, 100 * (pred - y) / y))
print("  implied chars/tok: alpha %.2f digit %.2f space %.2f punct %.2f"
      % tuple(1 / b if abs(b) > 1e-9 else float("nan") for b in beta[1:]))

sc = [acc["scaf"][j] / 500.0 for j in range(4)]
scaf_tok = sum(b * x for b, x in zip(beta[1:], sc))
print("\nSCAFFOLD mean: chars %.0f  (alpha %.0f digit %.0f space %.0f punct %.0f)"
      % (sum(sc), *sc))
print("SCAFFOLD predicted mean added tokens = %.0f  -> %.3f chars/tok" % (scaf_tok, sum(sc) / scaf_tok))
print("composition digit+punct fraction: scaffold %.3f  view(vo-sys) %.3f  raw(nl) %.3f"
      % ((sc[1] + sc[3]) / sum(sc),
         (acc["vo"][1] + acc["vo"][3]) / sum(acc["vo"]),
         (acc["nl"][1] + acc["nl"][3]) / sum(acc["nl"])))
json.dump({"beta": beta, "acc": acc, "tok": TOK, "scaf_rows": scaf_rows},
          open("/root/agent_memory_iclr/verify/_v_tokmodel.json", "w"))
