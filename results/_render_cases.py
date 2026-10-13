"""Render the error dossier into compact human-readable case files for manual typology coding."""
import json
import os
import sys

RES = "/root/agent_memory_iclr/results"
D = json.load(open(os.path.join(RES, "_err_dossier.json")))


def clip(s, n):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + " …"


def render(qids, path, ev_n=3, ev_len=340, view_n=5, view_len=200):
    lines = []
    for i, qid in enumerate(qids):
        d = D[qid]
        lines.append(f"\n===== [{i+1}] {qid} | {d['type']} | qdate={d['qdate']} | sess={d['n_sessions']} turns={d['n_units']} ev={d['n_ev']} chains={d['n_chains']}")
        if d["qc"]:
            c = d["qc"]
            lines.append(f"QC  : label={c['qlabel']} mode={c['mode']} f1={c['f1']} evcov={c['cov']} goldtok_raw20={c['gold_tok_in_raw20']} goldtok_view={c['gold_tok_in_view']}")
        else:
            lines.append("QC  : CORRECT")
        if d["full"]:
            f = d["full"]
            lines.append(f"FULL: f1={f['f1']} evcov={f['cov']} goldtok_raw30={f['gold_tok_in_raw30']}")
        else:
            lines.append("FULL: CORRECT")
        lines.append(f"Q   : {clip(d['question'], 450)}")
        lines.append(f"GOLD: {clip(d['gold'], 300)}")
        if d["qc"]:
            lines.append(f"P_QC: {clip(d['qc']['pred'], 900)}")
        if d["full"]:
            lines.append(f"P_FU: {clip(d['full']['pred'], 900)}")
        for e in d["evidence"][:ev_n]:
            lines.append(f"EV  : {clip(e, ev_len)}")
        if len(d["evidence"]) > ev_n:
            lines.append(f"EV  : (+{len(d['evidence'])-ev_n} more evidence turns)")
        for v in d["rel_view_hist"][:view_n]:
            lines.append(f"VH  : {clip(v, view_len)}")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(path, len(lines), "lines", os.path.getsize(path), "bytes")


qc_err = sorted([k for k, v in D.items() if v["qc"]], key=lambda k: (D[k]["type"], k))
fu_only = sorted([k for k, v in D.items() if v["full"] and not v["qc"]], key=lambda k: (D[k]["type"], k))
render(qc_err, os.path.join(RES, "_cases_qc.txt"))
render(fu_only, os.path.join(RES, "_cases_fullonly.txt"))
print("qc_err", len(qc_err), "full_only", len(fu_only))
