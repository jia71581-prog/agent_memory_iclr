"""Reconstruct the exact v5 context for every remaining v5 error (no API calls).

Writes /root/agent_memory_iclr/results/_v6inf_errs.json with, per failing qid:
question, question_date, gold, pred, gold-evidence turns, evidence-in-context flags,
the v5 view text and the v5 raw-turn block.
"""
import json
import os
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
sys.path.insert(0, "/root/agent_memory_iclr/results")
import lifecycle as LC
from _recon_ctx import build_store, evidence_turns, CACHE_MISS
import run_qc as RQ
from common import load_dataset
from query_classifier import classify_query

rows = [json.loads(l) for l in open("/root/agent_memory_iclr/results/full500_qc_v5.jsonl")]
errs = {r["qid"]: r for r in rows if not r.get("judge")}
data = load_dataset("s_cleaned")
out = []
for q in data:
    qid = q["question_id"]
    if qid not in errs:
        continue
    r = errs[qid]
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    store = build_store(q)
    view_txt = RQ._rank_view_lines(store, mode, q["question"], RQ.VIEW_CHAR_CAP)
    units, sel = RQ._select_turns_v5(q, q["question"], RQ.TOP_K)
    raw_txt = "\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in sel)
    ev = evidence_turns(q)
    got = {units[i][2] for i in sel}
    out.append({
        "qid": qid, "type": q["question_type"], "qlabel": label, "view_mode": mode,
        "question": q["question"], "question_date": q["question_date"],
        "gold": q["answer"], "pred": r.get("pred", ""), "f1": r.get("f1"),
        "n_ev": len(ev), "ev_in_raw": sum(1 for e in ev if e in got),
        "ev_turns": ev,
        "view_chars": len(view_txt), "raw_chars": len(raw_txt),
        "view": view_txt, "raw": raw_txt,
    })
    print(f"{qid} {q['question_type'][:20]:20s} ev {sum(1 for e in ev if e in got)}/{len(ev)} "
          f"view {len(view_txt)} raw {len(raw_txt)}", flush=True)

with open("/root/agent_memory_iclr/results/_v6inf_errs.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("cache misses:", CACHE_MISS[0], "cases:", len(out))
