"""Dump the ordering block for every currently-CORRECT 2-alternative ordering TR question,
compactly, for hand adjudication (match_item_date is too weak to trust)."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from v5_temporal import (detect_family, extract_events, precompute_arithmetic,
                         question_alternatives)
from v5_temporal_val import build_store_cached

R = "/root/agent_memory_iclr/results"
qc = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{R}/full500_qc.jsonl")}
data = [q for q in load_dataset("s_cleaned") if q["question_type"] == "temporal-reasoning"]

for q in data:
    qid = q["question_id"]
    family, unit = detect_family(q["question"])
    if family != "ordering":
        continue
    alts = question_alternatives(q["question"])
    if len(alts) != 2:
        continue
    store, _ = build_store_cached(q)
    events = extract_events(q, store, q["question"])
    lines = precompute_arithmetic(q["question"], q["question_date"], events, family, unit,
                                 q_for_anchor=q)
    print("=" * 100)
    print(f"{qid} | judge={qc.get(qid,{}).get('judge')} | Q: {q['question']}")
    print(f"GOLD: {q['answer']}")
    keep = False
    for ln in lines:
        if ln.startswith("Chronological order"):
            keep = True
        if keep:
            print("   ", ln[:150])
