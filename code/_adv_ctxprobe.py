"""Decisive mechanism test: was the gold-critical string ALREADY in the v4 context
(capped lifecycle view + top-20 BM25 raw turns)? Offline only."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from lifecycle import render_view
from query_classifier import classify_query
from run_baseline import bm25_rank
from _adv_build import build_store_offline
from v5_chains import merge_chains, render_aggregation_view

VIEW_CHAR_CAP = 20000
TOP_K = 20


def v4_context(q, store):
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    view_txt = render_view(store, mode)
    if len(view_txt) > VIEW_CHAR_CAP:
        kept, total = [], 0
        for line in view_txt.splitlines():
            if total + len(line) + 1 > VIEW_CHAR_CAP:
                break
            kept.append(line)
            total += len(line) + 1
        view_txt = "\n".join(kept)
    units = []
    for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                units.append((date, t["role"], t["content"]))
    order = sorted(bm25_rank(q["question"], [u[2] for u in units])[:TOP_K])
    raw_txt = "\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in order)
    return view_txt, raw_txt, mode


PROBES = {
    "6cb6f249": ["10-day", "10 day", "ten-day", "week-long", "week long"],
    "0a995998": ["dry cleaning", "exchanged boots", "Zara", "H&M"],
    "46a3abf7": ["5-gallon", "20-gallon", "1-gallon", "cousin"],
    "3a704032": ["succulent", "peace lily", "snake plant", "two weeks ago"],
    "88432d0a": ["chicken wings", "sourdough", "chocolate cake", "cookies"],
    "6d550036": ["Marketing Research", "high-priority"],
    "1a8a66a6": ["New Yorker", "Architectural Digest", "Forbes"],
}


def main():
    data = load_dataset()
    by = {q["question_id"]: q for q in data}
    out = {}
    for qid, probes in PROBES.items():
        q = by[qid]
        st, _ = build_store_offline(q)
        m = merge_chains(st)
        view, raw, mode = v4_context(q, st)
        v5view, _, _ = v4_context(q, m)
        blk = render_aggregation_view(m, q["question"])
        row = {"mode": mode, "view_chars": len(view), "raw_chars": len(raw),
               "blk_chars": len(blk), "gold": str(q["answer"])[:70], "probes": {}}
        for p in probes:
            row["probes"][p] = {
                "v4_view": p.lower() in view.lower(),
                "v4_raw": p.lower() in raw.lower(),
                "v5_view": p.lower() in v5view.lower(),
                "v5_block": p.lower() in blk.lower(),
            }
        out[qid] = row
        print(qid, "mode=%s view=%d raw=%d blk=%d" % (mode, len(view), len(raw), len(blk)))
        for p, d in row["probes"].items():
            print("   %-22s v4view=%-5s v4raw=%-5s v5view=%-5s v5block=%s"
                  % (p, d["v4_view"], d["v4_raw"], d["v5_view"], d["v5_block"]))
    with open("/root/agent_memory_iclr/results/_adv_ctx_probe.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
