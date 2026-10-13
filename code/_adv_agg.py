"""Adversarial recomputation: does the v5 aggregation block expose a count that MATCHES gold,
and what does it cost in context? Offline only."""
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from lifecycle import render_view
from query_classifier import classify_query
from _adv_build import build_store_offline
from v5_chains import (merge_chains, render_aggregation_view, aggregation_printed_groups,
                       aggregation_kind, is_aggregation_question)

WORD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "zero": 0,
        "a": 1, "an": 1, "single": 1, "twice": 2, "once": 1}


def gold_number(gold):
    g = str(gold).lower()
    m = re.search(r"\b(\d+)\b", g)
    if m:
        return int(m.group(1))
    for w, v in WORD.items():
        if re.search(rf"\b{w}\b", g):
            return v
    return None


def analyse(q, m):
    ques = q["question"]
    kept, (w, order) = aggregation_printed_groups(m, ques)
    block = render_aggregation_view(m, ques)
    gn = gold_number(q["answer"])
    sizes = [len(g["items"]) for g in kept]
    strong = [g["n_strong"] for g in kept]
    union = sum(sizes)
    cand = set(sizes) | set(strong) | {union, sum(strong)}
    return {
        "qid": q["question_id"], "type": q["question_type"], "gold": str(q["answer"])[:60],
        "gold_n": gn, "kind": aggregation_kind(ques),
        "terms": [g["term"] for g in kept], "sizes": sizes, "strong": strong,
        "union": union, "union_strong": sum(strong),
        "top_size": sizes[0] if sizes else 0, "top_strong": strong[0] if strong else 0,
        "gold_reachable": (gn in cand) if gn is not None else None,
        "top_size_eq": (gn == sizes[0]) if (gn is not None and sizes) else None,
        "top_strong_eq": (gn == strong[0]) if (gn is not None and strong) else None,
        "block_chars": len(block), "n_flags": block.count("may restate an earlier line"),
        "block": block,
    }


def main():
    qids = json.loads(sys.argv[1]) if len(sys.argv) > 1 else None
    data = load_dataset()
    out = []
    for q in data:
        if qids is not None and q["question_id"] not in qids:
            continue
        if not is_aggregation_question(q["question"])[0]:
            continue
        st, miss = build_store_offline(q)
        m = merge_chains(st)
        r = analyse(q, m)
        r["miss_sess"] = miss
        label, _ = classify_query(q["question"])
        mode = "historical" if label == "historical" else "current"
        r["v4_view_chars"] = len(render_view(st, mode))
        r["v5_view_chars"] = len(render_view(m, mode))
        r["question"] = q["question"]
        out.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "block"}, ensure_ascii=False),
              flush=True)
    with open("/root/agent_memory_iclr/results/_adv_agg.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("n_agg_questions", len(out))


if __name__ == "__main__":
    main()
