"""Does the aggregation block carry information the v4 context did not already have?
For every printed item line, test whether its value string already occurs verbatim in the
v4 context (capped view + top-20 BM25 turns). Offline only."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from _adv_build import build_store_offline
from _adv_ctxprobe import v4_context
from v5_chains import merge_chains, aggregation_printed_groups, is_aggregation_question

HAND = ["0a995998", "46a3abf7", "6cb6f249", "88432d0a", "4f54b7c9", "80ec1f4f", "6d550036",
        "gpt4_7fce9456", "1a8a66a6", "gpt4_f2262a51", "3a704032", "gpt4_2f8be40d",
        "2ce6a0f2", "bf659f65", "d682f1a2", "9d25d4e0"]


def main():
    only_hand = "--hand" in sys.argv
    data = load_dataset()
    out = []
    for q in data:
        qid = q["question_id"]
        if only_hand and qid not in HAND:
            continue
        if not is_aggregation_question(q["question"])[0]:
            continue
        st, _ = build_store_offline(q)
        m = merge_chains(st)
        view, raw, mode = v4_context(q, st)
        ctx = (view + "\n" + raw).lower()
        kept, _meta = aggregation_printed_groups(m, q["question"])
        n = new = 0
        newvals = []
        for g in kept:
            for score, c, attr, nt in g["items"]:
                n += 1
                v = str(c["value"]).strip().lower()
                if v and v not in ctx:
                    new += 1
                    newvals.append((g["term"], attr, c["value"]))
        row = {"qid": qid, "n_items": n, "n_new": new,
               "frac_new": round(new / n, 3) if n else None,
               "gold": str(q["answer"])[:60]}
        out.append(row)
        if only_hand:
            print(json.dumps(row, ensure_ascii=False))
            for t, a, v in newvals[:8]:
                print("      NEW [%s] %s = %s" % (t, a, v))
    with open("/root/agent_memory_iclr/results/_adv_newinfo%s.json"
              % ("_hand" if only_hand else ""), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    tot_n = sum(r["n_items"] for r in out)
    tot_new = sum(r["n_new"] for r in out)
    print("questions", len(out), "printed items", tot_n, "not already in v4 ctx", tot_new,
          "=", round(tot_new / max(1, tot_n), 3))
    zero = sum(1 for r in out if r["n_new"] == 0)
    print("questions where the block adds ZERO verbatim-new item:", zero, "/", len(out))


if __name__ == "__main__":
    main()
