"""Side effect of part (a): does merge_chains DELETE facts from the rendered 'current' view?
After a merge, only the last session group of the pooled chain is printed, so values that were
visible under a separate attribute spelling can vanish. Offline only."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from lifecycle import render_view
from query_classifier import classify_query
from _adv_build import build_store_offline
from v5_chains import merge_chains

STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def vals(store, mode):
    out = set()
    for k in store:
        chain = store[k]
        for c in chain:
            out.add(str(c["value"]).strip().lower())
    return out


def main():
    data = load_dataset()
    qc = {}
    with open("/root/agent_memory_iclr/results/full500_qc.jsonl") as f:
        for line in f:
            r = json.loads(line)
            qc[r["qid"]] = r
    tot_lost = tot_v4 = 0
    rows = []
    for i, q in enumerate(data):
        if i % STEP:
            continue
        st, _ = build_store_offline(q)
        m = merge_chains(st)
        label, _l = classify_query(q["question"])
        mode = "historical" if label == "historical" else "current"
        v4 = set(l.strip() for l in render_view(st, mode).splitlines())
        v5 = set(l.strip() for l in render_view(m, mode).splitlines())
        # value-level: which values survive in the rendered text?
        t4 = render_view(st, mode).lower()
        t5 = render_view(m, mode).lower()
        lost = [v for v in vals(st, mode) if v and v in t4 and v not in t5]
        tot_lost += len(lost)
        tot_v4 += len(st)
        rows.append({"qid": q["question_id"], "type": q["question_type"], "mode": mode,
                     "chains_v4": len(st), "chains_v5": len(m), "lost_values": len(lost),
                     "v4_judge": qc.get(q["question_id"], {}).get("judge"),
                     "sample_lost": lost[:5]})
    with open("/root/agent_memory_iclr/results/_adv_mergeloss.json", "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    import statistics as st_
    print("questions sampled", len(rows))
    print("median chains v4 -> v5: %d -> %d" % (
        st_.median([r["chains_v4"] for r in rows]), st_.median([r["chains_v5"] for r in rows])))
    print("median values lost from the rendered view per question:",
          st_.median([r["lost_values"] for r in rows]),
          " mean %.1f" % st_.mean([r["lost_values"] for r in rows]))
    print("questions with >=1 lost value:", sum(1 for r in rows if r["lost_values"]), "/", len(rows))
    ku = [r for r in rows if r["type"] == "knowledge-update"]
    if ku:
        print("knowledge-update subset n=%d, median lost %.1f, with loss %d"
              % (len(ku), st_.median([r["lost_values"] for r in ku]),
                 sum(1 for r in ku if r["lost_values"])))
    for r in sorted(rows, key=lambda r: -r["lost_values"])[:5]:
        print("  worst", r["qid"], r["type"], "lost", r["lost_values"], r["sample_lost"][:3])


if __name__ == "__main__":
    main()
