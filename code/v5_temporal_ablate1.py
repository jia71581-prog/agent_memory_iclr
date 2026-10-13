"""打分权重消融(零 API):在 TR 133 题上比较 cue/anchor/intent 三个软权重的组合。"""
import json, os, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_temporal as V
from common import load_dataset
from v5_temporal_measure import analyse
from v5_temporal_val import build_store_cached

CONF = [("plain IDF (1,1,1)", 1.0, 1.0, 1.0),
        ("cue only (1.3,1,1)", 1.3, 1.0, 1.0),
        ("cue+anchor (1.3,1.5,1)", 1.3, 1.5, 1.0),
        ("cue+anchor+intent (1.3,1.5,0.4)", 1.3, 1.5, 0.4),
        ("anchor heavy (1.3,2.0,0.4)", 1.3, 2.0, 0.4)]

data = [q for q in load_dataset("s_cleaned") if q["question_type"] == "temporal-reasoning"]
stores = {}
for q in data:
    stores[q["question_id"]] = build_store_cached(q)[0]
print("stores built", len(stores), flush=True)
out = {}
for name, c, a, i in CONF:
    V.W_CUE, V.W_ANCHOR, V.W_INTENT = c, a, i
    rows = [analyse(q, stores[q["question_id"]])[0] for q in data]
    dg = [r for r in rows if r["family"] in ("delta_between", "delta_since") and r["gold_vals"]]
    o = [r for r in rows if r["family"] == "ordering"]
    oi = [r for r in o if r.get("ordering_item_correct") is not None]
    al = [r for r in rows if r["family"] == "anchor_lookup"]
    m = {
        "deltaTL": sum(r["gold_derivable_from_timeline"] for r in dg) / len(dg),
        "ord_computable": len(oi) / len(o),
        "ord_correct": (sum(r["ordering_item_correct"] for r in oi) / len(oi)) if oi else 0,
        "ord_correct_of_all": (sum(bool(r.get("ordering_item_correct")) for r in o) / len(o)),
        "anchor_hit": sum(bool(r.get("anchor_hit")) for r in al) / len(al),
        "datesPres": st.mean([r["dates_present"] for r in rows]),
        "chars": st.median([r["chars"] for r in rows]),
    }
    out[name] = m
    print(f"{name:34s} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)
json.dump(out, open("/root/agent_memory_iclr/results/v5_ablate_weights.json", "w"), indent=1)
