"""组件消融(零 API):问句形式词剥离 / 谓词扫描 / 句内相对表达解析 三者各自的贡献。"""
import json, os, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_temporal as V
from common import load_dataset
from v5_temporal_measure import analyse
from v5_temporal_val import build_store_cached

data = [q for q in load_dataset("s_cleaned") if q["question_type"] == "temporal-reasoning"]
stores = {q["question_id"]: build_store_cached(q)[0] for q in data}
print("stores built", flush=True)

_orig_best = V.best_event_date
def no_intext(text, session_dt):
    d = V.parse_date(session_dt)
    return (V.day0(d), "") if d else (None, "")

CONF = [("final (stem+intext)", False, False, True),
        ("no stemming", False, False, True),
        ("no in-text resolution", False, False, False),
        ("no stem & no in-text", False, False, False)]
STEM = [True, False, True, False]
out = {}
for (name, qf, sw, it), stem in zip(CONF, STEM):
    V.USE_QFORM_STOP, V.USE_SWEEP, V.USE_STEM = qf, sw, stem
    V.best_event_date = _orig_best if it else no_intext
    rows = [analyse(q, stores[q["question_id"]])[0] for q in data]
    dg = [r for r in rows if r["family"] in ("delta_between", "delta_since") and r["gold_vals"]]
    o = [r for r in rows if r["family"] == "ordering"]
    oi = [r for r in o if r.get("ordering_item_correct") is not None]
    al = [r for r in rows if r["family"] == "anchor_lookup"]
    m = {"deltaTL": sum(r["gold_derivable_from_timeline"] for r in dg) / len(dg),
         "deltaLiteral": sum(r["gold_literal_in_arith"] for r in dg) / len(dg),
         "ord_computable": len(oi) / len(o),
         "ord_correct_of_all": sum(bool(r.get("ordering_item_correct")) for r in o) / len(o),
         "same_day_split": sum(bool(r.get("same_day_split")) for r in o),
         "anchor_hit": sum(bool(r.get("anchor_hit")) for r in al) / len(al),
         "datesPres": st.mean([r["dates_present"] for r in rows]),
         "chars": st.median([r["chars"] for r in rows])}
    out[name] = m
    print(f"{name:30s} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)
V.best_event_date = _orig_best
json.dump(out, open("/root/agent_memory_iclr/results/v5_ablate_components3.json", "w"), indent=1)
