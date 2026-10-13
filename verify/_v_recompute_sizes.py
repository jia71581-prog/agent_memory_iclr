"""Independent recompute of v5_temporal scaffold size + family detection over all 500."""
import json
import os
import statistics as st
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from v5_temporal import render_timeline, detect_family
from v5_temporal_val import build_store_cached

data = load_dataset("s_cleaned")
print("n questions:", len(data))
rows = []
for i, q in enumerate(data):
    store, miss = build_store_cached(q)
    fam, unit = detect_family(q["question"])
    txt = render_timeline(q, store, q["question"])
    rows.append({"qid": q["question_id"], "type": q["question_type"], "fam": fam,
                 "unit": unit, "chars": len(txt), "miss": miss,
                 "lines": txt.count("\n") + 1})
    if (i + 1) % 100 == 0:
        print("done", i + 1, flush=True)

json.dump(rows, open("/root/agent_memory_iclr/verify/_v_sizes.json", "w"))
c = [r["chars"] for r in rows]
c.sort()
print("ALL 500 chars: median %.0f mean %.1f p90 %.0f max %d min %d" % (
    st.median(c), st.mean(c), c[int(0.9 * len(c))], max(c), min(c)))
print("p90 alt (numpy-linear):", st.quantiles(c, n=10)[8])
fams = {}
for r in rows:
    fams[r["fam"]] = fams.get(r["fam"], 0) + 1
print("families:", sorted(fams.items(), key=lambda x: -x[1]))
print("other:", fams.get("other", 0), " non-other:", 500 - fams.get("other", 0))
print("cache-miss sessions total:", sum(r["miss"] for r in rows),
      " questions with any miss:", sum(1 for r in rows if r["miss"]))

# per type medians
byt = {}
for r in rows:
    byt.setdefault(r["type"], []).append(r["chars"])
for t, v in sorted(byt.items()):
    print("%-32s n=%3d median %.0f mean %.0f" % (t, len(v), st.median(v), st.mean(v)))

# TR vs non-TR detection
tr = [r for r in rows if r["type"].startswith("temporal")]
ntr = [r for r in rows if not r["type"].startswith("temporal")]
print("types present:", sorted(set(r["type"] for r in rows)))
print("TR n=%d fires=%d ; nonTR n=%d fires=%d" % (
    len(tr), sum(1 for r in tr if r["fam"] != "other"),
    len(ntr), sum(1 for r in ntr if r["fam"] != "other")))

gated = [r["chars"] for r in rows if r["fam"] != "other"]
print("gated n=%d mean chars %.1f sum %d amortized over 500 = %.1f" % (
    len(gated), st.mean(gated), sum(gated), sum(gated) / 500.0))
print("ungated sum chars %d mean %.1f" % (sum(c), st.mean(c)))
