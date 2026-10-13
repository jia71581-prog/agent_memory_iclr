"""Measurement 3: net context size, v4 vs v5, over all 500 questions (OFFLINE).
v4 context = render_view(assemble_chains(facts), mode)
v5 context = render_view(merge_chains(...), mode) + aggregation block when it fires
The view cap used by run_qc.py (VIEW_CHAR_CAP=20000) is applied to both so the numbers match
what the pipeline would actually send."""
import json
import pickle
import statistics as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifecycle import assemble_chains, render_view
from query_classifier import classify_query
from v5_chains import merge_chains, render_aggregation_view, is_aggregation_question

CAP = 20000
D = pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl", "rb"))


def capped(txt, cap=CAP):
    if len(txt) <= cap:
        return txt
    kept, total = [], 0
    for line in txt.splitlines():
        if total + len(line) + 1 > cap:
            break
        kept.append(line)
        total += len(line) + 1
    return "\n".join(kept)


rows = []
for x in D:
    label, _ = classify_query(x["question"])
    mode = "historical" if label == "historical" else "current"
    s4 = assemble_chains(x["facts"])
    s5 = merge_chains(s4)
    v4 = capped(render_view(s4, mode))
    v5v = capped(render_view(s5, mode))
    blk = render_aggregation_view(s5, x["question"])
    fired = bool(blk)
    rows.append({"qid": x["qid"], "fired": fired, "v4": len(v4), "v5_view": len(v5v),
                 "blk": len(blk), "v5_total": len(v5v) + (len(blk) + 2 if blk else 0),
                 "lines_v4": v4.count("\n") + 1, "lines_v5": v5v.count("\n") + 1})
json.dump(rows, open("/root/agent_memory_iclr/results/_v5_size.json", "w"))

f = lambda k, sub=None: st.median([r[k] for r in (sub or rows)])
fired = [r for r in rows if r["fired"]]
nof = [r for r in rows if not r["fired"]]
print(f"n=500, aggregation block fires on {len(fired)}")
print(f"{'group':30s} {'v4 view':>9s} {'v5 view':>9s} {'block':>7s} {'v5 total':>9s} {'net':>8s}")
for nm, sub in [("ALL 500", rows), (f"block fires ({len(fired)})", fired),
                (f"no block ({len(nof)})", nof)]:
    a, b, c, d = f("v4", sub), f("v5_view", sub), f("blk", sub), f("v5_total", sub)
    print(f"{nm:30s} {a:9.0f} {b:9.0f} {c:7.0f} {d:9.0f} {(d-a)/a:+8.2%}")
print(f"\nmean over 500: v4 {st.mean([r['v4'] for r in rows]):.0f} -> "
      f"v5 total {st.mean([r['v5_total'] for r in rows]):.0f} "
      f"({(st.mean([r['v5_total'] for r in rows])-st.mean([r['v4'] for r in rows]))/st.mean([r['v4'] for r in rows]):+.2%})")
print(f"view alone (merging only): {st.mean([r['v4'] for r in rows]):.0f} -> "
      f"{st.mean([r['v5_view'] for r in rows]):.0f} "
      f"({(st.mean([r['v5_view'] for r in rows])-st.mean([r['v4'] for r in rows]))/st.mean([r['v4'] for r in rows]):+.2%})")
print(f"view lines: v4 median {f('lines_v4'):.0f} -> v5 {f('lines_v5'):.0f}")
print(f"how often the 20000-char cap binds: v4 {sum(r['v4']>=CAP-200 for r in rows)}/500, "
      f"v5 {sum(r['v5_view']>=CAP-200 for r in rows)}/500")
tok = lambda c: c / 4
print(f"\napprox added prompt tokens from the block (chars/4): median "
      f"{tok(f('blk', fired)):.0f} on the {len(fired)} questions where it fires; "
      f"whole-run added prompt {sum(r['blk'] for r in rows)/4:.0f} tokens over 500 questions. "
      f"No extra API call: the block is pure rendering over the existing extraction cache.")
