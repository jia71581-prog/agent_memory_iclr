"""Test 3: (a) A3 provenance-cost numbers; (b) does extraction actually drop assistant content;
(c) does view_only refuse even when the gold string IS in its view (starvation vs discipline)."""
import json, re, sys, collections
sys.path.insert(0, "/root/agent_memory_iclr/code")
from _zver_core import build_ctx, load, norm, MISS
from v5_guards import is_decline

RES = "/root/agent_memory_iclr/results"
D = {q["question_id"]: q for q in load()}
A = {}
for t in ["qc", "qc_view_only", "qc_no_lifecycle"]:
    A[t] = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{RES}/full500_{t}.jsonl")}

qids = list(A["qc"].keys())
ans_ids = [q for q in qids if not q.endswith("_abs")]

out = {}
for i, qid in enumerate(qids):
    q = D[qid]
    c = build_ctx(q)
    g = norm(q["answer"])
    v, r = norm(c["view"]), norm(c["raw"])
    vf = norm(c["view_full"])
    out[qid] = {
        "in_view": g in v, "in_raw": g in r, "in_view_full": g in vf,
        "view_chars": len(c["view"]), "view_full_chars": len(c["view_full"]),
        "raw_chars": len(c["raw"]),
        "view_lines": len(c["view"].splitlines()), "view_full_lines": len(c["view_full"].splitlines()),
        "raw_roles": collections.Counter(u[1] for u in c["raw_units"]),
    }
    if i % 100 == 0:
        print("  ..%d" % i, flush=True)
json.dump({k: {**v, "raw_roles": dict(v["raw_roles"])} for k, v in out.items()},
          open("/root/agent_memory_iclr/results/_zver_contain.json", "w"))
print("cache misses:", len(MISS))

vonly = [q for q in ans_ids if out[q]["in_view"] and not out[q]["in_raw"]]
neither = [q for q in ans_ids if not out[q]["in_view"] and not out[q]["in_raw"]]
both = [q for q in ans_ids if out[q]["in_view"] and out[q]["in_raw"]]
ronly = [q for q in ans_ids if not out[q]["in_view"] and out[q]["in_raw"]]
print("\n=== A3 recompute (answerable n=%d, capped view vs top-20 raw) ===" % len(ans_ids))
print("gold in VIEW only  : %3d   qc correct %3d" % (len(vonly), sum(A["qc"][q]["judge"] for q in vonly)))
print("gold in RAW only   : %3d   qc correct %3d" % (len(ronly), sum(A["qc"][q]["judge"] for q in ronly)))
print("gold in BOTH       : %3d   qc correct %3d" % (len(both), sum(A["qc"][q]["judge"] for q in both)))
print("gold in NEITHER    : %3d   qc correct %3d" % (len(neither), sum(A["qc"][q]["judge"] for q in neither)))

print("\n=== (c) view_only behaviour conditioned on gold-in-view (answerable) ===")
for cond, name in [(lambda q: out[q]["in_view"], "gold IN view"),
                   (lambda q: not out[q]["in_view"], "gold NOT in view")]:
    s = [q for q in ans_ids if cond(q)]
    print("%-16s n=%3d  view_only: decline %3d (%.3f) correct %3d (%.3f) | qc: correct %3d (%.3f)" % (
        name, len(s),
        sum(is_decline(A["qc_view_only"][q]["pred"]) for q in s),
        sum(is_decline(A["qc_view_only"][q]["pred"]) for q in s) / len(s),
        sum(A["qc_view_only"][q]["judge"] for q in s),
        sum(A["qc_view_only"][q]["judge"] for q in s) / len(s),
        sum(A["qc"][q]["judge"] for q in s), sum(A["qc"][q]["judge"] for q in s) / len(s)))

print("\n=== context size (unmatched-budget check) ===")
import statistics
for name, key in [("qc view+raw", None), ("view_only view", "view_chars"), ("no_lifecycle raw", "raw_chars")]:
    if key is None:
        vals = [out[q]["view_chars"] + out[q]["raw_chars"] for q in qids]
    else:
        vals = [out[q][key] for q in qids]
    print("%-18s median=%7.0f mean=%7.0f" % (name, statistics.median(vals), statistics.mean(vals)))
print("cap fired on %d/500 questions; median lines deleted by cap = %.0f (%.1f%% of lines)" % (
    sum(1 for q in qids if out[q]["view_full_chars"] > 20000),
    statistics.median([out[q]["view_full_lines"] - out[q]["view_lines"] for q in qids]),
    100 * statistics.median([(out[q]["view_full_lines"] - out[q]["view_lines"]) / out[q]["view_full_lines"] for q in qids])))
print("gold recovered by uncapped view but lost to cap: %d answerable questions" % (
    sum(1 for q in ans_ids if out[q]["in_view_full"] and not out[q]["in_view"])))
