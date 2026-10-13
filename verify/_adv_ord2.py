"""Clean polarity-aware re-scoring of the binary ordering subfamily + baselines."""
import json, re, sys
sys.path.insert(0, "/root/agent_memory_iclr/code")
from v5_temporal_measure import contains

o = json.load(open("/root/agent_memory_iclr/verify/adv_ord.json"))
LATEST = re.compile(r"most recent|recently|latest|last\b", re.I)
rows = []
for r in o:
    if r["verdict"] is None:
        continue
    if len(r["gseq"]) >= 3:          # genuine list question
        rows.append(dict(r, kind="list", gold_idx=None, polarity=None, clean=r["verdict"]))
        continue
    cands, gold0 = r["cands"], r["gseq"][0]
    m = [i for i, c in enumerate(cands) if contains(c, gold0) or contains(gold0, c)]
    pol = "latest" if LATEST.search(r["question"].split("?")[0]) else "first"
    if len(m) != 1:
        rows.append(dict(r, kind="binary", gold_idx=None, polarity=pol, clean=None,
                         note="gold maps to %d candidates" % len(m)))
        continue
    gi = m[0]
    ds = [r["dates"][c] for c in cands]
    if len(set(ds)) < len(ds):
        clean = False; note = "tie (no order information)"
    else:
        want = min(range(len(ds)), key=lambda i: ds[i]) if pol == "first" else \
               max(range(len(ds)), key=lambda i: ds[i])
        clean = (want == gi); note = ""
    # position-only baselines under the same polarity-aware scoring
    rows.append(dict(r, kind="binary", gold_idx=gi, polarity=pol, clean=clean, note=note,
                     bl_first=(gi == 0), bl_last=(gi == len(cands) - 1)))
json.dump(rows, open("/root/agent_memory_iclr/verify/adv_ord2.json", "w"), ensure_ascii=False, indent=1)

b = [x for x in rows if x["kind"] == "binary" and x["clean"] is not None]
print("binary cleanly scorable:", len(b))
print("  reported-metric verdict : %d/%d = %.3f" % (sum(x["verdict"] for x in b), len(b),
                                                    sum(x["verdict"] for x in b)/len(b)))
print("  polarity-clean verdict  : %d/%d = %.3f" % (sum(x["clean"] for x in b), len(b),
                                                    sum(x["clean"] for x in b)/len(b)))
print("  baseline always-1st-mentioned: %d/%d = %.3f" % (sum(x["bl_first"] for x in b), len(b),
                                                         sum(x["bl_first"] for x in b)/len(b)))
print("  baseline always-2nd-mentioned: %d/%d = %.3f" % (sum(x["bl_last"] for x in b), len(b),
                                                         sum(x["bl_last"] for x in b)/len(b)))
print("  ties:", sum(1 for x in b if x["note"].startswith("tie")))
print("  polarity=latest cases:", sum(1 for x in b if x["polarity"] == "latest"),
      "| disagreements between reported and clean:",
      [(x["qid"], x["verdict"], x["clean"], x["polarity"], x["note"]) for x in b if x["verdict"] != x["clean"]])
print("  unscorable (gold->0 or 2 candidates):",
      [(x["qid"], x["note"]) for x in rows if x["kind"] == "binary" and x["clean"] is None])
# McNemar scaffold vs best position baseline
import math
for name in ("bl_last", "bl_first"):
    n01 = sum(1 for x in b if x["clean"] and not x[name])
    n10 = sum(1 for x in b if not x["clean"] and x[name])
    n = n01 + n10
    p = sum(math.comb(n, k) for k in range(min(n01, n10) + 1)) / 2 ** n * 2 if n else 1.0
    print(f"  McNemar vs {name}: scaffold-only-right={n01} baseline-only-right={n10} two-sided p={min(1.0,p):.4f}")
