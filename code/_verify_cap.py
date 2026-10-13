"""ADVERSARIAL VERIFIER step 3: what the model ACTUALLY sees.
run_qc.py caps the rendered view at VIEW_CHAR_CAP=20000 chars on line boundaries, with the
query classifier choosing current/historical. Recompute gold-evidence coverage of the CAPPED
view, before vs after merge_chains. Also cross-check coverage_diag.json n_chains."""
import json
import pickle
import statistics as st
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains, render_view, _norm
from v5_chains import merge_chains
from query_classifier import classify_query

CAP = 20000
MINE = pickle.load(open("/root/agent_memory_iclr/results/_verify_facts.pkl", "rb"))

cov = {r["qid"]: r for r in json.load(open("/root/agent_memory_iclr/results/coverage_diag.json"))}


def truncate(txt, cap=CAP):
    if len(txt) <= cap:
        return txt, False
    kept, total = [], 0
    for line in txt.splitlines():
        if total + len(line) + 1 > cap:
            break
        kept.append(line)
        total += len(line) + 1
    return "\n".join(kept), True


rows = []
nc_match = 0
for x in MINE:
    store = assemble_chains(x["facts"])
    merged = merge_chains(store)
    label, _ = classify_query(x["question"])
    mode = "historical" if label == "historical" else "current"

    vb_full = render_view(store, mode)
    va_full = render_view(merged, mode)
    vb, tb = truncate(vb_full)
    va, ta = truncate(va_full)

    # gold-evidence fact values for this question
    gs = set(x["gold_sidx"])
    gvals = sorted({f["value"] for f in x["facts"] if f["sidx"] in gs})
    def hit(view, vals):
        vl = view.lower()
        return sum(1 for v in vals if v.lower() in vl)
    hb, ha = hit(vb, gvals), hit(va, gvals)
    hbf, haf = hit(vb_full, gvals), hit(va_full, gvals)
    # does the gold ANSWER string survive?
    ansl = str(x["answer"]).lower().strip()
    ab = ansl in vb.lower()
    aa = ansl in va.lower()

    if cov.get(x["qid"], {}).get("n_chains") == len(store):
        nc_match += 1
    rows.append(dict(qid=x["qid"], type=x["type"], mode=mode,
                     cb=len(store), ca=len(merged),
                     lb=len(vb_full), la=len(va_full),
                     tb=tb, ta=ta,
                     capb=len(vb), capa=len(va),
                     lines_b=vb.count("\n") + 1, lines_a=va.count("\n") + 1,
                     g=len(gvals), hb=hb, ha=ha, hbf=hbf, haf=haf, ab=ab, aa=aa))

json.dump(rows, open("/root/agent_memory_iclr/results/_verify_cap.json", "w"))

print(f"coverage_diag.json n_chains == my recomputed v4 chain count for {nc_match}/500 questions")
cn = [cov[r['qid']]['n_chains'] for r in rows]
print(f"  coverage_diag n_chains median {st.median(cn)}  my chains_before median {st.median([r['cb'] for r in rows])}")

print(f"\nview truncated by the 20k cap: BEFORE {sum(r['tb'] for r in rows)}/500  "
      f"AFTER {sum(r['ta'] for r in rows)}/500")
print(f"capped view chars: before median {st.median([r['capb'] for r in rows]):.0f} "
      f"after {st.median([r['capa'] for r in rows]):.0f}  (budget is matched by construction)")
print(f"lines inside the cap: before median {st.median([r['lines_b'] for r in rows]):.0f} "
      f"after {st.median([r['lines_a'] for r in rows]):.0f}")
print(f"mode split: current {sum(1 for r in rows if r['mode']=='current')} "
      f"historical {sum(1 for r in rows if r['mode']=='historical')}")

G = sum(r["g"] for r in rows)
print(f"\n--- gold-evidence VALUE coverage (n gold values {G}) ---")
print(f"UNCAPPED view: before {sum(r['hbf'] for r in rows)} after {sum(r['haf'] for r in rows)} "
      f"({(sum(r['haf'] for r in rows)-sum(r['hbf'] for r in rows))/sum(r['hbf'] for r in rows):+.3%})")
print(f"CAPPED  view: before {sum(r['hb'] for r in rows)} after {sum(r['ha'] for r in rows)} "
      f"({(sum(r['ha'] for r in rows)-sum(r['hb'] for r in rows))/sum(r['hb'] for r in rows):+.3%})")
w = sum(1 for r in rows if r["ha"] < r["hb"]); b = sum(1 for r in rows if r["ha"] > r["hb"])
print(f"  capped: questions losing gold values {w}, gaining {b}, unchanged {500-w-b}")
print(f"gold ANSWER string in capped view: before {sum(r['ab'] for r in rows)} after {sum(r['aa'] for r in rows)}")

# per-question gold recall rate
rb = st.mean([r["hb"] / r["g"] for r in rows if r["g"]])
ra = st.mean([r["ha"] / r["g"] for r in rows if r["g"]])
print(f"mean per-question gold-value recall in capped view: {rb:.4f} -> {ra:.4f} ({ra-rb:+.4f})")
rbf = st.mean([r["hbf"] / r["g"] for r in rows if r["g"]])
raf = st.mean([r["haf"] / r["g"] for r in rows if r["g"]])
print(f"mean per-question gold-value recall in UNCAPPED view: {rbf:.4f} -> {raf:.4f} ({raf-rbf:+.4f})")
