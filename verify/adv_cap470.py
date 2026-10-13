"""ADVERSARIAL recompute #5: does 'gold content lost to the 20k cap' predict judge
correctness on the 470 answerable questions? And is the relevance-ordered fix budget-matched?"""
import json, pickle, sys, re, statistics as st
sys.path.insert(0, "/root/agent_memory_iclr/code")
import lifecycle as LC
from run_baseline import bm25_rank

data = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
views = pickle.load(open("/root/agent_memory_iclr/verify/_views.pkl", "rb"))
qc = {r["qid"]: r for r in (json.loads(l) for l in open("/root/agent_memory_iclr/results/full500_qc.jsonl"))}

def nrm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def contains(hay, needle):                       # v5_guards.v_cap's own rule
    h, n = nrm(hay), nrm(needle)
    if not n:
        return False
    if n in h:
        return True
    toks = [t for t in n.split() if len(t) > 2 and t not in ("the", "and", "for", "was", "were", "you", "your")]
    return bool(toks) and all(t in h for t in toks)

def relcap(full, question, budget=20000):        # v5_guards.relevance_capped_view
    lines = full.splitlines()
    if sum(len(l) + 1 for l in lines) <= budget:
        return "\n".join(lines)
    order = bm25_rank(question, lines)
    keep, total = [], 0
    for i in order:
        if total + len(lines[i]) + 1 > budget:
            continue
        keep.append(i); total += len(lines[i]) + 1
    return "\n".join(lines[i] for i in sorted(keep))

rows = []
for q in data:
    qid = q["question_id"]
    if qid.endswith("_abs"):
        continue
    V = views[qid]
    rel = relcap(V["full"], q["question"])
    g = q["answer"]
    rows.append(dict(qid=qid, type=q["question_type"], judge=qc[qid]["judge"],
                     cf=contains(V["full"], g), ca=contains(V["cap"], g), cr=contains(rel, g),
                     ch_a=len(V["cap"]), ch_r=len(rel),
                     ln_a=len(V["cap"].splitlines()), ln_r=len(rel.splitlines())))
json.dump(rows, open("/root/agent_memory_iclr/verify/_cap470.json", "w"))

n = len(rows)
print(f"n answerable = {n}")
print(f"gold-string coverage: full {sum(r['cf'] for r in rows)/n:.3f}  alpha-cap {sum(r['ca'] for r in rows)/n:.3f} "
      f"rel-cap {sum(r['cr'] for r in rows)/n:.3f}")
lost = [r for r in rows if r["cf"] and not r["ca"]]
rec = [r for r in lost if r["cr"]]
reg = [r for r in rows if r["cf"] and r["ca"] and not r["cr"]]
print(f"gold line lost to the alphabetical cap: {len(lost)}; recovered by relevance: {len(rec)}; regressions: {len(reg)}")

print("\n=== does cap loss predict a wrong answer? (qc arm) ===")
kept = [r for r in rows if r["cf"] and r["ca"]]
print(f"  gold in full AND survives cap : n={len(kept):3d} judge_acc {st.mean([r['judge'] for r in kept]):.3f}")
print(f"  gold in full, DELETED by cap  : n={len(lost):3d} judge_acc {st.mean([r['judge'] for r in lost]):.3f}")
never = [r for r in rows if not r["cf"]]
print(f"  gold not in the full view     : n={len(never):3d} judge_acc {st.mean([r['judge'] for r in never]):.3f}")
print(f"  recovered-by-relevance subset : n={len(rec):3d} judge_acc {st.mean([r['judge'] for r in rec]):.3f} "
      f"(wrong now: {sum(1 for r in rec if not r['judge'])})")
print(f"  regressed-by-relevance subset : n={len(reg):3d} judge_acc "
      f"{st.mean([r['judge'] for r in reg]) if reg else float('nan'):.3f}")

# 2x2 for cap-loss vs correctness among questions whose gold IS in the full view
a = sum(1 for r in kept if r["judge"]); b = len(kept) - a
c = sum(1 for r in lost if r["judge"]); d = len(lost) - c
print(f"\n  2x2 (gold in full view only): survived {a}/{a+b} correct, cap-deleted {c}/{c+d} correct")
try:
    from scipy.stats import fisher_exact
    print("  fisher p =", f"{fisher_exact([[a, b], [c, d]])[1]:.3f}")
except Exception as e:
    # exact test by hand
    from math import comb
    N, K, nn, k = a + b + c + d, a + c, a + b, a
    p = sum(comb(K, i) * comb(N - K, nn - i) for i in range(0, min(K, nn) + 1)
            if comb(K, i) * comb(N - K, nn - i) <= comb(K, k) * comb(N - K, nn - k)) / comb(N, nn)
    print(f"  fisher p (hand) = {p:.3f}")

print("\n=== budget parity of the relevance-ordered view ===")
print(f"  alpha-cap chars median {st.median([r['ch_a'] for r in rows]):.0f}  max {max(r['ch_a'] for r in rows)}")
print(f"  rel-cap   chars median {st.median([r['ch_r'] for r in rows]):.0f}  max {max(r['ch_r'] for r in rows)}")
print(f"  alpha-cap lines median {st.median([r['ln_a'] for r in rows]):.0f}; "
      f"rel-cap lines median {st.median([r['ln_r'] for r in rows]):.0f}")
print(f"  rel-cap keeps MORE chars than alpha on {sum(1 for r in rows if r['ch_r'] > r['ch_a'])}/{n} questions "
      f"(mean delta {st.mean([r['ch_r'] - r['ch_a'] for r in rows]):+.0f} chars, "
      f"{st.mean([r['ln_r'] - r['ln_a'] for r in rows]):+.1f} lines)")
