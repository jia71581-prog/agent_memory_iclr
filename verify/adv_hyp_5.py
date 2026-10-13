"""Test 5: (a) more permutation seeds; (b) junk-table control (same shape, nonsense words);
(c) subset-definition sensitivity (gold-count cutoff); (d) sanity checks on the claim's
structural assertions (21 categories, use_hyponyms=False really disables)."""
import json, pickle, sys, random, statistics as st, re
sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains
import v5_chains as V

D = {x["qid"]: x for x in pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl","rb"))}
G = pickle.load(open("/root/agent_memory_iclr/results/_v5_gold.pkl","rb"))
base = json.load(open("/root/agent_memory_iclr/verify/adv_hyp_rows.json"))
FULL = dict(V.HYPONYMS)
cats = list(FULL)
print("(d) HYPONYMS categories:", len(FULL), "| module-level constant:",
      isinstance(V.HYPONYMS, dict))
src = open("/root/agent_omitted", "w") if False else None
# does use_hyponyms=False really bypass the table? mutate the table and check invariance
V.HYPONYMS = {"clothing": ["zzzqqq"]}
f_off_a, _, _ = V.question_focus("How many items of clothing do I need?", False)
V.HYPONYMS = FULL
f_off_b, _, _ = V.question_focus("How many items of clothing do I need?", False)
f_on, _, _ = V.question_focus("How many items of clothing do I need?", True)
print("(d) use_hyponyms=False invariant to table content:", f_off_a == f_off_b,
      "| True expands focus:", len(f_on) > len(f_off_b))

stores, meta = {}, {}
for r in base:
    x = D[r["qid"]]
    stores[r["qid"]] = V.merge_chains(assemble_chains(x["facts"]))
    meta[r["qid"]] = (x["question"], set(G[r["qid"]]["gold_sidx"]), r["gnum"])

def measure(tbl, uh=True):
    V.HYPONYMS = tbl
    ok1 = ok2 = rec = tot = items = 0
    for qid, stt in stores.items():
        q, gs, gnum = meta[qid]
        kept, _ = V.aggregation_printed_groups(stt, q, uh)
        pr = [[it[1] for it in g["items"]] for g in kept]
        allit = [c for gi in pr for c in gi]
        gh = len({c.get("sidx") for c in allit} & gs)
        cov = gh == len(gs)
        ok1 += (len(allit) >= gnum) and cov
        ok2 += (max((len(gi) for gi in pr), default=0) >= gnum) and cov
        rec += gh; tot += len(gs); items += len(allit)
    V.HYPONYMS = FULL
    n = len(stores)
    return ok1, ok2, rec/tot, items

print("\n=== (a) permuted-table control, 8 seeds (semantics destroyed, shape preserved) ===")
res = []
for s in range(1, 9):
    rnd = random.Random(s); tgt = cats[:]
    while True:
        rnd.shuffle(tgt)
        if all(a != b for a, b in zip(cats, tgt)): break
    p = {c: FULL[t] for c, t in zip(cats, tgt)}
    r = measure(p); res.append(r)
    print(f"  seed {s}: D1 {r[0]:2d}/60  D2 {r[1]:2d}/60  recall {r[2]:.3f}  items {r[3]}")
print(f"  mean D1 {st.mean(x[0] for x in res):.2f}  D2 {st.mean(x[1] for x in res):.2f}  "
      f"recall {st.mean(x[2] for x in res):.4f}")
r5, rn = measure(FULL), measure(FULL, False)
print(f"  real table  : D1 {r5[0]}/60 D2 {r5[1]}/60 recall {r5[2]:.3f} items {r5[3]}")
print(f"  no table    : D1 {rn[0]}/60 D2 {rn[1]}/60 recall {rn[2]:.3f} items {rn[3]}")

print("\n=== (b) junk-table control (nonsense strings, same 21 keys & list lengths) ===")
alpha = "abcdefghijklmnopqrstuvwxyz"
for s in (11, 12):
    rnd = random.Random(s)
    junk = {c: ["".join(rnd.choice(alpha) for _ in range(rnd.randint(5, 9)))
                for _ in FULL[c]] for c in cats}
    r = measure(junk)
    print(f"  seed {s}: D1 {r[0]:2d}/60  D2 {r[1]:2d}/60  recall {r[2]:.3f}  items {r[3]}")

print("\n=== (c) subset-definition sensitivity: gold-count cutoff ===")
allrows = json.load(open("/root/agent_memory_iclr/results/_v5_agg2.json"))
for cut in (6, 8, 10, 12, 20, 999):
    sel = [r for r in allrows if r["gold_num"] is not None and r["enum"]
           or (r["gold_num"] is not None and re.search(r"\bhow many\b|\bnumber of\b",
               r["question"].lower()) and r["gold_num"] == int(r["gold_num"])
               and 1 <= r["gold_num"] <= cut)]
    sel = [r for r in sel if r["gold_num"] <= cut]
    if not sel: continue
    def dd(r, t, strict):
        k = "best_group" if strict else "items"
        return r[f"{t}_{k}"] >= r["gold_num"] and r[f"{t}_gold_hit"] == r["n_gold_sess"]
    n = len(sel)
    a1 = sum(dd(r,"v5",0) for r in sel); b1 = sum(dd(r,"nohyp",0) for r in sel)
    a2 = sum(dd(r,"v5",1) for r in sel); b2 = sum(dd(r,"nohyp",1) for r in sel)
    print(f"  gold<={cut:3d}  n={n:3d}  D1 {a1/n:.3f} vs {b1/n:.3f} (d={(a1-b1)/n:+.3f})  "
          f"D2 {a2/n:.3f} vs {b2/n:.3f} (d={(a2-b2)/n:+.3f})")
