import hashlib, json, os, re, sys
sys.path.insert(0, "/root/agent_memory_iclr/code")
import common
common.chat = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("blocked"))
import lifecycle; lifecycle.chat = common.chat
from lifecycle import assemble_chains, render_view

CACHE = "/root/agent_memory_iclr/results/lifecycle_cache/sessions_nt"
CAP = 20000
R = "/root/agent_memory_iclr/results/"
qc = {r["qid"]: r for r in (json.loads(l) for l in open(R + "full500_qc.jsonl")) if "judge" in r}
data = common.load_dataset("s_cleaned")
byid = {q["question_id"]: q for q in data}
nrm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def prod_cap(txt):
    """exact copy of run_qc.py semantics (splitlines)"""
    if len(txt) <= CAP:
        return txt
    kept, total = [], 0
    for line in txt.splitlines():
        if total + len(line) + 1 > CAP:
            break
        kept.append(line)
        total += len(line) + 1
    return "\n".join(kept)


n_diff_splitlines = 0
n_prefix_ok = 0
lost_prod = 0
gold_full = gold_cap = 0
mode_prefix_same = 0
for qid, r in qc.items():
    q = byid[qid]
    facts = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        text = "\n".join(f"[{t['role']}] {t['content']}" for t in sess if t.get("content"))[:24000]
        if not text: continue
        p = os.path.join(CACHE, hashlib.md5(text.encode()).hexdigest()[:16] + ".json")
        for ent, attr, val in json.load(open(p)):
            facts.append({"entity": ent, "attribute": attr, "value": val, "date": date, "sidx": si})
    store = assemble_chains(facts)
    keys = sorted(store)
    for mode in ("current", "historical"):
        full = render_view(store, mode)
        if len(full.splitlines()) != len(full.split("\n")):
            n_diff_splitlines += 1
        capped = prod_cap(full)
        klines = capped.split("\n")
        # MECHANISM: is the kept set exactly the alphabetically-first chains?
        expect = render_view({k: store[k] for k in keys[:len(klines)]}, mode).split("\n")
        if klines == expect:
            n_prefix_ok += 1
        if mode == r["view_mode"]:
            g = nrm(q["answer"])
            a = bool(g) and g in nrm(full)
            b = bool(g) and g in nrm(capped)
            gold_full += a; gold_cap += b
            lost_prod += (a and not b)
    # do the two modes keep the SAME chain prefix? (query-conditioning cannot change what survives much)
    kc = len(prod_cap(render_view(store, "current")).split("\n"))
    kh = len(prod_cap(render_view(store, "historical")).split("\n"))
    mode_prefix_same += (kc == kh)

print("views where splitlines() != split('\\n') (hidden line-break chars):", n_diff_splitlines, "/ 1000")
print("view/mode combos where kept set == alphabetically-first chains:", n_prefix_ok, "/ 1000")
print("questions where current and historical keep the same NUMBER of chains:", mode_prefix_same, "/ 500")
print("production-exact cap: gold in uncapped %d, capped %d, destroyed %d" % (gold_full, gold_cap, lost_prod))
