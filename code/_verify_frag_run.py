"""ADVERSARIAL VERIFIER step 2: recompute all reported fragmentation numbers from the
INDEPENDENTLY rebuilt facts, plus regression probes the design agent did not report."""
import json
import pickle
import statistics as st
import sys
from collections import Counter

sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains, render_view, _norm
from v5_chains import merge_chains, cluster_keys, _canon_tokens, _head, _ent_canon

MINE = pickle.load(open("/root/agent_memory_iclr/results/_verify_facts.pkl", "rb"))
THEIRS = pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl", "rb"))

# ---- provenance: does their pickle match an independent rebuild?
tb = {x["qid"]: x for x in THEIRS}
diff_q, diff_f = 0, 0
for x in MINE:
    y = tb.get(x["qid"])
    if y is None:
        diff_q += 1
        continue
    a = [(f["entity"], f["attribute"], f["value"], f["date"], f["sidx"]) for f in x["facts"]]
    b = [(f["entity"], f["attribute"], f["value"], f["date"], f["sidx"]) for f in y["facts"]]
    if a != b:
        diff_f += 1
print(f"PROVENANCE: qids missing from theirs {diff_q}; questions whose fact list differs {diff_f}")
print(f"  my total facts {sum(len(x['facts']) for x in MINE)}  their total {sum(len(x['facts']) for x in THEIRS)}")


def p90(v):
    s = sorted(v)
    return s[min(len(s) - 1, int(0.9 * len(s)))]


def visible_facts(store, mode="current"):
    """How many stored facts actually appear in the rendered view.
    accumulative chains print every value; other chains print only the LAST session group."""
    n = 0
    for chain in store.values():
        if chain[0]["relation"] == "accumulative":
            n += len(chain)
        else:
            gk = (chain[-1]["date"], chain[-1].get("sidx"))
            n += sum(1 for c in chain if (c["date"], c.get("sidx")) == gk)
    return n


def visible_values(store):
    out = set()
    for chain in store.values():
        if chain[0]["relation"] == "accumulative":
            for c in chain:
                out.add(_norm(c["value"]))
        else:
            gk = (chain[-1]["date"], chain[-1].get("sidx"))
            for c in chain:
                if (c["date"], c.get("sidx")) == gk:
                    out.add(_norm(c["value"]))
    return out


rows = []
edge_rules = Counter()
multi_clusters = 0
for x in MINE:
    store = assemble_chains(x["facts"])
    clusters, edges = cluster_keys(store)
    merged = merge_chains(store)
    for _, _, r in edges:
        edge_rules[r] += 1
    multi_clusters += sum(1 for c in clusters if len(c) > 1)

    # head-merge ceiling, computed with my own bucketing (same head fn - it is the claim's own def)
    heads = Counter()
    for k, ch in store.items():
        heads[(_ent_canon(ch[0]["entity"]), _head(_canon_tokens(ch[0]["attribute"])))] += 1
    n_ceil = len(heads)
    uniq_head = sum(1 for v in heads.values() if v == 1) / len(store)

    # regression probes
    vb, va = visible_facts(store), visible_facts(merged)
    sb, sa = visible_values(store), visible_values(merged)
    acc_b = sum(1 for c in store.values() if c[0]["relation"] == "accumulative")
    acc_a = sum(1 for c in merged.values() if c[0]["relation"] == "accumulative")
    accf_b = sum(len(c) for c in store.values() if c[0]["relation"] == "accumulative")
    accf_a = sum(len(c) for c in merged.values() if c[0]["relation"] == "accumulative")

    # gold-evidence facts visibility
    gs = set(x["gold_sidx"])
    gold_vis_b = gold_vis_a = gold_tot = 0
    if gs:
        gold_vals = {_norm(f["value"]) for f in x["facts"] if f["sidx"] in gs}
        gold_tot = len(gold_vals)
        gold_vis_b = len(gold_vals & sb)
        gold_vis_a = len(gold_vals & sa)

    rows.append(dict(qid=x["qid"], type=x["type"],
                     cb=len(store), ca=len(merged),
                     singb=sum(1 for c in store.values() if len(c) == 1),
                     singa=sum(1 for c in merged.values() if len(c) == 1),
                     maxb=max((len(c) for c in store.values()), default=0),
                     maxa=max((len(c) for c in merged.values()), default=0),
                     fb=sum(len(c) for c in store.values()),
                     fa=sum(len(c) for c in merged.values()),
                     edges=len(edges), ceil=n_ceil, uniq=uniq_head,
                     vcb=len(render_view(store, "current")), vca=len(render_view(merged, "current")),
                     vhb=len(render_view(store, "historical")), vha=len(render_view(merged, "historical")),
                     visb=vb, visa=va, uvb=len(sb), uva=len(sa),
                     accb=acc_b, acca=acc_a, accfb=accf_b, accfa=accf_a,
                     gtot=gold_tot, gvb=gold_vis_b, gva=gold_vis_a))

json.dump(rows, open("/root/agent_memory_iclr/results/_verify_frag.json", "w"))


def s(name, key, fmt="8.1f"):
    v = [r[key] for r in rows]
    print(f"{name:28s} median {st.median(v):{fmt}}  mean {st.mean(v):{fmt}}  p90 {p90(v):{fmt}}  "
          f"min {min(v)}  max {max(v)}")


print(f"\nn questions {len(rows)}")
for nm, k in [("chains BEFORE", "cb"), ("chains AFTER", "ca"),
              ("singletons BEFORE", "singb"), ("singletons AFTER", "singa"),
              ("maxchain BEFORE", "maxb"), ("maxchain AFTER", "maxa"),
              ("merge edges", "edges"), ("head-ceiling chains", "ceil"),
              ("view_cur chars BEFORE", "vcb"), ("view_cur chars AFTER", "vca"),
              ("view_hist chars BEFORE", "vhb"), ("view_hist chars AFTER", "vha")]:
    s(nm, k)
red = [1 - r["ca"] / r["cb"] for r in rows]
print(f"reduction: median {st.median(red):.4f} mean {st.mean(red):.4f} min {min(red):.4f} max {max(red):.4f}")
print(f"singleton share: before {sum(r['singb'] for r in rows)/sum(r['cb'] for r in rows):.4f} "
      f"after {sum(r['singa'] for r in rows)/sum(r['ca'] for r in rows):.4f}")
print(f"edges by rule {dict(edge_rules)}  total {sum(edge_rules.values())}  multi-key clusters {multi_clusters}")
print(f"facts in store: before {sum(r['fb'] for r in rows)} after {sum(r['fa'] for r in rows)}")

cm, bm = st.median([r["ceil"] for r in rows]), st.median([r["cb"] for r in rows])
print(f"\nCEILING ratio-of-medians: {bm} -> {cm} = {1-cm/bm:+.4f}")
per_q = [1 - r["ceil"] / r["cb"] for r in rows]
print(f"CEILING median of per-question reductions: {st.median(per_q):.4f} mean {st.mean(per_q):.4f}")
print(f"unique-head share: median {st.median([r['uniq'] for r in rows]):.4f} "
      f"mean {st.mean([r['uniq'] for r in rows]):.4f}")

print("\n--- REGRESSION PROBES (not reported by the design agent) ---")
for nm, k in [("facts visible in cur BEFORE", "visb"), ("facts visible in cur AFTER", "visa"),
              ("distinct vals vis BEFORE", "uvb"), ("distinct vals vis AFTER", "uva"),
              ("accumulative chains BEFORE", "accb"), ("accumulative chains AFTER", "acca"),
              ("facts in accum chains BEFORE", "accfb"), ("facts in accum chains AFTER", "accfa")]:
    s(nm, k)
tvb, tva = sum(r["visb"] for r in rows), sum(r["visa"] for r in rows)
print(f"TOTAL facts visible in 'current' view: {tvb} -> {tva} ({(tva-tvb)/tvb:+.3%})")
tub, tua = sum(r["uvb"] for r in rows), sum(r["uva"] for r in rows)
print(f"TOTAL distinct visible values:         {tub} -> {tua} ({(tua-tub)/tub:+.3%})")
worse = sum(1 for r in rows if r["uva"] < r["uvb"])
print(f"questions where merging REMOVES distinct visible values: {worse}/{len(rows)}")
gq = [r for r in rows if r["gtot"]]
if gq:
    gb, ga = sum(r["gvb"] for r in gq), sum(r["gva"] for r in gq)
    print(f"gold-evidence values visible (n={len(gq)} q): {gb} -> {ga} ({(ga-gb)/max(1,gb):+.3%})")
    print(f"  questions losing >=1 gold value: {sum(1 for r in gq if r['gva'] < r['gvb'])}; "
          f"gaining: {sum(1 for r in gq if r['gva'] > r['gvb'])}")
