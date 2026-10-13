import json, statistics as st, sys
from collections import Counter
rows=json.load(open('/root/agent_memory_iclr/results/v5_temporal_tr.json'))
def f(g,k): return sum(bool(r.get(k)) for r in g)/max(1,len(g))
print("== per-family recoverability (TR n=%d, incl. 6 abstention) =="%len(rows))
print(f"{'family':14s} {'n':>3s} {'datesPres':>9s} {'derivAns':>8s} {'derivEvid':>9s} {'derivTL':>8s} {'literal':>8s} {'medCand':>7s}")
for fam in ['delta_between','delta_since','ordering','anchor_lookup','count_before','other']:
    g=[r for r in rows if r['family']==fam]
    if not g: continue
    nc=[r['n_arith_candidates'] for r in g if r['n_arith_candidates']]
    print(f"{fam:14s} {len(g):3d} {st.mean([r['dates_present'] for r in g]):9.3f} "
          f"{f(g,'gold_derivable_from_ans_dates'):8.3f} {f(g,'gold_derivable_from_evidence'):9.3f} "
          f"{f(g,'gold_derivable_from_timeline'):8.3f} {f(g,'gold_literal_in_arith'):8.3f} {(st.median(nc) if nc else 0):7.1f}")
dg=[r for r in rows if r['family'] in ('delta_between','delta_since') and r['gold_vals']]
print(f"\ndelta families w/ numeric gold n={len(dg)}: derivEvid={f(dg,'gold_derivable_from_evidence'):.3f} derivTL={f(dg,'gold_derivable_from_timeline'):.3f} literalInArith={f(dg,'gold_literal_in_arith'):.3f}")
o=[r for r in rows if r['family']=='ordering']
oi=[r for r in o if r.get('ordering_item_correct') is not None]
print(f"\nordering n={len(o)}: ans_dates_present={f(o,'ordering_all_present'):.3f} same_day_collision={sum(bool(r.get('ans_same_day_collision')) for r in o)} split_by_intext={sum(bool(r.get('same_day_split')) for r in o)}")
print(f"  item-level: computable {len(oi)}/{len(o)}; correct {sum(r['ordering_item_correct'] for r in oi)}/{len(oi)} = {sum(r['ordering_item_correct'] for r in oi)/max(1,len(oi)):.3f}")
bi=[r for r in oi if r['ordering_n_items']==2]
print(f"  binary subfamily n={len(bi)} correct={sum(r['ordering_item_correct'] for r in bi)/max(1,len(bi)):.3f} (chance 0.5)")
li=[r for r in oi if r['ordering_n_items']>=3]
print(f"  list subfamily n={len(li)} correct={sum(r['ordering_item_correct'] for r in li)/max(1,len(li)):.3f} (chance 1/n!)")
print(f"  not computable (some named item has no timeline row): {[r['qid'] for r in o if r.get('ordering_item_correct') is None]}")
al=[r for r in rows if r['family']=='anchor_lookup']
print(f"\nanchor_lookup n={len(al)}: hit<=3d={f(al,'anchor_hit'):.3f} discriminative={f(al,'anchor_discriminative'):.3f} err_days={dict(Counter(r['anchor_err_days'] for r in al))}")
print(f"\nsize chars: median={st.median([r['chars'] for r in rows])} mean={round(st.mean([r['chars'] for r in rows]))} p90={sorted(r['chars'] for r in rows)[int(.9*len(rows))]} max={max(r['chars'] for r in rows)}")
