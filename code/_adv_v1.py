"""ADVERSARIAL recompute #1: reproduce the 5932/5357 headline and split it by
(i) merged vs unmerged store, (ii) relation label of the hidden facts."""
import json, pickle, sys, os, statistics as st
from collections import Counter
sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains, render_view
from v5_chains import merge_chains, is_aggregation_question

D = {x["qid"]: x for x in pickle.load(open("/root/agent_memory_iclr/results/_v5_facts.pkl", "rb"))}
G = pickle.load(open("/root/agent_memory_iclr/results/_v5_gold.pkl", "rb"))


def visible(chain):
    if chain[0]["relation"] == "accumulative":
        return chain
    last = (chain[-1]["date"], chain[-1].get("sidx"))
    return [c for c in chain if (c["date"], c.get("sidx")) == last]


def measure(store, gold_sidx):
    gf = gr = 0
    hidden_rel = Counter()
    vis_rel = Counter()
    for chain in store.values():
        vis = visible(chain)
        visids = {id(c) for c in vis}
        for c in chain:
            if c.get("sidx") in gold_sidx:
                gf += 1
                if c in vis:          # exactly as the design agent wrote it (dict ==)
                    gr += 1
                    vis_rel[c["relation"]] += 1
                else:
                    hidden_rel[c["relation"]] += 1
    return gf, gr, hidden_rel, vis_rel


tot = {"m_gf": 0, "m_gr": 0, "u_gf": 0, "u_gr": 0}
hid_m, hid_u = Counter(), Counter()
nq = 0
per_q = []
for qid, x in D.items():
    if not is_aggregation_question(x["question"])[0]:
        continue
    nq += 1
    gold_sidx = set(G[qid]["gold_sidx"])
    s4 = assemble_chains(x["facts"])
    s5 = merge_chains(s4)
    m_gf, m_gr, hm, _ = measure(s5, gold_sidx)
    u_gf, u_gr, hu, _ = measure(s4, gold_sidx)
    tot["m_gf"] += m_gf; tot["m_gr"] += m_gr
    tot["u_gf"] += u_gf; tot["u_gr"] += u_gr
    hid_m += hm; hid_u += hu
    per_q.append({"qid": qid, "m_gf": m_gf, "m_gr": m_gr, "u_gf": u_gf, "u_gr": u_gr,
                  "n_gold_sess": len(gold_sidx)})

print(f"triggered questions: {nq}")
print(f"MERGED store   : gold facts {tot['m_gf']}  rendered {tot['m_gr']}  "
      f"ratio {tot['m_gr']/tot['m_gf']:.4f}  hidden {1-tot['m_gr']/tot['m_gf']:.2%}")
print(f"UNMERGED v4    : gold facts {tot['u_gf']}  rendered {tot['u_gr']}  "
      f"ratio {tot['u_gr']/tot['u_gf']:.4f}  hidden {1-tot['u_gr']/tot['u_gf']:.2%}")
print("hidden-by-relation, merged  :", dict(hid_m))
print("hidden-by-relation, unmerged:", dict(hid_u))
json.dump(per_q, open("/root/agent_memory_iclr/results/_adv_v1.json", "w"))
