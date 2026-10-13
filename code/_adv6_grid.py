"""Grid search for the design agent's n=46 subset + cluster definition."""
import json, re, statistics as st, sys, hashlib, os
from multiprocessing import Pool
sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import _session_text, assemble_chains
from v5_chains import merge_chains, aggregation_printed_groups, aggregation_groups
from v5_shape import classify_shape
from _adv6_build import facts_from_cache, gold_number, is_enum_count

R = "/root/agent_memory_iclr/results"


def work(q):
    facts, _h, _m = facts_from_cache(q)
    store5 = merge_chains(assemble_chains(facts))
    kept, _meta = aggregation_printed_groups(store5, q["question"])
    raw, _ = aggregation_groups(store5, q["question"])
    sizes = [len(g["items"]) for g in kept]
    # top group = first in the sorted order (highest score); max group = largest
    def dv(items):
        return len({re.sub(r"[^a-z0-9]+", " ", it[1]["value"].lower()).strip() for it in items})
    top_dv = dv(kept[0]["items"]) if kept else 0
    max_dv = max((dv(g["items"]) for g in kept), default=0)
    strong_union = sum(1 for g in kept for it in g["items"] if it[3] >= 2)
    strong_top = sum(1 for it in (kept[0]["items"] if kept else []) if it[3] >= 2)
    strong_max = max((sum(1 for it in g["items"] if it[3] >= 2) for g in kept), default=0)
    return {"qid": q["question_id"], "top_dv": top_dv, "max_dv": max_dv,
            "union_dv": sum(sizes), "n_groups": len(kept), "sizes": sizes,
            "raw_union": sum(len(g["items"]) for g in raw),
            "strong_union": strong_union, "strong_top": strong_top, "strong_max": strong_max}


if __name__ == "__main__":
    data = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
    with Pool(16) as p:
        extra = {r["qid"]: r for r in p.map(work, data, chunksize=2)}
    base = {r["qid"]: r for r in json.load(open(f"{R}/_adv6_rows.json"))}
    for k, v in extra.items():
        base[k].update(v)
    rows = list(base.values())
    json.dump(rows, open(f"{R}/_adv6_rows2.json", "w"), ensure_ascii=False)

    QC = {}
    for l in open(f"{R}/full500_qc.jsonl"):
        d = json.loads(l)
        if "judge" in d:
            QC[d["qid"]] = d
    for r in rows:
        r["judge"] = QC[r["qid"]]["judge"]

    subsets = {
        "enum60": [r for r in rows if r["enum"]],
        "enum&AGGenum": [r for r in rows if r["enum"] and r["shape"] == "AGG.enum"],
        "AGGenum&howmany&N1-12": [r for r in rows if r["shape"] == "AGG.enum" and "how many" in r["question"].lower() and r["gold_num"] is not None and r["gold_num"] == int(r["gold_num"]) and 1 <= r["gold_num"] <= 12],
        "enum60&nonabs": [r for r in rows if r["enum"] and not r["qid"].endswith("_abs")],
        "enum60&has_group": [r for r in rows if r["enum"] and r["n_groups"] > 0],
        "AGGenum&N1-12": [r for r in rows if r["shape"] == "AGG.enum" and r["gold_num"] is not None and r["gold_num"] == int(r["gold_num"]) and 1 <= r["gold_num"] <= 12],
        "howmanyStart&N1-12": [r for r in rows if r["question"].lower().lstrip().startswith("how many") and r["gold_num"] is not None and r["gold_num"] == int(r["gold_num"]) and 1 <= r["gold_num"] <= 12],
        "howmanyStart&AGGenum&N1-12": [r for r in rows if r["question"].lower().lstrip().startswith("how many") and r["shape"] == "AGG.enum" and r["gold_num"] is not None and r["gold_num"] == int(r["gold_num"]) and 1 <= r["gold_num"] <= 12],
    }
    for name, sub in subsets.items():
        if not sub:
            continue
        print(f"\n-- {name}: n={len(sub)} goldMean={st.mean(r['gold_num'] for r in sub):.2f} "
              f"N>n_ev={sum(1 for r in sub if r['gold_num'] > r['n_ev'])}")
        for f in ("top_dv", "max_dv", "union_dv", "strong_union", "strong_top", "strong_max"):
            ge = sum(1 for r in sub if r[f] >= r["gold_num"])
            print(f"    {f:13s} mean={st.mean(r[f] for r in sub):6.2f} >=N {ge:3d}/{len(sub)} = {ge/len(sub):.3f}")
