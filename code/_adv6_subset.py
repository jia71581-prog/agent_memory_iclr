import json, re, statistics as st
R = "/root/agent_memory_iclr/results"
rows = json.load(open(f"{R}/_adv6_rows.json"))
QC = {}
for l in open(f"{R}/full500_qc.jsonl"):
    d = json.loads(l)
    if "judge" in d:
        QC[d["qid"]] = d
print("qc rows with judge", len(QC))
print("abs qids in dataset", sum(1 for r in rows if r["qid"].endswith("_abs")))
for r in rows:
    r["judge"] = QC.get(r["qid"], {}).get("judge")


def show(name, sub):
    if not sub:
        print(f"{name:52s} n=0")
        return
    mg = st.mean(r["gold_num"] for r in sub)
    md = st.mean(r["union_distinct"] for r in sub)
    mu = st.mean(r["union_items"] for r in sub)
    ge = sum(1 for r in sub if r["union_distinct"] >= r["gold_num"])
    geu = sum(1 for r in sub if r["union_items"] >= r["gold_num"])
    gt = sum(1 for r in sub if r["gold_num"] > r["n_ev"])
    print(f"{name:52s} n={len(sub):3d} goldMean={mg:5.2f} distMean={md:5.2f} unionMean={mu:5.2f} "
          f"dist>=N {ge:3d}/{len(sub)} union>=N {geu:3d} N>n_ev {gt:3d}")


E = [r for r in rows if r["enum"]]
AG = [r for r in rows if r["shape"] == "AGG.enum"]
show("is_enum_count (all 500)", E)
show("is_enum_count & shape==AGG.enum", [r for r in E if r["shape"] == "AGG.enum"])
show("is_enum_count & shape startswith AGG", [r for r in E if r["shape"].startswith("AGG")])
show("shape AGG.enum & N int 1..12", [r for r in AG if r["gold_num"] is not None and r["gold_num"] == int(r["gold_num"]) and 1 <= r["gold_num"] <= 12])
show("shape AGG.enum & 'how many' & N int 1..12", [r for r in AG if "how many" in r["question"].lower() and r["gold_num"] is not None and r["gold_num"] == int(r["gold_num"]) and 1 <= r["gold_num"] <= 12])
show("shape AGG.enum & 'how many' & N<=12 (allow 0)", [r for r in AG if "how many" in r["question"].lower() and r["gold_num"] is not None and r["gold_num"] == int(r["gold_num"]) and 0 <= r["gold_num"] <= 12])
show("'how many' & N int 1..12 (any shape)", [r for r in rows if "how many" in r["question"].lower() and r["gold_num"] is not None and r["gold_num"] == int(r["gold_num"]) and 1 <= r["gold_num"] <= 12])
show("is_enum_count & is_agg & kind==count", [r for r in E if r["kind"] == "count"])
show("is_enum_count minus GOLD_UNIT relax", [r for r in rows if "how many" in r["question"].lower() and r["gold_num"] is not None and r["gold_num"] == int(r["gold_num"]) and 1 <= r["gold_num"] <= 12 and r["shape"] in ("AGG.enum", "AGG.latest_count")])
