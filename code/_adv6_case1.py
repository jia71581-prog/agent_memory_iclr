"""Case audit: for the item-count under-counts, is the MISSING gold item actually present in
the printed cluster (precision problem) or absent from the cached extraction (recall problem)?"""
import json, re, sys
sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains
from v5_chains import merge_chains, aggregation_printed_groups
from _adv6_build import facts_from_cache

R = "/root/agent_memory_iclr/results"
data = {q["question_id"]: q for q in json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))}
QC = {}
for l in open(f"{R}/full500_qc.jsonl"):
    d = json.loads(l)
    if "judge" in d:
        QC[d["qid"]] = d
CASES = sys.argv[1:] or ["3a704032", "46a3abf7", "bf659f65", "2ce6a0f2", "d682f1a2"]
for qid in CASES:
    q = data[qid]
    facts, _h, _m = facts_from_cache(q)
    store5 = merge_chains(assemble_chains(facts))
    kept, _ = aggregation_printed_groups(store5, q["question"])
    sid2idx = {s: i for i, s in enumerate(q["haystack_session_ids"])}
    gold = {sid2idx[s] for s in q["answer_session_ids"] if s in sid2idx}
    print("=" * 100)
    print(f"{qid}  Q: {q['question']}\n  GOLD: {q['answer']}\n  PRED: {(QC[qid].get('pred') or '')[:200]}")
    print(f"  -- has_answer evidence turns ({len(gold)} gold sessions):")
    for si in sorted(gold):
        for t in q["haystack_sessions"][si]:
            if t.get("has_answer") and t.get("content"):
                print(f"     [s{si} {t['role']}] {t['content'][:300]}")
    print(f"  -- ALL cached facts extracted from those gold sessions:")
    for f in facts:
        if f["sidx"] in gold:
            print(f"     [s{f['sidx']}] {f['entity']} | {f['attribute']} | {f['value']}")
    print(f"  -- printed clusters ({len(kept)}):")
    for g in kept:
        print(f"     [{g['term']}] {g['entity']}: {len(g['items'])} items")
        for score, c, attr, nterms in g["items"][:40]:
            star = "*" if nterms >= 2 else " "
            mark = "G" if c.get("sidx") in gold else " "
            print(f"        {mark}{star} {attr} = {c['value']} (s{c.get('sidx')})")
