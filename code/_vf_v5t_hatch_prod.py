"""ADVERSARIAL: bug-4 blast radius under EXACT production gating
(render_timeline calls anchor_block only when family == 'anchor_lookup', for high[:2]).
Offline, no API.
"""
import importlib.util
import json
import re
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
import v5_temporal as VF

spec = importlib.util.spec_from_file_location("v5_temporal_rev", "/tmp/v5t_rev/v5_temporal_rev.py")
VR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(VR)

D = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


n_gated, n_calls, changed, rows = 0, 0, 0, []
for q in D:
    fam, _ = VF.detect_family(q["question"])
    if fam != "anchor_lookup":
        continue
    hits = VF.resolve_question_anchors(q["question"], q["question_date"])
    high = [h for h in hits if h["conf"] == "high"]
    if not high:
        continue
    n_gated += 1
    gs = set(q["answer_session_ids"])
    goldsent = set()
    for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"]):
        if sid in gs:
            for t in sess:
                if t.get("role") == "user":
                    for s in VF._sentences(t.get("content") or ""):
                        goldsent.add(norm(s))
    qchanged = False
    for a in high[:2]:
        n_calls += 1
        lf, wf = VF.anchor_block(q, a)
        lr, wr = VR.anchor_block(q, a)
        if lf == lr:
            continue
        qchanged = True
        dropped = [x for x in lr if x not in lf]

        def isg(line):
            b = norm(line).removesuffix("...").strip()
            return any(b[:80] in g for g in goldsent) if b else False
        rows.append({"qid": q["question_id"], "q": q["question"][:120],
                     "gold": str(q["answer"])[:60], "anchor": a["date"].strftime("%Y/%m/%d"),
                     "kind": a["kind"], "dropped": dropped,
                     "dropped_from_gold_session": [x for x in dropped if isg(x)],
                     "gold_rank_fixed": next((i for i, x in enumerate(lf) if isg(x)), None),
                     "gold_rank_rev": next((i for i, x in enumerate(lr) if isg(x)), None),
                     "gold_present_fixed": any(isg(x) for x in lf),
                     "gold_present_rev": any(isg(x) for x in lr)})
    changed += int(qchanged)

summ = {"questions_gated_for_anchor_block": n_gated, "anchor_block_calls": n_calls,
        "questions_changed_by_bug4_fix": changed, "blocks_changed": len(rows),
        "blocks_losing_gold_presence": sum(1 for r in rows
                                          if r["gold_present_rev"] and not r["gold_present_fixed"]),
        "blocks_gaining_gold_presence": sum(1 for r in rows
                                            if r["gold_present_fixed"] and not r["gold_present_rev"]),
        "gold_rank_improved": sum(1 for r in rows if r["gold_rank_fixed"] is not None
                                  and r["gold_rank_rev"] is not None
                                  and r["gold_rank_fixed"] < r["gold_rank_rev"]),
        "gold_rank_worsened": sum(1 for r in rows if r["gold_rank_fixed"] is not None
                                  and r["gold_rank_rev"] is not None
                                  and r["gold_rank_fixed"] > r["gold_rank_rev"])}
json.dump({"summary": summ, "rows": rows},
          open("/root/agent_memory_iclr/results/_vf_v5t_hatch_prod.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
for r in rows:
    print("-", r["qid"], r["anchor"], r["kind"], "| gold rank",
          r["gold_rank_rev"], "->", r["gold_rank_fixed"], "| dropped", len(r["dropped"]))
