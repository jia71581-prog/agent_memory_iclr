"""ADVERSARIAL: bug-4 (anchor-day escape hatch) real-corpus impact + the two named cases.
Fixed vs reverted hatch, over every question whose family is anchor_lookup.
Regression test: does the fix drop any line that came from a GOLD evidence session?
Offline, no API.
"""
import importlib.util
import json
import os
import re
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
import v5_temporal as VF                      # fixed

src = open("/root/agent_memory_iclr/code/v5_temporal.py").read()
OLD = '''                if abs((ev - anchor_dt).days) > window and not (
                        same_session and default_dated and ANCHOR_RE.search(sent)):'''
NEW = '''                if abs((ev - anchor_dt).days) > window and not (
                        same_session and ANCHOR_RE.search(sent)):'''
assert OLD in src
os.makedirs("/tmp/v5t_rev", exist_ok=True)
open("/tmp/v5t_rev/v5_temporal_rev.py", "w").write(src.replace(OLD, NEW))
spec = importlib.util.spec_from_file_location("v5_temporal_rev", "/tmp/v5t_rev/v5_temporal_rev.py")
VR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(VR)                    # reverted

D = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip().rstrip(".")


def gold_sentences(q):
    gs = set(q["answer_session_ids"])
    out = set()
    for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"]):
        if sid not in gs:
            continue
        for t in sess:
            if t.get("role") == "user":
                for s in VF._sentences(t.get("content") or ""):
                    out.add(norm(s))
    return out


rows = []
named = {}
for q in D:
    fam, unit = VF.detect_family(q["question"])
    anchors = VF.resolve_question_anchors(q["question"], q["question_date"])
    if not anchors:
        continue
    if fam != "anchor_lookup" and q["question_type"] != "temporal-reasoning":
        continue
    a = anchors[0]
    if a["conf"] != "high":
        continue
    lf, wf = VF.anchor_block(q, a, cap=10)
    lr, wr = VR.anchor_block(q, a, cap=10)
    if lf == lr and wf == wr:
        continue
    gs = gold_sentences(q)
    dropped = [x for x in lr if x not in lf]
    added = [x for x in lf if x not in lr]

    def is_gold(line):
        base = norm(line).rstrip(".").removesuffix("...").strip()
        return any(base[:80] in g for g in gs) if base else False

    rows.append({
        "qid": q["question_id"], "type": q["question_type"], "fam": fam,
        "q": q["question"], "qdate": q["question_date"], "gold": str(q["answer"])[:80],
        "anchor": a["date"].strftime("%Y/%m/%d"), "kind": a["kind"], "expr": a["expr"],
        "win_fixed": wf, "win_rev": wr,
        "n_fixed": len(lf), "n_rev": len(lr),
        "dropped": dropped, "added": added,
        "dropped_gold": [x for x in dropped if is_gold(x)],
        "added_gold": [x for x in added if is_gold(x)],
        "gold_line_rank_fixed": next((i for i, x in enumerate(lf) if is_gold(x)), None),
        "gold_line_rank_rev": next((i for i, x in enumerate(lr) if is_gold(x)), None),
    })
    if q["question_id"] in ("gpt4_4929293b", "gpt4_483dd43c"):
        named[q["question_id"]] = {"fixed": lf, "rev": lr}

summ = {
    "n_questions_with_high_anchor_changed": len(rows),
    "n_with_any_drop": sum(1 for r in rows if r["dropped"]),
    "n_dropping_a_gold_session_line": sum(1 for r in rows if r["dropped_gold"]),
    "n_gold_rank_worse": sum(1 for r in rows
                             if r["gold_line_rank_rev"] is not None
                             and (r["gold_line_rank_fixed"] is None
                                  or r["gold_line_rank_fixed"] > r["gold_line_rank_rev"])),
    "n_gold_rank_better": sum(1 for r in rows
                              if r["gold_line_rank_fixed"] is not None
                              and (r["gold_line_rank_rev"] is None
                                   or r["gold_line_rank_fixed"] < r["gold_line_rank_rev"])),
    "total_lines_dropped": sum(len(r["dropped"]) for r in rows),
    "total_lines_added": sum(len(r["added"]) for r in rows),
}
json.dump({"summary": summ, "rows": rows, "named": named},
          open("/root/agent_memory_iclr/results/_vf_v5t_hatch.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
print("\n== gpt4_4929293b (the cited case) ==")
for k, v in named.get("gpt4_4929293b", {}).items():
    print("--", k, len(v))
    for i, x in enumerate(v):
        print(f"  {i}. {x[:150]}")
