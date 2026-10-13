"""ADVERSARIAL VERIFIER #2 step 5: asymmetric risk of the aggregation block.
- can the block's printed counts fix the questions the pipeline currently gets WRONG?
- does it fire on abstention questions (where gold = 'no information'), inviting an answer?
- honest matched-budget view-line accounting."""
import json
import pickle
import statistics as st
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains, render_view
from query_classifier import classify_query
from v5_chains import merge_chains, render_aggregation_view

CAP = 20000
D = pickle.load(open("/root/agent_memory_iclr/results/_adv2_facts.pkl", "rb"))
C = {r["qid"]: r for r in json.load(open("/root/agent_memory_iclr/results/_adv2_counts.json"))}
COV = {r["qid"]: r for r in json.load(open("/root/agent_memory_iclr/results/coverage_diag.json"))}
JQC = {json.loads(l)["qid"]: json.loads(l) for l in
       open("/root/agent_memory_iclr/results/full500_qc.jsonl")}

fired = [q for q in C if C[q]["fired"]]
cnt = [q for q in fired if C[q]["kind"] == "count" and C[q]["gnum"] is not None]


def any_eq(r):
    return r["gnum"] in [g[1] for g in r["groups"]] + [r["union"], r["strong_union"], r["top_strong"]]


ok = [q for q in cnt if C[q]["judge"]]
bad = [q for q in cnt if not C[q]["judge"]]
print(f"=== asymmetric risk on the {len(cnt)} numeric 'count' questions where the block fires ===")
print(f"currently CORRECT {len(ok)} / currently WRONG {len(bad)}")
print(f"  ANY printed count == gold, among currently WRONG:   "
      f"{sum(any_eq(C[q]) for q in bad)}/{len(bad)} ({sum(any_eq(C[q]) for q in bad)/len(bad):.1%})"
      f"   <- upside the block can deliver")
print(f"  union != gold and over-counts, among currently CORRECT: "
      f"{sum(1 for q in ok if C[q]['union'] > C[q]['gnum'])}/{len(ok)} "
      f"({sum(1 for q in ok if C[q]['union'] > C[q]['gnum'])/len(ok):.1%})   <- downside exposure")
print(f"  median printed union among currently CORRECT: {st.median([C[q]['union'] for q in ok]):.0f} "
      f"vs median gold {st.median([C[q]['gnum'] for q in ok]):.0f}")

abs_q = [q for q in C if COV.get(q, {}).get("is_abs")]
abs_fire = [q for q in abs_q if C[q]["fired"]]
print(f"\n=== abstention questions (gold = unanswerable) ===")
print(f"total abstention questions {len(abs_q)}; block fires on {len(abs_fire)}")
print(f"  current judge acc on those {len(abs_fire)}: "
      f"{st.mean([C[q]['judge'] for q in abs_fire]):.3f}" if abs_fire else "")
if abs_fire:
    print(f"  they receive a block of median {st.median([len(str(C[q]['groups'])) for q in abs_fire]):.0f} "
          f"chars of 'candidate item sets' with median union {st.median([C[q]['union'] for q in abs_fire]):.0f} items")
    for q in abs_fire[:8]:
        print(f"    {q} union={C[q]['union']:>3} judge={'Y' if C[q]['judge'] else 'N'} "
              f"gold={C[q]['answer'][:40]!r} | {C[q]['question'][:70]}")

print("\n=== honest matched-budget accounting (block shares the 20k cap, placed first) ===")


def cap_lines(txt, cap=CAP):
    kept, total = [], 0
    for line in txt.splitlines():
        if total + len(line) + 1 > cap:
            break
        kept.append(line)
        total += len(line) + 1
    return kept


rows = []
for x in D:
    if not C[x["qid"]]["fired"]:
        continue
    label, _ = classify_query(x["question"])
    mode = "historical" if label == "historical" else "current"
    s4 = assemble_chains(x["facts"])
    s5 = merge_chains(s4)
    v4 = render_view(s4, mode)
    v5 = render_view(s5, mode)
    blk = render_aggregation_view(s5, x["question"])
    l4 = len(cap_lines(v4))
    l5 = len(cap_lines(v5))
    room = CAP - (len(blk) + 2)
    l5m = len(cap_lines(v5, room))
    rows.append((l4, l5, l5m, len(blk), blk.count("\n") + 1))
print(f"on the {len(rows)} firing questions, VIEW lines that survive:")
print(f"  v4                        median {st.median([r[0] for r in rows]):.0f}")
print(f"  v5 view, block outside cap (as reported) median {st.median([r[1] for r in rows]):.0f}")
print(f"  v5 view, block inside cap (matched)      median {st.median([r[2] for r in rows]):.0f} "
      f"-> {st.median([r[2] for r in rows])-st.median([r[0] for r in rows]):+.0f} vs v4, "
      f"i.e. {(st.median([r[0] for r in rows])-st.median([r[2] for r in rows]))/st.median([r[0] for r in rows]):.1%} of the view displaced")
print(f"  block lines median {st.median([r[4] for r in rows]):.0f}, block chars median {st.median([r[3] for r in rows]):.0f}")
