"""Case audit for 71017277: was the gold-bearing evidence already in the production qc context?"""
import pickle, sys, re, json
sys.path.insert(0, "/root/agent_memory_iclr/code"); sys.path.insert(0, "/root/agent_memory_iclr/results")
from _recon_ctx import qc_context, units_of, evidence_turns
tr={q["question_id"]:q for q in pickle.load(open("/root/agent_memory_iclr/vfy2/tr.pkl","rb"))}
for qid, pats in [("71017277", r"chandelier|aunt"), ("gpt4_f420262d", r"american airlines"),
                  ("gpt4_4929293b", r"cousin'?s wedding|bridesmaid")]:
    q=tr[qid]
    label, mode, view, raw_idx = qc_context(q)
    units=units_of(q)
    print("="*100); print(qid, "| Q:", q["question"], "| GOLD:", q["answer"], "| view_chars", len(view))
    print("-- lifecycle view lines matching /%s/:" % pats)
    for line in view.splitlines():
        if re.search(pats, line, re.I): print("   ", line[:220])
    print("-- BM25 top-20 raw turns matching:")
    for i in raw_idx:
        if re.search(pats, units[i][2], re.I):
            print(f"    [{units[i][0]}][{units[i][1]}] {units[i][2][:200]}")
    print("-- ground-truth evidence turn(s):")
    for e in evidence_turns(q): print("    ", e[:260])
