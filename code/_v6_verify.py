"""Run the REAL run_qc.answer_question path with the API hard-blocked, and diff v5 vs v6-AGG ctx.
Any attempted DeepSeek call (extraction cache miss or answer) raises immediately."""
import json, os, sys
os.environ["QC_V5"] = "1"
os.environ["QC_V6_AGG"] = "1"
sys.path.insert(0, "/root/agent_memory_iclr/code")

import common, lifecycle, run_qc

def _blocked(*a, **k):
    raise RuntimeError("API CALL ATTEMPTED - test must be offline")
common.chat = _blocked
lifecycle.chat = _blocked          # lifecycle imported chat by name

CAP = {}
def fake_answer_chat(system_prompt, user_content, ledger, op, max_tokens=6144):
    CAP["ctx"] = user_content
    CAP["sys"] = system_prompt
    return "STUB"
run_qc.answer_chat = fake_answer_chat

def ctx_for(q, v6):
    run_qc.V6_AGG = v6
    CAP.clear()
    run_qc.answer_question(q, None)
    return CAP["ctx"]

if __name__ == "__main__":
    data = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
    rows = {json.loads(l)["qid"]: json.loads(l) for l in open("/root/agent_memory_iclr/results/full500_qc_v5.jsonl")}
    ms_err = [r["qid"] for r in rows.values() if r["type"] == "multi-session" and not r["judge"]]
    want = set(ms_err)
    # add 12 controls: 6 firing-correct, 6 non-firing
    import v6_agg
    fire_ok, nofire = [], []
    for q in data:
        r = rows.get(q["question_id"])
        if not r or not r["judge"]:
            continue
        if v6_agg.is_agg_question(q["question"]) and len(fire_ok) < 6:
            fire_ok.append(q["question_id"])
        elif not v6_agg.is_agg_question(q["question"]) and len(nofire) < 6:
            nofire.append(q["question_id"])
    want |= set(fire_ok) | set(nofire)

    out, ident, changed = {}, 0, 0
    for q in data:
        qid = q["question_id"]
        if qid not in want:
            continue
        c5, c6 = ctx_for(q, False), ctx_for(q, True)
        fired = v6_agg.is_agg_question(q["question"])
        same = c5 == c6
        if same:
            ident += 1
        else:
            changed += 1
        assert fired or same, f"{qid}: non-firing question changed!"
        out[qid] = {"question": q["question"], "gold": q["answer"], "qdate": q["question_date"],
                    "fired": fired, "v5": c5, "v6": c6, "judge": rows[qid]["judge"],
                    "type": rows[qid]["type"], "pred": rows[qid]["pred"]}
        print(f"{qid:16s} fired={fired} identical={same} chars {len(c5)}->{len(c6)} "
              f"({100*(len(c6)/len(c5)-1):+.2f}%)", flush=True)
    json.dump(out, open("/root/agent_memory_iclr/results/_v6_verify.json", "w"), ensure_ascii=False)
    print(f"\nidentical={ident} changed={changed} (non-firing must all be identical)")
    # flag-off equivalence over the whole firing set is covered by the assert above
