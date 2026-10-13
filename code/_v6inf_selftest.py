"""v6-INF/VER 无 API 自检:
 1. 三种开关组合下首答 (system, user) 的逐字节关系(v5 基准 / VER-only / INF)
 2. _v6_verify_revise 的解析分支(OK / REVISED / 空 / 无法解析)—— chat 被替换为假桩,不联网
 3. 预算中性变体只削视图尾部、不动原文分节
"""
import json
import os
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
sys.path.insert(0, "/root/agent_memory_iclr/results")
from _recon_ctx import build_store
from common import load_dataset

QIDS = ["9ee3ecd6", "0edc2aef", "6e984301", "51a45a95", "d851d5ba", "gpt4_fe651585", "07741c45"]
data = [q for q in load_dataset("s_cleaned") if q["question_id"] in QIDS]


def capture(env):
    for k in ("QC_V6_INF", "QC_V6_VER", "QC_V6_INF_NEUTRAL", "QC_V6_VER_ALL", "QC_V6_SSP", "QC_V6_AGG"):
        os.environ.pop(k, None)
    os.environ["QC_V5"] = "1"
    os.environ.update(env)
    for m in [m for m in list(sys.modules) if m in ("run_qc",)]:
        del sys.modules[m]
    import run_qc as RQ
    got = {}
    RQ.build_lifecycle = lambda q, ledger: build_store(q)
    RQ.answer_chat = lambda s, u, ledger, op, max_tokens=6144: (got.__setitem__("p", (s, u)), "DRAFT")[1]
    RQ.REAL_VERIFY = RQ._v6_verify_revise
    RQ._v6_verify_revise = lambda s, u, d, ledger: (d, "stub")
    out = {}
    for q in data:
        RQ.answer_question(q, None)
        out[q["question_id"]] = got["p"]
    return out, RQ


base, _ = capture({})
ver, RQ_ver = capture({"QC_V6_VER": "1"})
inf, RQ_inf = capture({"QC_V6_INF": "1"})
neu, RQ_neu = capture({"QC_V6_INF": "1", "QC_V6_INF_NEUTRAL": "1"})

print("== 1. first-pass prompt identity")
for qid in base:
    b_s, b_u = base[qid]
    assert ver[qid] == (b_s, b_u), f"VER-only first pass differs from v5 on {qid}"
    i_s, i_u = inf[qid]
    scaf = RQ_inf.v6_scaffold_for([q for q in data if q["question_id"] == qid][0]["question"])
    assert i_u == b_u, f"INF changed the user content on {qid}"
    assert i_s == b_s + "\n\n" + scaf, f"INF system prompt malformed on {qid}"
    n_s, n_u = neu[qid]
    assert n_s == i_s
    print(f"  {qid:14s} v5={len(b_s)+len(b_u):6d} ch | INF={len(i_s)+len(i_u):6d} "
          f"({100*((len(i_s)+len(i_u))/(len(b_s)+len(b_u))-1):+.2f}%) | "
          f"NEUTRAL={len(n_s)+len(n_u):6d} ({100*((len(n_s)+len(n_u))/(len(b_s)+len(b_u))-1):+.2f}%) "
          f"| scaffold={len(scaf)}")
    # 预算中性只削视图尾部:原文分节逐字节不变
    mk = "\n\n### Retrieved raw conversation turns\n"
    if mk in b_u and mk in n_u:
        assert b_u[b_u.find(mk):] == n_u[n_u.find(mk):], f"NEUTRAL touched the raw section on {qid}"
print("  raw section byte-identical under the budget-neutral trim: OK")

print("\n== 2. verify-pass parsing (fake chat, no network)")
import common
CASES = [("OK", "draft", "ok"), ("OK\n", "draft", "ok"),
         ("REVISED\nYou need 100 more points.", "You need 100 more points.", "revised"),
         ("**REVISED**\nAlex.", "Alex.", "revised"),
         ("REVISED", "draft", "revised-empty"),
         ("", "draft", "empty"), ("hmm let me think", "draft", "unparsed")]
for out, want_ans, want_status in CASES:
    common.chat = lambda *a, **k: out
    got_ans, got_status = RQ_inf.REAL_VERIFY("sys", "user", "draft", None)
    ok = (got_ans == want_ans and got_status == want_status)
    print(f"  {out[:24]!r:28s} -> {got_status:14s} ans={got_ans[:26]!r} {'OK' if ok else 'MISMATCH'}")
    assert ok, (out, got_ans, got_status)

print("\n== 3. gate reasons on the read failing cases")
rows = {r["qid"]: r for r in (json.loads(l) for l in
        open("/root/agent_memory_iclr/results/full500_qc_v5.jsonl"))}
from v5_shape import classify_shape
for q in data:
    qid = q["question_id"]
    s, u = base[qid]
    ctx = u.rsplit("\n\nQuestion (asked on ", 1)[0]
    shape, _ = classify_shape(q["question"])
    print(f"  {qid:14s} shape={shape:18s} gate={RQ_inf._v6_verify_gate(shape, rows[qid]['pred'], ctx)}")
print("\nall assertions passed")
