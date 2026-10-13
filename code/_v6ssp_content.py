"""内容检查:6 个 LOSS 案例上,gold rubric 点名的具体细节在 v5 / v6 上下文里各出现几次。
这是"代码是否产生了预期的上下文变化"的直接证据,不是正确率代理指标。"""
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _v6ssp_verify as V

DATA = {q["question_id"]: q for q in json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))}

# 每个 LOSS 案例:gold rubric 明确点名、且 full(答对)在预测里引用过的具体细节
PROBE = {
    "1c0ddc50": ["Hardcore History", "The Dollop", "branch out", "40 minutes each way", "audiobook"],
    "6b7dfb22": ["Instagram", "30-day painting challenge", "flower", "tutorial"],
    "57f827a0": ["mid-century", "dresser"],
    "b6025781": ["quinoa", "roasted veg", "Caesar", "turkey", "avocado"],
    "32260d93": ["stand-up", "Netflix", "storytelling", "comedy special"],
    "1da05512": ["storage", "external hard drive", "capacity"],
}

m5 = V.load(v5="1")
m6 = V.load(v5="1", v6="1")
print(f"{'qid':10s} {'probe':28s} {'v5':>4s} {'v6':>4s}")
tot5 = tot6 = 0
for qid, probes in PROBE.items():
    q = DATA[qid]
    c5, _ = V.build_ctx(m5, q)
    c6, i6 = V.build_ctx(m6, q)
    assert i6["gated"]
    for p in probes:
        a, b = c5.lower().count(p.lower()), c6.lower().count(p.lower())
        tot5 += a > 0
        tot6 += b > 0
        flag = "  <-- NEW" if b > 0 and a == 0 else ("  <-- LOST" if a > 0 and b == 0 else "")
        print(f"{qid:10s} {p:28s} {a:4d} {b:4d}{flag}")
print(f"\nprobe strings present: v5 {tot5}/{sum(len(v) for v in PROBE.values())}  "
      f"v6 {tot6}/{sum(len(v) for v in PROBE.values())}")
