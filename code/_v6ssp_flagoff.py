"""关旗回归:当前 run_qc.py(含 v6-SSP 及兄弟分支)在只开 QC_V5=1 时,
prompt 必须与冻结的 v5 快照 run_qc_v5_frozen.py.bak 逐字节相同。"""
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DATA = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def capture(mod, q):
    got = {}

    def fake(system_prompt, user_content, ledger, op, max_tokens=6144):
        got["s"], got["u"] = system_prompt, user_content
        return "STUB"

    orig = mod.answer_chat
    mod.answer_chat = fake
    try:
        mod.answer_question(q, None)
    finally:
        mod.answer_chat = orig
    return got["s"], got["u"]


for k in list(os.environ):
    if k.startswith("QC_"):
        del os.environ[k]
os.environ["QC_V5"] = "1"
frozen = load_module("/tmp/run_qc_frozen_v5.py", "run_qc_frozen")
cur = load_module("/root/agent_memory_iclr/code/run_qc.py", "run_qc_cur")
print("frozen V5 =", frozen.V5, "| current V5 =", cur.V5, "V6SSP =", cur.V6SSP)

import collections
sample, seen = [], collections.Counter()
for q in DATA:
    if seen[q["question_type"]] < 25:
        seen[q["question_type"]] += 1
        sample.append(q)
bad = 0
for q in sample:
    sf, pf = capture(frozen, q)
    sc, pc = capture(cur, q)
    if sf != sc or pf != pc:
        bad += 1
        print("  MISMATCH", q["question_id"], q["question_type"], len(pf), len(pc))
print(f"{len(sample)-bad}/{len(sample)} prompts byte-identical to frozen v5 (QC_V5=1, 所有 v6 旗关闭)")
