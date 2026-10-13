"""Independent reviewer harness: dump the exact prompt each arm would send, with
every DeepSeek call made fatal (so a lifecycle cache miss is detected, not paid for).

usage: <env flags> python3 _rev_dump.py <out.jsonl> [module] [limit]
       module = run_qc (default) | frozen  (loads run_qc_v5_frozen.py.bak)
"""
import hashlib
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT = sys.argv[1]
WHICH = sys.argv[2] if len(sys.argv) > 2 else "run_qc"
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 0

import common
import lifecycle


class ApiCalled(RuntimeError):
    pass


API_ATTEMPTS = []


def _no_api(*a, **k):
    API_ATTEMPTS.append(1)
    raise ApiCalled("DeepSeek call attempted (cache miss) - refused by reviewer harness")


common.chat = _no_api
lifecycle.chat = _no_api

if WHICH == "frozen":
    import importlib.machinery
    path = os.path.join(HERE, "run_qc_v5_frozen.py.bak")
    spec = importlib.util.spec_from_loader(
        "run_qc_frozen", importlib.machinery.SourceFileLoader("run_qc_frozen", path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
else:
    import run_qc as mod

CAP = {}


def _cap_answer(system_prompt, user_content, ledger, op, max_tokens=6144):
    CAP["sys"] = system_prompt
    CAP["user"] = user_content
    return "STUB ANSWER"


mod.answer_chat = _cap_answer
if os.environ.get("REV_ABLATE"):
    mod.ABLATE = os.environ["REV_ABLATE"]

data = common.load_dataset("s_cleaned")
if LIMIT:
    data = data[:LIMIT]
ledger = common.CostLedger()
rows = []
t0 = time.time()
for i, q in enumerate(data):
    CAP.clear()
    rec = {"qid": q["question_id"], "type": q["question_type"]}
    try:
        ans, label, mode = mod.answer_question(q, ledger)
        u = CAP.get("user", "")
        s = CAP.get("sys", "")
        rec.update({
            "ok": True, "label": label, "mode": mode,
            "sys_sha": hashlib.sha256(s.encode()).hexdigest()[:16],
            "sys_len": len(s),
            "user_len": len(u),
            "user_sha": hashlib.sha256(u.encode()).hexdigest()[:16],
            "view_len": len(u.split("### Retrieved raw conversation turns\n")[0]),
            "n_raw_lines": (u.count("\n[") if "### Retrieved raw conversation turns" in u else 0),
            "n_sections": u.count("### "),
        })
    except Exception as e:
        rec.update({"ok": False, "err": f"{type(e).__name__}: {e}"})
    rows.append(rec)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(data)}  {time.time()-t0:.0f}s", flush=True)
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
bad = [r for r in rows if not r["ok"]]
print(f"done n={len(rows)} errors={len(bad)} api_attempts={len(API_ATTEMPTS)} "
      f"elapsed={time.time()-t0:.0f}s max_user_len={max((r.get('user_len',0) for r in rows), default=0)}")
for r in bad[:10]:
    print("  ERR", r["qid"], r["type"], r["err"])
