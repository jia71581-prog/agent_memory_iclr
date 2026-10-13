"""会话级抽取缓存预热:按内容哈希去重后并发抽取(全库 23,867 实例 → ~18,464 唯一)。
用法: python3 precompute_sessions.py <N题|all> [workers=16]
可反复断点续跑(已缓存会话跳过);任何会话失败会在结尾汇总并以非零退出码返回。
"""
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset, CostLedger
from lifecycle import extract_session_facts, _session_text, SESSION_CACHE_DIR
from run_baseline import stratified_sample

SCOPE = sys.argv[1] if len(sys.argv) > 1 else "all"
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 16


def main():
    data = load_dataset("s_cleaned")
    sample = data if SCOPE == "all" else stratified_sample(data, int(SCOPE))
    unique = {}
    n_inst = 0
    for q in sample:
        for sess in q["haystack_sessions"]:
            text = _session_text(sess)
            if not text:
                continue
            n_inst += 1
            key = hashlib.md5(text.encode()).hexdigest()[:16]
            if key not in unique and not os.path.exists(os.path.join(SESSION_CACHE_DIR, f"{key}.json")):
                unique[key] = sess
    print(f"scope={SCOPE}: {n_inst} session instances, {len(unique)} to extract (rest cached/dup)", flush=True)
    ledger = CostLedger()
    failed = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(extract_session_facts, sess, ledger): key for key, sess in unique.items()}
        for i, fut in enumerate(as_completed(futs)):
            try:
                fut.result()
            except Exception as e:
                failed.append(futs[fut])
                print(f"  FAIL {futs[fut]}: {e}", flush=True)
            if (i + 1) % 200 == 0:
                print(f"[{i+1}/{len(unique)}]", flush=True)
    ledger.save(f"/root/agent_memory_iclr/results/precompute_sessions_ledger_{SCOPE}.json")
    if failed:
        print(f"PRECOMPUTE SESSIONS DONE with {len(failed)} FAILURES: {failed[:10]}", flush=True)
        sys.exit(1)
    print("PRECOMPUTE SESSIONS DONE ok", flush=True)


if __name__ == "__main__":
    main()
