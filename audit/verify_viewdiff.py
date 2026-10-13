"""INDEPENDENT recomputation of the 'views really differ' claim. No trust in audit/*.json.

Blocks the network at the socket layer AND monkeypatches common.chat, so any cache miss
is a hard failure rather than a silent API call.
"""
import json, os, sys, re, socket, statistics as st, math
from itertools import zip_longest

class _Blocked(RuntimeError): pass
def _no_net(*a, **k): raise _Blocked("NETWORK BLOCKED by verifier")

sys.path.insert(0, "/root/agent_memory_iclr/code")
import common
# --- hard network block (after imports so ssl/http can still initialise) -------
socket.socket.connect = _no_net
socket.socket.connect_ex = _no_net
socket.create_connection = _no_net
_api = {"n": 0}
def _no_api(*a, **k):
    _api["n"] += 1
    raise _Blocked("CACHE MISS -> would need API")
common.chat = _no_api
import lifecycle
lifecycle.chat = _no_api
from lifecycle import build_lifecycle, render_view, _group_by_session
from query_classifier import classify_query

R = "/root/agent_memory_iclr/results/"

def load_arm(tag):
    """last row per qid that has a judge (mirrors runner's own dedup)."""
    latest = {}
    with open(R + f"full500_{tag}.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "judge" in r:
                latest[r["qid"]] = r
    return latest

qc, nq = load_arm("qc"), load_arm("qc_no_qc")
print("rows with judge: qc=%d no_qc=%d  shared qids=%d" % (len(qc), len(nq), len(set(qc) & set(nq))))

with open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json") as f:
    data = json.load(f)
byid = {q["question_id"]: q for q in data}
print("dataset n=%d  all qc qids present=%s" % (len(data), all(k in byid for k in qc)))

CAP = 20000  # run_qc.VIEW_CHAR_CAP

def cap(txt):
    """exact replica of run_qc.answer_question truncation."""
    if len(txt) <= CAP:
        return txt
    kept, tot = [], 0
    for line in txt.splitlines():
        if tot + len(line) + 1 > CAP:
            break
        kept.append(line)
        tot += len(line) + 1
    return "\n".join(kept)

_alnum = re.compile(r"[a-z0-9]+")
def toks(s):
    return sorted(_alnum.findall(s.lower()))

rows = []
miss = 0
for i, qid in enumerate(sorted(qc)):
    q = byid[qid]
    try:
        store = build_lifecycle(q, None)
    except _Blocked:
        miss += 1
        continue
    cur_full = render_view(store, "current")
    hist_full = render_view(store, "historical")
    cur, hist = cap(cur_full), cap(hist_full)

    # verify the recorded routing against the deterministic classifier
    label, _ = classify_query(q["question"])
    mode_expect_qc = "historical" if label == "historical" else "current"
    mode_qc, mode_nq = qc[qid]["view_mode"], nq[qid]["view_mode"]

    v_qc = cur if mode_qc == "current" else hist
    v_nq = cur if mode_nq == "current" else hist

    cl, hl = cur.splitlines(), hist.splitlines()
    keys = sorted(store)
    ngroups = {k: len(_group_by_session(store[k])) for k in keys}

    # line-level diff (both views emit exactly one line per chain in sorted(store) order,
    # so index i == chain keys[i]; use zip_longest so tail-only differences also count)
    n_diff = 0
    n_diff_cosmetic = 0   # differs but identical token multiset (separator-only, e.g. '; ' vs ' / ')
    n_diff_real = 0       # differs in actual tokens
    n_diff_multi = 0      # differing line belongs to a multi-session-group chain
    n_diff_single = 0
    for j, (a, b) in enumerate(zip_longest(cl, hl)):
        if a == b:
            continue
        n_diff += 1
        if a is not None and b is not None and toks(a) == toks(b):
            n_diff_cosmetic += 1
        else:
            n_diff_real += 1
        k = keys[j] if j < len(keys) else None
        if k is not None and ngroups[k] > 1:
            n_diff_multi += 1
        else:
            n_diff_single += 1
    n_prefix_diff = sum(1 for a, b in zip(cl, hl) if a != b)  # a6_cap.py's definition

    rows.append(dict(
        qid=qid, type=qc[qid]["type"], qlabel=qc[qid]["qlabel"], label_recomputed=label,
        mode_qc=mode_qc, mode_nq=mode_nq, mode_qc_expected=mode_expect_qc,
        routed_diff=mode_qc != mode_nq,
        same_view=v_qc == v_nq,
        len_qc=len(v_qc), len_nq=len(v_nq), len_cur=len(cur), len_hist=len(hist),
        full_len_cur=len(cur_full), full_len_hist=len(hist_full),
        n_chains=len(keys), kept_cur=len(cl), kept_hist=len(hl),
        n_multi=sum(1 for k in keys if ngroups[k] > 1),
        n_conflict=sum(1 for k in keys if any(c["relation"] == "contradiction" for c in store[k])),
        n_diff=n_diff, n_diff_prefix=n_prefix_diff,
        n_diff_cosmetic=n_diff_cosmetic, n_diff_real=n_diff_real,
        n_diff_multi=n_diff_multi, n_diff_single=n_diff_single,
        judge_qc=qc[qid]["judge"], judge_nq=nq[qid]["judge"], f1_qc=qc[qid]["f1"], f1_nq=nq[qid]["f1"],
    ))
    if (i + 1) % 100 == 0:
        print("  ...", i + 1, flush=True)

print("cache misses:", miss, " api attempts:", _api["n"], " rebuilt:", len(rows))
json.dump(rows, open("/root/agent_memory_iclr/audit/verify_viewdiff.json", "w"))
