"""Independent adversarial recomputation. Cache-only: never calls the API."""
import hashlib, json, os, re, sys
sys.path.insert(0, "/root/agent_memory_iclr/code")
import lifecycle as LC
from run_baseline import bm25_rank
from query_classifier import classify_query

RES = "/root/agent_memory_iclr/results"
SESS = os.path.join(RES, "lifecycle_cache/sessions_nt")
MISS = []


def cached_extract(session_turns, ledger=None):
    text = LC._session_text(session_turns)
    if not text:
        return []
    key = hashlib.md5(text.encode()).hexdigest()[:16]
    p = os.path.join(SESS, f"{key}.json")
    if not os.path.exists(p):
        MISS.append(key)
        return []
    with open(p) as f:
        return json.load(f)


LC.extract_session_facts = cached_extract  # hard block on any API call


def load():
    with open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json") as f:
        return json.load(f)


def arms():
    out = {}
    for tag in ["qc", "qc_view_only", "qc_no_lifecycle", "qc_no_qc", "full", "full_p2", "fifo", "decay"]:
        rows = [json.loads(l) for l in open(f"{RES}/full500_{tag}.jsonl")]
        out[tag] = {r["qid"]: r for r in rows}
    return out


def units_of(q):
    out = []
    for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                out.append((date, t["role"], t["content"]))
    return out


VIEW_CAP = 20000
TOP_K = 20


def build_ctx(q, cap_lines=True):
    """Reproduce run_qc.py context exactly: (view_txt_capped, view_txt_full, raw_txt, raw_units)."""
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    store = LC.build_lifecycle(q, None)
    full = LC.render_view(store, mode)
    view = full
    if len(view) > VIEW_CAP:
        kept, total = [], 0
        for line in view.splitlines():
            if total + len(line) + 1 > VIEW_CAP:
                break
            kept.append(line)
            total += len(line) + 1
        view = "\n".join(kept)
    units = units_of(q)
    order = sorted(bm25_rank(q["question"], [u[2] for u in units])[:TOP_K])
    raw_units = [units[i] for i in order]
    raw = "\n".join(f"[{u[0]}] [{u[1]}] {u[2]}" for u in raw_units)
    return {"mode": mode, "label": label, "view": view, "view_full": full, "raw": raw,
            "raw_units": raw_units}


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
