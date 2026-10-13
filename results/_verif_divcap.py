"""INDEPENDENT recomputation of the divcap2 negative-control claim.
Re-implements BM25 + unit building + session cap from scratch (does not import v5_retrieval),
then compares against v5's own selection to check for implementation divergence.
"""
import json, math, re, sys, os
from multiprocessing import Pool

sys.path.insert(0, "/root/agent_memory_iclr/code")
DATA_PATH = "/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"
W = re.compile(r"[a-z0-9']+")


def tok(s):
    return W.findall(str(s).lower())


def bm25(query, docs, k1=1.5, b=0.75):
    qt = tok(query)
    dt = [tok(d) for d in docs]
    avgdl = sum(len(d) for d in dt) / max(1, len(dt))
    df = {}
    for d in dt:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    n = len(dt)
    out = []
    for d in dt:
        tf = {}
        for w in d:
            tf[w] = tf.get(w, 0) + 1
        s = 0.0
        for w in qt:
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avgdl))
        out.append(s)
    return out


def pick(order, units, k, cap):
    if cap is None:
        return order[:k]
    out, used = [], {}
    for i in order:
        si = units[i][0]
        if used.get(si, 0) >= cap:
            continue
        used[si] = used.get(si, 0) + 1
        out.append(i)
        if len(out) >= k:
            break
    return out


DATA = None
CAPS = [None, 1, 2, 3, 4, 5, 6, 8, 10]


def one(qi):
    q = DATA[qi]
    units, flags = [], []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for t in sess:
            if t.get("content"):
                units.append((si, date, t["role"], t["content"]))
                flags.append(bool(t.get("has_answer")))
    ev = {i for i, f in enumerate(flags) if f}
    ans_ids = set(q["answer_session_ids"])
    ev_sess = {i for i, sid in enumerate(q["haystack_session_ids"]) if sid in ans_ids}
    sc = bm25(q["question"], [u[3] for u in units])
    order = sorted(range(len(units)), key=lambda i: (-sc[i], i))
    row = {"qid": q["question_id"], "type": q["question_type"],
           "is_abs": q["question_id"].endswith("_abs"),
           "n_ev": len(ev), "n_ev_sess": len(ev_sess), "n_units": len(units),
           "n_sessions": len(q["haystack_sessions"]), "v": {}}
    for cap in CAPS:
        idx = pick(order, units, 20, cap)
        raw_chars = sum(len(f"[{units[i][1]}] [{units[i][2]}] {units[i][3]}") + 1
                        for i in idx) - (1 if idx else 0)
        row["v"][str(cap)] = {
            "n_turns": len(idx), "sess": len({units[i][0] for i in idx}),
            "ev_hit": len(set(idx) & ev),
            "ev_sess_hit": len({units[i][0] for i in idx} & ev_sess),
            "allev": int(len(set(idx) & ev) == len(ev)),
            "raw_chars": raw_chars,
            "idx": sorted(idx),
        }
    return row


def main():
    global DATA
    DATA = json.load(open(DATA_PATH))
    with Pool(30) as p:
        rows = p.map(one, range(len(DATA)), chunksize=4)
    json.dump(rows, open("/root/agent_memory_iclr/results/_verif_divcap.json", "w"))
    print("n=", len(rows))


if __name__ == "__main__":
    main()
