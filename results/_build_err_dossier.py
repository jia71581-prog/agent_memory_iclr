"""Build an offline dossier for every error row in the qc and full arms.

No API calls. Reconstructs, per question:
  - question / gold / type / date
  - each arm's stored prediction
  - the exact retrieved raw-turn context (BM25, deterministic) and gold-evidence coverage
  - whether the gold answer string is literally present in the retrieved context / lifecycle view
  - the lifecycle chains relevant to the question, rendered both current and historical
  - the gold evidence turns (has_answer=True) verbatim (truncated)
"""
import json
import os
import re
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
sys.path.insert(0, "/root/agent_memory_iclr/results")
import lifecycle as LC
import _recon_ctx as RC
from run_baseline import bm25_rank

RES = "/root/agent_memory_iclr/results"
STOP = set("""a an the is are was were be been being do does did have has had of in on at to for with and or but if then
than that this these those i me my we our you your he she it they them his her their there here what which who whom when
where why how not no yes about after before during into over under again further once all any both each few more most
other some such only own same so too very can will just should now would could may might must does don't didn't im i'm""".split())


def toks(s):
    return [w for w in re.findall(r"[a-z0-9']+", str(s if s is not None else "").lower()) if w not in STOP and len(w) > 2]


def load(fn):
    return [json.loads(l) for l in open(os.path.join(RES, fn))]


def main():
    qc = {r["qid"]: r for r in load("full500_qc.jsonl")}
    fu = {r["qid"]: r for r in load("full500_full.jsonl")}
    qc_err = {k for k, r in qc.items() if not r["judge"]}
    fu_err = {k for k, r in fu.items() if not r["judge"]}
    need = qc_err | fu_err
    print("qc_err", len(qc_err), "fu_err", len(fu_err), "union", len(need), flush=True)

    data = LC.json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
    print("dataset loaded", len(data), flush=True)
    byq = {q["question_id"]: q for q in data}

    out = {}
    for n, qid in enumerate(sorted(need)):
        q = byq[qid]
        units = RC.units_of(q)
        ev = RC.evidence_turns(q)
        store = RC.build_store(q)
        view_cur = LC.render_view(store, "current")
        view_hist = LC.render_view(store, "historical")

        raw20 = sorted(bm25_rank(q["question"], [u[2] for u in units])[:20])
        raw30 = sorted(bm25_rank(q["question"], [u[2] for u in units])[:30])
        ctx20 = "\n".join(units[i][2] for i in raw20)
        ctx30 = "\n".join(units[i][2] for i in raw30)
        cov20 = sum(1 for e in ev if e in {units[i][2] for i in raw20})
        cov30 = sum(1 for e in ev if e in {units[i][2] for i in raw30})

        gtk = set(toks(q["answer"]))
        def frac_in(text):
            if not gtk:
                return None
            tt = set(toks(text))
            return round(len(gtk & tt) / len(gtk), 2)

        # relevant chains: share >=1 distinctive token with question or gold
        keytk = set(toks(q["question"])) | gtk
        rel_cur, rel_hist = [], []
        for line in view_cur.splitlines():
            if keytk & set(toks(line)):
                rel_cur.append(line)
        for line in view_hist.splitlines():
            if keytk & set(toks(line)):
                rel_hist.append(line)

        out[qid] = {
            "type": q["question_type"],
            "qdate": q["question_date"],
            "question": q["question"],
            "gold": str(q["answer"]),
            "n_units": len(units),
            "n_sessions": len(q["haystack_sessions"]),
            "n_ev": len(ev),
            "qc": None if qid not in qc_err else {
                "qlabel": qc[qid]["qlabel"], "mode": qc[qid]["view_mode"],
                "f1": qc[qid]["f1"], "pred": qc[qid]["pred"],
                "cov": f"{cov20}/{len(ev)}",
                "gold_tok_in_raw20": frac_in(ctx20),
                "gold_tok_in_view": frac_in(view_cur if qc[qid]["view_mode"] == "current" else view_hist),
            },
            "full": None if qid not in fu_err else {
                "f1": fu[qid]["f1"], "pred": fu[qid]["pred"],
                "cov": f"{cov30}/{len(ev)}",
                "gold_tok_in_raw30": frac_in(ctx30),
            },
            "qc_ok": qid in qc and qc[qid]["judge"],
            "full_ok": qid in fu and fu[qid]["judge"],
            "rel_view_current": rel_cur[:40],
            "rel_view_hist": rel_hist[:40],
            "evidence": [e[:900] for e in ev],
            "n_chains": len(store),
        }
        if (n + 1) % 10 == 0:
            print(n + 1, "done", flush=True)

    with open(os.path.join(RES, "_err_dossier.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("cache_miss", RC.CACHE_MISS[0])
    print("saved", len(out))


if __name__ == "__main__":
    main()
