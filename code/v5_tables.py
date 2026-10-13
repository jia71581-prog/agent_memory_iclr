"""v5 offline result tables. Reads results/prov_v5_offline.json (+ coverage_diag.json for the
deployed-arm judge labels and the published in_view/in_raw20 cross-check)."""
import json
import sys
from collections import Counter

R = "/root/agent_memory_iclr/results"
rows = json.load(open(f"{R}/prov_v5_offline.json"))
cov = {r["qid"]: r for r in json.load(open(f"{R}/coverage_diag.json"))}
V = ["baseline", "rankview", "divcap2", "prov_hybrid6", "prov_hybrid", "prov_only", "v5",
     "v5_divcap", "v5_deep", "v5_wide", "v5_noanchor", "v5_nocount", "v5_focus", "v5_bm",
     "bm25_k30", "bm25_k50", "realloc", "realloc_bm", "realloc_bm_a",
     "realloc8k", "realloc8k_bm", "v5_final", "v5f_bm"]
NAB = [r for r in rows if not r["is_abs"]]
TYPES = ["single-session-user", "single-session-assistant", "single-session-preference",
         "multi-session", "knowledge-update", "temporal-reasoning"]
SHORT = {"single-session-user": "SSU", "single-session-assistant": "SSA",
         "single-session-preference": "SSP", "multi-session": "MS",
         "knowledge-update": "KU", "temporal-reasoning": "TR"}


def f(g, name, k):
    return sum(r["v"][name][k] for r in g) / max(1, len(g))


def micro(g, name, k, tot):
    d = sum(r[tot] for r in g)
    return sum(r["v"][name][k] for r in g) / d if d else float("nan")


print("== 0. sanity: does variant 'baseline' reproduce the deployed arm? ==")
mism_raw = [r["qid"] for r in NAB if r["v"]["baseline"]["in_raw"] != cov[r["qid"]]["in_raw20"]]
print(f"in_raw vs coverage_diag.in_raw20 mismatches: {len(mism_raw)}/470")
print(f"baseline in_raw  {f(NAB,'baseline','in_raw'):.4f}  (coverage_diag in_raw20 "
      f"{sum(cov[r['qid']]['in_raw20'] for r in NAB)/len(NAB):.4f})")
print(f"baseline in_view (20k cap, as deployed) {f(NAB,'baseline','in_view'):.4f}  vs "
      f"coverage_diag in_view (NO cap) {sum(cov[r['qid']]['in_view'] for r in NAB)/len(NAB):.4f}")
print(f"view lines lost to the cap: mean kept chars {f(NAB,'baseline','view_chars'):.0f}")

print("\n== 1. gold-string coverage, non-abstention n=470 ==")
hdr = f"{'variant':13s} {'inView':>7s} {'inRaw':>7s} {'inCtx':>7s} " + " ".join(f"{SHORT[t]:>6s}" for t in TYPES)
print(hdr)
for name in V:
    per = []
    for t in TYPES:
        g = [r for r in NAB if r["type"] == t]
        per.append(f"{f(g,name,'in_ctx'):6.3f}")
    print(f"{name:13s} {f(NAB,name,'in_view'):7.4f} {f(NAB,name,'in_raw'):7.4f} "
          f"{f(NAB,name,'in_ctx'):7.4f} " + " ".join(per))

print("\n== 2. oracle evidence recall (has_answer turns; answer_session_ids), n=470 ==")
print(f"{'variant':13s} {'evTurnRec':>9s} {'allEvTurns':>11s} {'evSessRec':>9s} {'turnSess':>8s} "
      + " ".join(f"{SHORT[t]:>6s}" for t in TYPES) + "   <- allEvTurns per type")
for name in V:
    per = []
    for t in TYPES:
        g = [r for r in NAB if r["type"] == t]
        per.append(f"{sum(r['v'][name]['ev_turns_hit']==r['n_ev_turns'] for r in g)/len(g):6.3f}")
    print(f"{name:13s} {micro(NAB,name,'ev_turns_hit','n_ev_turns'):9.4f} "
          f"{sum(r['v'][name]['ev_turns_hit']==r['n_ev_turns'] for r in NAB)/len(NAB):11.4f} "
          f"{micro(NAB,name,'ev_sess_hit','n_ev_sessions'):9.4f} "
          f"{f(NAB,name,'n_turn_sessions'):8.2f} " + " ".join(per))

print("\n== 3. context size (mean per question; est tokens = chars/4) ==")
print(f"{'variant':13s} {'viewChars':>9s} {'rawChars':>9s} {'totChars':>9s} {'estTok':>7s} {'nTurns':>6s} {'vs base':>8s}")
base_tot = f(NAB, "baseline", "view_chars") + f(NAB, "baseline", "raw_chars")
for name in V:
    tot = f(NAB, name, "view_chars") + f(NAB, name, "raw_chars")
    print(f"{name:13s} {f(NAB,name,'view_chars'):9.0f} {f(NAB,name,'raw_chars'):9.0f} {tot:9.0f} "
          f"{tot/4:7.0f} {f(NAB,name,'n_turns'):6.2f} {tot/base_tot:8.3f}")

print("\n== 4. the deployed arm's error sets ==")
errs = [r for r in NAB if not cov[r["qid"]]["judge"]]
A = [r for r in errs if not (cov[r["qid"]]["in_view"] or cov[r["qid"]]["in_raw20"])]
B = [r for r in errs if cov[r["qid"]]["in_view"] or cov[r["qid"]]["in_raw20"]]
print(f"errors {len(errs)}; A evidence-missing (published defn, UNCAPPED view) {len(A)}; B present {len(B)}")
Ac = [r for r in errs if not r["v"]["baseline"]["in_ctx"]]
print(f"A' evidence-missing under the ACTUALLY DEPLOYED context (capped view) {len(Ac)}")
for label, S in (("A (published, n=%d)" % len(A), A), ("A' (deployed-context, n=%d)" % len(Ac), Ac)):
    print(f"\n  -- {label}: gold now in context --")
    for name in V:
        rec = sum(r["v"][name]["in_ctx"] for r in S)
        evc = sum(r["v"][name]["ev_turns_hit"] == r["n_ev_turns"] for r in S)
        print(f"   {name:13s} in_ctx {rec:3d}/{len(S)}  ({rec/max(1,len(S)):.3f})   allEvTurns {evc:3d}/{len(S)}")
print("\n  -- B (evidence present, wrong answer, n=%d): oracle completeness --" % len(B))
for name in V:
    evc = sum(r["v"][name]["ev_turns_hit"] == r["n_ev_turns"] for r in B)
    print(f"   {name:13s} allEvTurns {evc:3d}/{len(B)} ({evc/len(B):.3f})  "
          f"evTurnRec {micro(B,name,'ev_turns_hit','n_ev_turns'):.3f}")
print("  A types:", dict(Counter(r["type"] for r in A)), "\n  B types:", dict(Counter(r["type"] for r in B)))

print("\n== 5. aggregation completeness, counting questions ==")
C = [r for r in NAB if r["is_count"]]
CN = [r for r in C if r["gold_num"] is not None]
print(f"counting-style questions (non-abs): {len(C)}; with parseable numeric gold: {len(CN)}")
print(f"  by type: {dict(Counter(SHORT[r['type']] for r in C))}")
CMS = [r for r in CN if r["type"] == "multi-session"]
print(f"\n{'variant':13s} {'proxy>=gold':>11s} {'MSonly':>8s} {'meanProxy':>9s} {'allEvTurns':>11s} {'MSallEv':>8s}")
for name in V:
    ok = sum(r["v"][name]["agg"] >= r["gold_num"] for r in CN) / len(CN)
    okms = sum(r["v"][name]["agg"] >= r["gold_num"] for r in CMS) / max(1, len(CMS))
    print(f"{name:13s} {ok:11.3f} {okms:8.3f} {f(CN,name,'agg'):9.2f} "
          f"{sum(r['v'][name]['ev_turns_hit']==r['n_ev_turns'] for r in CN)/len(CN):11.3f} "
          f"{sum(r['v'][name]['ev_turns_hit']==r['n_ev_turns'] for r in CMS)/max(1,len(CMS)):8.3f}")
# restrict to the small-integer subset where the proxy is meaningful (gold<=10 items)
CS = [r for r in CN if r["gold_num"] <= 10]
print(f"\nsmall-count subset (numeric gold <= 10), n={len(CS)}  [proxy is session-granular, so it "
      f"cannot represent large numeric golds like 140 hours]")
for name in V:
    print(f"{name:13s} proxy>=gold {sum(r['v'][name]['agg']>=r['gold_num'] for r in CS)/len(CS):.3f}  "
          f"meanProxy {f(CS,name,'agg'):.2f}  allEvTurns {sum(r['v'][name]['ev_turns_hit']==r['n_ev_turns'] for r in CS)/len(CS):.3f}")

print("\n== 6. abstention questions n=30 (ours is the worst arm there; more evidence may hurt) ==")
AB = [r for r in rows if r["is_abs"]]
for name in V:
    print(f"{name:13s} view_chars {f(AB,name,'view_chars'):6.0f} raw_chars {f(AB,name,'raw_chars'):6.0f} "
          f"in_ctx(gold-string, meaningless for abs) {f(AB,name,'in_ctx'):.3f} "
          f"turnSess {f(AB,name,'n_turn_sessions'):.2f}")

print("\n== 7. per-question deltas: v5 vs baseline (non-abs) ==")
for name in ["rankview", "divcap2", "v5", "v5_bm"]:
    gain = [r["qid"] for r in NAB if r["v"][name]["in_ctx"] and not r["v"]["baseline"]["in_ctx"]]
    loss = [r["qid"] for r in NAB if r["v"]["baseline"]["in_ctx"] and not r["v"][name]["in_ctx"]]
    eg = [r["qid"] for r in NAB if (r["v"][name]["ev_turns_hit"] == r["n_ev_turns"]) and not (r["v"]["baseline"]["ev_turns_hit"] == r["n_ev_turns"])]
    el = [r["qid"] for r in NAB if (r["v"]["baseline"]["ev_turns_hit"] == r["n_ev_turns"]) and not (r["v"][name]["ev_turns_hit"] == r["n_ev_turns"])]
    print(f"{name:11s} gold-string  +{len(gain):3d} -{len(loss):3d}   allEvTurns +{len(eg):3d} -{len(el):3d}")
    if name == "v5":
        print("            evidence-set losses:", dict(Counter(next(x['type'] for x in NAB if x['qid']==qq) for qq in el)))
