"""Final offline report for v5 provenance-guided retrieval. No API.
Reads results/prov_v5_offline.json (+ coverage_diag.json for the deployed arm's judge labels)."""
import json
import math

R = "/root/agent_memory_iclr/results"
off = json.load(open(f"{R}/prov_v5_offline.json"))
cov = {r["qid"]: r for r in json.load(open(f"{R}/coverage_diag.json"))}
NAB = [r for r in off if not r["is_abs"]]
AB = [r for r in off if r["is_abs"]]
TYPES = ["single-session-user", "single-session-assistant", "single-session-preference",
         "multi-session", "knowledge-update", "temporal-reasoning"]
SH = {"single-session-user": "SSU", "single-session-assistant": "SSA",
      "single-session-preference": "SSP", "multi-session": "MS",
      "knowledge-update": "KU", "temporal-reasoning": "TR"}
MAIN = ["baseline", "rankview", "v5_drop_in", "v5_realloc", "v5_realloc_m",
        "final8k_bm_m", "divcap2", "bm25_k30", "bm25_k50"]
LBL = {"baseline": "baseline (deployed)", "rankview": "+ranked view only",
       "v5_drop_in": "v5 drop-in (20 turns)", "v5_realloc": "v5 realloc (8k view, 31 turns)",
       "v5_realloc_m": "v5 realloc, char-matched", "final8k_bm_m": "realloc CONTROL (no prov)",
       "divcap2": "diversity-only CONTROL", "bm25_k30": "BM25 k=30 (1.27x budget)",
       "bm25_k50": "BM25 k=50 (1.80x budget)"}


def mc(pairs):
    b = sum(1 for x, y in pairs if y and not x)
    c = sum(1 for x, y in pairs if x and not y)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    return b, c, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def g(n, k, G=NAB):
    return sum(r["v"][n][k] for r in G) / len(G)


def allev(n, G=NAB):
    return sum(r["v"][n]["ev_turns_hit"] == r["n_ev_turns"] for r in G) / len(G)


def evrec(n, G=NAB):
    return sum(r["v"][n]["ev_turns_hit"] for r in G) / sum(r["n_ev_turns"] for r in G)


bt = g("baseline", "view_chars") + g("baseline", "raw_chars")

print("=" * 108)
print("T1  CONTEXT BUDGET (mean per question, non-abstention n=470; est tokens = chars/4)")
print("=" * 108)
print(f"{'variant':28s} {'view':>7s} {'raw':>7s} {'total':>7s} {'estTok':>7s} {'turns':>6s} {'x base':>7s}")
for n in MAIN:
    t = g(n, "view_chars") + g(n, "raw_chars")
    print(f"{LBL[n]:28s} {g(n,'view_chars'):7.0f} {g(n,'raw_chars'):7.0f} {t:7.0f} {t/4:7.0f} "
          f"{g(n,'n_turns'):6.2f} {t/bt:7.3f}")

print("\n" + "=" * 108)
print("T2  GOLD-STRING COVERAGE (coverage_diag.contains; raw text content-only = comparable to")
print("    coverage_diag.json). n=470.  view coverage is measured on the CAPPED view actually sent.")
print("=" * 108)
print(f"{'variant':28s} {'inView':>7s} {'inRaw':>7s} {'inCtx':>7s} " + " ".join(f"{SH[t]:>6s}" for t in TYPES))
for n in MAIN:
    per = [f"{g(n,'in_ctx_c',[r for r in NAB if r['type']==t]):6.3f}" for t in TYPES]
    print(f"{LBL[n]:28s} {g(n,'in_view'):7.4f} {g(n,'in_raw_c'):7.4f} {g(n,'in_ctx_c'):7.4f} "
          + " ".join(per))
print(f"\n  reference: un-capped view coverage (what coverage_diag.py reported) = "
      f"{sum(cov[r['qid']]['in_view'] for r in NAB)/len(NAB):.4f}; whole-haystack ceiling = "
      f"{sum(cov[r['qid']]['in_hay'] for r in NAB)/len(NAB):.4f}")

print("\n" + "=" * 108)
print("T3  ORACLE EVIDENCE COMPLETENESS (dataset has_answer turns + answer_session_ids). n=470.")
print("    allEv = every has_answer turn of the question is inside the raw context.")
print("=" * 108)
print(f"{'variant':28s} {'evRec':>6s} {'allEv':>6s} {'sessRec':>7s} {'nSess':>6s} "
      + " ".join(f"{SH[t]:>6s}" for t in TYPES) + "   McNemar vs baseline")
for n in MAIN:
    per = [f"{allev(n,[r for r in NAB if r['type']==t]):6.3f}" for t in TYPES]
    b, c, p = mc([(r["v"]["baseline"]["ev_turns_hit"] == r["n_ev_turns"],
                   r["v"][n]["ev_turns_hit"] == r["n_ev_turns"]) for r in NAB])
    sr = sum(r["v"][n]["ev_sess_hit"] for r in NAB) / sum(r["n_ev_sessions"] for r in NAB)
    print(f"{LBL[n]:28s} {evrec(n):6.3f} {allev(n):6.3f} {sr:7.3f} {g(n,'n_turn_sessions'):6.2f} "
          + " ".join(per) + f"   +{b} -{c} p={p:.2e}")

print("\n  isolating the provenance component (paired, same turn count, same char budget):")
for a, b_ in [("v5_drop_in", "rankview"), ("v5_realloc", "final8k_bm"), ("v5_realloc_m", "final8k_bm_m")]:
    bb, cc, p = mc([(r["v"][b_]["ev_turns_hit"] == r["n_ev_turns"],
                     r["v"][a]["ev_turns_hit"] == r["n_ev_turns"]) for r in NAB])
    print(f"    {a:13s} vs {b_:14s} allEv {allev(b_):.3f} -> {allev(a):.3f}  +{bb} -{cc} p={p:.4f}")
print("  ablating the value-anchoring inside a provenance session (p6 family, 20 turns):")
bb, cc, p = mc([(r["v"]["p6_a0"]["ev_turns_hit"] == r["n_ev_turns"],
                 r["v"]["p6"]["ev_turns_hit"] == r["n_ev_turns"]) for r in NAB])
print(f"    anchor_weight 0 -> 1: allEv {allev('p6_a0'):.3f} -> {allev('p6'):.3f}  +{bb} -{cc} p={p:.4f}")

print("\n" + "=" * 108)
print("T4  THE 62 NON-ABSTENTION ERRORS OF THE DEPLOYED ARM")
print("=" * 108)
errs = [r for r in NAB if not cov[r["qid"]]["judge"]]
A = [r for r in errs if not (cov[r["qid"]]["in_view"] or cov[r["qid"]]["in_raw20"])]
B = [r for r in errs if cov[r["qid"]]["in_view"] or cov[r["qid"]]["in_raw20"]]
Ad = [r for r in errs if not r["v"]["baseline"]["in_ctx_c"]]
print(f"A  evidence-missing, published definition (un-capped view): {len(A)}")
print(f"A' evidence-missing in the context actually sent (capped view): {len(Ad)}")
print(f"B  evidence present but answered wrong: {len(B)}")
print(f"\n{'variant':28s} {'A:goldInCtx':>11s} {'A:allEv':>8s} {'A2:goldInCtx':>12s} {'B:allEv':>8s} {'B:evRec':>8s}")
for n in MAIN:
    print(f"{LBL[n]:28s} {sum(r['v'][n]['in_ctx_c'] for r in A):5d}/{len(A):<5d} "
          f"{sum(r['v'][n]['ev_turns_hit']==r['n_ev_turns'] for r in A):4d}/{len(A):<3d} "
          f"{sum(r['v'][n]['in_ctx_c'] for r in Ad):6d}/{len(Ad):<5d} "
          f"{sum(r['v'][n]['ev_turns_hit']==r['n_ev_turns'] for r in B):4d}/{len(B):<3d} "
          f"{evrec(n,B):8.3f}")

print("\n" + "=" * 108)
print("T5  AGGREGATION COMPLETENESS on counting questions (/how many|how much|total|number of/i)")
print("=" * 108)
C = [r for r in NAB if r["is_count"] and r["gold_num"] is not None]
CS = [r for r in C if r["gold_num"] <= 10]
CMS = [r for r in C if r["type"] == "multi-session"]
print(f"n counting = {len([r for r in NAB if r['is_count']])} (206 with a parseable numeric gold; "
      f"{len(CS)} with gold <= 10; {len(CMS)} multi-session)")
print(f"\n{'variant':28s} {'sessProxy':>9s} {'small<=10':>9s} {'chainProxy':>10s} {'allEv':>6s} {'MSallEv':>7s}")
for n in MAIN:
    print(f"{LBL[n]:28s} "
          f"{sum(r['v'][n]['agg']>=r['gold_num'] for r in C)/len(C):9.3f} "
          f"{sum(r['v'][n]['agg']>=r['gold_num'] for r in CS)/len(CS):9.3f} "
          f"{sum(r['v'][n]['chain_agg_in_view']>=r['gold_num'] for r in C)/len(C):10.3f} "
          f"{allev(n,C):6.3f} {allev(n,CMS):7.3f}")
print("\n  The session-granular proxy is GAMEABLE and must not be used as a headline: the")
print("  diversity-only control raises it (0.417 -> 0.451) while destroying real evidence")
print("  completeness (allEv 0.745 -> 0.587). Trust the oracle has_answer columns.")

print("\n" + "=" * 108)
print("T6  ABSTENTION (n=30, reported separately; ours is already the worst arm at 0.833)")
print("=" * 108)
print(f"{'variant':28s} {'viewChars':>9s} {'rawChars':>9s} {'turns':>6s} {'turnSess':>8s} {'total':>7s}")
for n in MAIN:
    print(f"{LBL[n]:28s} {g(n,'view_chars',AB):9.0f} {g(n,'raw_chars',AB):9.0f} "
          f"{g(n,'n_turns',AB):6.2f} {g(n,'n_turn_sessions',AB):8.2f} "
          f"{g(n,'view_chars',AB)+g(n,'raw_chars',AB):7.0f}")
print("  No offline metric can settle abstention: the failure is the answerer volunteering an")
print("  answer, not evidence coverage. v5_drop_in changes the abstention context least")
print("  (same size, +0.4 sessions); v5_realloc enlarges the raw block by 43% and touches 3.8")
print("  more sessions, which is the direction that CAUSED the abstention deficit. Untested.")
