"""Final consolidated adversarial checks."""
import re
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
from v5_temporal import (extract_events, detect_family, precompute_arithmetic, render_timeline,
                         _stem, _content_tokens, match_item_date, question_alternatives)
from v5_temporal_val import build_store_cached, data_by_id

d = data_by_id()
ids = ["982b5123_abs", "c8090214_abs", "gpt4_70e84552_abs", "gpt4_93159ced_abs",
       "gpt4_c27434e8_abs", "gpt4_fe651585_abs"]

print("=" * 100)
print("CHECK 1  per-question anatomy: is the SCAFFOLD empty, or only the ARITHMETIC block?")
print("=" * 100)
print(f"{'qid':22s} {'family':14s} {'#arith':>7s} {'#tl_rows':>9s} {'chars':>7s} {'ordering_block':>15s}")
for qid in ids:
    q = d[qid]
    store, _ = build_store_cached(q)
    fam, unit = detect_family(q["question"])
    ev = extract_events(q, store, q["question"])
    ar = precompute_arithmetic(q["question"], q["question_date"], ev, fam, unit, q_for_anchor=q)
    tl = render_timeline(q, store, q["question"])
    nrows = sum(1 for L in tl.splitlines() if L.startswith("["))
    ordb = any("Chronological order" in L for L in ar)
    print(f"{qid:22s} {fam:14s} {len(ar):7d} {nrows:9d} {len(tl):7d} {str(ordb):>15s}")

print()
print("=" * 100)
print("CHECK 2  gpt4_93159ced_abs pairwise delta table: how many rows exactly?")
print("=" * 100)
q = d["gpt4_93159ced_abs"]
store, _ = build_store_cached(q)
fam, unit = detect_family(q["question"])
ev = extract_events(q, store, q["question"])
ar = precompute_arithmetic(q["question"], q["question_date"], ev, fam, unit, q_for_anchor=q)
pw = [L for L in ar if re.match(r"^\s+\d{4}/\d{2}/\d{2} -> ", L)]
print(f"pairwise rows = {len(pw)}   (claim says 9)")
also = [L for L in ar if "also, from each of those dates" in L]
print(f"plus compact 'also' line: {len(also)}")
if also:
    print("  ", also[0].strip())
print("  NOTE: the compact line drops the YEAR -> two distinct dates both print as '05/28'"
      " with different day counts.")

print()
print("=" * 100)
print("CHECK 3  gpt4_70e84552_abs: does the scaffold really withhold the un-mentioned entity?")
print("=" * 100)
q = d["gpt4_70e84552_abs"]
store, _ = build_store_cached(q)
tl = render_timeline(q, store, q["question"])
for L in tl.splitlines():
    if re.search(r"\bcow", L, re.I):
        print("  DATED ROW MENTIONING A COW:", L.strip())
ev = extract_events(q, store, q["question"])
print("  match_item_date('fixing the fence') ->", match_item_date("fixing the fence", ev))
print("  tokens('fixing the fence') =", _content_tokens("fixing the fence"))
print("  tokens of the fence event  =", sorted(set(_content_tokens(
    [e.text for e in ev if "fence" in e.text.lower()][0])))[:14])
print(f"  _stem('fixing')={_stem('fixing')!r}  _stem('fixed')={_stem('fixed')!r}"
      "   -> never match; this is why the note flags an item that IS in the timeline")

print()
print("=" * 100)
print("CHECK 4  982b5123_abs: does the 'empty' scaffold supply a derivable wrong answer?")
print("=" * 100)
q = d["982b5123_abs"]
store, _ = build_store_cached(q)
tl = render_timeline(q, store, q["question"])
for L in tl.splitlines():
    if re.search(r"airbnb", L, re.I):
        print("  ", L.strip()[:190])

print()
print("=" * 100)
print("CHECK 5  c8090214_abs: are the Holiday Market rows correctly dated?")
print("=" * 100)
q = d["c8090214_abs"]
store, _ = build_store_cached(q)
tl = render_timeline(q, store, q["question"])
for L in tl.splitlines():
    if L.startswith("["):
        print("  ", L.strip()[:160])
print("  ground truth in the text: 'a week before Black Friday' said in a 2023/05/30 session")
print("  -> true event date ~2022/11/18; scaffold asserts 2023/05/30 = question date. WRONG DATE.")
print("  'Black Friday' is absent from holiday_dates():",
      "black friday" not in __import__("v5_temporal").holiday_dates(2023))
