"""Stronger check: does the anchor-day block line that matches gold actually come from a
has_answer=True evidence turn? Guards against coincidental single-token string matches."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from v5_temporal import (detect_family, resolve_question_anchors, parse_date, day0,
                         anchor_block, _sentences, _content_tokens)
from v5_temporal_measure import contains, contains_stem, norm

data = load_dataset("s_cleaned")
tr = [q for q in data if detect_family(q["question"])[0] == "anchor_lookup"]
out = []
for q in tr:
    qd = q["question_date"]
    anchors = [a for a in resolve_question_anchors(q["question"], qd) if a["conf"] == "high"]
    if not anchors:
        anchors = [a for a in resolve_question_anchors(q["question"], qd)
                   if a["kind"] in ("offset-coarse", "month-anchor")][:1]
    if not anchors:
        out.append({"qid": q["question_id"], "type": q["question_type"], "no_anchor": True})
        continue
    blk, w = anchor_block(q, anchors[0])
    # sentences that live in has_answer turns
    ev_sents, all_sents = set(), set()
    for date_s, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        for t in sess:
            for s in _sentences(t.get("content") or ""):
                k = norm(s)[:70]
                all_sents.add(k)
                if t.get("has_answer"):
                    ev_sents.add(k)
    def src(line):
        base = norm(line.replace("...", ""))[:70]
        if base in ev_sents:
            return "evidence"
        for k in ev_sents:
            if k.startswith(base[:50]) or base.startswith(k[:50]):
                return "evidence"
        return "other"
    hits = [(i + 1, x, src(x)) for i, x in enumerate(blk) if contains_stem(x, q["answer"])]
    out.append({"qid": q["question_id"], "type": q["question_type"], "w": w, "n": len(blk),
                "gold": str(q["answer"])[:80],
                "n_hits": len(hits),
                "hit_from_evidence": any(h[2] == "evidence" for h in hits),
                "first_hit_rank": hits[0][0] if hits else None,
                "first_hit_src": hits[0][2] if hits else None,
                "hit_lines": [(h[0], h[2], h[1][:120]) for h in hits[:3]],
                "n_evidence_lines_in_block": sum(1 for x in blk if src(x) == "evidence")})
json.dump(out, open("/root/agent_memory_iclr/results/_vf_anchor_prov.json", "w"),
          ensure_ascii=False, indent=1)
trr = [r for r in out if r["type"] == "temporal-reasoning"]
n = len(trr)
print("TR n =", n)
print(" stem hit:", sum(1 for r in trr if r.get("n_hits")), "/", n)
print(" hit traced to has_answer turn:", sum(1 for r in trr if r.get("hit_from_evidence")), "/", n)
print(" block contains >=1 has_answer sentence:",
      sum(1 for r in trr if r.get("n_evidence_lines_in_block")), "/", n)
for r in trr:
    if r.get("n_hits") and not r.get("hit_from_evidence"):
        print("  COINCIDENTAL?", r["qid"], "|gold:", r["gold"], "|line:", r["hit_lines"][:1])
nt = [r for r in out if r["type"] != "temporal-reasoning"]
print("non-TR n =", len(nt),
      "block has evidence line:", sum(1 for r in nt if r.get("n_evidence_lines_in_block")),
      "stem hit:", sum(1 for r in nt if r.get("n_hits")))
