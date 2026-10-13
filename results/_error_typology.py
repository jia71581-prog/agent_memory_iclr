"""Manual error typology for the qc and full arms (no LLM calls).

Every label below was assigned by reading question + gold + prediction + gold-evidence
turns, and (for every stale-fact / conflict-confusion / over-abstention assignment) by
checking with _recon_ctx whether the competing value was literally present in that arm's
reconstructed context.  Categories follow 12_qc_lifecycle_plan.md section 2 item 3.

  S   stale-fact          answer asserts a superseded value as current
  H   history-loss        past-state query on a superseded attribute answered with the
                          current value only, or with "I don't know"
  C   conflict-confusion  several values for one attribute coexist (multi-valued, or a
                          reliability conflict) and the answer picks wrongly / drops a
                          still-valid one / cannot resolve
  o1  retrieval-miss      required value never reached this arm's context
  o2a over-abstention     required value WAS in context, answer refused or went generic
  o2b under-abstention    gold is "not enough information", answer supplied a value
  o3  aggregation         all needed values in context; counting / summing / date error
  o4  grading-boundary    answer substantively right (or gold itself unsupported)
  o5  extraction-miss     empty prediction
"""
import json
import os
from collections import Counter, defaultdict

RES = "/root/agent_memory_iclr/results"

QC = {
    "031748ae_abs": "o2b", "07741c45": "S", "09ba9854_abs": "o2b", "09d032c9": "o1",
    "0a995998": "o3", "0edc2aef": "o1", "1903aded": "o1", "195a1a1b": "o1",
    "1a8a66a6": "C", "1d4e3b97": "o4", "2c63a862": "o1", "2ce6a0f2": "o1",
    "32260d93": "o1", "35a27287": "o2a", "370a8ff4": "o4", "37f165cf": "o2a",
    "3a704032": "o3", "46a3abf7": "C", "4f54b7c9": "o3", "51a45a95": "o2a",
    "61f8c8f8": "o1", "67e0d0f2": "o1", "69fee5aa": "S", "6cb6f249": "o1", "6d550036": "o3",
    "6e984301": "o3", "7024f17c": "o1", "71017277": "o4", "75f70248": "o4",
    "80ec1f4f": "o1", "80ec1f4f_abs": "o4", "852ce960": "S", "88432d0a": "o3",
    "8c18457d": "o3", "9a707b81": "o3", "9ee3ecd6": "o3", "a08a253f": "o1",
    "a89d7624": "o4", "a96c20ee_abs": "o2b", "b46e15ed": "o1", "bc149d6b": "o1",
    "bf659f65": "o1", "d01c6aa8": "o1", "d6233ab6": "o1", "d682f1a2": "o1",
    "d851d5ba": "o3", "dd2973ad": "o1", "e3038f8c": "o1", "eac54add": "o1",
    "ec81a493": "o2a", "f35224e0": "o1", "gpt4_1e4a8aec": "o1", "gpt4_2f8be40d": "o3",
    "gpt4_45189cb4": "o1", "gpt4_483dd43c": "o1", "gpt4_4929293b": "o1",
    "gpt4_59c863d7": "o1", "gpt4_65aabe59": "o1", "gpt4_731e37d7": "o1",
    "gpt4_7abb270c": "o5", "gpt4_7f6b06db": "o1", "gpt4_7fce9456": "o1",
    "gpt4_93159ced_abs": "o2b", "gpt4_e05b82a6": "o1", "gpt4_f2262a51": "o1",
    "gpt4_f420262c": "o1", "gpt4_f420262d": "C",
}

FULL = {
    "06f04340": "o1", "0977f2af": "H", "09ba9854": "C", "09ba9854_abs": "o2b",
    "09d032c9": "o1", "0a34ad58": "o4", "0a995998": "o3", "0edc2aef": "o1",
    "129d1232": "o1", "157a136e": "o1", "1903aded": "o1", "195a1a1b": "o1",
    "1a1907b4": "o1", "1a8a66a6": "o1", "1d4e3b97": "o4", "1da05512": "o2a",
    "2c63a862": "o1", "2ce6a0f2": "o1", "35a27287": "o2a", "370a8ff4": "o4",
    "37f165cf": "o2a", "38146c39": "o1", "3a704032": "o3", "41698283": "S",
    "46a3abf7": "C", "51a45a95": "o2a", "51c32626": "o1", "561fabcd": "C",
    "57f827a0": "o2a", "60159905": "o3", "67e0d0f2": "o1", "6d550036": "o3",
    "6e984302": "o1", "7024f17c": "o1", "71017277": "o4", "73d42213": "o2a",
    "75832dbd": "o2a", "75f70248": "o4", "80ec1f4f": "o1", "852ce960": "S",
    "8e91e7d9": "o1", "92a0aa75": "o1", "95228167": "o1", "9ee3ecd6": "o3",
    "a08a253f": "o1", "a1cc6108": "o1", "a2f3aa27": "o2a", "a3838d2b": "o1",
    "a89d7624": "o4", "a96c20ee_abs": "o2b", "a9f6b44c": "o1", "afdc33df": "o1",
    "b0479f84": "o4", "b46e15ed": "o1", "ba358f49": "o1", "bc149d6b": "o1",
    "bf659f65": "o2a", "c18a7dc8": "o1", "c4a1ceb8": "o1", "caf03d32": "o4",
    "d6233ab6": "o1", "d682f1a2": "o1", "e3038f8c": "o1", "eac54add": "o1",
    "edced276": "o1", "eeda8a6d": "o1", "f35224e0": "o1", "fca70973": "o4",
    "gpt4_15e38248": "o1", "gpt4_194be4b3": "o1", "gpt4_1e4a8aec": "o1",
    "gpt4_2f56ae70": "o1", "gpt4_2f8be40d": "o1", "gpt4_45189cb4": "o1",
    "gpt4_731e37d7": "o1", "gpt4_7abb270c": "o5", "gpt4_7f6b06db": "o1",
    "gpt4_7fce9456": "o1", "gpt4_8279ba03": "o1", "gpt4_93159ced": "H",
    "gpt4_a1b77f9c": "o1", "gpt4_a56e767c": "o1", "gpt4_ab202e7f": "o1",
    "gpt4_d6585ce8": "o1", "gpt4_d6585ce9": "o1", "gpt4_d84a3211": "o1",
    "gpt4_e061b84f": "o1", "gpt4_e061b84g": "o1", "gpt4_f2262a51": "o1",
    "gpt4_f420262c": "o1", "gpt4_fe651585": "C",
}

# cross-cutting mechanism flags observed while reading (not categories)
VIEW_INDUCED = {  # the lifecycle view supplied or reinforced the wrong value
    "qc": ["gpt4_f420262d", "d851d5ba", "88432d0a", "eac54add", "gpt4_2f8be40d"],
    "full": [],
}
DISTRACTOR_SUB = {  # answer imported a value from an unrelated session
    "qc": ["6e984301", "d01c6aa8", "gpt4_1e4a8aec", "gpt4_4929293b", "eac54add",
           "gpt4_f420262d", "gpt4_7f6b06db"],
    "full": ["gpt4_1e4a8aec", "eac54add", "gpt4_7f6b06db"],
}
CONFAB_ANCHOR = {"qc": ["8c18457d", "9a707b81", "gpt4_65aabe59"], "full": []}
DISPUTED_GOLD = ["370a8ff4", "71017277", "80ec1f4f_abs"]

MAJOR = {"S": "stale-fact", "H": "history-loss", "C": "conflict-confusion"}
OTHER = {"o1": "other:retrieval-miss", "o2a": "other:over-abstention",
         "o2b": "other:under-abstention", "o3": "other:aggregation",
         "o4": "other:grading-boundary", "o5": "other:extraction-miss"}
ALL = {**MAJOR, **OTHER}


def load(fn):
    return {r["qid"]: r for r in (json.loads(l) for l in open(os.path.join(RES, fn)))}


def main():
    qc, fu = load("full500_qc.jsonl"), load("full500_full.jsonl")
    hits = json.load(open(os.path.join(RES, "_ev_hits.json")))
    qce = {k for k, r in qc.items() if not r["judge"]}
    fue = {k for k, r in fu.items() if not r["judge"]}

    for name, lab, err in (("qc", QC, qce), ("full", FULL, fue)):
        assert set(lab) == err, (name, "missing", sorted(err - set(lab)),
                                 "extra", sorted(set(lab) - err))
        assert all(v in ALL for v in lab.values()), name
    print("label sets validated: qc=%d full=%d\n" % (len(QC), len(FULL)))

    rows = []
    order = ["S", "H", "C", "o1", "o2a", "o2b", "o3", "o4", "o5"]
    cq, cf = Counter(QC.values()), Counter(FULL.values())
    print(f"{'category':28s} {'qc':>10s} {'full':>10s} {'shift':>10s}")
    for k in order:
        a, b = cq[k], cf[k]
        pa, pb = 100 * a / len(QC), 100 * b / len(FULL)
        print(f"{ALL[k]:28s} {a:4d} ({pa:4.1f}%) {b:4d} ({pb:4.1f}%) {a-b:+4d} ({pa-pb:+5.1f}pp)")
        rows.append((ALL[k], a, pa, b, pb))
    print(f"{'TOTAL errors':28s} {len(QC):4d}        {len(FULL):4d}")
    maj_q = sum(cq[k] for k in MAJOR)
    maj_f = sum(cf[k] for k in MAJOR)
    print(f"\nlifecycle-attributable (S+H+C): qc {maj_q}/{len(QC)} = {100*maj_q/len(QC):.1f}%"
          f" | full {maj_f}/{len(FULL)} = {100*maj_f/len(FULL):.1f}%")
    print(f"of all 500 questions:           qc {100*maj_q/500:.1f}% | full {100*maj_f/500:.1f}%")

    # per question-type composition
    print("\nper question_type (qc | full), category counts:")
    tq, tf = defaultdict(Counter), defaultdict(Counter)
    for k, v in QC.items():
        tq[qc[k]["type"]][v] += 1
    for k, v in FULL.items():
        tf[fu[k]["type"]][v] += 1
    types = sorted(set(tq) | set(tf))
    for t in types:
        s_q = " ".join(f"{k}:{tq[t][k]}" for k in order if tq[t][k])
        s_f = " ".join(f"{k}:{tf[t][k]}" for k in order if tf[t][k])
        print(f"  {t:26s} qc[{sum(tq[t].values()):2d}] {s_q:34s} | full[{sum(tf[t].values()):2d}] {s_f}")

    # mechanism: was gold evidence in the arm's context?
    print("\nmechanism (gold-evidence coverage in that arm's raw context):")
    for name, lab, key in (("qc", QC, "ev_hit20"), ("full", FULL, "ev_hit30")):
        m = defaultdict(Counter)
        for q, v in lab.items():
            h = hits[q]
            if h["n_ev"] == 0:
                c = "abstention(no evidence)"
            elif all(h[key]):
                c = "complete"
            elif any(h[key]):
                c = "partial"
            else:
                c = "none"
            m[v][c] += 1
        print(f"  [{name}]")
        for k in order:
            if m[k]:
                print(f"    {ALL[k]:26s} " + "  ".join(f"{c}={n}" for c, n in sorted(m[k].items())))

    # cross-cutting flags
    print("\ncross-cutting mechanism flags:")
    for nm, d in (("view-induced wrong value", VIEW_INDUCED),
                  ("distractor substitution", DISTRACTOR_SUB),
                  ("confabulated date anchor", CONFAB_ANCHOR)):
        print(f"  {nm:26s} qc={len(d['qc'])} {d['qc']}")
        print(f"  {'':26s} full={len(d['full'])} {d['full']}")
    print(f"  disputed gold labels        {DISPUTED_GOLD}")

    # paired view: same question, both arms wrong -> category shift
    both = sorted(set(QC) & set(FULL))
    print(f"\nquestions wrong in BOTH arms: {len(both)}; category disagreements:")
    for q in both:
        if QC[q] != FULL[q]:
            print(f"    {q:20s} qc={ALL[QC[q]]:26s} full={ALL[FULL[q]]}")
    print(f"\nqc-only errors: {len(set(QC)-set(FULL))}   full-only errors: {len(set(FULL)-set(QC))}")
    print("qc-only by category:", dict(Counter(QC[q] for q in set(QC) - set(FULL))))
    print("full-only by category:", dict(Counter(FULL[q] for q in set(FULL) - set(QC))))

    json.dump({"qc": QC, "full": FULL, "legend": ALL,
               "view_induced": VIEW_INDUCED, "distractor_substitution": DISTRACTOR_SUB,
               "confabulated_anchor": CONFAB_ANCHOR, "disputed_gold": DISPUTED_GOLD},
              open(os.path.join(RES, "error_typology_labels.json"), "w"), indent=1)
    print("\nwrote error_typology_labels.json")


if __name__ == "__main__":
    main()
