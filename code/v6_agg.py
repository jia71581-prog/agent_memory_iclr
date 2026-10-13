"""v6-AGG:量化聚合题的"数量台账"(quantity ledger)视图补全 —— training-free、纯确定性、无额外 API 调用。

诊断依据(读 24 条 v5 multi-session 失败案例原文得到,见 run_qc.py 的 QC_V6_AGG 分支):
  机制 A(相关性剪枝):v5 的 _rank_view_lines 在 20000 字符预算下丢弃 22-46% 的链(24 题平均 36%),
    丢弃标准是链文本与问题的 BM25 分。决定性的数量操作数常落在被丢弃的一侧:
      157a136e  gold 43 = 75-32,`user::age = 32` 被剪枝(view/raw 都没有)→ 模型只能拒答
      92a0aa75  gold 1y5m = 3y9m - 2y4m,两个操作数(`tenure at company`、`job progression`)双双被剪枝
      731e37d7  gold $720 = 500+200+20,`paid for writing workshop = $200` 被剪枝 → 模型答 $520
  机制 B(current 视图坍塌):通用动词属性(has completed / owns / paid)把互不相干的数量并入一条链,
    被标成 supersession,current 模式只渲染最新一组,较早的数量被视图本身销毁:
      67e0d0f2  chain user::has completed = [8 edX courses(05/21), 12 courses on Coursera(05/27)]
                → 渲染成 "= 12 courses on Coursera",8 门 edX 消失 → 模型答 12(gold 20)
      e3038f8c  chain user::owns = [vase, rare coins, 25 rare coins(05/23), e-reader, Trek bike(05/29)]
                → 渲染成 "= a Trek Emonda road bike",25 枚币消失 → 模型答 74(gold 99)
  机制 C(窗口判定):"past month / last week" 的相对窗口靠日期字符串心算,v5 误判过
      2ba83207(把 05/26 的 Thrive Market 支出当成"上个月"排除)、7024f17c(把 05/20 的 30 分钟跑排除)。

干预:对"量化聚合题"(见 is_agg_question:how much / in total / how many <单位名词> / how long / the most money ...)
把 store 里**全部**带数量的事实强制进视图,去重、按日期排序、附相对问题日期的天数偏移,渲染成一个显式台账块;
台账占用的字符从 VIEW_CHAR_CAP 里扣除(预算中性,总上下文只多一个分节标题),且台账已完整覆盖的链从排序视图里跳过,
腾出的空间用来多放别的链。台账渲染走全链历史而非 current 最新值,故机制 B 被解除。

不做的事:不引入任何 gold/答案侧信息;不新增 API 调用;不改 v5 任何函数(本模块只在 QC_V6_AGG=1 分支被调用)。
"""
import re

from run_baseline import bm25_rank
from lifecycle import render_view

# ---------------------------------------------------------------- 题型门控

# 计量单位名词:出现在 "how many <unit>" 里说明答案是量纲量(可加总/相减),而不是物件计数。
_UNIT_NOUNS = (
    "year", "years", "month", "months", "week", "weeks", "day", "days",
    "hour", "hours", "minute", "minutes", "second", "seconds",
    "dollar", "dollars", "cent", "cents", "euro", "euros", "pound", "pounds",
    "point", "points", "page", "pages", "mile", "miles", "kilometer", "kilometers",
    "km", "kg", "gram", "grams", "pound", "lbs", "liter", "liters", "gallon", "gallons",
    "calorie", "calories", "step", "steps", "percent", "degree", "degrees",
    "money", "time", "times",
)
# 直接触发词(整句级线索)
_AGG_PHRASES = (
    "how much", "in total", "total number", "total amount", "total cost",
    "total money", "total spent", "altogether", "combined", "on average",
    "how long", "how old", "grand total", "sum of", "adds up",
)


def is_agg_question(question):
    """量化聚合题门控(纯规则)。物件计数题(how many plants / weddings / albums)刻意不触发:
    台账只装数量事实,对物件计数无用,触发只会白占预算。"""
    ql = " " + re.sub(r"[^a-z0-9$%' ]", " ", question.lower()) + " "
    ql = re.sub(r"\s+", " ", ql)
    if any(p in ql for p in _AGG_PHRASES):
        return True
    m = re.search(r"\bhow many ([a-z]+)", ql)
    if m and m.group(1) in _UNIT_NOUNS:
        return True
    if re.search(r"\b(most|least|highest|lowest|largest|biggest)\b", ql) and \
       re.search(r"\b(money|spend|spent|cost|paid|pay|expensive|amount|total)\b", ql):
        return True
    if re.search(r"\b(count|number) of\b", ql) or re.search(r"\b[a-z]+ count\b", ql):
        return True
    return False


# ---------------------------------------------------------------- 数量事实抽取

# 数量 token:货币/小数/千分位/百分号/裸数字,以及少量拼写数字。
_NUM_RE = re.compile(
    r"(?:[$€£]\s?\d[\d,.]*)"                      # $1,200 / €20
    r"|(?:\d[\d,.]*\s?(?:%|percent))"             # 15%
    r"|(?:\b\d[\d,.]*\b)",                        # 25 / 3.5 / 1,000
)

# 拼写数字:刻意剔除 one / once / couple —— "a loved one"、"once a week" 之类会把大量非数量事实
# 拉进台账(实测在 e3038f8c 上引入 4 条纯噪声行,白占预算)。
_WORD_NUM = re.compile(
    r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
    r"dozen|twice)\b", re.I)


def _numbers(text):
    return [m.group(0).strip() for m in _NUM_RE.finditer(text)]


def has_quantity(text):
    return bool(_NUM_RE.search(text)) or bool(_WORD_NUM.search(text))


_STOP = {"the", "a", "an", "of", "for", "to", "in", "on", "at", "and", "with", "user",
         "is", "was", "are", "were", "has", "have", "had", "their", "my", "i", "it"}


def _content_words(text):
    return {w for w in re.findall(r"[a-z0-9$]+", text.lower()) if w not in _STOP}


def _numkey(text):
    """数字多重集签名(去掉千分位/货币符号),用于识别"同一个数字被重复叙述"。"""
    out = []
    for n in _NUM_RE.finditer(text):
        s = re.sub(r"[^0-9.]", "", n.group(0))
        s = s.rstrip(".")
        if s:
            out.append(s)
    return tuple(sorted(out))


def collect_quantity_facts(store):
    """返回 store 中全部带数量的事实(全链历史,不受 current 坍塌影响),已去重。
    去重规则(确定性):
      1) 规范化 attribute+value 完全相同 → 合并;
      2) 数字签名相同且内容词高度重叠(一方是另一方子集,或 Jaccard >= 0.6)→ 保留信息更全的那条。
    另返回 covered:全部值都进了台账的链 key 集合(这些链可从排序视图里跳过,不丢信息)。"""
    picked = []            # [(sort_key, entity, attribute, value, date, sidx, chain_key)]
    per_chain_total, per_chain_taken = {}, {}
    for k in sorted(store):
        chain = store[k]
        per_chain_total[k] = len(chain)
        for f in chain:
            blob = f"{f['attribute']} {f['value']}"
            if not has_quantity(blob):
                continue
            picked.append((f["date"], f.get("sidx", 0), f["entity"], f["attribute"], f["value"], k))
            per_chain_taken[k] = per_chain_taken.get(k, 0) + 1
    # 去重
    kept = []
    for date, sidx, ent, attr, val, k in picked:
        blob = f"{attr} {val}"
        norm = " ".join(sorted(_content_words(blob)))
        nk = _numkey(blob)
        cw = _content_words(blob)
        dup_of = None
        for j, e in enumerate(kept):
            if e["norm"] == norm:
                dup_of = j
                break
            if nk and e["numkey"] == nk:
                inter = len(cw & e["cw"])
                union = len(cw | e["cw"]) or 1
                if cw <= e["cw"] or e["cw"] <= cw or inter / union >= 0.6:
                    dup_of = j
                    break
        if dup_of is None:
            kept.append({"date": date, "sidx": sidx, "entity": ent, "attribute": attr,
                         "value": val, "chain": k, "norm": norm, "numkey": nk, "cw": cw,
                         "dups": 0})
        else:
            e = kept[dup_of]
            e["dups"] += 1
            if len(cw) > len(e["cw"]):   # 保留信息更全的表述,日期取更早的一次陈述
                e.update({"attribute": attr, "value": val, "cw": cw, "norm": norm})
            if date < e["date"]:
                e["date"] = date
                e["sidx"] = sidx
    covered = {k for k, n in per_chain_total.items() if per_chain_taken.get(k, 0) == n}
    kept.sort(key=lambda e: (e["date"], e["sidx"], e["attribute"], e["value"]))
    return kept, covered


# ---------------------------------------------------------------- 渲染

_DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})")


def _day_index(date_str):
    m = _DATE_RE.search(date_str or "")
    if not m:
        return None
    y, mo, d = (int(x) for x in m.groups())
    # 儒略日式序号,足够做天数差(不依赖 datetime,避免时区/格式异常)
    a = (14 - mo) // 12
    yy = y + 4800 - a
    mm = mo + 12 * a - 3
    return d + (153 * mm + 2) // 5 + 365 * yy + yy // 4 - yy // 100 + yy // 400


def _rel_days(date_str, qdate):
    d1, d0 = _day_index(date_str), _day_index(qdate)
    if d1 is None or d0 is None:
        return ""
    delta = d0 - d1
    if delta <= 0:
        return "today"
    return f"{delta}d before"


LEDGER_HEADER = (
    "### Quantity ledger (EVERY quantity on record, de-duplicated, oldest first; "
    "'Nd before' = days before the question date). Each line is a distinct statement; "
    "lines repeating the same amount for the same thing were already merged. Use this list "
    "when the question asks for a total, a difference, a duration or a maximum: decide which "
    "lines fall inside the asked-for window, then compute over exactly those lines. "
    "When a later line updates an earlier one about the same thing (a running total, a count "
    "or a duration 'so far'), the latest-dated line is the current value, not the sum."
)

LINE_MAX = 170        # 单行上限:长散文值(章节列表、tagline)会一行吃掉 300+ 字符,截断而非丢弃


def render_ledger(facts, qdate, cap, question=None):
    """渲染台账正文(不含 header)。行按日期序输出;若全量超预算,则先按与问题的 BM25 分选出能装下的行,
    再按日期序渲染(溢出时的取舍是确定性的,且候选池只有数量事实,比 v5 的整链剪枝粒度细得多)。"""
    if not facts:
        return ""
    rendered = []
    for e in facts:
        rel = _rel_days(e["date"], qdate)
        day = e["date"].split()[0]
        line = f"- [{day}{(' | ' + rel) if rel else ''}] {e['entity']} | {e['attribute']} = {e['value']}"
        if len(line) > LINE_MAX:
            line = line[:LINE_MAX - 3] + "..."
        rendered.append(line)
    if sum(len(l) + 1 for l in rendered) <= cap:
        return "\n".join(rendered)
    # 溢出:按问题相关性挑,再恢复日期序
    texts = [f"{e['entity']} {e['attribute']} {e['value']}" for e in facts]
    order = bm25_rank(question or "", texts) if question else list(range(len(facts)))
    keep, total = set(), 0
    for i in order:
        if total + len(rendered[i]) + 1 > cap:
            continue
        keep.add(i)
        total += len(rendered[i]) + 1
    return "\n".join(rendered[i] for i in sorted(keep))


def rank_view_lines_skipping(store, mode, question, cap, skip):
    """v5 _rank_view_lines 的同构副本 + skip 集合(台账已完整覆盖的链不再重复渲染)。
    独立实现是为了不改动 v5 共享函数(多 agent 并发编辑 run_qc.py)。"""
    keys = [k for k in store if k not in skip]
    if not keys:
        return ""
    texts = [" ".join(f"{c['entity']} {c['attribute']} {c['value']}" for c in store[k]) for k in keys]
    order = bm25_rank(question, texts)
    kept, total = [], 0
    for i in order:
        line = render_view({keys[i]: store[keys[i]]}, mode)
        if total + len(line) + 1 > cap:
            continue
        kept.append((i, line))
        total += len(line) + 1
    kept.sort(key=lambda x: x[0])
    return "\n".join(l for _, l in kept)


def build_v6_sections(store, mode, question, qdate, view_cap, ledger_cap):
    """返回 (view_txt, ledger_txt)。台账正文 + 台账分节标题 + 视图 <= view_cap,
    即端到端上下文长度不超过同题 v5(标题开销也从视图预算里扣,预算中性 by construction)。"""
    facts, covered = collect_quantity_facts(store)
    budget = min(ledger_cap, max(0, view_cap // 2))
    ledger_txt = render_ledger(facts, qdate, budget, question)
    overhead = (len(LEDGER_HEADER) + 3) if ledger_txt else 0   # "### ..." 分节标题 + 段间换行
    remaining = max(0, view_cap - len(ledger_txt) - overhead)
    view_txt = rank_view_lines_skipping(store, mode, question, remaining, covered if ledger_txt else set())
    return view_txt, ledger_txt


if __name__ == "__main__":
    # 门控单测:量纲聚合题触发,物件计数题不触发
    fire = ["How much total money did I spend on attending workshops in the last four months?",
            "How many rare items do I have in total?",
            "What is the total number of online courses I've completed?",
            "How many years older is my grandma than me?",
            "How long have I been working in my current role?",
            "How many hours of jogging and yoga did I do last week?",
            "How many points do I need to earn to redeem a free skincare product at Sephora?",
            "Which grocery store did I spend the most money at in the past month?",
            "What was the page count of the two novels I finished in January and March?"]
    hold = ["How many plants did I acquire in the last month?",
            "How many weddings have I attended in this year?",
            "How many kitchen items did I replace or fix?",
            "How many music albums or EPs have I purchased or downloaded?",
            "What is my dog's name?",
            "Where did I go on vacation last summer?"]
    for q in fire:
        assert is_agg_question(q), q
    for q in hold:
        assert not is_agg_question(q), q
    # 数量识别 / 去重 / 相对日期
    assert has_quantity("paid $500 to attend the workshop") and has_quantity("8 edX courses")
    assert not has_quantity("a Trek Emonda road bike")
    assert _numkey("raised over $2,000 for a local animal shelter") == ("2000",)
    F = lambda a, v, d, s: {"entity": "user", "attribute": a, "value": v, "date": d, "sidx": s,
                            "relation": "initial"}
    store = {
        "user::helped raise": [F("helped raise", "over $2,000 for a local animal shelter", "2023/03/01 (Wed) 10:00", 1),
                               F("helped raise", "$2,000 for a local animal shelter on January 20th", "2023/03/05 (Sun) 10:00", 2)],
        "user::owns": [F("owns", "25 rare coins", "2023/03/02 (Thu) 10:00", 3),
                       F("owns", "a Trek Emonda road bike", "2023/03/09 (Thu) 10:00", 4)],
    }
    facts, covered = collect_quantity_facts(store)
    vals = [f["value"] for f in facts]
    assert len(facts) == 2, vals                     # 两条 $2,000 合并
    assert any("25 rare coins" in v for v in vals)   # current 坍塌本会藏掉这条
    assert covered == {"user::helped raise"}         # owns 链含无数量值 → 不跳过
    led = render_ledger(facts, "2023/03/10 (Fri) 12:00", 4000, "how much did I raise in total")
    assert "8d before" in led and "25 rare coins" in led, led
    assert render_ledger(facts, "2023/03/10 (Fri) 12:00", 10, "x") == ""
    # 噪声词剔除:"a loved one" / "once a week" 不算数量;长行截断
    assert not has_quantity("has a loved one with Alzheimer's disease")
    assert not has_quantity("goes swimming once a week")
    assert has_quantity("moving to Toronto in six weeks") and has_quantity("age = 32")
    long_store = {"user::x": [F("book chapter 1 sections", "It starts with you; " * 20,
                                "2023/03/01 (Wed) 10:00", 1)]}
    lf, _ = collect_quantity_facts(long_store)
    ll = render_ledger(lf, "2023/03/10 (Fri) 12:00", 4000, "q")
    assert len(ll) <= LINE_MAX and ll.endswith("..."), (len(ll), ll[-20:])
    print("v6_agg self-tests OK\n" + led)
