"""QC-Lifecycle v5 部件:时间脚手架(temporal scaffolding),training-free、零 API。

动机(实测):TR 题 n=127(非弃答)acc=0.858,18 个错题里 13 个"证据在场却答错"。
逐例读错误可见失败不在检索,而在**日期算术是推断出来的而不是机械算出来的**:
  - 锚点选错:多个同类事件里挑了错的那个(b46e15ed:连续两天的慈善活动在 2/14+2/15,
    模型拿 3/19 当锚 → 1 个月 vs gold 2)
  - 相对表达没落地:证据里写"a month ago"/"about a month ago",绝对日期从未出现,
    模型只能猜(gpt4_65aabe59 恒温器 vs mesh;gpt4_483dd43c GoT vs The Crown)
  - 问句侧相对表达没落地:"two weeks ago"/"four weeks ago"/"a week ago"/"last Saturday"
    没有被解析成绝对日期,模型挑了话题最像但日期错的会话(gpt4_1e4a8aec / eac54add / gpt4_4929293b)
  - 排序题要在 6 个日期上做全序,推理预算被烧光直接空答(gpt4_7abb270c)

因此本模块把"日期算术"从推理任务降级为查表任务,三块产出:
  1. ANCHOR RESOLUTION  - 问句里的相对/命名日期表达 → 绝对日期(确定性规则)
  2. TEMPORAL ARITHMETIC - 按题族预计算 delta / 全序 / 锚日事件(题族由规则判定)
  3. EVENT TIMELINE     - 与问题相关的带日期事件,按**事件时间**(不是会话时间)排序、
     去重,逐行标注相对问题日的有符号偏移,并标出相邻事件间隔与"连续两天"配对

对外接口(供 run_qc 集成):
  render_timeline(q, store, question) -> text
  annotate_offsets(lines, question_date) -> lines

设计约束:纯规则 + 正则,无训练、无 LLM 调用。store(lifecycle 事实链)可选;
给了就把事实行也纳入时间线候选,不给就只用原文轮句子。

离线实测(v5_temporal_measure.py / v5_temporal_ablate2.py,TR n=133 含 6 弃答题,零 API):
  题族判定(规则):delta_between 43 / ordering 43 / anchor_lookup 21 / delta_since 19 / other 6 / count_before 1
  证据会话日期出现在时间线里(必要条件):1.000 / 1.000 / 0.767 / 1.000
  delta 两族(数值 gold n=60):证据侧可推导上限 0.833,本脚手架时间线可推导 0.800,
      且预计算块里**字面**给出 gold 数值 0.800(候选数值中位数 6 个,非"把所有数都列一遍")
  ordering:条目级可判定 27/43,其中判对 17/27=0.630(二选一子族 16/24=0.667,随机 0.5);
      30 题证据会话同日(会话日期无法排序),其中 28 题靠句内相对表达拆成不同事件日
  anchor_lookup:问句锚点落在含 gold 的证据会话 ±3 天内 20/21=0.952;
      "锚日直取"块含 gold 19/21=0.905(词干版 20/21=0.952),gold 行排第 1 的有 12/17
  体积:全 500 题 median 2308 chars / mean 2591 / p90 4547 / max 8445
      (≈ +550 tokens mean,当前 qc prompt mean 11344 tokens,即 +4.8%;只在检测到时间题族时
      开启则摊薄到 +2.3%)
  承重部件消融:关掉句内相对表达解析 → delta 可推导 0.800→0.633、ordering 判对率 0.395→0.233、
      同日拆分 28→0。其余启发式(cue/anchor/intent 软权重、谓词扫描、问句形式词剥离)实测无收益,
      已默认关闭,不要凭直觉打开(见下方常量注释)。
"""
import os
import re
import sys
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------------
# 0. 日期基元
# ----------------------------------------------------------------------------

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
WEEKDAYS = {d: i for i, d in enumerate(
    ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])}
WEEKDAYS.update({d[:3]: i for d, i in list(WEEKDAYS.items())})
NUMWORDS = {
    "one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "couple": 2, "few": 3, "several": 3,
}
UNIT_DAYS = {"day": 1, "week": 7, "month": 30.44, "year": 365.25}

_DATE_RE = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})")


def parse_date(s):
    """解析数据集日期串 '2023/03/19 (Sun) 15:02' → datetime(含时分,缺省 00:00)。"""
    if isinstance(s, (datetime,)):
        return s
    m = _DATE_RE.search(str(s))
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    hm = re.search(r"(\d{1,2}):(\d{2})", str(s)[m.end():])
    h, mi = (int(hm.group(1)), int(hm.group(2))) if hm else (0, 0)
    try:
        return datetime(y, mo, d, h, mi)
    except ValueError:
        return None


def day0(dt):
    return datetime(dt.year, dt.month, dt.day)


def days_between(d1, d2):
    """d2 - d1,按日历天(丢弃时分,避免 03:31 vs 15:02 造成 ±1 抖动)。"""
    return (day0(d2) - day0(d1)).days


def months_between(d1, d2):
    """完整日历月数(d2 >= d1 时为非负)。'过了几个月'的人类口径。"""
    m = (d2.year - d1.year) * 12 + (d2.month - d1.month)
    if d2.day < d1.day:
        m -= 1
    return m


def _easter(year):
    """匿名格里高利算法。用于 Ash Wednesday / Maundy Thursday / Good Friday 等派生节日。"""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mo = (h + l - 7 * m + 114) // 31
    dy = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, mo, dy)


def _nth_weekday(year, month, weekday, n):
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(days=7 * (n - 1))


def holiday_dates(year):
    """固定/可推导节日 → (month, day)。只放数据里真实出现过的族。"""
    e = _easter(year)
    out = {
        "valentine's day": (2, 14), "valentines day": (2, 14), "valentine's": (2, 14),
        "christmas eve": (12, 24), "christmas": (12, 25),
        "new year's eve": (12, 31), "new year's day": (1, 1),
        "halloween": (10, 31), "independence day": (7, 4), "fourth of july": (7, 4),
        "july 4th": (7, 4), "st. patrick's day": (3, 17), "mother's day": None,
    }
    res = {k: v for k, v in out.items() if v}
    res["thanksgiving"] = (_nth_weekday(year, 11, 3, 4).month, _nth_weekday(year, 11, 3, 4).day)
    res["mother's day"] = (_nth_weekday(year, 5, 6, 2).month, _nth_weekday(year, 5, 6, 2).day)
    res["easter"] = (e.month, e.day)
    for name, off in [("ash wednesday", -46), ("maundy thursday", -3),
                      ("good friday", -2), ("palm sunday", -7)]:
        d = e + timedelta(days=off)
        res[name] = (d.month, d.day)
    return res


# ----------------------------------------------------------------------------
# 1. 偏移量格式化 / annotate_offsets
# ----------------------------------------------------------------------------

def offset_phrase(ev_dt, q_dt, verbose=True):
    """事件日期相对问题日期的有符号偏移短语。"""
    nd = days_between(ev_dt, q_dt)          # >0 = 事件在问题之前
    if nd == 0:
        return "same day as the question"
    side = "before" if nd > 0 else "after"
    a = abs(nd)
    wk = a / 7.0
    if not verbose:
        return f"{'-' if nd > 0 else '+'}{a}d"
    parts = [f"{a} days", f"{wk:.1f} weeks"]
    mo = abs(months_between(q_dt, ev_dt) if nd < 0 else months_between(ev_dt, q_dt))
    if a >= 25:
        parts.append(f"{mo} full months")
    return f"{' / '.join(parts)} {side} the question"


_LINE_DATE = re.compile(r"^\s*\[\s*(\d{4}/\d{1,2}/\d{1,2}[^\]]*)\]")


def annotate_offsets(lines, question_date):
    """给已有的上下文行盖上"相对问题日的有符号偏移"。

    lines: list[str],形如 '[2023/03/19 (Sun) 15:02] [user] ...' 或 '[2023/03/19] user | job = ...'。
           也接受单个字符串(按行切分)。无日期前缀的行原样透传。
    返回同长度的 list[str],日期括号被改写成 '[2023/03/19 (Sun) | 30 days / 4.3 weeks before the question]'。
    """
    if isinstance(lines, str):
        lines = lines.splitlines()
    q_dt = parse_date(question_date)
    if q_dt is None:
        return list(lines)
    out = []
    for ln in lines:
        m = _LINE_DATE.match(ln)
        if not m:
            out.append(ln)
            continue
        ev = parse_date(m.group(1))
        if ev is None:
            out.append(ln)
            continue
        wd = ev.strftime("%a")
        stamp = f"[{ev.year}/{ev.month:02d}/{ev.day:02d} ({wd}) | {offset_phrase(ev, q_dt)}]"
        out.append(stamp + ln[m.end():])
    return out


# ----------------------------------------------------------------------------
# 2. 文内相对时间表达 → 绝对日期(确定性)
# ----------------------------------------------------------------------------

_NUM = r"(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|a|an|a couple of|a few|several)"
_UNIT = r"(day|week|month|year)s?"
_WD = r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)"
_MO = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"


def _num(tok):
    tok = tok.strip().lower()
    tok = re.sub(r"^(a couple of|a few)$", lambda m: "couple" if "couple" in m.group(0) else "few", tok)
    if tok.isdigit():
        return int(tok)
    return NUMWORDS.get(tok)


def _shift(ref, n, unit):
    if unit.startswith("day"):
        return ref - timedelta(days=n)
    if unit.startswith("week"):
        return ref - timedelta(days=7 * n)
    if unit.startswith("month"):
        y, m = ref.year, ref.month - n
        while m <= 0:
            m += 12
            y -= 1
        d = min(ref.day, [31, 29 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 28,
                          31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
        return datetime(y, m, d, ref.hour, ref.minute)
    if unit.startswith("year"):
        return datetime(ref.year - n, ref.month, min(ref.day, 28), ref.hour, ref.minute)
    return None


def _prev_weekday(ref, wd, strict=True):
    """ref 之前(strict 时严格早于当天)最近的一个 wd。"""
    back = (ref.weekday() - wd) % 7
    if back == 0 and strict:
        back = 7
    return day0(ref) - timedelta(days=back)


def resolve_expressions(text, ref_date):
    """在 text 中找相对/命名时间表达,按 ref_date 解析成绝对日期。

    返回 [{'expr','date','conf','kind'}]:
      conf='high'   today/yesterday/N-ago/显式月日/命名节日/last+weekday
      conf='med'    last week|month|year / 月份(无日)/ 'for N weeks now' 起点
      conf='low'    recently / a few X ago
    确定性:同一 (text, ref_date) 永远同结果;不做任何模型调用。
    """
    ref = parse_date(ref_date)
    if ref is None or not text:
        return []
    t = text.lower()
    hits, seen = [], set()

    def add(expr, dt, conf, kind):
        if dt is None:
            return
        key = (expr, dt.date(), kind)
        if key in seen:
            return
        seen.add(key)
        hits.append({"expr": expr, "date": day0(dt), "conf": conf, "kind": kind})

    for m in re.finditer(r"\b(today|this morning|this afternoon|this evening|tonight|just now)\b", t):
        add(m.group(0), ref, "high", "same-day")
    for m in re.finditer(r"\bthe day before yesterday\b", t):
        add(m.group(0), ref - timedelta(days=2), "high", "offset")
    for m in re.finditer(r"\b(yesterday|last night)\b", t):
        add(m.group(0), ref - timedelta(days=1), "high", "offset")
    # N units ago  /  a week ago  /  a couple of days ago
    for m in re.finditer(r"\b(?:about|around|roughly|approximately|almost|nearly|over|some)?\s*" + _NUM + r"\s*(?:and a half\s*)?" + _UNIT + r"\s+ago\b", t):
        n = _num(m.group(1))
        if n is not None:
            conf = "low" if m.group(1).strip() in ("a few", "several", "few") else "high"
            add(m.group(0).strip(), _shift(ref, n, m.group(2)), conf, "offset")
    # last week / last month / last year
    for m in re.finditer(r"\blast (week|month|year)\b", t):
        add(m.group(0), _shift(ref, 1, m.group(1)), "med", "offset-coarse")
    # last / this past <weekday>
    for m in re.finditer(r"\b(?:last|this past|past|previous)\s+" + _WD + r"\b", t):
        add(m.group(0), _prev_weekday(ref, WEEKDAYS[m.group(1)[:3]]), "high", "weekday")
    # last weekend / the past weekend  → 最近的周六
    for m in re.finditer(r"\b(?:last|this past|the past|over the)\s+weekend\b", t):
        add(m.group(0), _prev_weekday(ref, 5), "high", "weekday")
    # on <weekday>  (裸 weekday:取最近的一个,含当天)
    for m in re.finditer(r"\bon\s+(?:a|the|that)?\s*" + _WD + r"\b", t):
        add(m.group(0), _prev_weekday(ref, WEEKDAYS[m.group(1)[:3]], strict=False), "med", "weekday")
    # 显式月日:March 15 / March 15th, 2023 / 15th of March
    for m in re.finditer(_MO + r"\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?\b", t):
        mo, dy = MONTHS[m.group(1)[:3]], int(m.group(2))
        yr = int(m.group(3)) if m.group(3) else ref.year
        try:
            dt = datetime(yr, mo, dy)
        except ValueError:
            continue
        conf, kind = "high", "explicit"
        if not m.group(3) and dt > ref + timedelta(days=1):
            # 会话日之后的显式日期 = 计划中的未来事件(机票/面试),不能当过去事件定时
            conf, kind = "med", "explicit-future"
        add(m.group(0), dt, conf, kind)
    for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+of\s+" + _MO + r"\b", t):
        mo, dy = MONTHS[m.group(2)[:3]], int(m.group(1))
        try:
            dt = datetime(ref.year, mo, dy)
        except ValueError:
            continue
        conf, kind = "high", "explicit"
        if dt > ref + timedelta(days=1):
            conf, kind = "med", "explicit-future"
        add(m.group(0), dt, conf, kind)
    # 数字日期 3/8 / 03/08 / 3-8 / 3/8/23(语料里真实出现:"graduation gift on the 3/8")
    for m in re.finditer(r"(?<![\d/\-])(\d{1,2})[/\-](\d{1,2})(?:[/\-](\d{2,4}))?(?![\d/\-])", t):
        a, b = int(m.group(1)), int(m.group(2))
        if not (1 <= a <= 12 and 1 <= b <= 31):
            continue
        after = t[m.end():m.end() + 24]
        before = t[max(0, m.start() - 18):m.start()]
        if re.search(r"^\s*(?:more\s+)?(episodes?|times?|days?|hours?|minutes?|weeks?|months?|years?|"
                     r"people|cups?|tsp|tbsp|miles?|km|kg|lbs?|oz|sets?|reps?|pages?|chapters?|"
                     r"servings?|glasses|bottles?|nights?|sessions?)\b", after):
            continue                                  # "2-3 episodes" / "1/2 cup" 不是日期
        if re.search(r"\b(at least|about|around|roughly|maybe|between|some|every|per)\s*$", before):
            continue
        if re.search(r"\b(cup|tsp|tbsp|teaspoon|tablespoon|inch|mile|hour|ratio|odds|split|scale|"
                     r"out of|24/7|lb|oz|km|cm|mm)\b", before + after):
            continue
        yr = int(m.group(3)) if m.group(3) else ref.year
        if yr < 100:
            yr += 2000
        try:
            dt = datetime(yr, a, b)
        except ValueError:
            continue
        conf, kind = "high", "explicit"
        if not m.group(3) and dt > ref + timedelta(days=1):
            conf, kind = "med", "explicit-future"
        add(m.group(0).strip(), dt, conf, kind)
    # 月份(无日):mid-January / early March / in December
    for m in re.finditer(r"\b(mid|early|late)?[\s-]*" + _MO + r"\b(?!\s+\d)", t):
        qual, mo = (m.group(1) or "").strip(), MONTHS[m.group(2)[:3]]
        if m.group(2) == "may" and not re.search(r"\b(in|since|during|of|last)\s+may\b", t):
            continue                                  # 'may' 作助动词,排除
        dy = {"early": 5, "mid": 15, "late": 25}.get(qual, 15)
        yr = ref.year if mo <= ref.month else ref.year - 1
        add(m.group(0).strip(), datetime(yr, mo, dy), "med", "month-anchor")
    # 命名节日
    hol = holiday_dates(ref.year)
    for name, (mo, dy) in hol.items():
        if name in t:
            yr = ref.year
            dt = datetime(yr, mo, dy)
            if dt > ref + timedelta(days=1):
                h2 = holiday_dates(yr - 1)[name]
                dt = datetime(yr - 1, h2[0], h2[1])
            add(name, dt, "high", "holiday")
    # 持续时长 → 起点:'for about 6 weeks now' / 'been X for 3 months'
    for m in re.finditer(r"\bfor\s+(?:about|around|roughly|almost|nearly|over)?\s*" + _NUM + r"\s*(?:and a half\s*)?" + _UNIT + r"\b(?:\s+now)?", t):
        n = _num(m.group(1))
        if n is not None:
            add(m.group(0).strip(), _shift(ref, n, m.group(2)), "med", "duration-start")
    for m in re.finditer(r"\b(finished|completed|read|watched|binge[- ]?watched|took me|took|spent|"
                         r"got through|wrapped up)\b[^.!?]{0,60}?\bin\s+(?:just|only|about|around)?\s*"
                         + _NUM + r"\s*" + _UNIT + r"\b", t):
        n = _num(m.group(2))
        if n is not None:
            add(m.group(0).strip()[:40], _shift(ref, n, m.group(3)), "med", "duration-start")
    for m in re.finditer(r"\b(recently|lately|the other day|a while back|not long ago)\b", t):
        add(m.group(0), ref - timedelta(days=7), "low", "vague")
    return hits


def best_event_date(text, session_dt):
    """会话内某句的"事件时间":有高置信文内表达则用它,否则用会话时间。
    返回 (event_dt, provenance_str)。"""
    ref = parse_date(session_dt)
    if ref is None:
        return None, ""
    hits = [h for h in resolve_expressions(text, ref)
            if h["conf"] == "high" and h["kind"] != "explicit-future"]
    if not hits:
        med = [h for h in resolve_expressions(text, ref)
               if h["conf"] == "med" and h["kind"] in ("offset-coarse", "month-anchor", "duration-start")]
        if med:
            h = min(med, key=lambda x: x["date"])
            return h["date"], f"~{h['expr']}"
        return day0(ref), ""
    same = [h for h in hits if h["kind"] == "same-day"]
    if same:
        return day0(ref), "today"
    h = min(hits, key=lambda x: x["date"])     # 最早的显式锚点 = 事件发生时刻
    return h["date"], h["expr"]


# ----------------------------------------------------------------------------
# 3. 题族判定 + 问句锚点解析
# ----------------------------------------------------------------------------

FAMILY_RULES = [
    ("delta_between", [
        r"how (many|much) \w+ (passed |had passed |have passed )?between",
        r"how many (days|weeks|months|years) (had |have )?passed between",
        r"how many (days|weeks|months|years) (before|prior to)",
        r"how (long|many \w+) (did it take|had (i|it)|have i|did i (take|spend|use))",
        r"how many \w+ (ago did i .* when i|had passed since i .* when i)",
        r"how many \w+ (have|had) i been .* when i",
        r"how long (had|have) i been .* when i",
        r"how long did i (use|have|own) .* before",
        r"how many \w+ before",
    ]),
    ("delta_since", [
        r"how many (days|weeks|months|years) ago",
        r"how many (days|weeks|months|years) (have|has|had) passed since",
        r"how long ago",
    ]),
    ("ordering", [
        r"what is the order of", r"in the order from", r"from earliest to latest",
        r"which .*\b(first|earliest)\b", r"who .*\bfirst\b",
        r"which .* most recently", r"most recent(ly)?\b.*\?",
        r"\bfirst, second and third\b", r"which came first",
    ]),
    ("count_before", [r"how many .* (before|prior to|since)"]),
    ("anchor_lookup", [
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an|a couple of|a few)\s+(day|week|month|year)s?\s+ago\b",
        r"\blast (week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend)\b",
        r"\b(this past|the past)\s+(week|month|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(yesterday|today)\b",
        r"\b(valentine|christmas|thanksgiving|halloween|easter|new year)\b",
        r"\bon (january|february|march|april|may|june|july|august|september|october|november|december)\b",
    ]),
]
_FAM = [(name, [re.compile(p) for p in pats]) for name, pats in FAMILY_RULES]
_UNIT_Q = re.compile(r"\bhow (?:many|long) (days|weeks|months|years)\b|\b(days|weeks|months|years)\s+ago\b")


def detect_family(question):
    """返回 (family, unit)。family ∈ delta_between|delta_since|ordering|anchor_lookup|count_before|other。
    优先级:delta_between > delta_since > ordering > count_before > anchor_lookup。"""
    q = question.lower()
    um = _UNIT_Q.search(q)
    unit = (um.group(1) or um.group(2)) if um else None
    if unit is None:
        for u in ("week", "month", "day", "year"):
            if re.search(r"how long", q) and u in q:
                unit = u + "s"
                break
    fam = "other"
    for name, pats in _FAM:
        if any(p.search(q) for p in pats):
            fam = name
            break
    return fam, unit


def resolve_question_anchors(question, question_date):
    """问句里的相对/命名日期表达 → 绝对日期。含 '<offset> 的那个 <weekday>' 复合形式。"""
    ref = parse_date(question_date)
    hits = resolve_expressions(question, question_date)
    # 复合:"the Wednesday two months ago" → 先偏移再吸附到该 weekday
    wd = re.search(r"\bthe\s+" + _WD + r"\b", question.lower())
    offs = [h for h in hits if h["kind"] in ("offset", "offset-coarse")]
    if wd and offs:
        base = min(offs, key=lambda x: x["date"])["date"]
        target = WEEKDAYS[wd.group(1)[:3]]
        cand = _prev_weekday(base + timedelta(days=3), target, strict=False)
        hits.append({"expr": f"{wd.group(0)} + {offs[0]['expr']}", "date": cand,
                     "conf": "high", "kind": "weekday-composite"})
    order = {"high": 0, "med": 1, "low": 2}
    hits.sort(key=lambda h: (order[h["conf"]], h["date"]))
    return hits


# ----------------------------------------------------------------------------
# 4. 候选事件抽取(相关性打分)
# ----------------------------------------------------------------------------

STOP = set("""a an the and or but if of to in on at for with from by as is are was were be been being
do does did doing done have has had i me my we our you your it its this that these those what which who
whom when where why how many much long ago did i's about there their them they he she his her not no yes
so than then too very can could will would should may might must just also more most some any all one
about between before after during since while over under again further once here now day days""".split())


# 只在**问句侧**剥掉的提问形式词:它们描述"问法"而不是内容,却常带高 IDF,
# 会把"order some essential oils"这类噪声顶到时间线前排(实测 ordering 族的主要污染源)。
QFORM_STOP = set("""order earliest latest first second third fourth fifth sixth among happened
event events thing things time times passed spend spent total mentioned narrator both which
week weeks month months year years old long ago far back""".split())


def _stem(w):
    """极简词干(确定性):只砍常见屈折后缀,保留 >=4 字符主干。
    不做词典化——'start' vs 'started'、'visit' vs 'visited' 之前完全匹配不上,是相关性打分的主要漏召回源。"""
    if not USE_STEM or len(w) < 5:
        return w
    for suf in ("ing", "ied", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            base = w[:-len(suf)]
            if suf == "ied":
                base += "y"
            if suf == "ing" and len(base) > 3 and base[-1] == base[-2] and base[-1] not in "aeiou":
                base = base[:-1]              # running -> run
            return base
    return w


def _content_tokens(s, qside=False):
    toks = [w for w in re.findall(r"[a-z0-9']+", s.lower()) if w not in STOP and len(w) > 2]
    if qside and USE_QFORM_STOP:
        toks = [w for w in toks if w not in QFORM_STOP]
    return [_stem(w) for w in toks]


def _sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and len(p.strip()) > 12]


# 意图/计划句 vs 已发生事件句:排序题里"I'm planning to visit the Modern Art Museum"会把
# 一个未发生的事件插进时间线并顶掉真事件,故软降权(不硬删,有些事件确实写在这种句子里)。
INTENT_RE = re.compile(r"\b(planning to|plan to|thinking of|thinking about|hoping to|"
                       r"would like to|looking forward to|going to|am about to|next week|"
                       r"next month|next time|haven't yet|have yet to)\b", re.I)
# 当日锚定句:事件时间 == 会话时间,是时间线上最可靠的一类行
ANCHOR_RE = re.compile(r"\b(today|yesterday|this morning|last night|tonight|just\s+(?:\w+ed|got|came|"
                       r"did|went|ate|had|saw|read|met|flew|took|made|left|won|drove|built|sent|paid|"
                       r"ran|spent|wrote|baked|caught|brought))\b", re.I)

# 三个打分软权重。**实测(v5_temporal_ablate1.py, TR n=133)四种组合对全部指标的影响都在噪声内**,
# 故默认全 1.0 = 纯 IDF 重叠,不引入无收益的超参。保留开关供论文消融表引用。
W_CUE, W_ANCHOR, W_INTENT = 1.0, 1.0, 1.0
# 下面两个默认关闭:v5_temporal_ablate2.py 实测(TR n=133)问句形式词剥离是净负(deltaTL 0.750→0.717,
# datesPresent 0.948→0.937),谓词扫描效果**恰好为零**(全部指标逐位相同)。留开关只为消融表可复现,
# 不要因为"听起来该有用"而打开。真正承重的部件是句内相对表达解析(关掉:deltaTL 0.750→0.617,
# 排序可判定率 0.628→0.488,排序正确率 0.395→0.186,同日题拆分 28→0)。
USE_QFORM_STOP = False
USE_SWEEP = False
USE_STEM = True              # 轻量词干:实测 deltaTL 0.750→0.800、证据日期覆盖 0.948→0.958,保留
# "问句点名的东西在记忆里根本没有"提示,针对弃答题。**默认关闭**:实测(TR ordering 43 题)
# 命中 2/3 弃答题,但在 40 道可答题上误报 4 次 → 精度 2/6=0.33。本臂本来就在弃答题上偏爱回答,
# 一个 0.33 精度的"可能无法回答"提示会把可答题推向拒答,净收益为负。集成方要用请自行打开并复核。
EMIT_MISSING_ITEM_NOTE = False

EVENT_CUE = re.compile(
    r"\b(today|yesterday|last night|just|recently|finally|ago|last (week|month|year|monday|tuesday|"
    r"wednesday|thursday|friday|saturday|sunday|weekend))\b|"
    r"\b(i|we)\s+(\w+ed|went|got|bought|took|made|had|met|did|saw|read|ran|flew|sold|began|started|"
    r"finished|attended|received|visited|joined|signed|set up|gave|found|left|won|wore|drove|built)\b",
    re.I)


class Event(object):
    __slots__ = ("ev_dt", "sess_dt", "text", "src", "score", "prov", "rank", "dates")

    def __init__(self, ev_dt, sess_dt, text, src, score, prov, rank=0, dates=None):
        self.ev_dt, self.sess_dt, self.text = ev_dt, sess_dt, text
        self.src, self.score, self.prov, self.rank = src, score, prov, rank
        self.dates = dates or [ev_dt]        # 该句能定出的全部候选日期(含 ev_dt)


def sentence_dates(text, session_dt):
    """一句话里全部可靠日期(高置信、非未来计划),用于两两算术的候选池。"""
    ref = parse_date(session_dt)
    if ref is None:
        return []
    out = {day0(ref)}
    for h in resolve_expressions(text, ref):
        if h["conf"] == "high" and h["kind"] != "explicit-future":
            out.add(h["date"])
    return sorted(out)


def _idf(q, docs_tokens):
    import math
    n = max(1, len(docs_tokens))
    df = {}
    for d in docs_tokens:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    return {w: math.log(1 + (n - df.get(w, 0) + 0.5) / (df.get(w, 0) + 0.5)) for w in q}


FAMILY_BUDGET = {          # 题族决定检索预算:排序题要"每个被点名的实体都有一行",预算必须更宽
    "ordering":      dict(max_events=26, rel_floor=0.10, per_date_turns=2, per_date_facts=2, sweep=True),
    "count_before":  dict(max_events=26, rel_floor=0.10, per_date_turns=2, per_date_facts=2, sweep=True),
    "delta_between": dict(max_events=18, rel_floor=0.25, per_date_turns=2, per_date_facts=1, sweep=False),
    "delta_since":   dict(max_events=18, rel_floor=0.25, per_date_turns=2, per_date_facts=1, sweep=False),
    "anchor_lookup": dict(max_events=20, rel_floor=0.18, per_date_turns=2, per_date_facts=1, sweep=False),
    "other":         dict(max_events=18, rel_floor=0.25, per_date_turns=2, per_date_facts=1, sweep=False),
}


def predicate_sweep(store, question, cap=14):
    """谓词扫描:把 store 里"属性与问句共享内容词"的事实链**全部**取出(同谓词不同宾语),
    这正是排序/计数题需要的完整实体集('user | visited = X' 的所有 X)。不受相关性下限约束。"""
    if not store:
        return []
    qtok = set(_content_tokens(question, qside=True))
    out = []
    for k in sorted(store):
        attr = k.split("::", 1)[1]
        if not (set(attr.split()) & qtok):
            continue
        for c in store[k]:
            out.append((c["date"], f"{c['entity']} | {c['attribute']} = {c['value']}"))
    seen, res = set(), []
    for date_s, txt in out:
        key = (date_s.split()[0], txt.lower()[:70])
        if key in seen:
            continue
        seen.add(key)
        res.append((date_s, txt))
        if len(res) >= cap:
            break
    return res


def extract_events(q, store=None, question=None, max_events=18, sent_chars=190,
                   rel_floor=0.25, per_date_turns=2, per_date_facts=1, family=None,
                   sweep=None):
    """从 haystack 用户轮 + (可选) lifecycle 事实链抽"与问题相关的带日期事件"。

    相关性 = 问题内容词的 IDF 加权命中数(纯代码,无 API,确定性)。
    事件时间优先用句内解析出的绝对日期(如 'a month ago' @ 会话 5/25 → 4/25),否则用会话日期。
    预算控制三道闸:相对分数下限 rel_floor(相对最高分)、每日期配额、全局 max_events。
    返回按事件时间排序的 Event 列表;每个 Event 的 .rank 是相关性排名(0 = 最相关)。
    """
    question = question or q["question"]
    if family is None:
        family = detect_family(question)[0]
    b = FAMILY_BUDGET.get(family)
    if b:
        max_events, rel_floor = b["max_events"], b["rel_floor"]
        per_date_turns, per_date_facts = b["per_date_turns"], b["per_date_facts"]
        if sweep is None:
            sweep = b["sweep"] and USE_SWEEP
    qtok = _content_tokens(question, qside=True)
    cands, all_tokens = [], []
    for date_s, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        for t in sess:
            c = t.get("content") or ""
            if t.get("role") != "user" or not c:
                continue
            for s in _sentences(c):
                st = _content_tokens(s)
                if not st:
                    continue
                all_tokens.append(st)
                cands.append((date_s, s, st, "turn"))
    if store:
        for k in sorted(store):
            for c in store[k]:
                s = f"{c['entity']} | {c['attribute']} = {c['value']}"
                st = _content_tokens(s)
                all_tokens.append(st)
                cands.append((c["date"], s, st, "fact"))
    idf = _idf(set(qtok), all_tokens)
    qset = set(qtok)
    scored = []
    for date_s, s, st, src in cands:
        hit = qset & set(st)
        if not hit:
            continue
        sc = sum(idf.get(w, 0.0) for w in hit)
        if src == "turn":
            if EVENT_CUE.search(s):
                sc *= W_CUE                 # 事件线索加权(软权重,不硬过滤)
            if ANCHOR_RE.search(s):
                sc *= W_ANCHOR              # 当日锚定 = 时间线最可靠的一类行
            if INTENT_RE.search(s):
                sc *= W_INTENT              # 计划/意图句降权(事件尚未发生)
        elif re.match(r"^\S+ \| (plans? to|planning|thinking of|interested in|looking forward|wants?)",
                      s, re.I):
            sc *= W_INTENT
        scored.append((sc, date_s, s, src))
    if not scored:
        return []
    scored.sort(key=lambda x: (-x[0], x[1]))
    floor = scored[0][0] * rel_floor
    if sweep:                       # 谓词扫描结果插到最前面,豁免相关性下限
        top = scored[0][0]
        sw = [(top + 1e-6, date_s, txt, "fact") for date_s, txt in predicate_sweep(store, question)]
        have = {(d.split()[0], t.lower()[:70]) for _, d, t, _ in scored}
        scored = [x for x in sw if (x[1].split()[0], x[2].lower()[:70]) not in have] + scored
    events, seen, per_date = [], set(), {}
    for rank, (sc, date_s, s, src) in enumerate(scored):
        if sc < floor:
            break
        sess_dt = parse_date(date_s)
        if sess_dt is None:
            continue
        ev_dt, prov = best_event_date(s, date_s) if src == "turn" else (day0(sess_dt), "")
        txt = re.sub(r"\s+", " ", s).strip()
        if len(txt) > sent_chars:
            txt = txt[:sent_chars].rsplit(" ", 1)[0] + "..."
        dk = (ev_dt.date(), src)
        quota = per_date_turns if src == "turn" else per_date_facts
        if per_date.get(dk, 0) >= quota:
            continue
        key = (ev_dt.date(), txt[:60].lower())
        if key in seen:
            continue
        seen.add(key)
        per_date[dk] = per_date.get(dk, 0) + 1
        dts = sentence_dates(s, date_s) if src == "turn" else [day0(sess_dt)]
        if ev_dt not in dts:
            dts = sorted(set(dts) | {ev_dt})
        events.append(Event(ev_dt, sess_dt, txt, src, sc, prov, rank, dts))
        if len(events) >= max_events:
            break
    events.sort(key=lambda e: (e.ev_dt, e.rank))
    return events


def top_dates_by_relevance(events, n, all_dates=True):
    """按相关性挑 n 个不同的候选日期,再按时间排序返回 [datetime]。
    delta_between 的两两 delta 必须建在最相关的日期上,否则 gold 那一对会被 cap 截掉;
    all_dates=True 时把同一句里的其它显式日期也纳入候选池。"""
    out = []
    for e in sorted(events, key=lambda x: x.rank):
        for d in (e.dates if all_dates else [e.ev_dt]):
            if d not in out:
                out.append(d)
            if len(out) >= n:
                return sorted(out)
    return sorted(out)


# ----------------------------------------------------------------------------
# 5. 预计算算术
# ----------------------------------------------------------------------------

def anchor_day_events(q, anchor_dt, cap=10, sent_chars=175, window=0, question=None):
    """锚日直取:问句解析出高置信锚点后,**不看词面相关性**把那一天发生的事件句全列出来。

    动机(实测 71017277):'I received a piece of jewelry last Saturday from whom?' 锚点解析完全正确
    (2023/03/04),但 'jewelry' 和证据里的 'crystal chandelier' 词面不重叠,相关性排序永远捞不到它。
    锚点型问题问的就是"那天发生了什么",所以那天该整天摊开,让模型自己认语义。
    """
    out = []
    qtok = set(_content_tokens(question or q.get("question", ""), qside=True))
    for date_s, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        sd = parse_date(date_s)
        if sd is None:
            continue
        same_session = abs((day0(sd) - anchor_dt).days) <= window
        for t in sess:
            if t.get("role") != "user":
                continue
            for sent in _sentences(t.get("content") or ""):
                ev, prov = best_event_date(sent, date_s)
                if ev is None:
                    continue
                # 兜底只在"句里没给别的日期"时用:'my cousin Rachel's baby shower in February'
                # 说在 6/15 的会话里,事件在 2 月,不能因为有 'just got back' 就算成锚日事件。
                default_dated = prov in ("", "today")
                if abs((ev - anchor_dt).days) > window and not (
                        same_session and default_dated and ANCHOR_RE.search(sent)):
                    continue
                cued = bool(ANCHOR_RE.search(sent)) or bool(EVENT_CUE.search(sent))
                # 锚日直取要宽:只要是第一人称陈述就留(EVENT_CUE 漏掉 "I'm still recovering from my
                # American Airlines flight" 这类完成体/被动式,实测 gpt4_f420262d 就栽在这)
                if not (cued or re.search(r"\bi\b|\bi'|\bmy\b|\bwe\b|\bour\b", sent, re.I)):
                    continue
                if not cued and sent.rstrip().endswith("?"):
                    continue                          # 纯提问句无事件信息
                if INTENT_RE.search(sent):
                    continue
                txt = re.sub(r"\s+", " ", sent).strip()
                if len(txt) > sent_chars:
                    txt = txt[:sent_chars].rsplit(" ", 1)[0] + "..."
                pri = 0 if ANCHOR_RE.search(sent) else (1 if cued else 2)
                # 同优先级内按问句词面重叠排序:不排除任何行(保住"不看相关性"的性质),
                # 只决定 cap 内的先后。实测 gold 行在 12/17 例排第 1。
                rel = len(qtok & set(_content_tokens(sent)))
                out.append((pri, -rel, day0(sd), txt))
    seen, res = set(), []
    for pri, nrel, sd, txt in sorted(out, key=lambda x: (x[0], x[1], x[2])):
        k = txt.lower()[:60]
        if k in seen:
            continue
        seen.add(k)
        res.append(txt)
        if len(res) >= cap:
            break
    return res


ANCHOR_WINDOW = {"offset": (1, 3), "weekday": (1, 3), "holiday": (1, 3), "explicit": (1, 3),
                 "weekday-composite": (1, 3), "same-day": (0, 1),
                 "offset-coarse": (4, 10), "month-anchor": (10, 16), "duration-start": (3, 7)}


def anchor_block(q, anchor, cap=10):
    """锚日直取 + 逐级放宽窗口:锚点日没有会话就扩到 ±3 天(实测 'four weeks ago' 解析出 02/28,
    真证据会话在 03/01,差 1 天;'last week'/'last month' 这类粗锚点差更多)。
    返回 (lines, used_window)。"""
    w0, w1 = ANCHOR_WINDOW.get(anchor["kind"], (1, 3))
    for w in (w0, w1):
        got = anchor_day_events(q, anchor["date"], cap=cap, window=w, question=q.get("question"))
        if got:
            return got, w
    return [], w1


_ALT_RE = re.compile(r"(?:\bfirst\b|\bearliest\b|\bmost recently\b|\blatest\b|\bsecond\b)[,:]?\s+(.+?)\?*$",
                     re.I)


def question_alternatives(question, max_items=4):
    """从问句里抽出被点名的候选项('Which task did I complete first, fixing the fence or
    purchasing three cows from Peter?' → 两项)。只用问句,不用 gold。"""
    m = _ALT_RE.search(question.strip())
    if not m:
        return []
    tail = m.group(1)
    parts = re.split(r",\s*(?:or\s+)?|\s+or\s+", tail)
    out = []
    for p in parts:
        p = p.strip(" .'\"")
        p = re.sub(r"^(the|a|an|my|i)\s+", "", p, flags=re.I).strip()
        if len(p) > 3 and len(_content_tokens(p)) >= 1:
            out.append(p)
    return out[:max_items] if len(out) >= 2 else []


def match_item_date(item, events, frac=0.6):
    """item 文本 → 时间线上最早匹配事件的日期(None = 时间线里找不到这个东西)。"""
    it = [w for w in _content_tokens(item) if len(w) > 2]
    if not it:
        return None
    best = None
    for e in events:
        et = set(_content_tokens(e.text))
        hit = sum(w in et for w in it)
        need = len(it) if len(it) <= 2 else max(2, int(round(frac * len(it))))
        if hit >= need and (best is None or e.ev_dt < best):
            best = e.ev_dt
    return best


def _delta_str(d1, d2, unit=None):
    nd = days_between(d1, d2)
    out = [f"{nd} days"]
    out.append(f"{nd / 7.0:.1f} weeks")
    out.append(f"{months_between(d1, d2)} full months")
    if unit:
        u = unit.rstrip("s")
        if u == "week":
            return f"{nd} days = {nd / 7.0:.1f} weeks (= {nd // 7} full weeks)"
        if u == "month":
            return f"{nd} days = {months_between(d1, d2)} full months ({nd / 30.44:.1f} calendar months)"
        if u == "day":
            return f"{nd} days"
    return " = ".join(out[:2])


def precompute_arithmetic(question, question_date, events, family, unit,
                          max_pairs=15, n_pair_dates=6, q_for_anchor=None):
    """按题族预计算确定性算术。返回文本行列表(可能为空)。全部纯算术,无 API。"""
    q_dt = parse_date(question_date)
    lines = []
    anchors = resolve_question_anchors(question, question_date)
    high = [a for a in anchors if a["conf"] == "high"]
    if not high:      # 没有高置信锚点时退到粗锚点('last week'/'last month'),窗口更宽
        high = [a for a in anchors if a["kind"] in ("offset-coarse", "month-anchor")][:1]
    if anchors:
        lines.append("Dates named or implied by the question, resolved:")
        for a in anchors[:6]:
            d = a["date"]
            lines.append(f"  \"{a['expr']}\" -> {d.year}/{d.month:02d}/{d.day:02d} ({d.strftime('%a')})"
                         f"  [{offset_phrase(d, q_dt)}; confidence {a['conf']}]")
    # 按相关性挑日期(而不是按时间取前几个),否则 cap 会把 gold 那一对截掉
    rel_dates = top_dates_by_relevance(events, n_pair_dates)
    date_gist = {}
    for e in sorted(events, key=lambda x: x.rank):
        date_gist.setdefault(e.ev_dt, e.text[:70])

    if family in ("delta_since", "count_before") or (family == "anchor_lookup" and not high):
        allsorted = top_dates_by_relevance(events, 10)
        if allsorted:
            lines.append(f"Elapsed time from each candidate event date to the question date "
                         f"({q_dt.year}/{q_dt.month:02d}/{q_dt.day:02d}):")
            for d in allsorted:
                lines.append(f"  {d.year}/{d.month:02d}/{d.day:02d} ({d.strftime('%a')}) "
                             f"-> {_delta_str(d, q_dt, unit)}   | {date_gist.get(d, '')}")
    if family == "delta_between" and len(rel_dates) >= 2:
        lines.append("Pairwise elapsed time between the most relevant event dates "
                     "(the arithmetic is already done; pick the pair the question names):")
        n = 0
        for i in range(len(rel_dates)):
            for j in range(i + 1, len(rel_dates)):
                a, b = rel_dates[i], rel_dates[j]
                lines.append(f"  {a.year}/{a.month:02d}/{a.day:02d} -> "
                             f"{b.year}/{b.month:02d}/{b.day:02d} : {_delta_str(a, b, unit)}")
                n += 1
                if n >= max_pairs:
                    break
            if n >= max_pairs:
                break
        lines.append("  (also, from each of those dates to the question date: "
                     + "; ".join(f"{d.month:02d}/{d.day:02d}={days_between(d, q_dt)}d"
                                 for d in rel_dates) + ")")
    if family == "ordering":
        alts = question_alternatives(question)
        missing = [a for a in alts if match_item_date(a, events) is None]
        if alts and missing and EMIT_MISSING_ITEM_NOTE:
            lines.append("Named in the question but NO dated event for it was found anywhere in memory "
                         "(so it may be unanswerable rather than orderable): "
                         + "; ".join(f'"{a}"' for a in missing))
    if family == "ordering" and events:
        seen_d = []
        lines.append("Chronological order of the candidate events (earliest -> latest):")
        i = 0
        for e in events:
            if e.ev_dt in seen_d:
                continue
            seen_d.append(e.ev_dt)
            i += 1
            d = e.ev_dt
            lines.append(f"  {i}. {d.year}/{d.month:02d}/{d.day:02d} ({d.strftime('%a')}) {e.text[:95]}")
    if family == "anchor_lookup" and high:
        for a in high[:2]:
            tgt = a["date"]
            near = sorted(events, key=lambda e: (abs((e.ev_dt - tgt).days), e.rank))
            same = [e for e in near if abs((e.ev_dt - tgt).days) == 0]
            if q_for_anchor is not None:
                day, w = anchor_block(q_for_anchor, a)
                if day:
                    span = ("on the resolved anchor day" if w == 0 else
                            f"within +/-{w} days of the resolved anchor")
                    lines.append(f"Everything the user reported doing {span} "
                                 f"{tgt.year}/{tgt.month:02d}/{tgt.day:02d} "
                                 f"(listed regardless of wording overlap with the question):")
                    for txt in day:
                        lines.append(f"  - {txt}")
            close = [e for e in near if 0 < abs((e.ev_dt - tgt).days) <= 3][:3]
            lines.append(f"Events dated exactly on the resolved anchor "
                         f"{tgt.year}/{tgt.month:02d}/{tgt.day:02d} (\"{a['expr']}\"):")
            if same:
                for e in same[:4]:
                    lines.append(f"  * {e.text[:150]}")
            else:
                lines.append("  (none dated exactly on that day)")
            if close:
                lines.append("  Nearest other dated events (within 3 days of the anchor):")
                for e in close:
                    d = e.ev_dt
                    lines.append(f"    {d.year}/{d.month:02d}/{d.day:02d} "
                                 f"({(d - tgt).days:+d}d vs anchor) {e.text[:110]}")
    return lines


# ----------------------------------------------------------------------------
# 6. 时间线渲染
# ----------------------------------------------------------------------------

def consecutive_pairs(events, top_n=8):
    """在相关性靠前的事件里找"事件日相差恰好 1 天"的配对(锚点消歧的关键线索)。
    返回 ([(dt_a, dt_b)], {事件下标: 标记串})。"""
    rel = sorted([e for e in events if e.rank < top_n], key=lambda e: e.ev_dt)
    pairs, marks = [], {}
    idx = {id(e): i for i, e in enumerate(events)}
    for i in range(1, len(rel)):
        a, b = rel[i - 1], rel[i]
        if (b.ev_dt - a.ev_dt).days == 1:
            if (a.ev_dt, b.ev_dt) not in pairs:
                pairs.append((a.ev_dt, b.ev_dt))
            marks[idx[id(a)]] = " <<CONSECUTIVE-DAY PAIR>>"
            marks[idx[id(b)]] = " <<CONSECUTIVE-DAY PAIR>>"
    return pairs, marks


def render_timeline(q, store=None, question=None, max_events=18, include_arithmetic=True):
    """时间脚手架主入口。返回可直接拼进上下文的文本(与事实视图/原文轮并列，不替代它们)。

    q: 数据集单题 dict(需 question_date / haystack_dates / haystack_sessions)
    store: lifecycle 事实链(可为 None)
    question: 问题文本(缺省取 q['question'])
    """
    question = question or q["question"]
    q_dt = parse_date(q["question_date"])
    family, unit = detect_family(question)
    events = extract_events(q, store, question, max_events=max_events)
    pairs, marks = consecutive_pairs(events)

    out = [f"### Temporal scaffold (question asked on {q_dt.year}/{q_dt.month:02d}/{q_dt.day:02d} "
           f"({q_dt.strftime('%a')}); question family detected: {family}"
           + (f"; unit asked: {unit}" if unit else "") + ")"]
    if include_arithmetic:
        arith = precompute_arithmetic(question, q["question_date"], events, family, unit,
                                      q_for_anchor=q)
        if pairs:
            arith.append("Consecutive-day pairs among the relevant events (adjacent calendar days): "
                         + "; ".join(f"{a.month:02d}/{a.day:02d} & {b.month:02d}/{b.day:02d}"
                                     for a, b in pairs))
        if arith:
            out.append("-- Resolved dates and pre-computed arithmetic "
                       "(computed by rule from the dates below; trust these numbers) --")
            out.extend(arith)
    if events:
        out.append("-- Event timeline, sorted by the date the event happened "
                   "(offsets are relative to the question date) --")
        prev = None
        for i, e in enumerate(events):
            d = e.ev_dt
            gap = "" if prev is None else f" (+{(d - prev).days}d)"
            prov = ""
            if e.prov and e.prov != "today":
                prov = (f" [date inferred from \"{e.prov}\" said on "
                        f"{e.sess_dt.year}/{e.sess_dt.month:02d}/{e.sess_dt.day:02d}]")
            elif day0(e.sess_dt) != d:
                prov = f" [reported {e.sess_dt.year}/{e.sess_dt.month:02d}/{e.sess_dt.day:02d}]"
            extra = [x for x in e.dates if x != d and x != day0(e.sess_dt)]
            ex = (" [text also dates: " + ", ".join(f"{x.year}/{x.month:02d}/{x.day:02d}"
                                                    for x in extra) + "]") if extra else ""
            out.append(f"[{d.year}/{d.month:02d}/{d.day:02d} ({d.strftime('%a')}) | "
                       f"{offset_phrase(d, q_dt)}]{gap}{marks.get(i, '')}{prov}{ex} {e.text}")
            prev = d
    return "\n".join(out)


# ----------------------------------------------------------------------------
# 7. 自检(无 API)
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    ref = "2023/05/25 (Thu) 11:29"
    r = {h["expr"]: h["date"].strftime("%Y/%m/%d") for h in resolve_expressions(
        "I set up my smart thermostat a month ago and upgraded the router yesterday. "
        "I started Game of Thrones about a month ago. I flew on Valentine's Day. "
        "I visited on March 15th and again last Saturday, in mid-January too.", ref)}
    assert r["a month ago"] == "2023/04/25", r
    assert r["yesterday"] == "2023/05/24", r
    assert r["valentine's day"] == "2023/02/14", r
    assert r["march 15th"] == "2023/03/15", r
    assert r["last saturday"] == "2023/05/20", r
    assert r["mid-january"] == "2023/01/15", r
    assert parse_date("2023/03/19 (Sun) 15:02").hour == 15
    assert days_between(parse_date("2023/02/15 (Wed) 19:39"), parse_date("2023/04/18 (Tue) 03:31")) == 62
    assert months_between(parse_date("2023/02/15"), parse_date("2023/04/18")) == 2
    assert months_between(parse_date("2023/03/19"), parse_date("2023/04/18")) == 0  # 3/19->4/18 未满一月
    assert days_between(parse_date("2023/02/11"), parse_date("2023/03/04")) == 21
    # annotate_offsets
    ann = annotate_offsets(["[2023/03/19 (Sun) 15:02] [user] did the Walk for Hunger",
                            "no date here"], "2023/04/18 (Tue) 03:31")
    assert "30 days" in ann[0] and "4.3 weeks" in ann[0] and "before the question" in ann[0], ann[0]
    assert ann[1] == "no date here"
    # 题族
    assert detect_family("How many months have passed since I participated in two charity events in a row, on consecutive days?")[0] == "delta_since"
    assert detect_family("How many days passed between my visit to MoMA and the exhibit?")[0] == "delta_between"
    assert detect_family("What is the order of the six museums I visited from earliest to latest?")[0] == "ordering"
    assert detect_family("What gardening-related activity did I do two weeks ago?")[0] == "anchor_lookup"
    assert detect_family("Which show did I start watching first, 'The Crown' or 'Game of Thrones'?")[0] == "ordering"
    assert detect_family("How many weeks have I been taking sculpting classes when I invested in my own set of tools?")[0] == "delta_between"
    # 问句锚点
    a = resolve_question_anchors("What did I do with Rachel on the Wednesday two months ago?", "2023/05/10 (Wed) 09:00")
    assert any(h["date"].weekday() == 2 and h["date"].month == 3 for h in a), a
    # 复活节派生
    h = holiday_dates(2023)
    assert h["easter"] == (4, 9) and h["maundy thursday"] == (4, 6) and h["ash wednesday"] == (2, 22), h
    # 新部件回归:数字日期 / 完成型时长 / 未来计划日期 / 词干 / 问句候选项
    r2 = {x["expr"]: (x["date"].strftime("%Y/%m/%d"), x["kind"]) for x in resolve_expressions(
        "I got a headphone for my brother as a graduation gift on the 3/8. I finished the season "
        "in just 14 days watching at least 2-3 episodes. My flight departs on August 15th.",
        "2023/03/29 (Wed) 21:27")}
    assert r2["3/8"] == ("2023/03/08", "explicit"), r2
    assert "2-3" not in r2, r2                                  # '2-3 episodes' 不是日期
    assert r2["august 15th"][1] == "explicit-future", r2         # 会话日之后=计划,不当过去事件
    assert any(v == ("2023/03/15", "duration-start") for v in r2.values()), r2
    assert _stem("started") == "start" and _stem("visited") == "visit" and _stem("running") == "run"
    alts = question_alternatives("Which task did I complete first, fixing the fence or "
                                 "purchasing three cows from Peter?")
    assert len(alts) == 2 and "fence" in alts[0] and "cows" in alts[1], alts
    assert offset_phrase(parse_date("2023/02/15"), parse_date("2023/04/18")).startswith("62 days")
    print("v5_temporal self-test OK")
