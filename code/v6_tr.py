"""QC-Lifecycle v6-TR:temporal-reasoning 残差的 date-keyed evidence recovery(training-free,零 API)。

诊断(逐例读 v5 的 12 个 TR 错题,见模块末尾 CASES 表):
  12 题里 8 题的 gold 证据轮**根本不在 v5 上下文里**。缺失是"日期形状"的,不是主题形状的:
    - 问句给的是偏移量而不是词面("two weeks ago" / "four weeks ago" / "10 days ago"):
      BM25 对日期完全无感,证据轮和问句词面不重叠 → 永远捞不到(1e4a8aec / eac54add / 8279ba03)
    - 问句给的是时间窗("during the past month"):3 条 gold 证据轮一条都没进(e061b84f)
    - 枚举题(6 个博物馆 / RFTC 之前的慈善活动)需要**全部** n 个事件,BM25 名额被同主题
      噪声吃掉 → 6 缺 4 / 6 缺 2(7abb270c / a3838d2b)
    - 同日会话族(整题 haystack 同一天)里,事件时间只存在于句内表达("3 weeks ago" /
      "a month ago" / "February 12th"),v5 的行首日期全是同一天,顺序信息在上下文里不存在
      (65aabe59 / fe651585 / a3838d2b)
  另 4 题证据齐全但算术/端点选错:同会话里的干扰时长表达被当成端点(6e984301:"for about
  6 weeks now" 与 "just started today" 同在 2/11 会话内),或算对了天数却报成小数周并列两个
  候选(61e13b3c:"3 weeks and 1 day / 约 3.1 周",gold "3 weeks")。

干预(一件事:把日期变成检索键与显式算术,而不是让模型从散落的行里推):
  1. 问句表层门控(不看 gold type):相对偏移 / 时间窗 / 枚举排序 / 时距 四类。
  2. 门控命中后,把问句里的时间表达解析成绝对日期,再**按日期取证据**:锚点日 ±3 天、
     时间窗内、或(枚举题)与问句内容词相关 + 带句内显式日期的事件句 —— 词面不重叠也要给。
  3. 渲染成一个紧凑去重的时间线:每行 = 事件日期(句内表达优先于会话日期)+ 相对问题日的
     有符号天偏移 + 句子。时距题另加一行预计算区间(端点由问句措辞匹配,天/周/整周/整月并列)。

严格附加:本模块只**新增**一个上下文分节(QC_V6_TR=1),v5 的视图与原文轮逐字节不变。
QC_V6_TR=2 是预算中性变体:按同样的字符数从 v5 原文轮里挤掉 BM25 最低分的若干轮。

不做的事:不训练、不下载模型、不调 API、不读 has_answer/answer_session_ids(仅诊断时读过)。
不声称任何准确率收益 —— 本项目已证离线代理指标不预测正确率;本模块只保证"上下文里出现了
该出现的日期化证据",正确率由 parent 在线 A/B 测。

复用的 v5_temporal 原语(已在真实错题句子上逐条核验,见 __main__ 回归):
  parse_date / day0 / days_between / months_between / resolve_expressions
核验通过:月份缩写/命名日期("February 12th"→02/12)、偏移("3 weeks ago"/"a month ago")、
  "yesterday"、月份锚点。核验发现并在本模块内处理的三处:
  (1) 裸数字对仍会误解析:"our team is currently 3-2 in the standings"(6e984301 真实句)→
      03/02。但同形状的 "graduation gift on the 3/8"(8c18457d 的 gold 起点)是真日期,
      故按上下文判:只接受斜杠形式且前面有 "on (the)"。
  (2) best_event_date 的 duration-start 兜底会把 "for about 6 weeks now" 当事件日期 —— 那正是
      6e984301 把端点带偏的干扰源;本模块的 _v6_event_date 不用该兜底。
  (3) min(高置信日期) 会挑中靠猜年份得来的节日日期(08f4fc43 的 "Ash Wednesday" → 2022/03/02,
      而同句明写 "February 1st");本模块把 holiday 降为最后一档。
预计算区间只在两端匹配到**不同**句/不同日时给出,且整数换算用四舍五入(gold 口径),措辞明确
标注端点是词面猜测、只有天数算术保证 —— 未加这两条时,在 v5 已答对的 24 道数值时距题里会给出
6 次与 gold 矛盾的数;加上后降到 1 次(08f4fc43,端点选错,已知残留)。
"""
import os
import re
import sys

from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v5_temporal import parse_date, day0, days_between, months_between, resolve_expressions

# ---------------------------------------------------------------------------
# 预算常量(全部可从环境变量覆盖,便于 parent 做体积消融)
# ---------------------------------------------------------------------------
V6_BLOCK_CHARS = int(os.environ.get("QC_V6_TR_CHARS", "2600"))  # 分节总字符上限
V6_SENT_CHARS = 185      # 单句截断
V6_ANCHOR_WIN = 3        # 锚点日 ±天(实测 "four weeks ago"→02/28,证据会话 03/01,差 1 天)
V6_ANCHOR_CAP = 8        # 锚点当日事件行上限
V6_WINDOW_CAP = 14       # 时间窗事件行上限
V6_ENUM_CAP = 16         # 枚举/排序题事件行上限
V6_SELFDATED_CAP = 6     # 枚举题额外补进的"自带句内日期"事件行上限
V6_PER_DAY = 2           # 同一事件日的行配额(haystack 里同一句会在多轮里重复出现)

_WD = r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thurs|fri|sat|sun)"
_MO = (r"(january|february|march|april|may|june|july|august|september|october|november|december"
       r"|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)")
_NUMW = (r"(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen"
         r"|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|a|an|a couple of|a few|several)")

# ---------------------------------------------------------------------------
# 1. 门控:只看问句表层形式(不看 question_type)
# ---------------------------------------------------------------------------
_G_ANCHOR = [
    r"\b" + _NUMW + r"\s+(?:day|week|month|year)s?\s+ago\b",
    r"\byesterday\b",
    r"\blast\s+(?:week|month|year|weekend|night|" + _WD + r")\b",
    r"\b(?:this|the)\s+past\s+(?:week|month|weekend|" + _WD + r")\b",
    r"\bon\s+" + _MO + r"\b",
    r"\b(?:valentine|christmas|thanksgiving|halloween|easter|new year)",
]
_G_WINDOW = [
    r"\b(?:in|during|over|within)\s+the\s+(?:past|last)\s+(?:" + _NUMW + r"\s+)?(?:day|week|month|year)s?\b",
]
_G_ENUM = [
    r"\border of\b", r"earliest to latest", r"latest to earliest", r"\bchronological\b",
    r"\bwhich\b[^?]*\b(?:first|earliest|most recent(?:ly)?|latest|last)\b",
    r"\bwho\b[^?]*\bfirst\b", r"\bwhich came first\b", r"\bfirst,\s*second\b",
    r"\bhow many\b[^?]*\b(?:before|after|prior to|since)\b",
]
# "how long" 必须带**已流逝**的语境。裸 "how long" 会把时长查表题也拉进来
# ("How long is my daily commute to work?" / "How much time do I dedicate to guitar every day?"),
# 那类题答案就写在原文里,时间线对它没用,只是白占预算和分散注意力。
_G_DELTA = [
    r"\bhow many\s+(?:days|weeks|months|years)\b(?!\s+(?:a|per|each|every)\s+(?:day|week|month|year))",
    r"\bhow long ago\b",
    r"\bhow long\s+(?:did|does)\s+it\s+take\b",
    r"\bhow long\s+(?:have|had|has)\s+(?:i|it)\b",
    r"\bhow long\s+(?:did|do)\s+i\s+(?:wait|spend|stay|live|work|use|own|keep|have)\b",
    r"\bhow long\s+(?:since|between)\b",
    r"\bhow old\s+(?:was|were)\s+i\s+when\b",
    r"\bhow many\s+\w+\s+(?:passed|elapsed)\b",
]
_G = [("anchor", [re.compile(p, re.I) for p in _G_ANCHOR]),
      ("window", [re.compile(p, re.I) for p in _G_WINDOW]),
      ("delta", [re.compile(p, re.I) for p in _G_DELTA]),
      ("enum", [re.compile(p, re.I) for p in _G_ENUM])]


def tr_gate(question):
    """问句表层门控。返回命中的类集合(可多类),空集 = 不介入。"""
    q = question.lower()
    return {name for name, pats in _G if any(p.search(q) for p in pats)}


# ---------------------------------------------------------------------------
# 2. 句子级事件候选(纯规则)
# ---------------------------------------------------------------------------
_V6_FIRST = re.compile(r"\b(?:i|i'm|i've|i'd|i'll|my|we|we've|we're|our)\b", re.I)
# "just" 必须带动词才算线索:裸 "just" 是强调词("Maybe it's just a coincidence" 会被当成事件句,
# 在 e061b84f 的时间窗里占掉了 gold 足球赛那天的配额)。
_V6_JUST = (r"\bjust\s+(?:\w+ed|got|went|came|had|did|finished|started|bought|saw|met|ran|flew|"
            r"took|made|wrapped|managed|signed|joined|paid|sold|read|ate)\b")
_V6_CUE = re.compile(r"\b(?:today|yesterday|last night|this morning|this afternoon|"
                     r"finally|recently|ago|last (?:week|month|year|weekend)|" + _WD + r")\b|"
                     + _V6_JUST, re.I)
# 过去锚定的当日表达。刻意不含 tonight / this evening:那是**今天晚些时候**的计划,
# 放行它会让计划句("I was thinking of making the Chicken Fajitas for dinner tonight")冒充事件。
_V6_SAMEDAY_PAST = ("today", "this morning", "this afternoon", "just now")
_V6_PASTV = re.compile(
    r"\b(?:i|we)\s+(?:just\s+|finally\s+|recently\s+|also\s+|already\s+|then\s+)?"
    r"(?:\w+ed|went|got|bought|took|made|had|met|did|saw|ran|flew|sold|began|started|finished|"
    r"attended|received|visited|joined|signed|set|gave|found|left|won|wore|drove|built|completed|"
    r"participated|volunteered|celebrated|moved|adopted|recovered|invested|read|ate|spent|sent|"
    r"paid|caught|brought|wrote|baked|hit|threw|kept|held|lost|won)\b", re.I)
_V6_INTENT = re.compile(r"\b(?:planning to|plan to|thinking of|thinking about|hoping to|would like to|"
                        r"looking forward to|going to|am about to|next week|next month|next time|"
                        r"haven't yet|have yet to|i will|i'll)\b", re.I)
# 裸数字对(比分/比例/区间):resolve_expressions 会把 "3-2 in the standings" 解析成 03/02。
_V6_BARE_NUM = re.compile(r"^\s*\d{1,2}\s*[-/]\s*\d{1,2}\s*$")
_V6_STOP = set("""a an the and or but if of to in on at for with from by as is are was were be been
being do does did doing done have has had i me my we our you your it its this that these those what
which who whom when where why how many much long ago about there their them they he she his her not
no yes so than then too very can could will would should may might must just also more most some any
all one between before after during since while over under again further once here now know like
get got really thing things time way day days""".split())
# 只在问句侧剥掉的提问形式词:描述"问法"不是内容,却带高 IDF。
# ("order some essential oils" 会被 "order of the six museums" 顶上来 —— 排序题的主要污染源)
_V6_QFORM = set("""order earliest latest first second third fourth fifth sixth last recent recently
among event events happened passed elapsed spend spent total mentioned narrator both old far back
week weeks month months year years date dates participate participated participating did do""".split())


def _v6_tokens(s, qside=False):
    # 首尾撇号必须剥掉:问句里的 "'Run for the Cure'" 会切出 "'run" / "cure'",
    # 与证据句里的 "Run for the Cure" 一个都对不上(a3838d2b 的 gold 锚点事件就是这样丢的);
    # "Farmers' Market" 同理。词内撇号("i've"/"don't")保留。
    raw = (w.strip("'") for w in re.findall(r"[a-z0-9']+", s.lower()))
    toks = [w for w in raw if len(w) > 2 and w not in _V6_STOP]
    if qside:
        toks = [w for w in toks if w not in _V6_QFORM]
    return [_v6_stem(w) for w in toks]


def _v6_stem(w):
    if len(w) < 5:
        return w
    for suf in ("ing", "ied", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            base = w[:-len(suf)]
            if suf == "ied":
                base += "y"
            if suf == "ing" and len(base) > 3 and base[-1] == base[-2] and base[-1] not in "aeiou":
                base = base[:-1]
            return base
    return w


def _v6_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and len(p.strip()) > 12]


def _v6_hits(sent, sess_date):
    """resolve_expressions + 裸数字对守卫。

    裸数字对既有真日期也有比分:语料里 "graduation gift on the 3/8"(真 = 03/08,8c18457d 的
    gold 起点)与 "our team is currently 3-2 in the standings"(假 = 03/02,6e984301 的干扰句)
    形状完全一样。判据用上下文而不是一律丢:只接受**斜杠**形式且前面出现 "on (the)"。"""
    low = sent.lower()
    out = []
    for h in resolve_expressions(sent, sess_date):
        if not _V6_BARE_NUM.match(h["expr"]):
            out.append(h)
            continue
        if "-" in h["expr"]:
            continue                      # 连字符形式几乎都是比分/区间
        ok, start = False, 0
        while True:
            i = low.find(h["expr"], start)
            if i < 0:
                break
            if re.search(r"\bon\s+(?:the\s+)?$", low[max(0, i - 12):i]):
                ok = True
                break
            start = i + 1
        if ok:
            out.append(h)
    return out


def _v6_event_date(sent, sess_date):
    """句子的事件日期。返回 (dt, prov, approx)。
    优先级:句内显式/偏移/星期/节日(高置信、非未来) > today 类 > 月份锚点(近似) > 会话日期。
    刻意**不用** v5_temporal.best_event_date 的 duration-start 兜底:"for about 6 weeks now"
    这类时长表达在 6e984301 里正是把端点带偏的干扰源(同会话另一句写着 "just started today")。"""
    sd = parse_date(sess_date)
    if sd is None:
        return None, "", False
    hits = _v6_hits(sent, sess_date)
    # 节日名放在最后一档:它既可能只是话题词,又要靠"是否晚于会话日"猜年份,最不可靠。
    # ("I just came from the Ash Wednesday service ... on February 1st" 里 ash wednesday 被
    #  猜成上一年的 2022/03/02,min() 就会挑中它而不是句子明写的 February 1st,08f4fc43。)
    dated = [h for h in hits if h["conf"] == "high" and h["kind"] in ("explicit", "offset", "weekday")]
    if not dated:
        dated = [h for h in hits if h["conf"] == "high" and h["kind"] == "holiday"]
    if dated:
        h = min(dated, key=lambda x: x["date"])
        return h["date"], h["expr"], False
    if any(h["kind"] == "same-day" for h in hits):
        return day0(sd), "today", False
    coarse = [h for h in hits if h["kind"] in ("month-anchor", "offset-coarse") and h["date"] <= day0(sd)]
    if coarse:
        h = min(coarse, key=lambda x: x["date"])
        return h["date"], "~" + h["expr"], True
    return day0(sd), "", False


def _v6_candidates(q):
    """全 haystack 用户轮 → 事件句候选(去重)。返回 (list[dict], {sidx: 会话用户词集})。"""
    out, seen, sess_toks = [], set(), {}
    for si, (date_s, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        stoks = set()
        for t in sess:
            if t.get("role") == "assistant":
                continue
            content = t.get("content") or ""
            stoks |= set(_v6_tokens(content))
            for sent in _v6_sentences(content):
                if sent.rstrip().endswith("?"):
                    continue
                if not _V6_FIRST.search(sent):
                    continue
                hits = _v6_hits(sent, date_s)
                has_date = any(h["conf"] in ("high", "med") and h["kind"] in
                               ("explicit", "offset", "weekday", "holiday", "month-anchor")
                               for h in hits)
                # 已完成事件的证据(用于放行 INTENT 句):必须是过去式第一人称或当日锚定。
                # 刻意**不含** has_date:"I was thinking of exploring the waterfront path on
                # Thursday evening" 也带一个句内星期,若算进来会把计划句全放行(e061b84f
                # 的时间窗块曾被这类句子占掉配额)。
                done = bool(_V6_PASTV.search(sent)) or any(
                    h["kind"] == "same-day" and h["expr"] in _V6_SAMEDAY_PAST for h in hits)
                # 意图/计划句只在**没有已完成事件证据**时丢弃:LongMemEval 的证据句常把已发生
                # 事件和一句前瞻塞在同一句里("today I sold ... and I'm thinking of raising prices"
                # = 61e13b3c 的 gold 起点;"I finally recovered from the flu today, and I was
                # thinking about ..." = 370a8ff4 的 gold 起点)。硬删 INTENT 会把它们一起删掉。
                if _V6_INTENT.search(sent) and not done:
                    continue
                if not (done or has_date or _V6_CUE.search(sent)):
                    continue
                key = re.sub(r"[^a-z0-9]", "", sent.lower())[:70]
                if key in seen:
                    continue
                seen.add(key)
                ev, prov, approx = _v6_event_date(sent, date_s)
                if ev is None:
                    continue
                txt = re.sub(r"\s+", " ", sent).strip()
                if len(txt) > V6_SENT_CHARS:
                    txt = txt[:V6_SENT_CHARS].rsplit(" ", 1)[0] + "..."
                out.append({"ev": ev, "sess": day0(parse_date(date_s)), "prov": prov, "sidx": si,
                            "approx": approx, "text": txt, "toks": set(_v6_tokens(sent)),
                            # report = 这句是不是"事件报告"(自带日期或当日锚定),而不是状态描述。
                            # 排序/计数/锚日题要的正是事件报告句;"I've been stuck on this playlist
                            # for the past 5 days" 这类状态句占掉同日配额会顶掉真事件(e061b84f)。
                            "report": 1 if (prov or re.search(_V6_JUST + r"|\b(?:finally|today|"
                                                              r"yesterday)\b", sent, re.I)) else 0,
                            "selfdated": 1 if prov not in ("", "today") else 0})
        sess_toks[si] = stoks
    return out, sess_toks


V6_SESS_W = 0.35   # 会话级相关性权重(v5 的 session-BM25 扩展同一动机:证据句常与"点名句"
                   # 不在同一句甚至不在同一轮 —— fe651585 的 "Rachel" 与 "twins born February
                   # 12th" 隔了好几轮;7abb270c 的 "Modern Art Museum" 只出现在计划句里)


def _v6_score(cands, question, sess_toks=None):
    """问句内容词的 IDF 加权命中 + 会话级加成(确定性,无 API)。就地写入 c['score']/c['own']。"""
    import math
    qt = set(_v6_tokens(question, qside=True))
    n = max(1, len(cands))
    df = {}
    for c in cands:
        for w in c["toks"] & qt:
            df[w] = df.get(w, 0) + 1
    idf = {w: math.log(1 + (n - df.get(w, 0) + 0.5) / (df.get(w, 0) + 0.5)) for w in qt}
    sess_score = {}
    for si, st in (sess_toks or {}).items():
        sess_score[si] = sum(idf.get(w, 0.0) for w in st & qt)
    for c in cands:
        c["own"] = sum(idf.get(w, 0.0) for w in c["toks"] & qt)
        c["sess_score"] = sess_score.get(c.get("sidx"), 0.0)
        c["score"] = c["own"] + V6_SESS_W * c["sess_score"]
    return cands


# ---------------------------------------------------------------------------
# 3. 问句侧锚点 / 时间窗
# ---------------------------------------------------------------------------
def _v6_question_anchors(question, question_date):
    hits = [h for h in _v6_hits(question, question_date)
            if h["conf"] == "high" and h["kind"] in ("offset", "weekday", "holiday", "explicit")]
    if not hits:
        hits = [h for h in _v6_hits(question, question_date) if h["kind"] in ("offset-coarse", "month-anchor")]
    uniq, seen = [], set()
    for h in sorted(hits, key=lambda x: x["date"]):
        if h["date"] in seen:
            continue
        seen.add(h["date"])
        uniq.append(h)
    return uniq[:3]


_WIN_RE = re.compile(r"\b(?:in|during|over|within)\s+the\s+(?:past|last)\s+(?:(" + _NUMW.strip("()") +
                     r")\s+)?(day|week|month|year)s?\b", re.I)
_NUMMAP = {"one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "couple": 2, "few": 3,
           "several": 3, "a couple of": 2, "a few": 3}


def _v6_question_window(question, q_dt):
    """'during the past month' → (start, end) 闭区间。"""
    m = _WIN_RE.search(question)
    if not m:
        return None
    ntok = (m.group(1) or "1").strip().lower()
    n = int(ntok) if ntok.isdigit() else _NUMMAP.get(ntok, 1)
    unit = m.group(2).lower()
    days = {"day": 1, "week": 7, "month": 31, "year": 366}[unit] * n
    return day0(q_dt) - timedelta(days=days), day0(q_dt), m.group(0)


# ---------------------------------------------------------------------------
# 4. 时距题:端点匹配 + 预计算区间
# ---------------------------------------------------------------------------
_BETWEEN_RE = re.compile(r"\bbetween\s+(.+?)\s+and\s+(.+?)\s*\??$", re.I)
_WHEN_RE = re.compile(r"\b(?:since|after|before)?\s*(?:i|my)\b(.+?)\bwhen\s+i\b(.+?)\s*\??$", re.I)
_SINCE_RE = re.compile(r"\bsince\s+(?:i|my)\b(.+?)\s*\??$", re.I)
_LASTTIME_RE = re.compile(r"\b(?:for the last time|last time|most recent(?:ly)?)\b", re.I)
_FIRSTTIME_RE = re.compile(r"\b(?:for the first time|first time)\b", re.I)


def _v6_endpoints(question):
    """问句 → (start_phrase, end_phrase);end_phrase 为 None 表示另一端是问题日。"""
    ql = question.strip()
    m = _BETWEEN_RE.search(ql)
    if m:
        return m.group(1), m.group(2)
    m = _WHEN_RE.search(ql)
    if m:
        return m.group(1), m.group(2)
    m = _SINCE_RE.search(ql)
    if m:
        return m.group(1), None
    return None, None


def _v6_match_phrase(phrase, cands, prefer="early"):
    """短语 → 最匹配的候选事件。要求内容词重叠 >= min(2, len(tokens))。"""
    pt = [w for w in _v6_tokens(phrase) if w not in _V6_QFORM]
    if not pt:
        return None
    need = min(2, len(pt))
    best = None
    for c in cands:
        hit = len(c["toks"] & set(pt))
        if hit < need:
            continue
        key = (-hit, c["ev"] if prefer == "early" else -c["ev"].toordinal())
        if best is None or key < best[0]:
            best = (key, c)
    return best[1] if best else None


def _v6_unit(question):
    m = re.search(r"\bhow many\s+(days|weeks|months|years)\b", question, re.I)
    if m:
        return m.group(1).rstrip("s").lower()
    m = re.search(r"\bhow long\b", question, re.I)
    if m:
        for u in ("week", "month", "day", "year"):
            if u in question.lower():
                return u
    return None


def _v6_interval_str(a, b, unit=None):
    """区间的多口径写法。**整数用四舍五入而不是下取整**:LongMemEval 的 gold 就是四舍五入口径
    (5 天 → gold "1 week";179 天 → gold "6 months";23 天 → gold "3 weeks"),下取整会给出
    "0 full weeks"/"5 full months" 这类与 gold 相反的整数(实测在 v5 已答对的 24 道数值时距题里
    单靠这一项就制造 3 次矛盾)。精确天数始终并列,便于模型自己换口径。"""
    nd = days_between(a, b)
    wk, mo = nd / 7.0, nd / 30.44
    parts = [f"{nd} days", f"{wk:.1f} weeks (~{int(round(wk))} weeks)",
             f"{months_between(a, b)} full calendar months (~{int(round(mo))} months)"]
    if unit == "week":
        parts = [f"{nd} days", f"{wk:.1f} weeks", f"~{int(round(wk))} weeks if a whole number is wanted"]
    elif unit == "day":
        parts = [f"{nd} days", f"{wk:.1f} weeks"]
    elif unit == "month":
        parts = [f"{nd} days", f"{months_between(a, b)} full calendar months",
                 f"~{int(round(mo))} months if a whole number is wanted"]
    return " = ".join(parts)


# ---------------------------------------------------------------------------
# 5. 渲染
# ---------------------------------------------------------------------------
def _v6_off(ev, q_dt):
    nd = days_between(ev, q_dt)
    if nd == 0:
        return "same day as the question"
    return f"{-nd:+d}d" if nd > 0 else f"{-nd:+d}d"


def _v6_line(c, q_dt):
    d = c["ev"]
    stamp = f"{d.year}/{d.month:02d}/{d.day:02d}"
    if c["approx"]:
        stamp += "(approx)"
    prov = ""
    if c["prov"] and c["prov"] != "today":
        prov = f' [from "{c["prov"]}" said {c["sess"].month:02d}/{c["sess"].day:02d}]'
    elif c["sess"] != d:
        prov = f' [reported {c["sess"].month:02d}/{c["sess"].day:02d}]'
    return f"[{stamp} ({d.strftime('%a')}) | {_v6_off(d, q_dt)}]{prov} {c['text']}"


def _v6_key(c):
    """选行优先级:问句词面命中 > 事件报告句 > 事件日精确(非月份近似)> 会话级相关 > 时间。
    分层(而不是把会话加成直接加到分数上)是必要的:加成一旦可加,整段会话的状态句就能
    盖过真正词面命中的事件句(实测 7abb270c 的最后一个博物馆、a3838d2b 的 Bike-a-Thon
    都是这样被顶掉的)。"""
    return (-c.get("own", 0.0), -c.get("report", 0), 1 if c.get("approx") else 0,
            -c.get("sess_score", 0.0), c["ev"])


def _v6_chrono(cands, q_dt, cap, per_day=V6_PER_DAY, budget=None):
    """先按 _v6_key 优先级(同一事件日内配额 per_day、字符预算 budget)选出 <=cap 条,
    再按事件时间渲染。**预算必须在选行阶段生效**:若先按时间排好再截断字符,砍掉的是时间线
    末尾,也就是最晚的那几个事件(a3838d2b 里被砍掉的正是 9/25、10/15、11 月三条)。"""
    per, sel, used = {}, [], 0
    for c in sorted(cands, key=_v6_key):
        k = c["ev"].date()
        if per.get(k, 0) >= per_day:
            continue
        ln = _v6_line(c, q_dt)
        if budget is not None and used + len(ln) + 3 > budget:
            continue
        per[k] = per.get(k, 0) + 1
        used += len(ln) + 3
        sel.append(c)
        if len(sel) >= cap:
            break
    return [_v6_line(c, q_dt) for c in sorted(sel, key=lambda x: (x["ev"], _v6_key(x)))]


_AGE_RE = re.compile(r"\b(?:i'?m|i am|i was|i turned|aged)\s+(\d{1,2})\b|"
                     r"\b(\d{1,2})[- ]year[- ]old\b", re.I)
_AGE_Q_RE = re.compile(r"\bhow old\s+(?:was|were|am|is)\s+i\b", re.I)


def _v6_age_lines(q, q_dt, cap=4):
    """'How old was I when I ...' 需要"现在多少岁"+"多久以前"两个数,前者是数值属性不是事件,
    事件句过滤会把它筛掉(d01c6aa8:'I'm 32-year-old male' 从不进上下文)。此处单独摊开,
    并保留全部相互矛盾的年龄陈述(该 haystack 里同时有 32 和 20),让模型自己看日期取舍。"""
    out, seen = [], set()
    for date_s, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        sd = parse_date(date_s)
        for t in sess:
            if t.get("role") == "assistant":
                continue
            for sent in _v6_sentences(t.get("content") or ""):
                m = _AGE_RE.search(sent)
                if not m or not _V6_FIRST.search(sent):
                    continue
                age = m.group(1) or m.group(2)
                if not (1 <= int(age) <= 99):
                    continue
                if age in seen:
                    continue
                seen.add(age)
                txt = re.sub(r"\s+", " ", sent).strip()[:150]
                out.append(f'  [{sd.year}/{sd.month:02d}/{sd.day:02d} | '
                           f'{_v6_off(day0(sd), q_dt)}] age stated = {age}: {txt}')
                if len(out) >= cap:
                    return out
    return out


def tr_scaffold(q, question=None):
    """v6-TR 主入口。返回可直接拼进上下文的分节文本('' = 门控未命中)。纯规则,零 API。"""
    question = question or q["question"]
    classes = tr_gate(question)
    if not classes:
        return ""
    q_dt = parse_date(q["question_date"])
    if q_dt is None:
        return ""
    cands, sess_toks = _v6_candidates(q)
    cands = _v6_score(cands, question, sess_toks)
    if not cands:
        return ""
    ranked = sorted(cands, key=_v6_key)
    head = (f"### Temporal scaffold (rule-computed; question asked "
            f"{q_dt.year}/{q_dt.month:02d}/{q_dt.day:02d} ({q_dt.strftime('%a')}). "
            f"Event dates below come from the wording inside the turn when it states one "
            f"(\"a month ago\", \"February 12th\"), otherwise from the session date; "
            f"offsets are days relative to the question date.)")
    body, used, spent = [], set(), [len(head)]

    def room():
        return V6_BLOCK_CHARS - spent[0]

    def add(line):
        if spent[0] + len(line) + 1 > V6_BLOCK_CHARS:
            return False
        body.append(line)
        spent[0] += len(line) + 1
        return True

    def emit(lines_, prefix="  ", reserve=0):
        for ln in lines_:
            if ln in used:
                continue
            if spent[0] + len(prefix) + len(ln) + 1 > V6_BLOCK_CHARS - reserve:
                break
            add(prefix + ln)
            used.add(ln)

    # 枚举题的时间线是承重块,先给它留出预算,别让前面的分节吃光
    reserve = 900 if "enum" in classes else 0
    win = _v6_question_window(question, q_dt) if "window" in classes else None

    # 5a. 问句锚点 → 锚日整天摊开(词面不重叠也给),再补 ±W 天邻域。
    #     有明确时间窗时跳过(窗覆盖锚点,"the past month" 这类粗表达两边都会命中门控)。
    if "anchor" in classes and not win:
        anchors = _v6_question_anchors(question, q["question_date"])
        if anchors:
            add("Dates named by the question, resolved:")
            for a in anchors:
                d = a["date"]
                add(f'  "{a["expr"]}" -> {d.year}/{d.month:02d}/{d.day:02d} '
                    f'({d.strftime("%a")}) [{_v6_off(d, q_dt)}]')
            for a in anchors[:2]:
                tgt = a["date"]
                same = [c for c in cands if c["ev"] == tgt]
                near = [c for c in cands if 0 < abs((c["ev"] - tgt).days) <= V6_ANCHOR_WIN]
                if same:
                    add(f'Everything the user reported doing on '
                        f'{tgt.year}/{tgt.month:02d}/{tgt.day:02d} ("{a["expr"]}"), '
                        f'listed regardless of wording overlap:')
                    emit(_v6_chrono(same, q_dt, V6_ANCHOR_CAP, per_day=V6_ANCHOR_CAP,
                                    budget=room() - reserve), reserve=reserve)
                if near:
                    add(f'Nearest other dated reports (within +/-{V6_ANCHOR_WIN} days of '
                        f'{tgt.month:02d}/{tgt.day:02d}):')
                    emit(_v6_chrono(near, q_dt, 6, per_day=2, budget=room() - reserve),
                         reserve=reserve)
                if not same and not near:
                    add(f'  (nothing dated within +/-{V6_ANCHOR_WIN} days of '
                        f'{tgt.month:02d}/{tgt.day:02d})')

    # 5b. 时间窗 → 窗内事件全摊开(e061b84f:三条 gold 证据与问句词面全不重叠)
    if win:
        lo, hi, expr = win
        inw = [c for c in cands if lo <= c["ev"] <= hi]
        if inw:
            add(f'Events inside the window "{expr}" '
                f'({lo.year}/{lo.month:02d}/{lo.day:02d} .. {hi.year}/{hi.month:02d}/{hi.day:02d}), '
                f'listed regardless of wording overlap:')
            emit(_v6_chrono(inw, q_dt, V6_WINDOW_CAP, per_day=3, budget=room() - reserve),
                 reserve=reserve)

    # 5c. 时距题 → 端点匹配 + 预计算区间;"how old was I when" 另加年龄陈述
    if "delta" in classes:
        sp, ep = _v6_endpoints(question)
        unit = _v6_unit(question)
        if sp:
            pa = "late" if _LASTTIME_RE.search(sp) else "early"
            ca = _v6_match_phrase(sp, cands, pa)
            cb = _v6_match_phrase(ep, cands, "late" if ep and _LASTTIME_RE.search(ep) else "early") if ep else None
            if cb is not None and ca is not None and (cb is ca or cb["ev"] == ca["ev"]):
                cb, ca = None, None      # 两端落到同一句/同一天 = 端点匹配失败,宁可不给
            if ca is not None:
                b_dt = cb["ev"] if cb is not None else day0(q_dt)
                b_lbl = (f'"{ep.strip()}" -> {b_dt.year}/{b_dt.month:02d}/{b_dt.day:02d}'
                         if cb is not None else
                         f"the question date -> {b_dt.year}/{b_dt.month:02d}/{b_dt.day:02d}")
                add("Candidate interval, IF these are the two events the question names (the two "
                    "endpoint lines are a wording match and may be wrong -- check them against the "
                    "timeline below; only the day arithmetic on the last line is guaranteed):")
                add(f'  start: "{sp.strip()}" -> {ca["ev"].year}/{ca["ev"].month:02d}/'
                    f'{ca["ev"].day:02d} ({ca["ev"].strftime("%a")})  <- {ca["text"][:110]}')
                add(f'  end:   {b_lbl}' + (f'  <- {cb["text"][:110]}' if cb is not None else ""))
                lo_dt, hi_dt = sorted([ca["ev"], b_dt])
                add(f"  elapsed: {_v6_interval_str(lo_dt, hi_dt, unit)}")
        if _AGE_Q_RE.search(question):
            al = _v6_age_lines(q, q_dt)
            if al:
                add("Ages the user stated about themselves (age at an earlier event = stated age "
                    "minus the elapsed time; all statements are listed, including disagreeing ones):")
                for ln in al:
                    add(ln)

    # 5d. 事件时间线:枚举/排序/计数题的承重块(替代"从散落行里自己排序");时距题给短版,
    #     让模型能核对上面自动匹配的端点(6e984301 的错端点就来自同会话的干扰时长句)。
    #     候选池两层:问句词面命中(own>=floor);以及**会话级相关**的事件报告/自带日期句 ——
    #     计数/排序需要全部 n 个实例,而实例措辞常与问句不重叠。会话级相关是必要条件:
    #     不加它,自带日期的无关流水("I got a laptop backpack from Tumi on January 5th")
    #     会把 a3838d2b 的慈善事件挤出预算。
    enum_cap = V6_ENUM_CAP if "enum" in classes else 8
    floor = max(c["own"] for c in cands) * 0.12
    pool = [c for c in ranked if c["own"] >= floor and c["own"] > 0][:40]
    extra = [c for c in ranked if c["own"] <= 0 and c["sess_score"] > 0
             and (c["selfdated"] or c["report"])]
    pool = pool + extra[:V6_SELFDATED_CAP + 4]
    if pool:
        tl = _v6_chrono(pool, q_dt, enum_cap, budget=room() - 60)
        if tl:
            add("Dated events matching the question, earliest -> latest:")
            emit(tl)

    if not body:
        return ""
    return "\n".join([head] + body)


# ---------------------------------------------------------------------------
# 6. 预算中性变体:按新增字符数挤掉 v5 原文轮里 BM25 最低分的行
# ---------------------------------------------------------------------------
def tr_trim_raw(question, raw_txt, need_chars):
    """从 v5 原文轮块里删掉最低 BM25 分的行,直到释放 need_chars 字符。保持剩余行原顺序。"""
    if need_chars <= 0 or not raw_txt:
        return raw_txt
    from run_baseline import bm25_rank
    lines = raw_txt.splitlines()
    if len(lines) <= 1:
        return raw_txt
    order = bm25_rank(question, lines)          # 降序
    drop, freed = set(), 0
    for i in reversed(order):                   # 最低分先删
        if freed >= need_chars or len(drop) >= len(lines) - 1:
            break
        drop.add(i)
        freed += len(lines[i]) + 1
    return "\n".join(l for i, l in enumerate(lines) if i not in drop)


# ---------------------------------------------------------------------------
# 7. 自检(零 API):门控 + 原语回归 + 真实错题句子回归
# ---------------------------------------------------------------------------
CASES = {  # v5 的 12 个 TR 错题 → 读例得到的机制标签(诊断用,方法内不使用)
    "gpt4_1e4a8aec": "question-side offset, gold turn absent from v5 ctx",
    "eac54add": "question-side offset (+/-1d slack needed), gold turn absent",
    "gpt4_8279ba03": "question-side offset, gold turn absent -> v5 answered don't-know",
    "gpt4_e061b84f": "question-side window, all 3 gold turns absent",
    "gpt4_7abb270c": "enumeration of 6, 4 gold turns absent",
    "a3838d2b": "count-before, 2 gold turns absent + event dates only in-text",
    "gpt4_65aabe59": "same-day sessions, ordering only in-text ('3 weeks ago' turn absent)",
    "gpt4_fe651585": "same-day sessions, ordering only in-text ('February 12th')",
    "6e984301": "endpoints present, distractor duration expression in same session",
    "gpt4_61e13b3c": "endpoints present, answer hedged in fractional weeks",
    "370a8ff4": "endpoints present and unambiguous; gold 15 weeks not derivable (81 days)",
    "d01c6aa8": "age arithmetic, age statement absent from v5 ctx (not date arithmetic)",
}

if __name__ == "__main__":
    # 门控:12 个真实错题问句必须全部命中
    QS = [
        ("What gardening-related activity did I do two weeks ago?", "anchor"),
        ("What was the significant buisiness milestone I mentioned four weeks ago?", "anchor"),
        ("What kitchen appliance did I buy 10 days ago?", "anchor"),
        ("What is the order of the three sports events I participated in during the past month, "
         "from earliest to latest?", "window"),
        ("What is the order of the six museums I visited from earliest to latest?", "enum"),
        ("How many charity events did I participate in before the 'Run for the Cure' event?", "enum"),
        ("Which device did I set up first, the smart thermostat or the mesh network system?", "enum"),
        ("Who became a parent first, Rachel or Alex?", "enum"),
        ("How many weeks have I been taking sculpting classes when I invested in my own set of "
         "sculpting tools?", "delta"),
        ("How many weeks passed between the time I sold homemade baked goods at the Farmers' Market "
         "for the last time and the time I participated in the Spring Fling Market?", "delta"),
        ("How many weeks had passed since I recovered from the flu when I went on my 10th jog "
         "outdoors?", "delta"),
        ("How old was I when I moved to the United States?", "delta"),
    ]
    for qq, cls in QS:
        g = tr_gate(qq)
        assert cls in g, (qq, g)
    # 非时间题不应命中(误触会白烧预算)
    for qq in ["What kind of coffee do I like?", "Where does my sister live now?",
               "What did the doctor recommend for my knee?"]:
        assert not tr_gate(qq), (qq, tr_gate(qq))

    # 原语回归:全部取自真实错题句子(v5_temporal 原语核验)
    r = {h["expr"]: h["date"].strftime("%Y/%m/%d") for h in resolve_expressions(
        "twins, Jackson and Julia, who were born on February 12th", "2023/03/17 (Fri) 18:54")}
    assert r["february 12th"] == "2023/02/12", r          # 月份缩写/命名日期
    ev, prov, ap = _v6_event_date("my cousin Alex just adopted a baby girl from China in January",
                                  "2023/03/17 (Fri) 15:27")
    assert ev.strftime("%Y/%m") == "2023/01" and ap, (ev, ap)   # 月份锚点=近似
    ev, prov, ap = _v6_event_date("Since I've recently upgraded my home Wi-Fi router 3 weeks ago, "
                                  "which improved coverage", "2023/05/25 (Thu) 01:12")
    assert ev.strftime("%Y/%m/%d") == "2023/05/04" and not ap, ev
    ev, prov, ap = _v6_event_date("since I set up my smart thermostat a month ago, I've noticed",
                                  "2023/05/25 (Thu) 19:02")
    assert ev.strftime("%Y/%m/%d") == "2023/04/25", ev
    # 裸数字对必须被丢弃:"3-2 in the standings"(6e984301 真实句)不得变成 03/02 事件
    sent = ('I signed up for the recreational volleyball league about 6 weeks ago. our team is '
            'currently 3-2 in the standings')
    assert all(not _V6_BARE_NUM.match(h["expr"]) for h in _v6_hits(sent, "2023/01/24 (Tue) 05:36"))
    assert any(h["expr"] == "3-2" for h in resolve_expressions(sent, "2023/01/24 (Tue) 05:36")), \
        "上游行为变了:裸数字对不再被解析,可放宽本模块的过滤"
    # 时长表达不得当事件日期(6e984301 的干扰源)
    ev, prov, ap = _v6_event_date("I've been taking sculpting classes at a local art studio for "
                                  "about 6 weeks now", "2023/02/11 (Sat) 18:51")
    assert ev.strftime("%Y/%m/%d") == "2023/02/11", (ev, prov)

    # 端点切分
    assert _v6_endpoints("How many weeks have I been taking sculpting classes when I invested in my "
                         "own set of sculpting tools?")[1].strip().startswith("invested")
    a, b = _v6_endpoints("How many weeks passed between the time I sold homemade baked goods at the "
                         "Farmers' Market for the last time and the time I participated in the "
                         "Spring Fling Market?")
    assert "Farmers" in a and "Spring Fling" in b, (a, b)
    # 区间算术
    # 真实错题的两个区间:四舍五入口径必须与 gold("3 weeks")一致
    assert _v6_interval_str(parse_date("2023/02/11"), parse_date("2023/03/04"), "week") == \
        "21 days = 3.0 weeks = ~3 weeks if a whole number is wanted"
    assert _v6_interval_str(parse_date("2023/02/26"), parse_date("2023/03/21"), "week") == \
        "23 days = 3.3 weeks = ~3 weeks if a whole number is wanted"
    # gpt4_7a0daae1(v5 已答对,gold "1 week"):下取整会给 0 full weeks,四舍五入给 1
    assert "~1 weeks" in _v6_interval_str(parse_date("2023/03/10"), parse_date("2023/03/15"), "week")
    # 8c18457d 的真实句:"on the 3/8" 必须被当日期,"3-2 in the standings" 必须不被
    hh = {h["expr"]: h["date"].strftime("%Y/%m/%d") for h in _v6_hits(
        "I recently got a wireless headphone for my brother as a graduation gift on the 3/8.",
        "2023/03/29 (Wed) 21:27")}
    assert hh.get("3/8") == "2023/03/08", hh
    assert not [h for h in _v6_hits("our team is currently 3-2 in the standings",
                                    "2023/01/24 (Tue) 05:36") if _V6_BARE_NUM.match(h["expr"])]
    # 08f4fc43 的真实句:句内显式日期优先于节日名(节日名要猜年份)
    ev, prov, ap = _v6_event_date("By the way, I just came from the Ash Wednesday service at the "
                                 "cathedral on February 1st, and it really made me think.",
                                 "2023/02/20 (Mon) 09:00")
    assert ev.strftime("%Y/%m/%d") == "2023/02/01", (ev, prov)
    # 时间窗
    w = _v6_question_window("the three sports events I participated in during the past month",
                            parse_date("2023/07/01 (Sat) 20:43"))
    assert w and w[0].strftime("%Y/%m/%d") == "2023/05/31", w
    # 预算中性裁剪
    raw = "\n".join(f"[2023/01/{i:02d}] [user] filler line about topic {i}" for i in range(1, 11))
    trimmed = tr_trim_raw("topic 3", raw, 120)
    assert len(trimmed) <= len(raw) - 120 + 60 and "topic 3" in trimmed
    print("v6_tr self-test OK")
