"""QC-Lifecycle 项目共享工具:数据加载、DeepSeek 客户端、判分、成本记账。"""
import json
import os
import time
import urllib.request
import urllib.error

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
API_URL = "https://api.deepseek.com/chat/completions"


def load_dataset(variant="s_cleaned"):
    fname = {"s_cleaned": "longmemeval_s_cleaned.json", "oracle": "longmemeval_oracle.json"}[variant]
    with open(os.path.join(DATA_DIR, fname)) as f:
        return json.load(f)


class CostLedger:
    """逐操作 token 记账(MERIT 口径)。"""

    def __init__(self):
        self.rows = []

    def add(self, op, model, usage):
        self.rows.append({
            "op": op, "model": model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0),
            "cache_hit": usage.get("prompt_cache_hit_tokens", 0),
        })

    def summary(self):
        out = {}
        for r in self.rows:
            k = (r["op"], r["model"])
            s = out.setdefault(k, {"calls": 0, "prompt": 0, "completion": 0})
            s["calls"] += 1
            s["prompt"] += r["prompt_tokens"]
            s["completion"] += r["completion_tokens"]
        return {f"{op}/{m}": v for (op, m), v in out.items()}

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"rows": self.rows, "summary": self.summary()}, f, ensure_ascii=False, indent=1)


def chat(messages, model="deepseek-flash", max_tokens=1024, temperature=0.0,
         ledger=None, op="generic", retries=3, thinking=True):
    """DeepSeek chat。注意:两模型均为推理型,max_tokens 需给 reasoning 留余量。
    thinking=False 关闭推理(实测 thinking={type:disabled} 生效,reasoning_tokens=0),
    用于抽取等简单高吞吐操作,输出成本降 ~30x。
    环境代理指向 127.0.0.1:15721(仅 Anthropic),必须绕过 → 用无代理 opener。"""
    if not API_KEY:
        raise RuntimeError("Set DEEPSEEK_API_KEY before calling DeepSeek.")
    body = {
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
    }
    if not thinking:
        body["thinking"] = {"type": "disabled"}
    payload = json.dumps(body).encode()
    req = urllib.request.Request(API_URL, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    })
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_err = None
    for attempt in range(retries):
        try:
            with opener.open(req, timeout=300) as resp:
                d = json.loads(resp.read())
            if ledger is not None:
                ledger.add(op, model, d.get("usage", {}))
            return d["choices"][0]["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(2 ** attempt * 3)
    raise RuntimeError(f"DeepSeek 调用失败({op}): {last_err}")


ANSWER_MODEL = "deepseek-flash"  # 2026-09-17 用户指定:回答用 flash(judge 仍 v4-pro,判分≠答题防自评偏好)

# 全臂共用的回答 system prompt。2026-09-17 修:此前 run_baseline 与 run_qc 各用一套措辞,
# 基线臂因措辞更易拒答(470 可答题上拒答 26 次 vs qc 4 次,全部判错),
# 使 qc 对 full 的 47 个胜场里 18 个来自基线拒答 → 头条 +4.8pp 中约 3.1pp 是提示词假象。
# 现统一为同一文本;各臂的来源差异只通过上下文分节标题体现,不进 system prompt。
SHARED_ANSWER_PROMPT = (
    "Answer the question using the memory context provided. The context may contain several "
    "sections; use all of them together. Be concise. Answer whenever the context supports an "
    "answer, even partially; say you don't know only if the context contains nothing relevant."
)


def answer_chat(system_prompt, user_content, ledger, op, max_tokens=6144):
    """统一的回答调用:所有系统同模型、同 token 预算、同空答重试策略(消除臂间不对称)。"""
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}]
    ans = chat(messages, model=ANSWER_MODEL, max_tokens=max_tokens, ledger=ledger, op=op)
    if not ans.strip():  # 推理型模型 reasoning 吃光配额 → 空正文,加倍配额重试一次
        ans = chat(messages, model=ANSWER_MODEL, max_tokens=max_tokens * 2,
                   ledger=ledger, op=f"{op}-retry")
    return ans


_f1_word = None


def _f1_tokens(s):
    """SQuAD 式规范化:小写、按字母数字取词、去冠词(标点/markdown 不再破坏匹配)。"""
    global _f1_word
    if _f1_word is None:
        import re
        _f1_word = re.compile(r"[a-z0-9]+")
    return [t for t in _f1_word.findall(str(s).lower()) if t not in ("a", "an", "the")]


def token_f1(pred, gold):
    p, g = _f1_tokens(pred), _f1_tokens(gold)
    if not p or not g:
        return 0.0
    common = {}
    for t in p:
        common[t] = common.get(t, 0)
    overlap = 0
    gc = {}
    for t in g:
        gc[t] = gc.get(t, 0) + 1
    pc = {}
    for t in p:
        pc[t] = pc.get(t, 0) + 1
    for t, c in pc.items():
        overlap += min(c, gc.get(t, 0))
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


JUDGE_PROMPT = """You are grading a model answer against a gold answer for a memory QA benchmark.

Question ({qtype}): {question}
Gold answer: {gold}
Model answer: {pred}

Rules:
- The model answer is CORRECT if it contains the gold answer's key information, even with extra wording.
- For abstention questions (gold indicates unanswerable/no information), CORRECT means the model declines or states it doesn't know.
- Otherwise it is INCORRECT.

Reply with exactly one word: CORRECT or INCORRECT."""


def llm_judge(question, qtype, gold, pred, ledger=None, n=1):
    """固定 judge = deepseek-v4-pro,温度 0;n>1 时多数票。
    空输出/无判定词(reasoning 耗尽配额)→ 加倍配额重试一次;仍不可解析则抛异常,
    由调用方记为错误行(断点续跑时重试),绝不静默计为 INCORRECT。"""
    msg = [{"role": "user", "content": JUDGE_PROMPT.format(qtype=qtype, question=question, gold=gold, pred=pred)}]
    votes = []
    for _ in range(n):
        out = chat(msg, model="deepseek-v4-pro", max_tokens=2048, ledger=ledger, op="judge")
        if "CORRECT" not in out.upper():
            out = chat(msg, model="deepseek-v4-pro", max_tokens=8192, ledger=ledger, op="judge-retry")
        if "CORRECT" not in out.upper():
            raise RuntimeError(f"judge unparseable: {out[:80]!r}")
        votes.append("INCORRECT" not in out.upper())
    return sum(votes) > len(votes) / 2
