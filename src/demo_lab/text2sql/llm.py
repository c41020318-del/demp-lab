"""LLM 层：把「问题 + 上下文」变成 SQL。

这里是整个链路里唯一需要替换的地方（对应原文里继承 OpenAI_Chat 那一步）。
两个实现随时互换：
    OpenAICompatLLM  真实大模型，走 OpenAI 兼容协议（DeepSeek / 通义 / 本地 vLLM 都能用）
    RuleBasedLLM     离线替身，不联网也能跑通全链路（默认）
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Protocol

from demo_lab.text2sql.core import GeneratedSQL, SqlRequest, tokenize


class SqlGenerationError(RuntimeError):
    """LLM 生成失败（没命中模板、也没找到可参考的样例）。"""


class SupportsGenerateSQL(Protocol):
    """只要实现 generate_sql(request) -> GeneratedSQL，就能插进来当 LLM 用。"""

    def generate_sql(self, request: SqlRequest) -> GeneratedSQL: ...


# ============================================================
# 提示词
# ============================================================

_PROMPT_HEADER = (
    "你是一个资深的 {dialect} 数据分析师，负责把用户的中文问题翻译成 SQL。\n"
    "必须遵守以下规则：\n"
    "1. 只输出一条 SQL 语句，不要解释，不要 Markdown 以外的多余文字；\n"
    "2. 只能写只读查询（SELECT / WITH），禁止 INSERT / UPDATE / DELETE / DROP；\n"
    "3. 只使用下面给出的表和字段，不要臆造；\n"
    "4. 优先参考「历史问答样例」的写法风格。"
)

_PROMPT_FOOTER = "请把上面的自然语言问题翻译成 SQL：\n{question}"


def build_prompt(request: SqlRequest) -> str:
    """把 SqlRequest 拼成一整段提示词。

    顺序是有讲究的：规则 -> 表结构 -> 术语 -> 样例 -> 问题。
    把「问题」放最后，等于告诉模型「前面都是参考资料，这是你的任务」。
    """
    blocks: list[str] = [_PROMPT_HEADER.format(dialect=request.dialect)]

    if request.ddl_text:
        blocks.append("## 表结构\n```sql\n" + request.ddl_text.strip() + "\n```")

    if request.documentation:
        blocks.append("## 业务术语\n" + request.documentation.strip())

    if request.examples:
        example_blocks = []
        for example in request.examples:
            example_blocks.append(
                f"问题：{example.question}\nSQL：\n```sql\n{example.sql.strip()}\n```"
            )
        blocks.append("## 历史问答样例\n" + "\n\n".join(example_blocks))

    blocks.append(_PROMPT_FOOTER.format(question=request.question))
    return "\n\n".join(blocks)


# ============================================================
# 从 LLM 的原始回复里抠出 SQL
# ============================================================

_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(raw: str) -> str:
    """从 ```sql ... ``` 代码块里取出 SQL；没有代码块就当整段都是 SQL。"""
    blocks = _FENCE_RE.findall(raw)
    text = blocks[-1] if blocks else raw
    return text.strip().rstrip(";").strip()


# ============================================================
# 离线替身：规则 LLM
# ============================================================

_NUMBER_RE = re.compile(r"\d+|[一二三四五六七八九十]+")
_LIMIT_RE = re.compile(r"(?i)\blimit\s+\d+")
_CN_NUMERALS = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def extract_number(question: str, default: int | None = None) -> int | None:
    """从问题里抽出第一个数字，支持阿拉伯数字和简单中文数字。

    用途：把「前 3 名」里的 3 填进模板的 {limit}、「过去 90 天」里的 90 填进 {days}。
    """
    match = _NUMBER_RE.search(question)
    if match is None:
        return default

    raw = match.group()
    if raw.isdigit():
        return int(raw)

    # 中文数字：十 -> 10，十五 -> 15，二十 -> 20，其余情况退回默认值
    if raw in _CN_NUMERALS:
        return _CN_NUMERALS[raw]
    if raw.startswith("十") and len(raw) == 2:
        return 10 + _CN_NUMERALS.get(raw[1], 0)
    if len(raw) == 3 and raw[1] == "十":
        return _CN_NUMERALS.get(raw[0], 0) * 10 + _CN_NUMERALS.get(raw[2], 0)
    return default


def replace_limit(sql: str, limit: int | None) -> str:
    """把样例 SQL 里的 LIMIT N 换成问题里的数字。"""
    if limit is None:
        return sql
    return _LIMIT_RE.sub(f"LIMIT {limit}", sql)


def similarity(a: str, b: str) -> float:
    """两个句子的 Jaccard 相似度（分词集合的交集 / 并集）。"""
    tokens_a, tokens_b = set(tokenize(a)), set(tokenize(b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


@dataclass
class TemplateRule:
    """一条「问法 -> SQL 模板」的规则，相当于给人看的 if-else。

    groups 是多组关键词：
        组内是「与」的关系（都要出现才算命中）
        组间是「或」的关系（任一组命中即可）
    例如 [["最高", "金额"], ["花钱", "最多"]] 表示两种说法走同一个模板。
    """

    groups: list[list[str]]
    sql: str
    note: str = ""

    def matches(self, question: str) -> bool:
        return any(all(keyword in question for keyword in group) for group in self.groups)


class RuleBasedLLM:
    """LLM 的离线替身（和 rag.embedder.HashingEmbedder 是同一个思路）。

    生成 SQL 的两条优先级：
        1. 命中显式定义的 TemplateRule -> 套模板，数字从问题里抽
        2. 否则找最相似的历史样例 -> 复用它的 SQL，并按本次问题的数字改写 LIMIT

    注意这不是"理解"自然语言，而是模式匹配；真上线请换成 OpenAICompatLLM。
    """

    def __init__(
        self,
        template_rules: list[TemplateRule] | None = None,
        reuse_threshold: float = 0.3,
    ):
        self.template_rules: list[TemplateRule] = list(template_rules or [])
        self.reuse_threshold = reuse_threshold

    def generate_sql(self, request: SqlRequest) -> GeneratedSQL:
        # ---- 路径 1：显式模板优先，结果最可控 ----
        for rule in self.template_rules:
            if rule.matches(request.question):
                sql = rule.sql.format(
                    limit=extract_number(request.question, default=5),
                    days=extract_number(request.question, default=90),
                )
                return GeneratedSQL(sql=sql, source="template", note=rule.note or "命中 SQL 模板")

        # ---- 路径 2：复用最相似的历史样例 ----
        for example in request.examples:
            score = similarity(request.question, example.question)
            if score >= self.reuse_threshold:
                sql = replace_limit(example.sql, extract_number(request.question))
                return GeneratedSQL(
                    sql=sql,
                    source="example-reuse",
                    note=f"复用样例「{example.question}」(相似度={score:.2f})",
                )

        raise SqlGenerationError(
            f"替身 LLM 无法回答「{request.question}」："
            "既没有命中 SQL 模板，也没有相似度超过阈值的历史样例。"
            "可以再 train 一条 question->sql 样例，或换成 OpenAICompatLLM。"
        )


# ============================================================
# 联网版：OpenAI 兼容协议的大模型（DeepSeek 等）
# ============================================================


@dataclass
class OpenAICompatLLM:
    """真实 LLM：所有 OpenAI 协议兼容的服务都能接。

    DeepSeek 只要换 base_url 和 api_key 就能用，不需要装额外的 SDK，
    这一点和原文里 `OpenAI(api_key=..., base_url="https://api.deepseek.com")` 完全一致。
    """

    model: str = "deepseek-chat"
    api_key: str | None = None
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.0  # Text-to-SQL 要的是确定性，别让模型自由发挥
    client: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # 外部已经传进来 client 就直接用（方便测试时塞一个假客户端）
        if self.client is not None:
            return

        # openai 是可选依赖，放在这里导入，没装也不影响离线 demo
        try:
            from openai import OpenAI  # noqa: PLC0415 - 故意延迟导入
        except ImportError as exc:  # pragma: no cover - 取决于运行环境
            raise ImportError("联网版请先安装依赖: pip install openai") from exc

        api_key = self.api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("缺少 API Key：请传 api_key，或设置环境变量 DEEPSEEK_API_KEY")

        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

    @classmethod
    def from_env(cls) -> OpenAICompatLLM:
        """从环境变量读取配置，省得把 key 写进代码。"""
        return cls(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )

    def generate_sql(self, request: SqlRequest) -> GeneratedSQL:
        prompt = build_prompt(request)

        response = self.client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content or ""

        return GeneratedSQL(sql=extract_sql(raw), source="llm", note=f"model={self.model}")


__all__ = [
    "GeneratedSQL",
    "OpenAICompatLLM",
    "RuleBasedLLM",
    "SqlGenerationError",
    "SqlRequest",
    "SupportsGenerateSQL",
    "TemplateRule",
    "build_prompt",
    "extract_number",
    "extract_sql",
    "replace_limit",
    "similarity",
]
