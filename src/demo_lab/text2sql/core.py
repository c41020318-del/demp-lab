"""Text-to-SQL 的「数据结构 + 文本处理」层。

对应真实 Vanna 里的三类训练数据：
    ddl            表结构（让模型知道有哪些表、哪些字段）
    documentation  业务术语（「高价值客户」到底指什么）
    sql            问法 -> SQL 的对照样例（few-shot）

这一层不依赖任何外部库，也不依赖同包的其他文件，
所以它可以被 store / llm / engine 随意引用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 训练材料的三种类型，写成常量是为了避免到处写字符串打错字
KIND_DDL = "ddl"
KIND_DOCUMENTATION = "documentation"
KIND_SQL = "sql"
KINDS = (KIND_DDL, KIND_DOCUMENTATION, KIND_SQL)

# 英文单词 / 数字 / 下划线标识符
_ASCII_TOKEN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+")
# 连续的中文片段（英文已被上一个正则切走，这里只留下汉字）
_CJK_SEGMENT = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """极简中文分词：英文按词切，中文按「单字 + 二元组」切。

    为什么中文要额外加一层二元组？
    因为 BM25 和向量都只看 token 是否重叠。
    「华东」如果只切成「华」「东」两个单字，粒度太细，
    问「华东」时「华北」也会跟着沾光。补一层二元组后两者就有了区分度。
    """
    tokens: list[str] = []

    # 第 1 路：英文 / 数字，统一转小写，保证 SELECT 和 select 是同一个 token
    tokens.extend(_ASCII_TOKEN.findall(text.lower()))

    # 第 2 路：中文，单字 + 相邻二元组
    for segment in _CJK_SEGMENT.findall(text):
        tokens.extend(segment)  # 单字
        if len(segment) > 1:
            tokens.extend(segment[i : i + 2] for i in range(len(segment) - 1))  # 二元组

    return tokens


@dataclass
class TrainingItem:
    """一条训练材料。

    @dataclass 自动生成 __init__ / __repr__ / __eq__，省掉样板代码。
    """

    kind: str  # ddl / documentation / sql
    content: str  # 原文：DDL 文本、术语说明、或者「问题」
    sql: str | None = None  # 只有 kind == sql 时才有值
    item_id: str = ""  # 空则由仓库自动编号
    tokens: list[str] = field(default_factory=list)  # content 的分词结果

    def __post_init__(self) -> None:
        # 数据校验放在这里，一旦有人写错 kind，构造时就报错而不是等到检索时才发现
        if self.kind not in KINDS:
            raise ValueError(f"未知的训练类型 {self.kind!r}，可选值: {KINDS}")
        if not self.content.strip():
            raise ValueError("content must not be empty")
        # 没传 tokens 就现场算一个
        if not self.tokens:
            self.tokens = tokenize(self.content)

    # ---- 三个工厂方法，让调用方不用记 kind 字符串 ----

    @classmethod
    def from_ddl(cls, ddl: str) -> TrainingItem:
        return cls(kind=KIND_DDL, content=ddl.strip())

    @classmethod
    def from_documentation(cls, documentation: str) -> TrainingItem:
        return cls(kind=KIND_DOCUMENTATION, content=documentation.strip())

    @classmethod
    def from_question_sql(cls, question: str, sql: str) -> TrainingItem:
        return cls(kind=KIND_SQL, content=question.strip(), sql=sql.strip())


@dataclass
class Example:
    """喂给 LLM 的 few-shot 样例：一个问题对应一条正确 SQL。"""

    question: str
    sql: str


@dataclass
class SqlRequest:
    """一次 SQL 生成请求的完整上下文（DDL + 术语 + 相似样例 + 问题）。

    把它单独拎出来，是为了让「谁来生成 SQL」这件事可替换：
    规则替身和真实大模型收到的是同一份输入。
    """

    question: str
    ddl_text: str = ""
    documentation: str = ""
    examples: list[Example] = field(default_factory=list)
    dialect: str = "sqlite"


@dataclass
class GeneratedSQL:
    """生成结果：SQL 本身 + 它从哪来的（排查问题时很有用）。"""

    sql: str
    source: str  # "llm" / "template" / "example-reuse"
    note: str = ""


@dataclass
class QueryResult:
    """SQL 执行结果。真实 Vanna 这里返回 pandas.DataFrame，demo 只用标准库。"""

    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_table(self, max_rows: int = 20) -> str:
        """渲染成对齐的文本表格，方便在控制台打印。"""
        if not self.rows:
            return "(没有查到数据)"

        header = " | ".join(self.columns)
        lines = [header, "-" * len(header)]
        lines.extend(" | ".join(str(value) for value in row) for row in self.rows[:max_rows])
        if len(self.rows) > max_rows:
            lines.append(f"... 共 {len(self.rows)} 行")
        return "\n".join(lines)


@dataclass
class AskResult:
    """ask() 的返回值：问题 + SQL + 结果 + 追问候选。"""

    question: str
    generated: GeneratedSQL
    result: QueryResult
    followup_questions: list[str] = field(default_factory=list)

    @property
    def sql(self) -> str:
        return self.generated.sql
