"""Vanna 风格的 Text-to-SQL demo（零依赖版）。

对标原文的 ChromaDB_VectorStore + OpenAI_Chat 组合，这里换成：
    向量/检索 -> 复用 demo_lab.rag 的 BM25 + HashingEmbedder + RRF
    LLM      -> OpenAICompatLLM（联网）/ RuleBasedLLM（离线替身）
    数据库   -> SQLite（开箱即用），也可以 connect_to_mysql 接真实 MySQL

对外只需要 `from demo_lab.text2sql import MiniVanna` 就能用。
"""

from demo_lab.text2sql.core import (
    KIND_DDL,
    KIND_DOCUMENTATION,
    KIND_SQL,
    AskResult,
    Example,
    GeneratedSQL,
    QueryResult,
    SqlRequest,
    TrainingItem,
    tokenize,
)
from demo_lab.text2sql.engine import MiniVanna
from demo_lab.text2sql.llm import (
    OpenAICompatLLM,
    RuleBasedLLM,
    SqlGenerationError,
    SupportsGenerateSQL,
    TemplateRule,
    build_prompt,
    extract_sql,
)
from demo_lab.text2sql.store import RetrievedItem, TrainingStore

# __all__ 相当于给 IDE 和读者的"公开接口清单"
__all__ = [
    # 主编排
    "MiniVanna",
    # 训练仓库
    "TrainingStore",
    "RetrievedItem",
    # LLM
    "RuleBasedLLM",
    "OpenAICompatLLM",
    "SupportsGenerateSQL",
    "TemplateRule",
    "SqlGenerationError",
    "build_prompt",
    "extract_sql",
    # 数据结构
    "TrainingItem",
    "Example",
    "SqlRequest",
    "GeneratedSQL",
    "QueryResult",
    "AskResult",
    # 常量与工具
    "KIND_DDL",
    "KIND_DOCUMENTATION",
    "KIND_SQL",
    "tokenize",
]
