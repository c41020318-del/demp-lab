"""MiniVanna：把「训练仓库 + LLM + 数据库」串起来的主编排类。

对标原文里的 class MyVanna(ChromaDB_VectorStore, OpenAI_Chat)，区别在于：
本实现不依赖 vanna / chromadb / MySQL 服务端，全部走标准库 + 本项目已有组件，
并且把「向量存储」和「LLM」换成了两个可注入的对象（构造时传入即生效）。

一次完整问答的流程：
    提问 -> 检索相似样例/DDL/术语 -> 拼提示词 -> LLM 生成 SQL -> 执行 -> （可选）回流训练
"""

from __future__ import annotations

from typing import Any

from demo_lab.text2sql.core import (
    AskResult,
    Example,
    GeneratedSQL,
    QueryResult,
    SqlRequest,
    TrainingItem,
)
from demo_lab.text2sql.llm import RuleBasedLLM, SupportsGenerateSQL, build_prompt
from demo_lab.text2sql.store import TrainingStore

# demo 为了安全只允许只读查询
_ALLOWED_STATEMENT_PREFIXES = ("select", "with")


class MiniVanna:
    """极简 Vanna：train() 喂知识，ask() 提问拿结果。"""

    def __init__(
        self,
        llm: SupportsGenerateSQL | None = None,
        store: TrainingStore | None = None,
        dialect: str = "sqlite",
    ):
        # 不传 LLM 就用离线替身（默认不联网也能跑）
        self.llm: SupportsGenerateSQL = llm or RuleBasedLLM()
        self.store = store or TrainingStore()
        self.dialect = dialect
        self._connection: Any = None

    # ============================================================
    # 第 1 步：连数据库
    # ============================================================

    def connect_to_sqlite(self, database: str = ":memory:") -> MiniVanna:
        """连 SQLite（:memory: 或文件路径都可以）。

        返回 self 是为了能链式调用：vn.connect_to_sqlite("demo.db").train(...)
        """
        import sqlite3  # 标准库，用到时才导入

        self._connection = sqlite3.connect(database)
        self.dialect = "sqlite"
        return self

    def connect_to_mysql(
        self,
        host: str,
        dbname: str,
        user: str,
        password: str,
        port: int = 3306,
    ) -> MiniVanna:
        """连 MySQL（参数与原文 vn.connect_to_mysql 保持一致）。

        pymysql 是可选依赖，用到时才导入，装了就能跑：
            pip install pymysql
        """
        try:
            import pymysql  # noqa: PLC0415 - 故意延迟导入
        except ImportError as exc:  # pragma: no cover - 取决于运行环境
            raise ImportError("连 MySQL 请先安装驱动: pip install pymysql") from exc

        self._connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=dbname,
            port=port,
            charset="utf8mb4",
        )
        self.dialect = "mysql"
        return self

    def set_connection(self, connection: Any, dialect: str = "sqlite") -> MiniVanna:
        """直接塞一个已有连接进来（比如外面建好的 SQLite in-memory 连接）。"""
        self._connection = connection
        self.dialect = dialect
        return self

    def _require_connection(self) -> Any:
        if self._connection is None:
            raise RuntimeError("还没有连数据库，请先调用 connect_to_sqlite() 或 connect_to_mysql()")
        return self._connection

    # ============================================================
    # 第 2 步：训练
    # ============================================================

    def train(
        self,
        ddl: str | None = None,
        documentation: str | None = None,
        question: str | None = None,
        sql: str | None = None,
    ) -> list[TrainingItem]:
        """一次调用可以喂任意一种/多种训练数据，返回新增的材料列表。"""
        added: list[TrainingItem] = []

        if ddl:
            added.append(self.store.add(TrainingItem.from_ddl(ddl)))
        if documentation:
            added.append(self.store.add(TrainingItem.from_documentation(documentation)))
        if question or sql:
            # question 和 sql 必须成对出现，否则这条训练数据没有意义
            if not (question and sql):
                raise ValueError("question 和 sql 必须成对传入")
            added.append(self.store.add(TrainingItem.from_question_sql(question, sql)))

        if not added:
            raise ValueError("至少需要传入一种训练数据：ddl / documentation / question+sql")
        return added

    def train_from_schema(self) -> list[TrainingItem]:
        """自动把 SQLite 里的表结构读出来喂给模型（省掉手写 DDL）。

        真实场景下这一步等价于查 information_schema / SHOW CREATE TABLE。
        """
        connection = self._require_connection()
        rows = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND sql IS NOT NULL"
        ).fetchall()
        if not rows:
            raise RuntimeError("数据库里读不到任何表结构")
        return self.train(ddl="\n\n".join(row[0] for row in rows))

    def training_summary(self) -> dict[str, int]:
        """各类训练材料的数量。"""
        return self.store.counts()

    # ============================================================
    # 第 3 步：生成 SQL
    # ============================================================

    def build_request(self, question: str) -> SqlRequest:
        """把检索到的上下文拼成一次完整的生成请求。"""
        hits = self.store.similar_sql(question, top_k=3)
        return SqlRequest(
            question=question,
            dialect=self.dialect,
            ddl_text=self.store.related_ddl(),
            documentation=self.store.related_documentation(question, top_k=2),
            examples=[Example(hit.item.content, hit.item.sql or "") for hit in hits],
        )

    def generate_sql(self, question: str) -> GeneratedSQL:
        """检索上下文 -> 拼请求 -> 交给 LLM。"""
        return self.llm.generate_sql(self.build_request(question))

    def build_prompt(self, question: str) -> str:
        """把这次提问的完整提示词吐出来（调试时非常有用）。"""
        return build_prompt(self.build_request(question))

    # ============================================================
    # 第 4 步：执行 SQL
    # ============================================================

    def run_sql(self, sql: str) -> QueryResult:
        """执行 SQL 并返回列 + 行。

        安全起见 demo 只允许 SELECT / WITH：让 LLM 拥有写权限是危险的，
        生产环境还要再加表白名单、超时和返回行数上限。
        """
        connection = self._require_connection()
        cleaned = sql.strip().rstrip(";").strip()

        words = cleaned.split()
        prefix = words[0].lower() if words else ""
        if prefix not in _ALLOWED_STATEMENT_PREFIXES:
            raise ValueError(
                f"只允许执行只读查询（SELECT / WITH），收到的是 {prefix.upper() or '空语句'}"
            )

        cursor = connection.execute(cleaned)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        return QueryResult(columns=columns, rows=rows)

    # ============================================================
    # 第 5 步：一问到底
    # ============================================================

    def ask(self, question: str, auto_train: bool = True) -> AskResult:
        """问一句中文，返回 SQL + 结果表。

        auto_train=True 时，只要 SQL 执行成功，就把这条「问法 -> SQL」回流到训练库，
        等价于原文里 vn.ask(question=..., auto_train=True) 的效果。
        """
        generated = self.generate_sql(question)
        result = self.run_sql(generated.sql)

        if auto_train and not self.store.has_pair(question, generated.sql):
            self.store.add(TrainingItem.from_question_sql(question, generated.sql))

        return AskResult(
            question=question,
            generated=generated,
            result=result,
            followup_questions=self._followups(question),
        )

    # ============================================================
    # 辅助
    # ============================================================

    def _followups(self, current_question: str, limit: int = 3) -> list[str]:
        """推荐追问：把训练库里已经会答的其他问题列出来。"""
        candidates = [
            item.content
            for item in self.store.items("sql")
            if item.content.strip() != current_question.strip()
        ]
        return candidates[:limit]
