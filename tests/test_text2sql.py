import sqlite3

import pytest

from demo_lab.text2sql import (
    Example,
    MiniVanna,
    QueryResult,
    RuleBasedLLM,
    SqlGenerationError,
    SqlRequest,
    TemplateRule,
    TrainingItem,
    build_prompt,
    extract_sql,
    tokenize,
)
from demo_lab.text2sql.llm import extract_number, similarity
from demo_lab.text2sql.store import TrainingStore

DDL = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    region TEXT,
    signup_date TEXT
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    amount REAL,
    order_date TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);
"""

TOP_SPENDERS_SQL = """
SELECT c.name, SUM(o.amount) AS total_spent
FROM customers c
JOIN orders o ON c.id = o.customer_id
GROUP BY c.id, c.name
ORDER BY total_spent DESC
LIMIT 5;
"""

RULES = [
    TemplateRule(
        groups=[["活跃客户"]],
        sql="SELECT COUNT(DISTINCT c.id) AS cnt FROM customers c "
        "JOIN orders o ON c.id = o.customer_id "
        "WHERE o.order_date >= date('now', '-{days} days');",
        note="活跃客户模板",
    ),
]

CUSTOMERS = [("张伟", "华东"), ("李娜", "华东"), ("王强", "华北")]
ORDERS = [(1, 6000.0), (1, 5500.0), (2, 3200.0), (3, 9000.0)]


def _new_connection() -> sqlite3.Connection:
    """建一个带演示数据的 SQLite 内存连接。"""
    connection = sqlite3.connect(":memory:")
    connection.executescript(DDL)
    connection.executemany("INSERT INTO customers (name, region) VALUES (?, ?)", CUSTOMERS)
    connection.executemany("INSERT INTO orders (customer_id, amount) VALUES (?, ?)", ORDERS)
    connection.commit()
    return connection


def _build_vanna(rules=None) -> MiniVanna:
    """造一个「已连库 + 已训练」的 MiniVanna。"""
    vn = MiniVanna(llm=RuleBasedLLM(template_rules=rules or RULES))
    vn.set_connection(_new_connection())
    vn.train(ddl=DDL)
    vn.train(documentation="「活跃客户」指过去 90 天内有下单记录的客户。")
    vn.train(question="消费金额最高的 5 位客户是谁？", sql=TOP_SPENDERS_SQL)
    return vn


def build_request(question: str, examples=None) -> SqlRequest:
    """小工具：构造 SqlRequest，省得每个测试都重复写一堆字段。"""
    return SqlRequest(
        question=question,
        examples=[Example(q, s) for q, s in (examples or [])],
    )


# ---------- 分词 ----------


def test_tokenize_splits_ascii_and_chinese():
    tokens = tokenize("华东 iPhone 15")
    assert "iphone" in tokens
    assert "15" in tokens
    assert "华东" in tokens  # 中文補了二元组


def test_tokenize_lowercases_ascii():
    assert "select" in tokenize("SELECT *")


# ---------- 训练仓库 ----------


def test_store_counts_three_kinds():
    store = TrainingStore()
    store.add(TrainingItem.from_ddl(DDL))
    store.add(TrainingItem.from_documentation("术语说明"))
    store.add(TrainingItem.from_question_sql("问题？", "SELECT 1;"))
    assert store.counts() == {"ddl": 1, "documentation": 1, "sql": 1}


def test_store_duplicate_item_id_raises():
    store = TrainingStore()
    store.add(TrainingItem.from_ddl("CREATE TABLE t (id INT);"))
    store.add(TrainingItem.from_ddl("CREATE TABLE t2 (id INT);"))
    with pytest.raises(ValueError, match="重复"):
        store.add(TrainingItem(kind="ddl", content="x", item_id="ddl-1"))


def test_store_search_hits_expected_item():
    vn = _build_vanna()
    hits = vn.store.similar_sql("消费金额最高的 5 位客户是谁？", top_k=1)
    assert hits
    assert hits[0].item.content == "消费金额最高的 5 位客户是谁？"


def test_store_related_documentation_is_filtered_by_question():
    vn = _build_vanna()
    text = vn.store.related_documentation("活跃客户有多少？")
    assert "活跃客户" in text


def test_store_has_pair_ignores_trailing_semicolon():
    store = TrainingStore()
    store.add(TrainingItem.from_question_sql("q", "SELECT 1;"))
    assert store.has_pair("q", "SELECT 1")


# ---------- 提示词 ----------


def test_build_prompt_contains_all_context():
    prompt = build_prompt(
        SqlRequest(
            question="高价值客户有哪些？",
            ddl_text=DDL,
            documentation="「高价值客户」指累计金额超过 10000 元的客户。",
            examples=[Example("样例问题", "SELECT 1;")],
        )
    )
    assert DDL.strip() in prompt
    assert "高价值客户" in prompt
    assert "样例问题" in prompt
    assert "SELECT 1;" in prompt


def test_extract_sql_from_markdown_fence():
    raw = "好的，SQL 如下：\n```sql\nSELECT 1;\n```\n希望有帮助。"
    assert extract_sql(raw) == "SELECT 1"


def test_extract_sql_without_fence():
    assert extract_sql("SELECT 1;") == "SELECT 1"


# ---------- LLM 替身 ----------


def test_rule_llm_template_has_priority_over_example():
    llm = RuleBasedLLM(template_rules=RULES)
    generated = llm.generate_sql(
        build_request(question="过去 90 天内的活跃客户有多少？", examples=[("消费最高的客户", "X")])
    )
    assert generated.source == "template"
    assert "date('now', '-90 days')" in generated.sql


def test_rule_llm_reuses_example_and_rewrites_limit():
    llm = RuleBasedLLM()
    generated = llm.generate_sql(
        build_request(
            question="消费金额最高的前 3 名客户是谁？",
            examples=[("消费金额最高的 5 位客户是谁？", TOP_SPENDERS_SQL)],
        )
    )
    assert generated.source == "example-reuse"
    assert "LIMIT 3" in generated.sql


def test_rule_llm_raises_when_nothing_matches():
    llm = RuleBasedLLM()
    with pytest.raises(SqlGenerationError):
        llm.generate_sql(build_request(question="今天天气怎么样？", examples=[]))


def test_extract_number_supports_chinese():
    assert extract_number("前 3 名", default=5) == 3
    assert extract_number("过期十五天", default=90) == 15
    assert extract_number("没有任何数字", default=7) == 7


def test_similarity_of_identical_text_is_one():
    assert similarity("abc", "abc") == 1.0


# ---------- 引擎 ----------


def test_run_sql_rejects_non_readonly():
    vn = _build_vanna()
    with pytest.raises(ValueError, match="只读"):
        vn.run_sql("DELETE FROM customers")


def test_ask_end_to_end_returns_rows():
    vn = _build_vanna()
    answer = vn.ask("消费金额最高的 5 位客户是谁？", auto_train=False)
    assert answer.generated.source in ("template", "example-reuse")
    assert answer.result.row_count > 0
    assert answer.result.columns[0] == "name"
    assert answer.result.rows[0][0] == "张伟"


def test_ask_requires_connection():
    vn = MiniVanna(llm=RuleBasedLLM(template_rules=RULES))
    vn.train(ddl=DDL)
    # generate_sql 这一步不碰数据库，所以要用能命中模板的问题才能走到执行阶段
    with pytest.raises(RuntimeError, match="连数据库"):
        vn.ask("过去 90 天内的活跃客户有多少？")


def test_train_requires_paired_question_and_sql():
    vn = MiniVanna()
    with pytest.raises(ValueError, match="成对"):
        vn.train(question="只有问题没有 SQL")


def test_train_requires_any_argument():
    vn = MiniVanna()
    with pytest.raises(ValueError, match="至少"):
        vn.train()


def test_train_from_schema_reads_sqlite_master():
    fresh = MiniVanna(llm=RuleBasedLLM(template_rules=RULES)).connect_to_sqlite(":memory:")
    fresh._connection.executescript(DDL)
    added = fresh.train_from_schema()
    assert len(added) == 1
    assert "customers" in added[0].content


def test_auto_train_is_idempotent():
    vn = _build_vanna()
    before = vn.training_summary()["sql"]
    question = "消费金额最高的 5 位客户是谁？"
    vn.ask(question, auto_train=True)
    mid = vn.training_summary()["sql"]
    # 这条样例本来就在训练库里，不应该重复灌进去
    assert mid == before
    vn.ask("消费金额最高的前 3 名客户是谁？", auto_train=True)
    assert vn.training_summary()["sql"] == mid + 1


def test_query_result_renders_table():
    table = QueryResult(columns=["a"], rows=[(1,)]).to_table()
    assert "a" in table
    assert "1" in table


def test_query_result_empty_rows():
    assert QueryResult(columns=[], rows=[]).to_table() == "(没有查到数据)"
