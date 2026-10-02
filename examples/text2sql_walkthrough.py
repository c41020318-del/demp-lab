"""Text-to-SQL（Vanna 风格）最小实现 demo —— 中文可视化走查。

离线运行（默认，规则替身当 LLM）：
    uv run python examples/text2sql_walkthrough.py

联网运行（换成真实的 DeepSeek 大模型，其余代码一行不改）：
    set DEEPSEEK_API_KEY=sk-xxxx
    uv run python examples/text2sql_walkthrough.py
"""

from __future__ import annotations

import os
import sqlite3
from datetime import date, timedelta

from demo_lab.text2sql import MiniVanna, OpenAICompatLLM, RuleBasedLLM, TemplateRule

# ============================================================
# demo 用的配置
# ============================================================

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

DOCUMENTATION = """
「高价值客户」指累计订单金额超过 10000 元的客户。
「活跃客户」指过去 90 天内有下单记录的客户。
"""

# 业务术语里写的阈值，模板 SQL 要和它保持一致
HIGH_VALUE_THRESHOLD = 10000

TEMPLATE_RULES = [
    TemplateRule(
        groups=[["最高", "客户"], ["最高", "金额"], ["消费", "最高"], ["花钱", "最多"]],
        sql=(
            "SELECT c.name, SUM(o.amount) AS total_spent\n"
            "FROM customers c\n"
            "JOIN orders o ON c.id = o.customer_id\n"
            "GROUP BY c.id, c.name\n"
            "ORDER BY total_spent DESC\n"
            "LIMIT {limit};"
        ),
        note="消费排行模板：JOIN + SUM + GROUP BY + ORDER BY DESC + LIMIT",
    ),
    TemplateRule(
        groups=[["活跃客户"], ["活跃", "客户"]],
        sql=(
            "SELECT COUNT(DISTINCT c.id) AS active_customers\n"
            "FROM customers c\n"
            "JOIN orders o ON c.id = o.customer_id\n"
            "WHERE o.order_date >= date('now', '-{days} days');"
        ),
        note="活跃客户模板：术语定义 -> COUNT(DISTINCT) + 时间窗口过滤",
    ),
    TemplateRule(
        groups=[["高价值客户"], ["高价值"]],
        sql=(
            "SELECT c.name, SUM(o.amount) AS total_spent\n"
            "FROM customers c\n"
            "JOIN orders o ON c.id = o.customer_id\n"
            "GROUP BY c.id, c.name\n"
            f"HAVING SUM(o.amount) > {HIGH_VALUE_THRESHOLD}\n"
            "ORDER BY total_spent DESC;"
        ),
        note=f"高价值客户模板：按文档里的阈值 HAVING SUM(amount) > {HIGH_VALUE_THRESHOLD}",
    ),
    TemplateRule(
        groups=[["大区", "客户"], ["各", "区", "客户"], ["区", "多少", "客户"], ["区域", "分布"]],
        sql=(
            "SELECT region, COUNT(*) AS customer_count\n"
            "FROM customers\n"
            "GROUP BY region\n"
            "ORDER BY customer_count DESC;"
        ),
        note="分组统计模板：GROUP BY region + COUNT(*)",
    ),
]

# (客户名, 大区, 注册日)
CUSTOMERS = [
    ("张伟", "华东", 400),
    ("李娜", "华东", 520),
    ("王强", "华北", 300),
    ("刘洋", "华南", 210),
    ("陈静", "华北", 150),
    ("赵磊", "华东", 90),
    ("孙悦", "华南", 60),
    ("周涛", "华北", 30),
]

# (客户 id, 金额, 下单距今多少天)
ORDERS = [
    (1, 6000.0, 200),
    (1, 5500.0, 20),
    (2, 3200.0, 400),
    (2, 4100.0, 350),
    (3, 9000.0, 120),
    (3, 2000.0, 30),
    (4, 1500.0, 80),
    (5, 8000.0, 95),
    (6, 4500.0, 10),
    (6, 1200.0, 5),
    (7, 7000.0, 45),
    (8, 300.0, 15),
]


# ============================================================
# 工具函数
# ============================================================


def line(title: str = "") -> None:
    print("\n" + "=" * 66)
    if title:
        print(f"  {title}")
        print("=" * 66)


def build_demo_database() -> sqlite3.Connection:
    """建一个内存 SQLite 库并塞入演示数据。

    这一步对标原文里的 vn.connect_to_mysql(...)：真实项目换成 MySQL 即可，
    下面的 DDL 也特意写成 MySQL 兼容的语法（只有自增关键字不同）。
    """
    connection = sqlite3.connect(":memory:")
    connection.executescript(DDL)

    today = date.today()

    def days_ago(days: int) -> str:
        return (today - timedelta(days=days)).isoformat()

    customer_rows = [(name, region, days_ago(d)) for name, region, d in CUSTOMERS]
    order_rows = [(cid, amount, days_ago(d)) for cid, amount, d in ORDERS]

    connection.executemany(
        "INSERT INTO customers (name, region, signup_date) VALUES (?, ?, ?)", customer_rows
    )
    connection.executemany(
        "INSERT INTO orders (customer_id, amount, order_date) VALUES (?, ?, ?)", order_rows
    )
    connection.commit()
    return connection


def choose_llm():
    """有 API Key 就用真实大模型，没有就用离线规则替身。"""
    if os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"):
        print("  检测到 API Key -> 使用 OpenAICompatLLM（DeepSeek 兼容 OpenAI 协议）")
        return OpenAICompatLLM.from_env()
    print("  没检测到 API Key -> 使用 RuleBasedLLM（离线替身，模式匹配出 SQL）")
    return RuleBasedLLM(template_rules=TEMPLATE_RULES)


def ask_and_print(vn: MiniVanna, question: str, show_prompt: bool = False) -> None:
    """提问并打印：来源 -> SQL -> 结果表。"""
    print(f"\n  [提问] {question}")

    if show_prompt:
        print("\n  --- 实际发给 LLM 的提示词 ---")
        for row in vn.build_prompt(question).splitlines():
            print(f"  | {row}")
        print("  -----------------------------")

    answer = vn.ask(question=question, auto_train=True)
    print(f"  [来源] {answer.generated.source}  {answer.generated.note}")
    print("  [SQL]")
    for row in answer.sql.splitlines():
        print(f"      {row}")
    print(f"\n  [结果] 共 {answer.result.row_count} 行:")
    for row in answer.result.to_table().splitlines():
        print(f"      {row}")
    if answer.followup_questions:
        print(f"  [追问] {' / '.join(answer.followup_questions)}")


# ============================================================
# 主流程
# ============================================================


def main() -> None:
    line("第 1 步: 准备 demo 数据库（对标 vn.connect_to_mysql）")
    connection = build_demo_database()
    customer_count = connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    order_count = connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    print(f"  SQLite 内存库建好了: customers {customer_count} 行, orders {order_count} 行")
    print("  关键改动：用标准库 sqlite3 代替 MySQL，任何人 clone 下来都能直接跑。")

    line("第 2 步: 选择 LLM（对标 OpenAI_Chat / DeepSeek 客户端）")
    llm = choose_llm()

    line("第 3 步: 组装 MiniVanna 并连库")
    vn = MiniVanna(llm=llm).set_connection(connection, dialect="sqlite")
    print("  MiniVanna = 训练仓库(检索) + LLM(生成) + 数据库连接(执行)")

    line("第 4 步: 训练 DDL —— 告诉模型有哪些表和字段")
    vn.train(ddl=DDL)
    print("  已登记 1 条 DDL 训练数据")

    line("第 5 步: 训练文档 —— 解释业务术语")
    vn.train(documentation=DOCUMENTATION)
    print("  已登记 1 条文档训练数据")
    print("  作用：不解释的话，模型不知道「高价值客户」是金额 1 万还是 10 万。")

    line("第 6 步: 训练样例 —— 给一组「问法 -> 正确 SQL」")
    vn.train(
        question="消费金额最高的 5 位客户是谁？",
        sql=(
            "SELECT c.name, SUM(o.amount) AS total_spent\n"
            "FROM customers c\n"
            "JOIN orders o ON c.id = o.customer_id\n"
            "GROUP BY c.id, c.name\n"
            "ORDER BY total_spent DESC\n"
            "LIMIT 5;"
        ),
    )
    print("  已登记 1 条 question->sql 训练数据")
    print(f"  当前训练库: {vn.training_summary()}")

    # 先复制一份「只有 few-shot 样例」的平行实例，供第 11 步对比使用
    fewshot_only = MiniVanna(llm=RuleBasedLLM()).set_connection(connection)
    for item in vn.store.items():
        fewshot_only.store.add(item)

    line("第 7 步: 提问① —— 命中 SQL 模板，数字直接从问题里抽")
    print("  注意优先级：模板 > 相似样例。模板结果可控，所以放在第一位。")
    ask_and_print(vn, "消费金额最高的 5 位客户是谁？", show_prompt=True)

    line("第 8 步: 提问② —— 靠业务术语翻译成时间窗口")
    ask_and_print(vn, "过去 90 天内的活跃客户有多少位？")

    line("第 9 步: 提问③ —— 靠业务术语翻译成聚合阈值")
    ask_and_print(vn, "高价值客户有哪些？")

    line("第 10 步: 提问④ —— 换个数字，模板里的 LIMIT 会跟着变")
    ask_and_print(vn, "消费金额最高的前 3 名客户是谁？")

    line("第 11 步: 把模板摘掉，只看 few-shot 检索还能不能答对")
    print("  平行实例里没有任何模板，训练库只有最初那一条样例（不涉及前面几轮的 auto_train）。")
    ask_and_print(fewshot_only, "消费金额最高的前 3 名客户是谁？")
    print("\n  它复用了「前 5 名」的样例 SQL，并把 LIMIT 5 改写成了 LIMIT 3。")

    line("第 12 步: auto_train 的效果")
    print(f"  训练库从最初的 1 条样例涨到现在: {vn.training_summary()}")
    print("  每次成功的问答都会回流成新的「问法 -> SQL」样例，越用越准。")

    line("演示结束")
    print("  想换成真实环境，只需要改两处：")
    print("    1. LLM:   set DEEPSEEK_API_KEY=sk-xxx  （或 OpenAICompatLLM(api_key=...)）")
    print("    2. 数据库: vn.connect_to_mysql(host=..., dbname=..., user=..., password=...)")


if __name__ == "__main__":
    main()
