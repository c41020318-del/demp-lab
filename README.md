# demo-lab

个人 demo 库，用来测试各种技术方案。

## 环境

- Python 3.12+
- uv 管理依赖

```bash
uv sync
uv run pytest -v
```

## demo 列表

### RAG 混合检索（BM25 + 向量 + RRF）

```bash
uv run python examples/rag_walkthrough.py
```

### Text-to-SQL（Vanna 风格，零依赖版）

跑通「中文问题 -> SQL -> 执行结果」，向量检索复用上面的 RAG 模块，
数据库用标准库 sqlite3，LLM 可在「联网 DeepSeek」和「离线规则替身」之间切换。

```bash
uv run python examples/text2sql_walkthrough.py

# 连真实大模型（DeepSeek 兼容 OpenAI 协议）只需一个环境变量
DEEPSEEK_API_KEY=sk-xxx uv run python examples/text2sql_walkthrough.py
```

模块位置 `src/demo_lab/text2sql/`：

- `core.py` 训练数据结构（ddl / documentation / question->sql）与中文分词
- `store.py` 训练仓库：混合召回（BM25 + 向量 + RRF）+ 向量精排
- `llm.py` 提示词拼装，以及 `OpenAICompatLLM` / `RuleBasedLLM` 两种实现
- `engine.py` `MiniVanna`：train / generate_sql / run_sql / ask 的完整编排