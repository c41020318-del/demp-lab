"""训练材料仓库：负责「存」和「检索」。

真实 Vanna 这一步用 ChromaDB 做向量检索；本项目已经有 RAG 混合检索器，
直接复用即可 —— BM25（字面） + HashingEmbedder（语义） + RRF（融合），依赖为 0。

对外只暴露四个检索入口：
    related_ddl()           取全部表结构
    related_documentation() 按问题取相关业务术语
    similar_sql()           按问题取最相似的「问法 -> SQL」样例
    has_pair()              判断某个「问法 -> SQL」是否已经训练过（防重复）
"""

from __future__ import annotations

from dataclasses import dataclass

from demo_lab.rag import HashingEmbedder, HybridRetriever, cosine
from demo_lab.rag.core import Document
from demo_lab.text2sql.core import KIND_DDL, KIND_DOCUMENTATION, KIND_SQL, TrainingItem, tokenize


@dataclass
class RetrievedItem:
    """一条检索命中：材料本身 + 两路召回的分。"""

    item: TrainingItem
    score: float  # 精排分：问题与材料的向量余弦相似度，0~1
    rrf_score: float  # 融合分：混合召回阶段的 RRF 分，决定「有没有被召回来」

    def __repr__(self) -> str:  # pragma: no cover - 仅用于调试展示
        return f"RetrievedItem({self.item.item_id}, score={self.score:.4f})"


class TrainingStore:
    """训练材料的增删查。

    为什么要 `items(kind=...)` 而不是每种类型各一个列表？
    因为检索时要按类型隔离：查「样例」不应该被 DDL 文本干扰。
    """

    def __init__(self, embedder: HashingEmbedder | None = None):
        # dim 调大到 128：SQL 里的英文标识符很多，维度小了容易碰撞
        self.embedder = embedder or HashingEmbedder(dim=128)
        self._items: list[TrainingItem] = []
        self._by_id: dict[str, TrainingItem] = {}
        self._retrievers: dict[str, HybridRetriever] = {}  # 每种类型一个独立索引
        self._seq = 0

    # ============================================================
    # 写入
    # ============================================================

    def add(self, item: TrainingItem) -> TrainingItem:
        """登记一条训练材料，返回带 item_id 的对象。"""
        self._seq += 1
        item.item_id = item.item_id or f"{item.kind}-{self._seq}"

        if item.item_id in self._by_id:
            raise ValueError(f"重复的训练材料 id: {item.item_id}")

        self._items.append(item)
        self._by_id[item.item_id] = item

        # 索引失效：新数据加进来了，旧索引必须丢掉重新建
        self._retrievers.pop(item.kind, None)
        return item

    # ============================================================
    # 读取
    # ============================================================

    def items(self, kind: str | None = None) -> list[TrainingItem]:
        """按类型取出材料列表，不传 kind 就是全部。"""
        if kind is None:
            return list(self._items)
        return [item for item in self._items if item.kind == kind]

    def __len__(self) -> int:
        return len(self._items)

    def counts(self) -> dict[str, int]:
        """各类材料的数量，用于打印「训练进度」。"""
        return {
            KIND_DDL: len(self.items(KIND_DDL)),
            KIND_DOCUMENTATION: len(self.items(KIND_DOCUMENTATION)),
            KIND_SQL: len(self.items(KIND_SQL)),
        }

    # ============================================================
    # 检索
    # ============================================================

    def search(self, text: str, kind: str, top_k: int = 3) -> list[RetrievedItem]:
        """在指定类型的材料里做「混合召回 + 向量精排」。

        两阶段而不是一步到位：
            阶段 1 混合召回：BM25 或 向量，只要有一路觉得像就先进候选池（宽进）
            阶段 2 向量精排：用余弦相似度重新排序（严出）
        """
        pool = self.items(kind)
        if not pool or not text.strip():
            return []

        tokens = tokenize(text)

        # 阶段 1：复用现成的混合检索器
        retriever = self._retriever_for(kind)
        hits = retriever.search(tokens, top_k=len(pool))

        # 阶段 2：逐个算余弦相似度
        query_vec = self.embedder.embed(tokens)
        results = [
            RetrievedItem(
                item=self._by_id[hit.doc_id],
                score=cosine(query_vec, self.embedder.embed(self._by_id[hit.doc_id].tokens)),
                rrf_score=hit.score,
            )
            for hit in hits
        ]

        # 按精排分降序（并列时用融合分兜底）
        results.sort(key=lambda r: (r.score, r.rrf_score), reverse=True)
        return results[:top_k]

    def related_ddl(self) -> str:
        """表结构全量给：表通常就几张，与其冒着漏表的风险做截断，不如全带上。"""
        return "\n\n".join(item.content for item in self.items(KIND_DDL))

    def related_documentation(self, question: str, top_k: int = 2) -> str:
        """按问题挑最相关的业务术语。"""
        hits = self.search(question, KIND_DOCUMENTATION, top_k=top_k)
        return "\n".join(hit.item.content for hit in hits)

    def similar_sql(self, question: str, top_k: int = 3) -> list[RetrievedItem]:
        """最相似的「问法 -> SQL」样例，用于 few-shot。"""
        return self.search(question, KIND_SQL, top_k=top_k)

    def has_pair(self, question: str, sql: str) -> bool:
        """这个「问法 -> SQL」是不是已经训练过了？避免 auto_train 反复灌重复数据。"""
        target_sql = sql.strip().rstrip(";").strip()
        return any(
            item.content == question.strip() and _normalize(item.sql or "") == target_sql
            for item in self.items(KIND_SQL)
        )

    # ============================================================
    # 内部实现
    # ============================================================

    def _retriever_for(self, kind: str) -> HybridRetriever:
        """按类型懒加载索引：第一次用到才建，之后缓存；写入时会自动失效。"""
        cached = self._retrievers.get(kind)
        if cached is not None:
            return cached

        pool = self.items(kind)
        if not pool:
            raise ValueError(f"没有任何 {kind} 类型的训练材料")

        documents = [
            Document(doc_id=item.item_id, text=item.content, tokens=item.tokens) for item in pool
        ]
        retriever = HybridRetriever(documents, embedder=self.embedder)
        self._retrievers[kind] = retriever
        return retriever


def _normalize(sql: str) -> str:
    """去掉首尾空白和结尾分号，用于比较两条 SQL 是否等价。"""
    return sql.strip().rstrip(";").strip()
