"""混合检索：BM25 稀疏召回 + 向量稠密召回 + RRF 融合。

这是 RAG 的"项目经理"，对外只暴露 HybridRetriever 一个类。
"""

from __future__ import annotations

from demo_lab.bm25.core import BM25  # 复用已有的 BM25
from demo_lab.rag.core import Document, ScoredDoc, rrf_fuse  # 数据 + 融合
from demo_lab.rag.embedder import HashingEmbedder, cosine  # 嵌入 + 相似度


class HybridRetriever:
    """混合检索器：初始化时索引所有文档，检索时融合两路结果。"""

    def __init__(
        self,
        documents: list[Document],
        embedder: HashingEmbedder | None = None,  # 允许外部换嵌入器
    ):
        # 空文档没意义，直接报错，让调用方尽早发现问题
        if not documents:
            raise ValueError("documents must not be empty")

        # 保存文档列表
        self.documents = documents

        # 文档 id 列表（顺序和 documents 一致）
        self.doc_ids = [d.doc_id for d in documents]

        # id -> 下标，用于 O(1) 查找
        self.id_to_index = {d.doc_id: i for i, d in enumerate(documents)}

        # ---------- 稀疏索引：BM25 ----------
        # BM25 需要"分词后的语料"，从 Document.tokens 抽取
        corpus = [d.tokens for d in documents]
        self.bm25 = BM25(corpus)

        # ---------- 稠密索引：向量 ----------
        # 如果调用方没传嵌入器，用默认的 HashingEmbedder
        self.embedder = embedder or HashingEmbedder()

        # 给每个文档预计算向量，检索时不用重复算
        self.doc_vectors = self.embedder.embed_batch(corpus)

    # ============================================================
    # 单路召回
    # ============================================================

    def bm25_search(self, query_tokens: list[str], top_k: int = 10) -> list[ScoredDoc]:
        """BM25 召回：字面匹配。"""
        # BM25 返回的是 [(doc_idx, score), ...]
        hits = self.bm25.search(query_tokens, top_k=top_k)

        # 把 doc_idx 转回 doc_id，并包装成 ScoredDoc
        return [
            ScoredDoc(doc_id=self.doc_ids[idx], score=score, source="bm25") for idx, score in hits
        ]

    def vector_search(self, query_tokens: list[str], top_k: int = 10) -> list[ScoredDoc]:
        """向量召回：语义相似。"""
        # 把查询词也编码成向量
        q_vec = self.embedder.embed(query_tokens)

        # 对每个文档向量算余弦相似度
        scored = [(i, cosine(q_vec, doc_vec)) for i, doc_vec in enumerate(self.doc_vectors)]

        # 只保留相似度 > 0 的（相似度 <= 0 表示方向相反或无关）
        scored = [x for x in scored if x[1] > 0]

        # 降序排序
        scored.sort(key=lambda x: x[1], reverse=True)

        # 包装成 ScoredDoc 并截断
        return [
            ScoredDoc(doc_id=self.doc_ids[i], score=s, source="vector") for i, s in scored[:top_k]
        ]

    # ============================================================
    # 混合召回（对外的主要接口）
    # ============================================================

    def search(
        self,
        query_tokens: list[str],
        top_k: int = 5,  # 最终返回几条
        recall_k: int = 20,  # 每路召回几条候选
        rrf_k: int = 60,  # RRF 平滑参数
    ) -> list[ScoredDoc]:
        """先两路各召回 recall_k 条，RRF 融合后取 top_k 条。

        为什么 recall_k 比 top_k 大？
        因为融合的输入越丰富，融合后的结果越准。
        典型漏斗：宽召回 -> 融合 -> 精排 -> 输出。
        """
        # 第一路：BM25
        bm25_hits = self.bm25_search(query_tokens, top_k=recall_k)

        # 第二路：向量
        vector_hits = self.vector_search(query_tokens, top_k=recall_k)

        # RRF 融合
        fused = rrf_fuse([bm25_hits, vector_hits], k=rrf_k, top_k=top_k)

        return fused

    # ============================================================
    # 辅助方法
    # ============================================================

    def get_document(self, doc_id: str) -> Document:
        """按 doc_id 取回原始文档。"""
        return self.documents[self.id_to_index[doc_id]]
