"""RAG 混合检索的「数据 + 融合算法」核心。

这个文件不依赖任何其他 RAG 文件，是纯工具/数据结构，
所以被 hybrid.py 引用，也可以被其他检索器复用。
"""

# 让类型注解延迟求值，允许写 list[Document] 这种"自己引用自己"的类型
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """一条待检索的文档。

    @dataclass 是 Python 自带的语法糖，自动生成 __init__ / __repr__ / __eq__。
    """

    doc_id: str  # 唯一标识，比如 "d1"
    text: str  # 原始文本，用于展示
    tokens: list[str] = field(default_factory=list)  # 分词结果，BM25 和向量都依赖它
    metadata: dict = field(default_factory=dict)  # 预留字段，比如来源、时间


@dataclass
class ScoredDoc:
    """召回结果：哪个文档、得了多少分、来自哪一路召回。"""

    doc_id: str
    score: float
    source: str  # "bm25" / "vector" / "hybrid"，调试时能看出这条是谁给的


def rrf_fuse(
    rankings: list[list[ScoredDoc]],  # 多路召回结果，每路是一个降序列表
    k: int = 60,  # 平滑参数，常用 60
    top_k: int | None = None,  # 返回几条，None 表示全返回
) -> list[ScoredDoc]:
    """Reciprocal Rank Fusion（倒数排名融合）。

    核心思想：只看「排名」，不看「原始分数」。
    这样即使 BM25 的分数是 0~10，向量的分数是 0~1，也能公平融合。

    公式：score(doc) = Σ over 每一路 1 / (k + rank + 1)
    排名越靠前，1/(k+rank) 越大，累加后得分越高。
    """

    # 用一个字典累加每个文档的融合分
    fused: dict[str, float] = {}

    # 遍历每一路召回
    for ranking in rankings:
        # enumerate 得到 (排名, 文档)，排名从 0 开始
        for rank, doc in enumerate(ranking):
            # 排名 0 加 1/(k+1)，排名 1 加 1/(k+2)，以此类推
            # fused.get(doc.doc_id, 0.0) 表示如果没见过这个文档就从 0 开始累加
            fused[doc.doc_id] = fused.get(doc.doc_id, 0.0) + 1.0 / (k + rank + 1)

    # 把字典转成 ScoredDoc 列表
    result = [
        ScoredDoc(doc_id=doc_id, score=score, source="hybrid") for doc_id, score in fused.items()
    ]

    # 按融合分降序
    result.sort(key=lambda x: x.score, reverse=True)

    # 按 top_k 截断
    return result[:top_k] if top_k else result
