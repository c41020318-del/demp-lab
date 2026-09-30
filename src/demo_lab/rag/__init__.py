"""RAG 模块的对外接口。

外部只需要 `from demo_lab.rag import HybridRetriever` 就能用，
不用关心内部怎么分文件的。
"""

# Document / ScoredDoc 是数据结构，rrf_fuse 是融合算法
from demo_lab.rag.core import Document, ScoredDoc, rrf_fuse

# HashingEmbedder 是 demo 用的嵌入器，cosine 是相似度函数
from demo_lab.rag.embedder import HashingEmbedder, cosine

# HybridRetriever 是"项目经理"，对外只暴露这一个类
from demo_lab.rag.hybrid import HybridRetriever

# __all__ 决定 `from demo_lab.rag import *` 会导出哪些名字
# 也相当于给 IDE 和文档一个"公开接口清单"
__all__ = [
    "Document",
    "ScoredDoc",
    "rrf_fuse",
    "HashingEmbedder",
    "cosine",
    "HybridRetriever",
]
