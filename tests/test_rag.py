import pytest

from demo_lab.rag import Document, HashingEmbedder, HybridRetriever, rrf_fuse
from demo_lab.rag.core import ScoredDoc


def _build_retriever() -> HybridRetriever:
    docs = [
        Document("d1", "iPhone 15 Pro 电池容量", tokens=["iphone", "15", "pro", "电池", "容量"]),
        Document("d2", "iPhone 15 价格与发布", tokens=["iphone", "15", "价格", "发布"]),
        Document("d3", "华为 Mate 60 电池续航", tokens=["华为", "mate", "60", "电池", "续航"]),
        Document("d4", "小米 14 快充功率", tokens=["小米", "14", "快充", "功率"]),
        Document("d5", "苹果 A17 Pro 芯片性能", tokens=["苹果", "a17", "pro", "芯片", "性能"]),
    ]
    return HybridRetriever(docs)


# ---------- 单路召回 ----------


def test_bm25_search_hits_expected_doc():
    r = _build_retriever()
    hits = r.bm25_search(["iphone", "电池"], top_k=5)
    assert hits, "BM25 should return hits"
    assert hits[0].source == "bm25"
    assert hits[0].doc_id == "d1"


def test_vector_search_returns_results():
    r = _build_retriever()
    hits = r.vector_search(["iphone", "电池"], top_k=5)
    assert hits, "vector should return hits"
    assert hits[0].source == "vector"


# ---------- 混合召回 ----------


def test_hybrid_search_returns_expected_top1():
    r = _build_retriever()
    hits = r.search(["iphone", "电池"], top_k=3)
    assert hits, "hybrid should return hits"
    assert hits[0].source == "hybrid"
    assert hits[0].doc_id == "d1"


def test_hybrid_top_k_respected():
    r = _build_retriever()
    hits = r.search(["iphone", "电池"], top_k=2)
    assert len(hits) == 2


def test_hybrid_empty_query_returns_empty():
    r = _build_retriever()
    assert r.search([], top_k=3) == []


# ---------- RRF ----------


def test_rrf_fuse_merges_both_rankings():
    a = [ScoredDoc("d1", 10.0, "bm25"), ScoredDoc("d2", 5.0, "bm25")]
    b = [ScoredDoc("d2", 0.9, "vector"), ScoredDoc("d3", 0.8, "vector")]
    fused = rrf_fuse([a, b], k=60)
    ids = [x.doc_id for x in fused]
    # d2 在两路都出现，应该排第一
    assert ids[0] == "d2"
    assert set(ids) == {"d1", "d2", "d3"}


def test_rrf_respects_top_k():
    a = [ScoredDoc(f"d{i}", 1.0, "bm25") for i in range(10)]
    fused = rrf_fuse([a], k=60, top_k=3)
    assert len(fused) == 3


# ---------- 嵌入器 ----------


def test_embedder_is_deterministic():
    e = HashingEmbedder(dim=32)
    v1 = e.embed(["a", "b", "c"])
    v2 = e.embed(["a", "b", "c"])
    assert v1 == v2


def test_embedder_norm_is_one():
    e = HashingEmbedder(dim=32)
    v = e.embed(["a", "b", "c"])
    norm = sum(x * x for x in v) ** 0.5
    assert abs(norm - 1.0) < 1e-9


def test_embedder_overlap_similarity():
    e = HashingEmbedder(dim=256)
    a = e.embed(["iphone", "电池", "容量"])
    b = e.embed(["iphone", "电池", "续航"])
    c = e.embed(["小米", "快充"])
    from demo_lab.rag.embedder import cosine

    assert cosine(a, b) > cosine(a, c)


# ---------- 边界 ----------


def test_empty_documents_raises():
    with pytest.raises(ValueError):
        HybridRetriever([])


def test_get_document_by_id():
    r = _build_retriever()
    doc = r.get_document("d1")
    assert doc.doc_id == "d1"
