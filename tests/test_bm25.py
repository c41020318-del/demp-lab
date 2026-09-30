from demo_lab.bm25.core import BM25


def test_bm25_ranks_matching_doc_first():
    corpus = [
        ["iphone", "15", "电池", "容量"],
        ["iphone", "15", "价格"],
        ["华为", "mate", "60", "电池"],
    ]
    bm25 = BM25(corpus)
    results = bm25.search(["iphone", "电池"], top_k=2)

    assert results, "should return at least one result"
    assert results[0][0] == 0, "doc 0 should rank first"


def test_bm25_ignores_unknown_terms():
    corpus = [["a", "b"], ["c", "d"]]
    bm25 = BM25(corpus)
    assert bm25.search(["zzz"]) == []
