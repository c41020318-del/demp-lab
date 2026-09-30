"""RAG 混合检索 demo —— 中文可视化走查。

运行: uv run python examples/rag_walkthrough.py
"""

from demo_lab.rag import Document, HybridRetriever


def line(title: str = "") -> None:
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def main() -> None:
    line("第 1 步: 准备文档库")
    docs = [
        Document("d1", "iPhone 15 Pro 电池容量", tokens=["iphone", "15", "pro", "电池", "容量"]),
        Document("d2", "iPhone 15 价格与发布", tokens=["iphone", "15", "价格", "发布"]),
        Document("d3", "华为 Mate 60 电池续航", tokens=["华为", "mate", "60", "电池", "续航"]),
        Document("d4", "小米 14 快充功率", tokens=["小米", "14", "快充", "功率"]),
        Document("d5", "苹果 A17 Pro 芯片性能", tokens=["苹果", "a17", "pro", "芯片", "性能"]),
    ]
    for d in docs:
        print(f"  [{d.doc_id}] {d.text}")
        print(f"        分词: {d.tokens}")

    line("第 2 步: 构建混合检索器")
    retriever = HybridRetriever(docs)
    print("  ✅ BM25 索引建好了 (统计了每个词出现在哪些文档里)")
    print("  ✅ 向量索引建好了 (每个文档变成 64 维数字向量)")

    query = ["iphone", "电池"]
    print(f"\n  我们的查询词: {query}")
    print("  人话: 找同时和 'iphone' 和 '电池' 相关的文档")

    line("第 3 步: BM25 召回 (字面匹配)")
    print("  BM25 只看字面: 查询词在文档里出现了几次、这个词有多罕见")
    bm25_hits = retriever.bm25_search(query, top_k=5)
    if not bm25_hits:
        print("  (没找到任何文档)")
    for rank, hit in enumerate(bm25_hits, 1):
        text = retriever.get_document(hit.doc_id).text
        print(f"    第{rank}名 [{hit.doc_id}] 分数={hit.score:.4f}  {text}")

    line("第 4 步: 向量召回 (语义匹配)")
    print("  向量看语义: 查询词和文档都被变成数字向量, 比较方向是否接近")
    vector_hits = retriever.vector_search(query, top_k=5)
    if not vector_hits:
        print("  (没找到任何文档)")
    for rank, hit in enumerate(vector_hits, 1):
        text = retriever.get_document(hit.doc_id).text
        print(f"    第{rank}名 [{hit.doc_id}] 分数={hit.score:.4f}  {text}")

    line("第 5 步: RRF 融合 (投票合并两路结果)")
    print("  规则: 一个文档在某一路排名越靠前, 得分越高; 两路都上榜的分数叠加")
    print("  公式: score = Σ 1 / (60 + 排名)")
    fused = retriever.search(query, top_k=5)
    for rank, hit in enumerate(fused, 1):
        text = retriever.get_document(hit.doc_id).text
        print(f"    第{rank}名 [{hit.doc_id}] 融合分={hit.score:.6f}  {text}")

    line("第 6 步: 最终答案")
    if fused:
        best = fused[0]
        print(f"  🎯 最相关的文档是 [{best.doc_id}]: {retriever.get_document(best.doc_id).text}")
    else:
        print("  ❌ 没有找到任何相关文档")

    line("演示结束")


if __name__ == "__main__":
    main()
