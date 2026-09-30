"""BM25 最小实现。

BM25 是一个"打分函数"，用来给「查询词 -> 文档」的相关性打分。
它不是模型，不需要训练，只有两个可调参数 k1 和 b。

公式（对每个查询词累加）：
    score(D, Q) = Σ IDF(q) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * |D| / avgdl))

其中：
    f      = 查询词 q 在文档 D 中出现的次数（词频 TF）
    |D|    = 文档 D 的长度（token 数）
    avgdl  = 所有文档的平均长度
    IDF(q) = 逆文档频率，越罕见的词权重越高
"""

import math
from collections import Counter


class BM25:
    def __init__(
        self,
        corpus: list[list[str]],  # 语料库：一个列表，每个元素是一个文档的分词结果
        k1: float = 1.5,  # 词频饱和参数，通常 1.2 ~ 2.0
        b: float = 0.75,  # 长度归一化强度，0=不归一化，1=完全归一化
    ):
        # 把参数存到实例上，后面 _score_term 会用到
        self.k1 = k1
        self.b = b
        self.corpus = corpus

        # 文档总数 N
        self.N = len(corpus)

        # 每个文档的长度（token 数），用于长度归一化
        self.doc_len = [len(doc) for doc in corpus]

        # 所有文档的平均长度，作为长度归一化的基准
        self.avgdl = sum(self.doc_len) / self.N if self.N else 0.0

        # 每个文档的词频表：doc_freqs[i] 是第 i 个文档的 Counter
        # Counter 是 dict 子类，统计 "词 -> 出现次数"
        self.doc_freqs = [Counter(doc) for doc in corpus]

        # 文档频率 DF：某个词出现在多少个文档里（不是出现多少次）
        # 用 set(doc) 去重，保证一个文档里同一个词只算 1 次
        self.df = Counter()
        for doc in corpus:
            for term in set(doc):
                self.df[term] += 1

        # 预计算每个词的 IDF，避免每次查询重复算
        # IDF = ln((N - n + 0.5) / (n + 0.5) + 1)
        #   N = 总文档数，n = 包含该词的文档数
        # 末尾 +1 是为了让 IDF 非负（罕见词 IDF 大，常见词 IDF 接近 0）
        self.idf = {}
        for term, n in self.df.items():
            self.idf[term] = math.log((self.N - n + 0.5) / (n + 0.5) + 1)

    def _score_term(self, term: str, doc_idx: int) -> float:
        """计算「单个查询词」对「某个文档」的贡献分。"""

        # 如果这个词在任何文档里都没出现过，直接 0 分
        if term not in self.idf:
            return 0.0

        # 这个词在该文档里出现了几次（TF）
        f = self.doc_freqs[doc_idx].get(term, 0)

        # 出现 0 次，贡献 0 分
        if f == 0:
            return 0.0

        # 取出预计算好的 IDF
        idf = self.idf[term]

        # 当前文档长度
        dl = self.doc_len[doc_idx]

        # 分母 = TF 饱和项 + 长度归一化项
        # k1 * (1 - b + b * dl / avgdl)：
        #   dl / avgdl 越大（文档越长），惩罚越重
        #   b = 0 时完全忽略长度；b = 1 时完全按长度惩罚
        denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)

        # BM25 核心公式：IDF * (TF * (k1 + 1)) / 分母
        # (k1 + 1) 是归一化系数，让 TF 饱和后的最大值不超过 (k1 + 1)
        return idf * (f * (self.k1 + 1)) / denom

    def score(self, query: list[str], doc_idx: int) -> float:
        """一个查询对单个文档的总分 = 每个查询词贡献之和。"""
        return sum(self._score_term(q, doc_idx) for q in query)

    def search(self, query: list[str], top_k: int = 5) -> list[tuple]:
        """返回 [(doc_idx, score), ...]，按分数降序，只保留正分的文档。"""
        # 给每个文档打分
        scores = [(i, self.score(query, i)) for i in range(self.N)]

        # 过滤掉 0 分的（没有任何查询词命中）
        scores = [s for s in scores if s[1] > 0]

        # 按分数从高到低排序
        scores.sort(key=lambda x: x[1], reverse=True)

        # 取前 top_k 条
        return scores[:top_k]
