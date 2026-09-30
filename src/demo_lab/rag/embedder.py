"""极简嵌入器：把 token 列表映射成稠密向量。

⚠️ 这不是真正的语义嵌入，只是个 demo 替身。
   它有 demo 需要的两个关键性质：
   1. 确定性：同样的输入永远得到同样的向量。
   2. 词重叠越多，向量越"像"。

生产环境请把这个类替换成 sentence-transformers / BGE / OpenAI embedding，
只要保持 "embed(tokens) -> list[float]" 这个接口不变即可。
"""

from __future__ import annotations

import hashlib  # 用于把 token 稳定地映射到一个整数
import math  # 用于算 L2 范数


class HashingEmbedder:
    """哈希嵌入器：把每个 token 哈希到一个固定的维度索引上。

    比如 dim=64，那么 "iphone" 会固定映射到 0~63 中的某一个位置，
    该位置的数值 +1。多个 token 落到同一个位置就累加。
    """

    def __init__(self, dim: int = 64):
        self.dim = dim  # 向量维度

    def _hash_to_index(self, token: str) -> int:
        """把一个 token 稳定地映射到 [0, dim) 的下标。"""
        # md5 保证同一个 token 永远得到同样的十六进制串
        h = hashlib.md5(token.encode("utf-8")).hexdigest()

        # 把十六进制串当成整数，再对 dim 取模，得到下标
        return int(h, 16) % self.dim

    def embed(self, tokens: list[str]) -> list[float]:
        """把 token 列表转成 L2 归一化后的向量。"""
        # 先建一个全 0 的向量
        vec = [0.0] * self.dim

        # 遍历每个 token，落到对应下标上 +1
        for token in tokens:
            vec[self._hash_to_index(token)] += 1.0

        # 归一化：让向量长度为 1
        return self._l2_normalize(vec)

    def embed_batch(self, token_lists: list[list[str]]) -> list[list[float]]:
        """批量嵌入，对每个文档调一次 embed。"""
        return [self.embed(tokens) for tokens in token_lists]

    @staticmethod
    def _l2_normalize(vec: list[float]) -> list[float]:
        """L2 归一化：把向量除以自己的长度，使长度变成 1。

        为什么要归一化？
        归一化后，两个向量的余弦相似度 = 它们的点积，
        计算更快，数值也更稳定。
        """
        # 所有元素平方和再开根号
        norm = math.sqrt(sum(v * v for v in vec))

        # 全 0 向量没法归一化，直接返回
        if norm == 0:
            return vec

        # 每个元素除以 norm
        return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """两个已 L2 归一化的向量的余弦相似度。

    因为已归一化，余弦 = 点积，所以这里只需要逐元素相乘再求和。
    zip(..., strict=True) 保证两个向量长度一致，否则报错。
    """
    return sum(x * y for x, y in zip(a, b, strict=True))
