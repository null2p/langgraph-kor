"""테스트를 위한 임베딩 유틸리티입니다."""

import math
import random
from collections import Counter, defaultdict
from typing import Any

from langchain_core.embeddings import Embeddings


class CharacterEmbeddings(Embeddings):
    """랜덤 프로젝션을 사용한 간단한 문자 빈도 기반 임베딩입니다."""

    def __init__(self, dims: int = 50, seed: int = 42):
        """임베딩 차원과 랜덤 시드로 초기화합니다."""
        self._rng = random.Random(seed)
        self.dims = dims
        # 각 문자에 대한 프로젝션 벡터를 지연 생성합니다
        self._char_projections: defaultdict[str, list[float]] = defaultdict(
            lambda: [
                self._rng.gauss(0, 1 / math.sqrt(self.dims)) for _ in range(self.dims)
            ]
        )

    def _embed_one(self, text: str) -> list[float]:
        """단일 텍스트를 임베딩합니다."""
        counts = Counter(text)
        total = sum(counts.values())

        if total == 0:
            return [0.0] * self.dims

        embedding = [0.0] * self.dims
        for char, count in counts.items():
            weight = count / total
            char_proj = self._char_projections[char]
            for i, proj in enumerate(char_proj):
                embedding[i] += weight * proj

        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """문서 목록을 임베딩합니다."""
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """쿼리 문자열을 임베딩합니다."""
        return self._embed_one(text)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, CharacterEmbeddings) and self.dims == other.dims
