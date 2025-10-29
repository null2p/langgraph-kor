from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langgraph.cache.base import BaseCache, FullKey, Namespace, ValueT
from langgraph.checkpoint.serde.base import SerializerProtocol


class RedisCache(BaseCache[ValueT]):
    """TTL 지원을 포함한 Redis 기반 캐시 구현입니다."""

    def __init__(
        self,
        redis: Any,
        *,
        serde: SerializerProtocol | None = None,
        prefix: str = "langgraph:cache:",
    ) -> None:
        """Redis 클라이언트로 캐시를 초기화합니다.

        Args:
            redis: Redis 클라이언트 인스턴스 (동기 또는 비동기)
            serde: 값에 사용할 시리얼라이저
            prefix: 모든 캐시된 값에 대한 키 접두사
        """
        super().__init__(serde=serde)
        self.redis = redis
        self.prefix = prefix

    def _make_key(self, ns: Namespace, key: str) -> str:
        """네임스페이스와 키로부터 Redis 키를 생성합니다."""
        ns_str = ":".join(ns) if ns else ""
        return f"{self.prefix}{ns_str}:{key}" if ns_str else f"{self.prefix}{key}"

    def _parse_key(self, redis_key: str) -> tuple[Namespace, str]:
        """Redis 키를 네임스페이스와 키로 다시 파싱합니다."""
        if not redis_key.startswith(self.prefix):
            raise ValueError(
                f"Key {redis_key} does not start with prefix {self.prefix}"
            )

        remaining = redis_key[len(self.prefix) :]
        if ":" in remaining:
            parts = remaining.split(":")
            key = parts[-1]
            ns_parts = parts[:-1]
            return (tuple(ns_parts), key)
        else:
            return (tuple(), remaining)

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """주어진 키에 대한 캐시된 값을 가져옵니다."""
        if not keys:
            return {}

        # Redis 키 생성
        redis_keys = [self._make_key(ns, key) for ns, key in keys]

        # MGET을 사용하여 Redis에서 값 가져오기
        try:
            raw_values = self.redis.mget(redis_keys)
        except Exception:
            # Redis를 사용할 수 없는 경우 빈 딕셔너리 반환
            return {}

        values: dict[FullKey, ValueT] = {}
        for i, raw_value in enumerate(raw_values):
            if raw_value is not None:
                try:
                    # 값 역직렬화
                    encoding, data = raw_value.split(b":", 1)
                    values[keys[i]] = self.serde.loads_typed((encoding.decode(), data))
                except Exception:
                    # 손상된 항목 건너뛰기
                    continue

        return values

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """주어진 키에 대한 캐시된 값을 비동기적으로 가져옵니다."""
        return self.get(keys)

    def set(self, mapping: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """주어진 키와 TTL에 대한 캐시된 값을 설정합니다."""
        if not mapping:
            return

        # 효율적인 배치 작업을 위해 파이프라인 사용
        pipe = self.redis.pipeline()

        for (ns, key), (value, ttl) in mapping.items():
            redis_key = self._make_key(ns, key)
            encoding, data = self.serde.dumps_typed(value)

            # "encoding:data" 형식으로 저장
            serialized_value = f"{encoding}:".encode() + data

            if ttl is not None:
                pipe.setex(redis_key, ttl, serialized_value)
            else:
                pipe.set(redis_key, serialized_value)

        try:
            pipe.execute()
        except Exception:
            # Redis를 사용할 수 없는 경우 조용히 실패
            pass

    async def aset(self, mapping: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """주어진 키와 TTL에 대한 캐시된 값을 비동기적으로 설정합니다."""
        self.set(mapping)

    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """주어진 네임스페이스에 대한 캐시된 값을 삭제합니다.
        네임스페이스가 제공되지 않으면 모든 캐시된 값을 지웁니다."""
        try:
            if namespaces is None:
                # 접두사를 가진 모든 키 지우기
                pattern = f"{self.prefix}*"
                keys = self.redis.keys(pattern)
                if keys:
                    self.redis.delete(*keys)
            else:
                # 특정 네임스페이스 지우기
                keys_to_delete = []
                for ns in namespaces:
                    ns_str = ":".join(ns) if ns else ""
                    pattern = (
                        f"{self.prefix}{ns_str}:*" if ns_str else f"{self.prefix}*"
                    )
                    keys = self.redis.keys(pattern)
                    keys_to_delete.extend(keys)

                if keys_to_delete:
                    self.redis.delete(*keys_to_delete)
        except Exception:
            # Redis를 사용할 수 없는 경우 조용히 실패
            pass

    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """주어진 네임스페이스에 대한 캐시된 값을 비동기적으로 삭제합니다.
        네임스페이스가 제공되지 않으면 모든 캐시된 값을 지웁니다."""
        self.clear(namespaces)
