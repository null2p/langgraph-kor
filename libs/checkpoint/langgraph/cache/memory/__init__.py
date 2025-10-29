from __future__ import annotations

import datetime
import threading
from collections.abc import Mapping, Sequence

from langgraph.cache.base import BaseCache, FullKey, Namespace, ValueT
from langgraph.checkpoint.serde.base import SerializerProtocol


class InMemoryCache(BaseCache[ValueT]):
    def __init__(self, *, serde: SerializerProtocol | None = None):
        super().__init__(serde=serde)
        self._cache: dict[Namespace, dict[str, tuple[str, bytes, float | None]]] = {}
        self._lock = threading.RLock()

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """주어진 키에 대한 캐시된 값을 가져옵니다."""
        with self._lock:
            if not keys:
                return {}
            now = datetime.datetime.now(datetime.timezone.utc).timestamp()
            values: dict[FullKey, ValueT] = {}
            for ns_tuple, key in keys:
                ns = Namespace(ns_tuple)
                if ns in self._cache and key in self._cache[ns]:
                    enc, val, expiry = self._cache[ns][key]
                    if expiry is None or now < expiry:
                        values[(ns, key)] = self.serde.loads_typed((enc, val))
                    else:
                        del self._cache[ns][key]
            return values

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """주어진 키에 대한 캐시된 값을 비동기적으로 가져옵니다."""
        return self.get(keys)

    def set(self, keys: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """주어진 키에 대한 캐시된 값을 설정합니다."""
        with self._lock:
            now = datetime.datetime.now(datetime.timezone.utc)
            for (ns, key), (value, ttl) in keys.items():
                if ttl is not None:
                    delta = datetime.timedelta(seconds=ttl)
                    expiry: float | None = (now + delta).timestamp()
                else:
                    expiry = None
                if ns not in self._cache:
                    self._cache[ns] = {}
                self._cache[ns][key] = (
                    *self.serde.dumps_typed(value),
                    expiry,
                )

    async def aset(self, keys: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """주어진 키에 대한 캐시된 값을 비동기적으로 설정합니다."""
        self.set(keys)

    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """주어진 네임스페이스에 대한 캐시된 값을 삭제합니다.
        네임스페이스가 제공되지 않으면 모든 캐시된 값을 지웁니다."""
        with self._lock:
            if namespaces is None:
                self._cache.clear()
            else:
                for ns in namespaces:
                    if ns in self._cache:
                        del self._cache[ns]

    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """주어진 네임스페이스에 대한 캐시된 값을 비동기적으로 삭제합니다.
        네임스페이스가 제공되지 않으면 모든 캐시된 값을 지웁니다."""
        self.clear(namespaces)
