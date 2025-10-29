from __future__ import annotations

import asyncio
import datetime
import sqlite3
import threading
from collections.abc import Mapping, Sequence

from langgraph.cache.base import BaseCache, FullKey, Namespace, ValueT
from langgraph.checkpoint.serde.base import SerializerProtocol


class SqliteCache(BaseCache[ValueT]):
    """SQLite를 사용하는 파일 기반 캐시입니다."""

    def __init__(
        self,
        *,
        path: str,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """파일 경로로 캐시를 초기화합니다."""
        super().__init__(serde=serde)
        # SQLite 백업 저장소
        self._conn = sqlite3.connect(
            path,
            check_same_thread=False,
        )
        # 스레드 간 공유 연결에 대한 액세스를 직렬화합니다
        self._lock = threading.RLock()
        # 더 나은 동시성 및 원자성
        self._conn.execute("PRAGMA journal_mode=WAL;")
        # 스키마: key -> (expiry, encoding, value)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS cache (
                ns TEXT,
                key TEXT,
                expiry REAL,
                encoding TEXT NOT NULL,
                val BLOB NOT NULL,
                PRIMARY KEY (ns, key)
            )"""
        )
        self._conn.commit()

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """주어진 키에 대한 캐시된 값을 가져옵니다."""
        with self._lock, self._conn:
            now = datetime.datetime.now(datetime.timezone.utc).timestamp()
            if not keys:
                return {}
            placeholders = ",".join("(?, ?)" for _ in keys)
            params: list[str] = []
            for ns_tuple, key in keys:
                params.extend((",".join(ns_tuple), key))
            cursor = self._conn.execute(
                f"SELECT ns, key, expiry, encoding, val FROM cache WHERE (ns, key) IN ({placeholders})",
                tuple(params),
            )
            values: dict[FullKey, ValueT] = {}
            rows = cursor.fetchall()
            for ns, key, expiry, encoding, raw in rows:
                if expiry is not None and now > expiry:
                    # 만료된 항목 삭제
                    self._conn.execute(
                        "DELETE FROM cache WHERE (ns, key) = (?, ?)", (ns, key)
                    )
                    continue
                values[(tuple(ns.split(",")), key)] = self.serde.loads_typed(
                    (encoding, raw)
                )
            return values

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """주어진 키에 대한 캐시된 값을 비동기적으로 가져옵니다."""
        return await asyncio.to_thread(self.get, keys)

    def set(self, mapping: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """주어진 키와 TTL에 대한 캐시 값을 설정합니다."""
        with self._lock, self._conn:
            now = datetime.datetime.now(datetime.timezone.utc)
            for key, (value, ttl) in mapping.items():
                if ttl is not None:
                    delta = datetime.timedelta(seconds=ttl)
                    expiry: float | None = (now + delta).timestamp()
                else:
                    expiry = None
                encoding, raw = self.serde.dumps_typed(value)
                self._conn.execute(
                    "INSERT OR REPLACE INTO cache (ns, key, expiry, encoding, val) VALUES (?, ?, ?, ?, ?)",
                    (",".join(key[0]), key[1], expiry, encoding, raw),
                )

    async def aset(self, mapping: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """주어진 키와 TTL에 대한 캐시 값을 비동기적으로 설정합니다."""
        await asyncio.to_thread(self.set, mapping)

    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """주어진 네임스페이스에 대한 캐시된 값을 삭제합니다.
        네임스페이스가 제공되지 않으면 모든 캐시된 값을 삭제합니다."""
        with self._lock, self._conn:
            if namespaces is None:
                self._conn.execute("DELETE FROM cache")
            else:
                placeholders = ",".join("?" for _ in namespaces)
                self._conn.execute(
                    f"DELETE FROM cache WHERE (ns) IN ({placeholders})",
                    tuple(",".join(key) for key in namespaces),
                )

    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """주어진 네임스페이스에 대한 캐시된 값을 비동기적으로 삭제합니다.
        네임스페이스가 제공되지 않으면 모든 캐시된 값을 삭제합니다."""
        await asyncio.to_thread(self.clear, namespaces)

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
