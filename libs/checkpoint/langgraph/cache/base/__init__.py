from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Generic, TypeVar

from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

ValueT = TypeVar("ValueT")
Namespace = tuple[str, ...]
FullKey = tuple[Namespace, str]


class BaseCache(ABC, Generic[ValueT]):
    """캐시의 베이스 클래스입니다."""

    serde: SerializerProtocol = JsonPlusSerializer(pickle_fallback=True)

    def __init__(self, *, serde: SerializerProtocol | None = None) -> None:
        """시리얼라이저로 캐시를 초기화합니다."""
        self.serde = serde or self.serde

    @abstractmethod
    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """주어진 키에 대한 캐시된 값을 가져옵니다."""

    @abstractmethod
    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """주어진 키에 대한 캐시된 값을 비동기적으로 가져옵니다."""

    @abstractmethod
    def set(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """주어진 키와 TTL에 대한 캐시된 값을 설정합니다."""

    @abstractmethod
    async def aset(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """주어진 키와 TTL에 대한 캐시된 값을 비동기적으로 설정합니다."""

    @abstractmethod
    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """주어진 네임스페이스에 대한 캐시된 값을 삭제합니다.
        네임스페이스가 제공되지 않으면 모든 캐시된 값을 지웁니다."""

    @abstractmethod
    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """주어진 네임스페이스에 대한 캐시된 값을 비동기적으로 삭제합니다.
        네임스페이스가 제공되지 않으면 모든 캐시된 값을 지웁니다."""
