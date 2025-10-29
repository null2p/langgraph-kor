from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Generic

from typing_extensions import Self

from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel, Value
from langgraph.errors import EmptyChannelError

__all__ = ("Topic",)


def _flatten(values: Sequence[Value | list[Value]]) -> Iterator[Value]:
    for value in values:
        if isinstance(value, list):
            yield from value
        else:
            yield value


class Topic(
    Generic[Value],
    BaseChannel[Sequence[Value], Value | list[Value], list[Value]],
):
    """설정 가능한 PubSub 토픽입니다.

    Args:
        typ: 채널에 저장된 값의 타입입니다.
        accumulate: 단계 간에 값을 누적할지 여부입니다. `False`이면 각 단계 후에 채널이 비워집니다.
    """

    __slots__ = ("values", "accumulate")

    def __init__(self, typ: type[Value], accumulate: bool = False) -> None:
        super().__init__(typ)
        # 속성
        self.accumulate = accumulate
        # 상태
        self.values = list[Value]()

    def __eq__(self, value: object) -> bool:
        return isinstance(value, Topic) and value.accumulate == self.accumulate

    @property
    def ValueType(self) -> Any:
        """채널에 저장된 값의 타입입니다."""
        return Sequence[self.typ]  # type: ignore[name-defined]

    @property
    def UpdateType(self) -> Any:
        """채널이 받는 업데이트의 타입입니다."""
        return self.typ | list[self.typ]  # type: ignore[name-defined]

    def copy(self) -> Self:
        """채널의 복사본을 반환합니다."""
        empty = self.__class__(self.typ, self.accumulate)
        empty.key = self.key
        empty.values = self.values.copy()
        return empty

    def checkpoint(self) -> list[Value]:
        return self.values

    def from_checkpoint(self, checkpoint: list[Value]) -> Self:
        empty = self.__class__(self.typ, self.accumulate)
        empty.key = self.key
        if checkpoint is not MISSING:
            if isinstance(checkpoint, tuple):
                # 하위 호환성
                empty.values = checkpoint[1]
            else:
                empty.values = checkpoint
        return empty

    def update(self, values: Sequence[Value | list[Value]]) -> bool:
        updated = False
        if not self.accumulate:
            updated = bool(self.values)
            self.values = list[Value]()
        if flat_values := tuple(_flatten(values)):
            updated = True
            self.values.extend(flat_values)
        return updated

    def get(self) -> Sequence[Value]:
        if self.values:
            return list(self.values)
        else:
            raise EmptyChannelError

    def is_available(self) -> bool:
        return bool(self.values)
