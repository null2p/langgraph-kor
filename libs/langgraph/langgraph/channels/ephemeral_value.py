from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Generic

from typing_extensions import Self

from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel, Value
from langgraph.errors import EmptyChannelError, InvalidUpdateError

__all__ = ("EphemeralValue",)


class EphemeralValue(Generic[Value], BaseChannel[Value, Value, Value]):
    """직전 단계에서 받은 값을 저장하고, 그 후에 지웁니다."""

    __slots__ = ("value", "guard")

    value: Value | Any
    guard: bool

    def __init__(self, typ: Any, guard: bool = True) -> None:
        super().__init__(typ)
        self.guard = guard
        self.value = MISSING

    def __eq__(self, value: object) -> bool:
        return isinstance(value, EphemeralValue) and value.guard == self.guard

    @property
    def ValueType(self) -> type[Value]:
        """채널에 저장된 값의 타입입니다."""
        return self.typ

    @property
    def UpdateType(self) -> type[Value]:
        """채널이 받는 업데이트의 타입입니다."""
        return self.typ

    def copy(self) -> Self:
        """채널의 복사본을 반환합니다."""
        empty = self.__class__(self.typ, self.guard)
        empty.key = self.key
        empty.value = self.value
        return empty

    def from_checkpoint(self, checkpoint: Value) -> Self:
        empty = self.__class__(self.typ, self.guard)
        empty.key = self.key
        if checkpoint is not MISSING:
            empty.value = checkpoint
        return empty

    def update(self, values: Sequence[Value]) -> bool:
        if len(values) == 0:
            if self.value is not MISSING:
                self.value = MISSING
                return True
            else:
                return False
        if len(values) != 1 and self.guard:
            raise InvalidUpdateError(
                f"At key '{self.key}': EphemeralValue(guard=True) can receive only one value per step. Use guard=False if you want to store any one of multiple values."
            )

        self.value = values[-1]
        return True

    def get(self) -> Value:
        if self.value is MISSING:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not MISSING

    def checkpoint(self) -> Value:
        return self.value
