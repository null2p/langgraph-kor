from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Generic

from typing_extensions import Self

from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel, Value
from langgraph.errors import (
    EmptyChannelError,
    ErrorCode,
    InvalidUpdateError,
    create_error_message,
)

__all__ = ("LastValue", "LastValueAfterFinish")


class LastValue(Generic[Value], BaseChannel[Value, Value, Value]):
    """수신된 마지막 값을 저장하며, 단계당 최대 하나의 값을 받을 수 있습니다."""

    __slots__ = ("value",)

    value: Value | Any

    def __init__(self, typ: Any, key: str = "") -> None:
        super().__init__(typ, key)
        self.value = MISSING

    def __eq__(self, value: object) -> bool:
        return isinstance(value, LastValue)

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
        empty = self.__class__(self.typ, self.key)
        empty.value = self.value
        return empty

    def from_checkpoint(self, checkpoint: Value) -> Self:
        empty = self.__class__(self.typ, self.key)
        if checkpoint is not MISSING:
            empty.value = checkpoint
        return empty

    def update(self, values: Sequence[Value]) -> bool:
        if len(values) == 0:
            return False
        if len(values) != 1:
            msg = create_error_message(
                message=f"At key '{self.key}': Can receive only one value per step. Use an Annotated key to handle multiple values.",
                error_code=ErrorCode.INVALID_CONCURRENT_GRAPH_UPDATE,
            )
            raise InvalidUpdateError(msg)

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


class LastValueAfterFinish(
    Generic[Value], BaseChannel[Value, Value, tuple[Value, bool]]
):
    """수신된 마지막 값을 저장하지만, finish() 후에만 사용 가능합니다.
    사용 가능하게 되면 값을 지웁니다."""

    __slots__ = ("value", "finished")

    value: Value | Any
    finished: bool

    def __init__(self, typ: Any, key: str = "") -> None:
        super().__init__(typ, key)
        self.value = MISSING
        self.finished = False

    def __eq__(self, value: object) -> bool:
        return isinstance(value, LastValueAfterFinish)

    @property
    def ValueType(self) -> type[Value]:
        """채널에 저장된 값의 타입입니다."""
        return self.typ

    @property
    def UpdateType(self) -> type[Value]:
        """채널이 받는 업데이트의 타입입니다."""
        return self.typ

    def checkpoint(self) -> tuple[Value | Any, bool] | Any:
        if self.value is MISSING:
            return MISSING
        return (self.value, self.finished)

    def from_checkpoint(self, checkpoint: tuple[Value | Any, bool] | Any) -> Self:
        empty = self.__class__(self.typ)
        empty.key = self.key
        if checkpoint is not MISSING:
            empty.value, empty.finished = checkpoint
        return empty

    def update(self, values: Sequence[Value | Any]) -> bool:
        if len(values) == 0:
            return False

        self.finished = False
        self.value = values[-1]
        return True

    def consume(self) -> bool:
        if self.finished:
            self.finished = False
            self.value = MISSING
            return True

        return False

    def finish(self) -> bool:
        if not self.finished and self.value is not MISSING:
            self.finished = True
            return True
        else:
            return False

    def get(self) -> Value:
        if self.value is MISSING or not self.finished:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not MISSING and self.finished
