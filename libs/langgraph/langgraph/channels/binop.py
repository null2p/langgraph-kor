import collections.abc
from collections.abc import Callable, Sequence
from typing import Generic

from typing_extensions import NotRequired, Required, Self

from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel, Value
from langgraph.errors import EmptyChannelError

__all__ = ("BinaryOperatorAggregate",)


# typing_extensions에서 가져옴
def _strip_extras(t):  # type: ignore[no-untyped-def]
    """주어진 타입에서 Annotated, Required, NotRequired를 제거합니다."""
    if hasattr(t, "__origin__"):
        return _strip_extras(t.__origin__)
    if hasattr(t, "__origin__") and t.__origin__ in (Required, NotRequired):
        return _strip_extras(t.__args__[0])

    return t


class BinaryOperatorAggregate(Generic[Value], BaseChannel[Value, Value, Value]):
    """현재 값과 각 새 값에 이진 연산자를 적용한 결과를 저장합니다.

    ```python
    import operator

    total = Channels.BinaryOperatorAggregate(int, operator.add)
    ```
    """

    __slots__ = ("value", "operator")

    def __init__(self, typ: type[Value], operator: Callable[[Value, Value], Value]):
        super().__init__(typ)
        self.operator = operator
        # typing이나 collections.abc의 특수 형태는 인스턴스화할 수 없으므로
        # 구체적인 대응 형태로 교체해야 합니다
        typ = _strip_extras(typ)
        if typ in (collections.abc.Sequence, collections.abc.MutableSequence):
            typ = list
        if typ in (collections.abc.Set, collections.abc.MutableSet):
            typ = set
        if typ in (collections.abc.Mapping, collections.abc.MutableMapping):
            typ = dict
        try:
            self.value = typ()
        except Exception:
            self.value = MISSING

    def __eq__(self, value: object) -> bool:
        return isinstance(value, BinaryOperatorAggregate) and (
            value.operator is self.operator
            if value.operator.__name__ != "<lambda>"
            and self.operator.__name__ != "<lambda>"
            else True
        )

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
        empty = self.__class__(self.typ, self.operator)
        empty.key = self.key
        empty.value = self.value
        return empty

    def from_checkpoint(self, checkpoint: Value) -> Self:
        empty = self.__class__(self.typ, self.operator)
        empty.key = self.key
        if checkpoint is not MISSING:
            empty.value = checkpoint
        return empty

    def update(self, values: Sequence[Value]) -> bool:
        if not values:
            return False
        if self.value is MISSING:
            self.value = values[0]
            values = values[1:]
        for value in values:
            self.value = self.operator(self.value, value)
        return True

    def get(self) -> Value:
        if self.value is MISSING:
            raise EmptyChannelError()
        return self.value

    def is_available(self) -> bool:
        return self.value is not MISSING

    def checkpoint(self) -> Value:
        return self.value
