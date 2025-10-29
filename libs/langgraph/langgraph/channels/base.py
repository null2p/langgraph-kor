from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from typing_extensions import Self

from langgraph._internal._typing import MISSING
from langgraph.errors import EmptyChannelError

Value = TypeVar("Value")
Update = TypeVar("Update")
Checkpoint = TypeVar("Checkpoint")

__all__ = ("BaseChannel",)


class BaseChannel(Generic[Value, Update, Checkpoint], ABC):
    """모든 채널의 베이스 클래스입니다."""

    __slots__ = ("key", "typ")

    def __init__(self, typ: Any, key: str = "") -> None:
        self.typ = typ
        self.key = key

    @property
    @abstractmethod
    def ValueType(self) -> Any:
        """채널에 저장된 값의 타입입니다."""

    @property
    @abstractmethod
    def UpdateType(self) -> Any:
        """채널이 받는 업데이트의 타입입니다."""

    # 직렬화/역직렬화 메서드

    def copy(self) -> Self:
        """채널의 복사본을 반환합니다.
        기본적으로 `checkpoint()`와 `from_checkpoint()`에 위임합니다.
        서브클래스는 더 효율적인 구현으로 이 메서드를 재정의할 수 있습니다."""
        return self.from_checkpoint(self.checkpoint())

    def checkpoint(self) -> Checkpoint | Any:
        """채널의 현재 상태의 직렬화 가능한 표현을 반환합니다.
        채널이 비어 있거나 (아직 업데이트되지 않음) 체크포인트를 지원하지 않으면
        `EmptyChannelError`를 발생시킵니다."""
        try:
            return self.get()
        except EmptyChannelError:
            return MISSING

    @abstractmethod
    def from_checkpoint(self, checkpoint: Checkpoint | Any) -> Self:
        """새로운 동일한 채널을 반환하며, 선택적으로 체크포인트에서 초기화합니다.
        체크포인트에 복잡한 데이터 구조가 포함된 경우 복사되어야 합니다."""

    # 읽기 메서드

    @abstractmethod
    def get(self) -> Value:
        """채널의 현재 값을 반환합니다.

        채널이 비어 있으면 (아직 업데이트되지 않음) `EmptyChannelError`를 발생시킵니다."""

    def is_available(self) -> bool:
        """채널이 사용 가능하면 (비어 있지 않음) `True`를, 그렇지 않으면 `False`를 반환합니다.
        서브클래스는 `get()`을 호출하고 `EmptyChannelError`를 잡는 것보다
        더 효율적인 구현을 제공하도록 이 메서드를 재정의해야 합니다.
        """
        try:
            self.get()
            return True
        except EmptyChannelError:
            return False

    # 쓰기 메서드

    @abstractmethod
    def update(self, values: Sequence[Update]) -> bool:
        """주어진 업데이트 시퀀스로 채널의 값을 업데이트합니다.
        시퀀스의 업데이트 순서는 임의적입니다.
        이 메서드는 각 단계의 끝에 모든 채널에 대해 Pregel에 의해 호출됩니다.
        업데이트가 없으면 빈 시퀀스로 호출됩니다.
        업데이트 시퀀스가 유효하지 않으면 `InvalidUpdateError`를 발생시킵니다.
        채널이 업데이트되면 `True`를, 그렇지 않으면 `False`를 반환합니다."""

    def consume(self) -> bool:
        """구독된 작업이 실행되었음을 채널에 알립니다. 기본적으로 no-op입니다.
        채널은 이 메서드를 사용하여 상태를 수정하고
        값이 다시 소비되는 것을 방지할 수 있습니다.

        채널이 업데이트되면 `True`를, 그렇지 않으면 `False`를 반환합니다.
        """
        return False

    def finish(self) -> bool:
        """Pregel 실행이 완료됨을 채널에 알립니다. 기본적으로 no-op입니다.
        채널은 이 메서드를 사용하여 상태를 수정하고 완료를 방지할 수 있습니다.

        채널이 업데이트되면 `True`를, 그렇지 않으면 `False`를 반환합니다.
        """
        return False
