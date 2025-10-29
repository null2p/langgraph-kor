from __future__ import annotations

from typing_extensions import TypeVar

from langgraph._internal._typing import StateLike

__all__ = (
    "StateT",
    "StateT_co",
    "StateT_contra",
    "InputT",
    "OutputT",
    "ContextT",
)

StateT = TypeVar("StateT", bound=StateLike)
"""그래프의 상태를 나타내는 데 사용되는 타입 변수입니다."""

StateT_co = TypeVar("StateT_co", bound=StateLike, covariant=True)

StateT_contra = TypeVar("StateT_contra", bound=StateLike, contravariant=True)

ContextT = TypeVar("ContextT", bound=StateLike | None, default=None)
"""그래프 실행 범위 컨텍스트를 나타내는 데 사용되는 타입 변수입니다.

기본값은 `None`입니다.
"""

ContextT_contra = TypeVar(
    "ContextT_contra", bound=StateLike | None, contravariant=True, default=None
)

InputT = TypeVar("InputT", bound=StateLike, default=StateT)
"""상태 그래프의 입력을 나타내는 데 사용되는 타입 변수입니다.

기본값은 `StateT`입니다.
"""

OutputT = TypeVar("OutputT", bound=StateLike, default=StateT)
"""상태 그래프의 출력을 나타내는 데 사용되는 타입 변수입니다.

기본값은 `StateT`입니다.
"""

NodeInputT = TypeVar("NodeInputT", bound=StateLike)
"""노드의 입력을 나타내는 데 사용되는 타입 변수입니다."""

NodeInputT_contra = TypeVar("NodeInputT_contra", bound=StateLike, contravariant=True)
