"""LangGraph 전용 경고입니다."""

from __future__ import annotations

__all__ = (
    "LangGraphDeprecationWarning",
    "LangGraphDeprecatedSinceV05",
    "LangGraphDeprecatedSinceV10",
)


class LangGraphDeprecationWarning(DeprecationWarning):
    """LangGraph 전용 지원 중단 경고입니다.

    속성:
        message: 경고에 대한 설명입니다.
        since: 지원 중단이 도입된 LangGraph 버전입니다.
        expected_removal: 해당 기능이 제거될 것으로 예상되는 LangGraph 버전입니다.

    명확한 버전 정보가 포함된 지원 중단 경고의 훌륭한 표준을 설정한
    Pydantic의 `PydanticDeprecationWarning` 클래스에서 영감을 받았습니다.
    """

    message: str
    since: tuple[int, int]
    expected_removal: tuple[int, int]

    def __init__(
        self,
        message: str,
        *args: object,
        since: tuple[int, int],
        expected_removal: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(message, *args)
        self.message = message.rstrip(".")
        self.since = since
        self.expected_removal = (
            expected_removal if expected_removal is not None else (since[0] + 1, 0)
        )

    def __str__(self) -> str:
        message = (
            f"{self.message}. Deprecated in LangGraph V{self.since[0]}.{self.since[1]}"
            f" to be removed in V{self.expected_removal[0]}.{self.expected_removal[1]}."
        )
        return message


class LangGraphDeprecatedSinceV05(LangGraphDeprecationWarning):
    """LangGraph v0.5.0부터 지원이 중단된 기능을 정의하는 `LangGraphDeprecationWarning`의 특정 서브클래스입니다"""

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args, since=(0, 5), expected_removal=(2, 0))


class LangGraphDeprecatedSinceV10(LangGraphDeprecationWarning):
    """LangGraph v1.0.0부터 지원이 중단된 기능을 정의하는 `LangGraphDeprecationWarning`의 특정 서브클래스입니다"""

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args, since=(1, 0), expected_removal=(2, 0))
