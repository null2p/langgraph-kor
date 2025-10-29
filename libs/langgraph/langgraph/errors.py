from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any
from warnings import warn

# EmptyChannelError is re-exported from langgraph.channels.base
from langgraph.checkpoint.base import EmptyChannelError  # noqa: F401
from typing_extensions import deprecated

from langgraph.types import Command, Interrupt
from langgraph.warnings import LangGraphDeprecatedSinceV10

__all__ = (
    "EmptyChannelError",
    "ErrorCode",
    "GraphRecursionError",
    "InvalidUpdateError",
    "GraphBubbleUp",
    "GraphInterrupt",
    "NodeInterrupt",
    "ParentCommand",
    "EmptyInputError",
    "TaskNotFound",
)


class ErrorCode(Enum):
    GRAPH_RECURSION_LIMIT = "GRAPH_RECURSION_LIMIT"
    INVALID_CONCURRENT_GRAPH_UPDATE = "INVALID_CONCURRENT_GRAPH_UPDATE"
    INVALID_GRAPH_NODE_RETURN_VALUE = "INVALID_GRAPH_NODE_RETURN_VALUE"
    MULTIPLE_SUBGRAPHS = "MULTIPLE_SUBGRAPHS"
    INVALID_CHAT_HISTORY = "INVALID_CHAT_HISTORY"


def create_error_message(*, message: str, error_code: ErrorCode) -> str:
    return (
        f"{message}\n"
        "For troubleshooting, visit: https://python.langchain.com/docs/"
        f"troubleshooting/errors/{error_code.value}"
    )


class GraphRecursionError(RecursionError):
    """그래프가 최대 단계 수를 초과했을 때 발생합니다.

    이는 무한 루프를 방지합니다. 최대 단계 수를 늘리려면,
    더 높은 `recursion_limit`를 지정한 config로 그래프를 실행하세요.

    문제 해결 가이드:

    - [`GRAPH_RECURSION_LIMIT`](https://docs.langchain.com/oss/python/langgraph/GRAPH_RECURSION_LIMIT)

    예제:

        graph = builder.compile()
        graph.invoke(
            {"messages": [("user", "Hello, world!")]},
            # config는 두 번째 위치 인자입니다
            {"recursion_limit": 1000},
        )
    """

    pass


class InvalidUpdateError(Exception):
    """유효하지 않은 업데이트 세트로 채널을 업데이트하려고 할 때 발생합니다.

    문제 해결 가이드:

    - [`INVALID_CONCURRENT_GRAPH_UPDATE`](https://docs.langchain.com/oss/python/langgraph/INVALID_CONCURRENT_GRAPH_UPDATE)
    - [`INVALID_GRAPH_NODE_RETURN_VALUE`](https://docs.langchain.com/oss/python/langgraph/INVALID_GRAPH_NODE_RETURN_VALUE)
    """

    pass


class GraphBubbleUp(Exception):
    pass


class GraphInterrupt(GraphBubbleUp):
    """서브그래프가 중단되어 루트 그래프에 의해 억제될 때 발생합니다.
    직접 발생하거나 사용자에게 노출되지 않습니다."""

    def __init__(self, interrupts: Sequence[Interrupt] = ()) -> None:
        super().__init__(interrupts)


@deprecated(
    "NodeInterrupt is deprecated. Please use [`interrupt`][langgraph.types.interrupt] instead.",
    category=None,
)
class NodeInterrupt(GraphInterrupt):
    """노드가 실행을 중단하기 위해 발생시킵니다."""

    def __init__(self, value: Any, id: str | None = None) -> None:
        warn(
            "NodeInterrupt is deprecated. Please use `langgraph.types.interrupt` instead.",
            LangGraphDeprecatedSinceV10,
            stacklevel=2,
        )
        if id is None:
            super().__init__([Interrupt(value=value)])
        else:
            super().__init__([Interrupt(value=value, id=id)])


class ParentCommand(GraphBubbleUp):
    args: tuple[Command]

    def __init__(self, command: Command) -> None:
        super().__init__(command)


class EmptyInputError(Exception):
    """그래프가 빈 입력을 받았을 때 발생합니다."""

    pass


class TaskNotFound(Exception):
    """실행기가 작업을 찾을 수 없을 때 발생합니다 (분산 모드용)."""

    pass
