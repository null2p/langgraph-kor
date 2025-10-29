from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Generic, cast

from langgraph.store.base import BaseStore
from typing_extensions import TypedDict, Unpack

from langgraph._internal._constants import CONF, CONFIG_KEY_RUNTIME
from langgraph.config import get_config
from langgraph.types import _DC_KWARGS, StreamWriter
from langgraph.typing import ContextT

__all__ = ("Runtime", "get_runtime")


def _no_op_stream_writer(_: Any) -> None: ...


class _RuntimeOverrides(TypedDict, Generic[ContextT], total=False):
    context: ContextT
    store: BaseStore | None
    stream_writer: StreamWriter
    previous: Any


@dataclass(**_DC_KWARGS)
class Runtime(Generic[ContextT]):
    """실행 범위 컨텍스트 및 기타 런타임 유틸리티를 번들링하는 편의 클래스입니다.

    !!! version-added "버전 v0.6.0에서 추가됨"

    예제:

    ```python
    from typing import TypedDict
    from langgraph.graph import StateGraph
    from dataclasses import dataclass
    from langgraph.runtime import Runtime
    from langgraph.store.memory import InMemoryStore


    @dataclass
    class Context:  # (1)!
        user_id: str


    class State(TypedDict, total=False):
        response: str


    store = InMemoryStore()  # (2)!
    store.put(("users",), "user_123", {"name": "Alice"})


    def personalized_greeting(state: State, runtime: Runtime[Context]) -> State:
        '''런타임 컨텍스트와 스토어를 사용하여 개인화된 인사말을 생성합니다.'''
        user_id = runtime.context.user_id  # (3)!
        name = "unknown_user"
        if runtime.store:
            if memory := runtime.store.get(("users",), user_id):
                name = memory.value["name"]

        response = f"Hello {name}! Nice to see you again."
        return {"response": response}


    graph = (
        StateGraph(state_schema=State, context_schema=Context)
        .add_node("personalized_greeting", personalized_greeting)
        .set_entry_point("personalized_greeting")
        .set_finish_point("personalized_greeting")
        .compile(store=store)
    )

    result = graph.invoke({}, context=Context(user_id="user_123"))
    print(result)
    # > {'response': 'Hello Alice! Nice to see you again.'}
    ```

    1. 런타임 컨텍스트의 스키마를 정의합니다.
    2. 메모리 및 기타 정보를 영속화할 스토어를 생성합니다.
    3. 런타임 컨텍스트를 사용하여 `user_id`에 액세스합니다.
    """

    context: ContextT = field(default=None)  # type: ignore[assignment]
    """그래프 실행을 위한 정적 컨텍스트입니다. 예: `user_id`, `db_conn` 등.

    '실행 종속성'으로도 생각할 수 있습니다."""

    store: BaseStore | None = field(default=None)
    """그래프 실행을 위한 스토어로, 영속성과 메모리를 가능하게 합니다."""

    stream_writer: StreamWriter = field(default=_no_op_stream_writer)
    """사용자 정의 스트림에 기록하는 함수입니다."""

    previous: Any = field(default=None)
    """주어진 스레드의 이전 반환 값입니다.

    체크포인터가 제공된 경우에만 함수형 API에서 사용할 수 있습니다.
    """

    def merge(self, other: Runtime[ContextT]) -> Runtime[ContextT]:
        """두 런타임을 병합합니다.

        다른 런타임에 값이 제공되지 않으면 현재 런타임의 값이 사용됩니다.
        """
        return Runtime(
            context=other.context or self.context,
            store=other.store or self.store,
            stream_writer=other.stream_writer
            if other.stream_writer is not _no_op_stream_writer
            else self.stream_writer,
            previous=other.previous or self.previous,
        )

    def override(
        self, **overrides: Unpack[_RuntimeOverrides[ContextT]]
    ) -> Runtime[ContextT]:
        """주어진 재정의로 런타임을 새 런타임으로 교체합니다."""
        return replace(self, **overrides)


DEFAULT_RUNTIME = Runtime(
    context=None,
    store=None,
    stream_writer=_no_op_stream_writer,
    previous=None,
)


def get_runtime(context_schema: type[ContextT] | None = None) -> Runtime[ContextT]:
    """현재 그래프 실행의 런타임을 가져옵니다.

    인자:
        context_schema: 런타임의 반환 타입을 타입 힌팅하는 데 사용되는 선택적 스키마입니다.

    반환:
        현재 그래프 실행의 런타임입니다.
    """

    # TODO: 이상적인 세계에서는 설정과 독립적인
    # 런타임용 컨텍스트 매니저를 가질 것입니다. 이것은
    # configurable 패킹의 제거로부터 이어질 것입니다
    runtime = cast(Runtime[ContextT], get_config()[CONF].get(CONFIG_KEY_RUNTIME))
    return runtime
