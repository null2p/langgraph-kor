from __future__ import annotations

import sys
from collections import deque
from collections.abc import Callable, Hashable, Sequence
from dataclasses import asdict, dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    NamedTuple,
    TypeVar,
    final,
)
from warnings import warn

from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointMetadata
from typing_extensions import Unpack, deprecated
from xxhash import xxh3_128_hexdigest

from langgraph._internal._cache import default_cache_key
from langgraph._internal._fields import get_cached_annotated_keys, get_update_as_tuples
from langgraph._internal._retry import default_retry_on
from langgraph._internal._typing import MISSING, DeprecatedKwargs
from langgraph.warnings import LangGraphDeprecatedSinceV10

if TYPE_CHECKING:
    from langgraph.pregel.protocol import PregelProtocol


try:
    from langchain_core.messages.tool import ToolOutputMixin
except ImportError:

    class ToolOutputMixin:  # type: ignore[no-redef]
        pass


__all__ = (
    "All",
    "Checkpointer",
    "StreamMode",
    "StreamWriter",
    "RetryPolicy",
    "CachePolicy",
    "Interrupt",
    "StateUpdate",
    "PregelTask",
    "PregelExecutableTask",
    "StateSnapshot",
    "Send",
    "Command",
    "Durability",
    "interrupt",
)

Durability = Literal["sync", "async", "exit"]
"""그래프 실행의 내구성 모드입니다.
- `"sync"`: 다음 단계가 시작되기 전에 변경 사항이 동기적으로 영속화됩니다.
- `"async"`: 다음 단계가 실행되는 동안 변경 사항이 비동기적으로 영속화됩니다.
- `"exit"`: 그래프가 종료될 때만 변경 사항이 영속화됩니다."""

All = Literal["*"]
"""그래프가 모든 노드에서 중단되어야 함을 나타내는 특수 값입니다."""

Checkpointer = None | bool | BaseCheckpointSaver
"""서브그래프에 사용할 체크포인터의 타입입니다.
- True는 이 서브그래프의 영속적 체크포인팅을 활성화합니다.
- False는 부모 그래프에 체크포인터가 있어도 체크포인팅을 비활성화합니다.
- None은 부모 그래프로부터 체크포인터를 상속받습니다."""

StreamMode = Literal[
    "values", "updates", "checkpoints", "tasks", "debug", "messages", "custom"
]
"""stream 메서드가 출력을 방출하는 방식입니다.

- `"values"`: 중단을 포함하여 각 단계 후 상태의 모든 값을 방출합니다.
    함수형 API와 함께 사용되는 경우, 값은 워크플로우 끝에 한 번 방출됩니다.
- `"updates"`: 각 단계 후 노드 또는 작업 이름과 노드 또는 작업이 반환한 업데이트만 방출합니다.
    동일한 단계에서 여러 업데이트가 이루어지면 (예: 여러 노드가 실행되면) 해당 업데이트들이 별도로 방출됩니다.
- `"custom"`: `StreamWriter`를 사용하여 노드 또는 작업 내부에서 사용자 정의 데이터를 방출합니다.
- `"messages"`: 노드 또는 작업 내부의 모든 LLM 호출에 대한 메타데이터와 함께 LLM 메시지를 토큰별로 방출합니다.
- `"checkpoints"`: 체크포인트가 생성될 때 `get_state()`가 반환하는 것과 동일한 형식으로 이벤트를 방출합니다.
- `"tasks"`: 작업이 시작되고 완료될 때 결과와 오류를 포함하여 이벤트를 방출합니다.
- `"debug"`: 디버깅 목적으로 `"checkpoints"` 및 `"tasks"` 이벤트를 방출합니다.
"""

StreamWriter = Callable[[Any], None]
"""단일 인자를 받아 출력 스트림에 기록하는 `Callable`입니다.
키워드 인자로 요청되면 항상 노드에 주입되지만,
`stream_mode="custom"`을 사용하지 않으면 작동하지 않습니다."""

_DC_KWARGS = {"kw_only": True, "slots": True, "frozen": True}


class RetryPolicy(NamedTuple):
    """노드 재시도 설정입니다.

    !!! version-added "버전 0.2.24에서 추가됨"
    """

    initial_interval: float = 0.5
    """첫 번째 재시도가 발생하기 전에 경과해야 하는 시간입니다. 초 단위입니다."""
    backoff_factor: float = 2.0
    """각 재시도 후 간격이 증가하는 배수입니다."""
    max_interval: float = 128.0
    """재시도 간에 경과할 수 있는 최대 시간입니다. 초 단위입니다."""
    max_attempts: int = 3
    """첫 번째 시도를 포함하여 포기하기 전에 시도할 최대 횟수입니다."""
    jitter: bool = True
    """재시도 간 간격에 무작위 지터를 추가할지 여부입니다."""
    retry_on: (
        type[Exception] | Sequence[type[Exception]] | Callable[[Exception], bool]
    ) = default_retry_on
    """재시도를 트리거해야 하는 예외 클래스 목록이거나, 재시도를 트리거해야 하는 예외에 대해 `True`를 반환하는 callable입니다."""


KeyFuncT = TypeVar("KeyFuncT", bound=Callable[..., str | bytes])


@dataclass(**_DC_KWARGS)
class CachePolicy(Generic[KeyFuncT]):
    """노드 캐싱 설정입니다."""

    key_func: KeyFuncT = default_cache_key  # type: ignore[assignment]
    """노드의 입력으로부터 캐시 키를 생성하는 함수입니다.
    기본값은 pickle로 입력을 해싱합니다."""

    ttl: int | None = None
    """캐시 항목의 수명(초)입니다. `None`이면 항목이 만료되지 않습니다."""


_DEFAULT_INTERRUPT_ID = "placeholder-id"


@final
@dataclass(init=False, slots=True)
class Interrupt:
    """노드에서 발생한 중단에 대한 정보입니다.

    !!! version-added "버전 0.2.24에서 추가됨"

    !!! version-changed "버전 v0.4.0에서 변경됨"
        * `interrupt_id`가 속성으로 도입됨

    !!! version-changed "버전 v0.6.0에서 변경됨"

        다음 속성이 제거되었습니다:

        * `ns`
        * `when`
        * `resumable`
        * `interrupt_id`, `id`로 대체되어 지원 중단됨
    """

    value: Any
    """중단과 연관된 값입니다."""

    id: str
    """중단의 ID입니다. 중단을 직접 재개하는 데 사용할 수 있습니다."""

    def __init__(
        self,
        value: Any,
        id: str = _DEFAULT_INTERRUPT_ID,
        **deprecated_kwargs: Unpack[DeprecatedKwargs],
    ) -> None:
        self.value = value

        if (
            (ns := deprecated_kwargs.get("ns", MISSING)) is not MISSING
            and (id == _DEFAULT_INTERRUPT_ID)
            and (isinstance(ns, Sequence))
        ):
            self.id = xxh3_128_hexdigest("|".join(ns).encode())
        else:
            self.id = id

    @classmethod
    def from_ns(cls, value: Any, ns: str) -> Interrupt:
        return cls(value=value, id=xxh3_128_hexdigest(ns.encode()))

    @property
    @deprecated("`interrupt_id` is deprecated. Use `id` instead.", category=None)
    def interrupt_id(self) -> str:
        warn(
            "`interrupt_id` is deprecated. Use `id` instead.",
            LangGraphDeprecatedSinceV10,
            stacklevel=2,
        )
        return self.id


class StateUpdate(NamedTuple):
    values: dict[str, Any] | None
    as_node: str | None = None
    task_id: str | None = None


class PregelTask(NamedTuple):
    """Pregel 작업입니다."""

    id: str
    name: str
    path: tuple[str | int | tuple, ...]
    error: Exception | None = None
    interrupts: tuple[Interrupt, ...] = ()
    state: None | RunnableConfig | StateSnapshot = None
    result: Any | None = None


if sys.version_info > (3, 11):
    _T_DC_KWARGS = {"weakref_slot": True, "slots": True, "frozen": True}
else:
    _T_DC_KWARGS = {"frozen": True}


class CacheKey(NamedTuple):
    """작업의 캐시 키입니다."""

    ns: tuple[str, ...]
    """캐시 항목의 네임스페이스입니다."""
    key: str
    """캐시 항목의 키입니다."""
    ttl: int | None
    """캐시 항목의 수명(초)입니다."""


@dataclass(**_T_DC_KWARGS)
class PregelExecutableTask:
    name: str
    input: Any
    proc: Runnable
    writes: deque[tuple[str, Any]]
    config: RunnableConfig
    triggers: Sequence[str]
    retry_policy: Sequence[RetryPolicy]
    cache_key: CacheKey | None
    id: str
    path: tuple[str | int | tuple, ...]
    writers: Sequence[Runnable] = ()
    subgraphs: Sequence[PregelProtocol] = ()


class StateSnapshot(NamedTuple):
    """단계 시작 시 그래프 상태의 스냅샷입니다."""

    values: dict[str, Any] | Any
    """채널의 현재 값입니다."""
    next: tuple[str, ...]
    """이 단계의 각 작업에서 실행할 노드의 이름입니다."""
    config: RunnableConfig
    """이 스냅샷을 가져오는 데 사용된 설정입니다."""
    metadata: CheckpointMetadata | None
    """이 스냅샷과 연관된 메타데이터입니다."""
    created_at: str | None
    """스냅샷 생성 타임스탬프입니다."""
    parent_config: RunnableConfig | None
    """있는 경우 부모 스냅샷을 가져오는 데 사용된 설정입니다."""
    tasks: tuple[PregelTask, ...]
    """이 단계에서 실행할 작업입니다. 이미 시도된 경우 오류를 포함할 수 있습니다."""
    interrupts: tuple[Interrupt, ...]
    """이 단계에서 발생하여 해결이 보류 중인 중단입니다."""


class Send:
    """그래프의 특정 노드로 전송할 메시지 또는 패킷입니다.

    `Send` 클래스는 `StateGraph`의 조건부 엣지 내에서 사용되어
    다음 단계에서 사용자 정의 상태로 노드를 동적으로 호출합니다.

    중요한 점은 전송된 상태가 핵심 그래프의 상태와 다를 수 있어,
    유연하고 동적인 워크플로우 관리가 가능하다는 것입니다.

    한 가지 예로, 그래프가 다른 상태로 동일한 노드를 여러 번 병렬로 호출한 후
    결과를 메인 그래프의 상태로 다시 집계하는 "map-reduce" 워크플로우가 있습니다.

    속성:
        node (str): 메시지를 전송할 대상 노드의 이름입니다.
        arg (Any): 대상 노드로 전송할 상태 또는 메시지입니다.

    예제:
        >>> from typing import Annotated
        >>> import operator
        >>> class OverallState(TypedDict):
        ...     subjects: list[str]
        ...     jokes: Annotated[list[str], operator.add]
        >>> from langgraph.types import Send
        >>> from langgraph.graph import END, START
        >>> def continue_to_jokes(state: OverallState):
        ...     return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]
        >>> from langgraph.graph import StateGraph
        >>> builder = StateGraph(OverallState)
        >>> builder.add_node("generate_joke", lambda state: {"jokes": [f"Joke about {state['subject']}"]})
        >>> builder.add_conditional_edges(START, continue_to_jokes)
        >>> builder.add_edge("generate_joke", END)
        >>> graph = builder.compile()
        >>>
        >>> # 두 주제로 호출하면 각 주제에 대한 농담이 생성됩니다
        >>> graph.invoke({"subjects": ["cats", "dogs"]})
        {'subjects': ['cats', 'dogs'], 'jokes': ['Joke about cats', 'Joke about dogs']}
    """

    __slots__ = ("node", "arg")

    node: str
    arg: Any

    def __init__(self, /, node: str, arg: Any) -> None:
        """
        `Send` 클래스의 새 인스턴스를 초기화합니다.

        인자:
            node: 메시지를 전송할 대상 노드의 이름입니다.
            arg: 대상 노드로 전송할 상태 또는 메시지입니다.
        """
        self.node = node
        self.arg = arg

    def __hash__(self) -> int:
        return hash((self.node, self.arg))

    def __repr__(self) -> str:
        return f"Send(node={self.node!r}, arg={self.arg!r})"

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, Send)
            and self.node == value.node
            and self.arg == value.arg
        )


N = TypeVar("N", bound=Hashable)


@dataclass(**_DC_KWARGS)
class Command(Generic[N], ToolOutputMixin):
    """그래프의 상태를 업데이트하고 노드에 메시지를 전송하는 하나 이상의 명령입니다.

    !!! version-added "버전 0.2.24에서 추가됨"

    인자:
        graph: 명령을 전송할 그래프입니다. 지원되는 값은:

            - `None`: 현재 그래프
            - `Command.PARENT`: 가장 가까운 부모 그래프
        update: 그래프의 상태에 적용할 업데이트입니다.
        resume: 실행을 재개할 값입니다. [`interrupt()`][langgraph.types.interrupt]와 함께 사용됩니다.
            다음 중 하나일 수 있습니다:

            - 중단 ID를 재개 값에 매핑
            - 다음 중단을 재개할 단일 값
        goto: 다음 중 하나일 수 있습니다:

            - 다음으로 이동할 노드의 이름 (지정된 `graph`에 속하는 모든 노드)
            - 다음으로 이동할 노드 이름의 시퀀스
            - `Send` 객체 (제공된 입력으로 노드를 실행)
            - `Send` 객체의 시퀀스
    """

    graph: str | None = None
    update: Any | None = None
    resume: dict[str, Any] | Any | None = None
    goto: Send | Sequence[Send | N] | N = ()

    def __repr__(self) -> str:
        # None이 아닌 모든 값을 가져옵니다
        contents = ", ".join(
            f"{key}={value!r}" for key, value in asdict(self).items() if value
        )
        return f"Command({contents})"

    def _update_as_tuples(self) -> Sequence[tuple[str, Any]]:
        if isinstance(self.update, dict):
            return list(self.update.items())
        elif isinstance(self.update, (list, tuple)) and all(
            isinstance(t, tuple) and len(t) == 2 and isinstance(t[0], str)
            for t in self.update
        ):
            return self.update
        elif keys := get_cached_annotated_keys(type(self.update)):
            return get_update_as_tuples(self.update, keys)
        elif self.update is not None:
            return [("__root__", self.update)]
        else:
            return []

    PARENT: ClassVar[Literal["__parent__"]] = "__parent__"


def interrupt(value: Any) -> Any:
    """노드 내에서 재개 가능한 예외로 그래프를 중단합니다.

    `interrupt` 함수는 그래프 실행을 일시 중지하고 클라이언트에 값을 노출하여
    human-in-the-loop 워크플로우를 가능하게 합니다. 이 값은 컨텍스트를 전달하거나
    실행을 재개하는 데 필요한 입력을 요청할 수 있습니다.

    주어진 노드에서 이 함수의 첫 번째 호출은 `GraphInterrupt` 예외를 발생시켜
    실행을 중단합니다. 제공된 `value`는 예외와 함께 포함되어
    그래프를 실행하는 클라이언트로 전송됩니다.

    그래프를 재개하는 클라이언트는 [`Command`][langgraph.types.Command] 프리미티브를
    사용하여 중단에 대한 값을 지정하고 실행을 계속해야 합니다.
    그래프는 노드의 시작 부분부터 **모든 로직을 다시 실행**하며 재개됩니다.

    노드에 여러 `interrupt` 호출이 포함된 경우, LangGraph는 노드 내 순서를
    기반으로 재개 값을 중단과 매칭합니다. 이 재개 값 목록은 노드를 실행하는
    특정 작업에만 적용되며 작업 간에 공유되지 않습니다.

    `interrupt`를 사용하려면 체크포인터를 활성화해야 합니다. 이 기능은
    그래프 상태를 영속화하는 데 의존하기 때문입니다.

    예제:
        ```python
        import uuid
        from typing import Optional
        from typing_extensions import TypedDict

        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.constants import START
        from langgraph.graph import StateGraph
        from langgraph.types import interrupt, Command


        class State(TypedDict):
            \"\"\"The graph state.\"\"\"

            foo: str
            human_value: Optional[str]
            \"\"\"Human value will be updated using an interrupt.\"\"\"


        def node(state: State):
            answer = interrupt(
                # This value will be sent to the client
                # as part of the interrupt information.
                \"what is your age?\"
            )
            print(f\"> Received an input from the interrupt: {answer}\")
            return {\"human_value\": answer}


        builder = StateGraph(State)
        builder.add_node(\"node\", node)
        builder.add_edge(START, \"node\")

        # A checkpointer must be enabled for interrupts to work!
        checkpointer = InMemorySaver()
        graph = builder.compile(checkpointer=checkpointer)

        config = {
            \"configurable\": {
                \"thread_id\": uuid.uuid4(),
            }
        }

        for chunk in graph.stream({\"foo\": \"abc\"}, config):
            print(chunk)

        # > {'__interrupt__': (Interrupt(value='what is your age?', id='45fda8478b2ef754419799e10992af06'),)}

        command = Command(resume=\"some input from a human!!!\")

        for chunk in graph.stream(Command(resume=\"some input from a human!!!\"), config):
            print(chunk)

        # > Received an input from the interrupt: some input from a human!!!
        # > {'node': {'human_value': 'some input from a human!!!'}}
        ```

    인자:
        value: 그래프가 중단될 때 클라이언트에 노출할 값입니다.

    반환:
        Any: 동일한 노드(정확하게는 동일한 작업) 내에서 후속 호출 시, 첫 번째 호출 중에 제공된 값을 반환합니다

    예외:
        GraphInterrupt: 노드 내의 첫 번째 호출 시, 실행을 중단하고 제공된 값을 클라이언트에 노출합니다.
    """
    from langgraph._internal._constants import (
        CONFIG_KEY_CHECKPOINT_NS,
        CONFIG_KEY_SCRATCHPAD,
        CONFIG_KEY_SEND,
        RESUME,
    )
    from langgraph.config import get_config
    from langgraph.errors import GraphInterrupt

    conf = get_config()["configurable"]
    # 중단 인덱스 추적
    scratchpad = conf[CONFIG_KEY_SCRATCHPAD]
    idx = scratchpad.interrupt_counter()
    # 이전 재개 값 찾기
    if scratchpad.resume:
        if idx < len(scratchpad.resume):
            conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])
            return scratchpad.resume[idx]
    # 현재 재개 값 찾기
    v = scratchpad.get_null_resume(True)
    if v is not None:
        assert len(scratchpad.resume) == idx, (scratchpad.resume, idx)
        scratchpad.resume.append(v)
        conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])
        return v
    # 재개 값을 찾지 못함
    raise GraphInterrupt(
        (
            Interrupt.from_ns(
                value=value,
                ns=conf[CONFIG_KEY_CHECKPOINT_NS],
            ),
        )
    )
