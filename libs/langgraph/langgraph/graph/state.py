from __future__ import annotations

import inspect
import logging
import typing
import warnings
from collections import defaultdict
from collections.abc import Awaitable, Callable, Hashable, Sequence
from functools import partial
from inspect import isclass, isfunction, ismethod, signature
from types import FunctionType
from types import NoneType as NoneType
from typing import (
    Any,
    Generic,
    Literal,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.cache.base import BaseCache
from langgraph.checkpoint.base import Checkpoint
from langgraph.store.base import BaseStore
from pydantic import BaseModel, TypeAdapter
from typing_extensions import NotRequired, Required, Self, Unpack, is_typeddict

from langgraph._internal._constants import (
    INTERRUPT,
    NS_END,
    NS_SEP,
    TASKS,
)
from langgraph._internal._fields import (
    get_cached_annotated_keys,
    get_field_default,
    get_update_as_tuples,
)
from langgraph._internal._pydantic import create_model
from langgraph._internal._runnable import coerce_to_runnable
from langgraph._internal._typing import EMPTY_SEQ, MISSING, DeprecatedKwargs
from langgraph.channels.base import BaseChannel
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.last_value import LastValue, LastValueAfterFinish
from langgraph.channels.named_barrier_value import (
    NamedBarrierValue,
    NamedBarrierValueAfterFinish,
)
from langgraph.constants import END, START, TAG_HIDDEN
from langgraph.errors import (
    ErrorCode,
    InvalidUpdateError,
    ParentCommand,
    create_error_message,
)
from langgraph.graph._branch import BranchSpec
from langgraph.graph._node import StateNode, StateNodeSpec
from langgraph.managed.base import (
    ManagedValueSpec,
    is_managed_value,
)
from langgraph.pregel import Pregel
from langgraph.pregel._read import ChannelRead, PregelNode
from langgraph.pregel._write import (
    ChannelWrite,
    ChannelWriteEntry,
    ChannelWriteTupleEntry,
)
from langgraph.types import (
    All,
    CachePolicy,
    Checkpointer,
    Command,
    RetryPolicy,
    Send,
)
from langgraph.typing import ContextT, InputT, NodeInputT, OutputT, StateT
from langgraph.warnings import LangGraphDeprecatedSinceV05, LangGraphDeprecatedSinceV10

__all__ = ("StateGraph", "CompiledStateGraph")

logger = logging.getLogger(__name__)

_CHANNEL_BRANCH_TO = "branch:to:{}"


def _warn_invalid_state_schema(schema: type[Any] | Any) -> None:
    if isinstance(schema, type):
        return
    if typing.get_args(schema):
        return
    warnings.warn(
        f"Invalid state_schema: {schema}. Expected a type or Annotated[type, reducer]. "
        "Please provide a valid schema to ensure correct updates.\n"
        " See: https://langchain-ai.github.io/langgraph/reference/graphs/#stategraph"
    )


def _get_node_name(node: StateNode[Any, ContextT]) -> str:
    try:
        return getattr(node, "__name__", node.__class__.__name__)
    except AttributeError:
        raise TypeError(f"Unsupported node type: {type(node)}")


class StateGraph(Generic[StateT, ContextT, InputT, OutputT]):
    """공유 상태를 읽고 쓰는 방식으로 노드들이 통신하는 그래프입니다.
    각 노드의 시그니처는 State -> Partial<State>입니다.

    각 상태 키는 선택적으로 리듀서 함수로 어노테이션할 수 있으며,
    이 함수는 여러 노드로부터 받은 해당 키의 값들을 집계하는 데 사용됩니다.
    리듀서 함수의 시그니처는 `(Value, Value) -> Value`입니다.

    Args:
        state_schema: 상태를 정의하는 스키마 클래스입니다.
        context_schema: 런타임 컨텍스트를 정의하는 스키마 클래스입니다.
            이를 사용하여 `user_id`, `db_conn` 등과 같은 불변 컨텍스트 데이터를 노드에 노출할 수 있습니다.
        input_schema: 그래프의 입력을 정의하는 스키마 클래스입니다.
        output_schema: 그래프의 출력을 정의하는 스키마 클래스입니다.

    !!! warning "`config_schema` 지원 중단"
        `config_schema` 파라미터는 v0.6.0에서 지원 중단되었으며 v2.0.0에서 제거될 예정입니다.
        대신 `context_schema`를 사용하여 실행 범위 컨텍스트의 스키마를 지정하십시오.

    Example:
        ```python
        from langchain_core.runnables import RunnableConfig
        from typing_extensions import Annotated, TypedDict
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import StateGraph
        from langgraph.runtime import Runtime


        def reducer(a: list, b: int | None) -> list:
            if b is not None:
                return a + [b]
            return a


        class State(TypedDict):
            x: Annotated[list, reducer]


        class Context(TypedDict):
            r: float


        graph = StateGraph(state_schema=State, context_schema=Context)


        def node(state: State, runtime: Runtime[Context]) -> dict:
            r = runtime.context.get("r", 1.0)
            x = state["x"][-1]
            next_value = x * r * (1 - x)
            return {"x": next_value}


        graph.add_node("A", node)
        graph.set_entry_point("A")
        graph.set_finish_point("A")
        compiled = graph.compile()

        step1 = compiled.invoke({"x": 0.5}, context={"r": 3.0})
        # {'x': [0.5, 0.75]}
        ```
    """

    edges: set[tuple[str, str]]
    nodes: dict[str, StateNodeSpec[Any, ContextT]]
    branches: defaultdict[str, dict[str, BranchSpec]]
    channels: dict[str, BaseChannel]
    managed: dict[str, ManagedValueSpec]
    schemas: dict[type[Any], dict[str, BaseChannel | ManagedValueSpec]]
    waiting_edges: set[tuple[tuple[str, ...], str]]

    compiled: bool
    state_schema: type[StateT]
    context_schema: type[ContextT] | None
    input_schema: type[InputT]
    output_schema: type[OutputT]

    def __init__(
        self,
        state_schema: type[StateT],
        context_schema: type[ContextT] | None = None,
        *,
        input_schema: type[InputT] | None = None,
        output_schema: type[OutputT] | None = None,
        **kwargs: Unpack[DeprecatedKwargs],
    ) -> None:
        if (config_schema := kwargs.get("config_schema", MISSING)) is not MISSING:
            warnings.warn(
                "`config_schema` is deprecated and will be removed. Please use `context_schema` instead.",
                category=LangGraphDeprecatedSinceV10,
                stacklevel=2,
            )
            if context_schema is None:
                context_schema = cast(type[ContextT], config_schema)

        if (input_ := kwargs.get("input", MISSING)) is not MISSING:
            warnings.warn(
                "`input` is deprecated and will be removed. Please use `input_schema` instead.",
                category=LangGraphDeprecatedSinceV05,
                stacklevel=2,
            )
            if input_schema is None:
                input_schema = cast(type[InputT], input_)

        if (output := kwargs.get("output", MISSING)) is not MISSING:
            warnings.warn(
                "`output` is deprecated and will be removed. Please use `output_schema` instead.",
                category=LangGraphDeprecatedSinceV05,
                stacklevel=2,
            )
            if output_schema is None:
                output_schema = cast(type[OutputT], output)

        self.nodes = {}
        self.edges = set()
        self.branches = defaultdict(dict)
        self.schemas = {}
        self.channels = {}
        self.managed = {}
        self.compiled = False
        self.waiting_edges = set()

        self.state_schema = state_schema
        self.input_schema = cast(type[InputT], input_schema or state_schema)
        self.output_schema = cast(type[OutputT], output_schema or state_schema)
        self.context_schema = context_schema

        self._add_schema(self.state_schema)
        self._add_schema(self.input_schema, allow_managed=False)
        self._add_schema(self.output_schema, allow_managed=False)

    @property
    def _all_edges(self) -> set[tuple[str, str]]:
        return self.edges | {
            (start, end) for starts, end in self.waiting_edges for start in starts
        }

    def _add_schema(self, schema: type[Any], /, allow_managed: bool = True) -> None:
        if schema not in self.schemas:
            _warn_invalid_state_schema(schema)
            channels, managed, type_hints = _get_channels(schema)
            if managed and not allow_managed:
                names = ", ".join(managed)
                schema_name = getattr(schema, "__name__", "")
                raise ValueError(
                    f"Invalid managed channels detected in {schema_name}: {names}."
                    " Managed channels are not permitted in Input/Output schema."
                )
            self.schemas[schema] = {**channels, **managed}
            for key, channel in channels.items():
                if key in self.channels:
                    if self.channels[key] != channel:
                        if isinstance(channel, LastValue):
                            pass
                        else:
                            raise ValueError(
                                f"Channel '{key}' already exists with a different type"
                            )
                else:
                    self.channels[key] = channel
            for key, managed in managed.items():
                if key in self.managed:
                    if self.managed[key] != managed:
                        raise ValueError(
                            f"Managed value '{key}' already exists with a different type"
                        )
                else:
                    self.managed[key] = managed

    @overload
    def add_node(
        self,
        node: StateNode[NodeInputT, ContextT],
        *,
        defer: bool = False,
        metadata: dict[str, Any] | None = None,
        input_schema: None = None,
        retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
        cache_policy: CachePolicy | None = None,
        destinations: dict[str, str] | tuple[str, ...] | None = None,
        **kwargs: Unpack[DeprecatedKwargs],
    ) -> Self:
        """상태 그래프에 새 노드를 추가합니다. 입력 스키마는 상태 스키마로 추론됩니다.
        함수/runnable의 이름을 노드 이름으로 사용합니다.
        """
        ...

    @overload
    def add_node(
        self,
        node: StateNode[NodeInputT, ContextT],
        *,
        defer: bool = False,
        metadata: dict[str, Any] | None = None,
        input_schema: type[NodeInputT],
        retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
        cache_policy: CachePolicy | None = None,
        destinations: dict[str, str] | tuple[str, ...] | None = None,
        **kwargs: Unpack[DeprecatedKwargs],
    ) -> Self:
        """상태 그래프에 새 노드를 추가합니다. 입력 스키마가 지정됩니다.
        함수/runnable의 이름을 노드 이름으로 사용합니다.
        """
        ...

    @overload
    def add_node(
        self,
        node: str,
        action: StateNode[NodeInputT, ContextT],
        *,
        defer: bool = False,
        metadata: dict[str, Any] | None = None,
        input_schema: None = None,
        retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
        cache_policy: CachePolicy | None = None,
        destinations: dict[str, str] | tuple[str, ...] | None = None,
        **kwargs: Unpack[DeprecatedKwargs],
    ) -> Self:
        """상태 그래프에 새 노드를 추가합니다. 입력 스키마는 상태 스키마로 추론됩니다."""
        ...

    @overload
    def add_node(
        self,
        node: str | StateNode[NodeInputT, ContextT],
        action: StateNode[NodeInputT, ContextT] | None = None,
        *,
        defer: bool = False,
        metadata: dict[str, Any] | None = None,
        input_schema: type[NodeInputT],
        retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
        cache_policy: CachePolicy | None = None,
        destinations: dict[str, str] | tuple[str, ...] | None = None,
        **kwargs: Unpack[DeprecatedKwargs],
    ) -> Self:
        """상태 그래프에 새 노드를 추가합니다. 입력 스키마가 지정됩니다."""
        ...

    def add_node(
        self,
        node: str | StateNode[NodeInputT, ContextT],
        action: StateNode[NodeInputT, ContextT] | None = None,
        *,
        defer: bool = False,
        metadata: dict[str, Any] | None = None,
        input_schema: type[NodeInputT] | None = None,
        retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
        cache_policy: CachePolicy | None = None,
        destinations: dict[str, str] | tuple[str, ...] | None = None,
        **kwargs: Unpack[DeprecatedKwargs],
    ) -> Self:
        """상태 그래프에 새 노드를 추가합니다.

        Args:
            node: 이 노드가 실행할 함수 또는 runnable입니다.
                문자열이 제공되면 노드 이름으로 사용되며, action이 함수 또는 runnable로 사용됩니다.
            action: 노드와 연결된 액션입니다.
                `node`가 문자열(노드 이름)인 경우 노드 함수 또는 runnable로 사용됩니다.
            defer: 실행이 끝나기 직전까지 노드 실행을 지연할지 여부입니다.
            metadata: 노드와 연결된 메타데이터입니다.
            input_schema: 노드의 입력 스키마입니다. (기본값: 그래프의 상태 스키마)
            retry_policy: 노드의 재시도 정책입니다.
                시퀀스가 제공되면 첫 번째로 일치하는 정책이 적용됩니다.
            cache_policy: 노드의 캐시 정책입니다.
            destinations: 노드가 라우팅할 수 있는 목적지를 나타냅니다.
                `Command` 객체를 반환하는 노드가 있는 엣지 없는 그래프에 유용합니다.
                `dict`가 제공되면 키는 대상 노드 이름으로 사용되고 값은 엣지의 레이블로 사용됩니다.
                `tuple`이 제공되면 값이 대상 노드 이름으로 사용됩니다.

                !!! note

                    이것은 그래프 렌더링에만 사용되며 그래프 실행에는 영향을 미치지 않습니다.

        Example:
            ```python
            from typing_extensions import TypedDict

            from langchain_core.runnables import RunnableConfig
            from langgraph.graph import START, StateGraph


            class State(TypedDict):
                x: int


            def my_node(state: State, config: RunnableConfig) -> State:
                return {"x": state["x"] + 1}


            builder = StateGraph(State)
            builder.add_node(my_node)  # node name will be 'my_node'
            builder.add_edge(START, "my_node")
            graph = builder.compile()
            graph.invoke({"x": 1})
            # {'x': 2}
            ```

        Example: Customize the name:
            ```python
            builder = StateGraph(State)
            builder.add_node("my_fair_node", my_node)
            builder.add_edge(START, "my_fair_node")
            graph = builder.compile()
            graph.invoke({"x": 1})
            # {'x': 2}
            ```

        Returns:
            Self: 메서드 체이닝을 허용하는 상태 그래프의 인스턴스입니다.
        """
        if (retry := kwargs.get("retry", MISSING)) is not MISSING:
            warnings.warn(
                "`retry` is deprecated and will be removed. Please use `retry_policy` instead.",
                category=LangGraphDeprecatedSinceV05,
            )
            if retry_policy is None:
                retry_policy = retry  # type: ignore[assignment]

        if (input_ := kwargs.get("input", MISSING)) is not MISSING:
            warnings.warn(
                "`input` is deprecated and will be removed. Please use `input_schema` instead.",
                category=LangGraphDeprecatedSinceV05,
            )
            if input_schema is None:
                input_schema = cast(type[NodeInputT] | None, input_)

        if not isinstance(node, str):
            action = node
            if isinstance(action, Runnable):
                node = action.get_name()
            else:
                node = getattr(action, "__name__", action.__class__.__name__)
            if node is None:
                raise ValueError(
                    "Node name must be provided if action is not a function"
                )
        if self.compiled:
            logger.warning(
                "Adding a node to a graph that has already been compiled. This will "
                "not be reflected in the compiled graph."
            )
        if not isinstance(node, str):
            action = node
            node = cast(str, getattr(action, "name", getattr(action, "__name__", None)))
            if node is None:
                raise ValueError(
                    "Node name must be provided if action is not a function"
                )
        if action is None:
            raise RuntimeError
        if node in self.nodes:
            raise ValueError(f"Node `{node}` already present.")
        if node == END or node == START:
            raise ValueError(f"Node `{node}` is reserved.")

        for character in (NS_SEP, NS_END):
            if character in node:
                raise ValueError(
                    f"'{character}' is a reserved character and is not allowed in the node names."
                )

        inferred_input_schema = None

        ends: tuple[str, ...] | dict[str, str] = EMPTY_SEQ
        try:
            if (
                isfunction(action)
                or ismethod(action)
                or ismethod(getattr(action, "__call__", None))
            ) and (
                hints := get_type_hints(getattr(action, "__call__"))
                or get_type_hints(action)
            ):
                if input_schema is None:
                    first_parameter_name = next(
                        iter(
                            inspect.signature(
                                cast(FunctionType, action)
                            ).parameters.keys()
                        )
                    )
                    if input_hint := hints.get(first_parameter_name):
                        if isinstance(input_hint, type) and get_type_hints(input_hint):
                            inferred_input_schema = input_hint
                if rtn := hints.get("return"):
                    # Handle Union types
                    rtn_origin = get_origin(rtn)
                    if rtn_origin is Union:
                        rtn_args = get_args(rtn)
                        # Look for Command in the union
                        for arg in rtn_args:
                            arg_origin = get_origin(arg)
                            if arg_origin is Command:
                                rtn = arg
                                rtn_origin = arg_origin
                                break

                    # Check if it's a Command type
                    if (
                        rtn_origin is Command
                        and (rargs := get_args(rtn))
                        and get_origin(rargs[0]) is Literal
                        and (vals := get_args(rargs[0]))
                    ):
                        ends = vals
        except (NameError, TypeError, StopIteration):
            pass

        if destinations is not None:
            ends = destinations

        if input_schema is not None:
            self.nodes[node] = StateNodeSpec[NodeInputT, ContextT](
                coerce_to_runnable(action, name=node, trace=False),  # type: ignore[arg-type]
                metadata,
                input_schema=input_schema,
                retry_policy=retry_policy,
                cache_policy=cache_policy,
                ends=ends,
                defer=defer,
            )
        elif inferred_input_schema is not None:
            self.nodes[node] = StateNodeSpec(
                coerce_to_runnable(action, name=node, trace=False),  # type: ignore[arg-type]
                metadata,
                input_schema=inferred_input_schema,
                retry_policy=retry_policy,
                cache_policy=cache_policy,
                ends=ends,
                defer=defer,
            )
        else:
            self.nodes[node] = StateNodeSpec[StateT, ContextT](
                coerce_to_runnable(action, name=node, trace=False),  # type: ignore[arg-type]
                metadata,
                input_schema=self.state_schema,
                retry_policy=retry_policy,
                cache_policy=cache_policy,
                ends=ends,
                defer=defer,
            )

        input_schema = input_schema or inferred_input_schema
        if input_schema is not None:
            self._add_schema(input_schema)

        return self

    def add_edge(self, start_key: str | list[str], end_key: str) -> Self:
        """시작 노드(또는 시작 노드 목록)에서 종료 노드로의 방향성 엣지를 추가합니다.

        단일 시작 노드가 제공되면 그래프는 해당 노드가 완료될 때까지 기다린 후
        종료 노드를 실행합니다. 여러 시작 노드가 제공되면
        그래프는 모든 시작 노드가 완료될 때까지 기다린 후 종료 노드를 실행합니다.

        Args:
            start_key: 엣지의 시작 노드 키입니다.
            end_key: 엣지의 종료 노드 키입니다.

        Raises:
            ValueError: 시작 키가 `'END'`이거나 시작 키 또는 종료 키가 그래프에 없는 경우 발생합니다.

        Returns:
            Self: 메서드 체이닝을 허용하는 상태 그래프의 인스턴스입니다.
        """
        if self.compiled:
            logger.warning(
                "Adding an edge to a graph that has already been compiled. This will "
                "not be reflected in the compiled graph."
            )

        if isinstance(start_key, str):
            if start_key == END:
                raise ValueError("END cannot be a start node")
            if end_key == START:
                raise ValueError("START cannot be an end node")

            # run this validation only for non-StateGraph graphs
            if not hasattr(self, "channels") and start_key in set(
                start for start, _ in self.edges
            ):
                raise ValueError(
                    f"Already found path for node '{start_key}'.\n"
                    "For multiple edges, use StateGraph with an Annotated state key."
                )

            self.edges.add((start_key, end_key))
            return self

        for start in start_key:
            if start == END:
                raise ValueError("END cannot be a start node")
            if start not in self.nodes:
                raise ValueError(f"Need to add_node `{start}` first")
        if end_key == START:
            raise ValueError("START cannot be an end node")
        if end_key != END and end_key not in self.nodes:
            raise ValueError(f"Need to add_node `{end_key}` first")

        self.waiting_edges.add((tuple(start_key), end_key))
        return self

    def add_conditional_edges(
        self,
        source: str,
        path: Callable[..., Hashable | Sequence[Hashable]]
        | Callable[..., Awaitable[Hashable | Sequence[Hashable]]]
        | Runnable[Any, Hashable | Sequence[Hashable]],
        path_map: dict[Hashable, str] | list[str] | None = None,
    ) -> Self:
        """시작 노드에서 여러 목적지 노드로의 조건부 엣지를 추가합니다.

        Args:
            source: 시작 노드입니다. 이 조건부 엣지는 이 노드를 종료할 때 실행됩니다.
            path: 다음 노드를 결정하는 호출 가능 객체입니다.
                `path_map`을 지정하지 않으면 하나 이상의 노드를 반환해야 합니다.
                `'END'`를 반환하면 그래프 실행이 중지됩니다.
            path_map: 경로를 노드 이름에 매핑하는 선택적 매핑입니다.
                생략하면 `path`가 반환하는 경로가 노드 이름이어야 합니다.

        Returns:
            Self: 메서드 체이닝을 허용하는 그래프의 인스턴스입니다.

        !!! warning
            `path` 함수의 반환 값에 대한 타입 힌트(예: `-> Literal["foo", "__end__"]:`)
            또는 path_map이 없으면 그래프 시각화는 엣지가 그래프의 모든 노드로 전환될 수 있다고 가정합니다.

        """  # noqa: E501
        if self.compiled:
            logger.warning(
                "Adding an edge to a graph that has already been compiled. This will "
                "not be reflected in the compiled graph."
            )

        # find a name for the condition
        path = coerce_to_runnable(path, name=None, trace=True)
        name = path.name or "condition"
        # validate the condition
        if name in self.branches[source]:
            raise ValueError(
                f"Branch with name `{path.name}` already exists for node `{source}`"
            )
        # save it
        self.branches[source][name] = BranchSpec.from_path(path, path_map, True)
        if schema := self.branches[source][name].input_schema:
            self._add_schema(schema)
        return self

    def add_sequence(
        self,
        nodes: Sequence[
            StateNode[NodeInputT, ContextT]
            | tuple[str, StateNode[NodeInputT, ContextT]]
        ],
    ) -> Self:
        """제공된 순서대로 실행될 노드 시퀀스를 추가합니다.

        Args:
            nodes: `StateNode`(`state` 인자를 받는 호출 가능 객체) 또는 `(name, StateNode)` 튜플의 시퀀스입니다.
                이름이 제공되지 않으면 노드 객체(예: `Runnable` 또는 `Callable` 이름)에서 이름이 추론됩니다.
                각 노드는 제공된 순서대로 실행됩니다.

        Raises:
            ValueError: 시퀀스가 비어 있는 경우 발생합니다.
            ValueError: 시퀀스에 중복된 노드 이름이 포함된 경우 발생합니다.

        Returns:
            Self: 메서드 체이닝을 허용하는 상태 그래프의 인스턴스입니다.
        """
        if len(nodes) < 1:
            raise ValueError("Sequence requires at least one node.")

        previous_name: str | None = None
        for node in nodes:
            if isinstance(node, tuple) and len(node) == 2:
                name, node = node
            else:
                name = _get_node_name(node)

            if name in self.nodes:
                raise ValueError(
                    f"Node names must be unique: node with the name '{name}' already exists. "
                    "If you need to use two different runnables/callables with the same name (for example, using `lambda`), please provide them as tuples (name, runnable/callable)."
                )

            self.add_node(name, node)
            if previous_name is not None:
                self.add_edge(previous_name, name)

            previous_name = name

        return self

    def set_entry_point(self, key: str) -> Self:
        """그래프에서 호출할 첫 번째 노드를 지정합니다.

        `add_edge(START, key)`를 호출하는 것과 동일합니다.

        Parameters:
            key (str): 진입점으로 설정할 노드의 키입니다.

        Returns:
            Self: 메서드 체이닝을 허용하는 그래프의 인스턴스입니다.
        """
        return self.add_edge(START, key)

    def set_conditional_entry_point(
        self,
        path: Callable[..., Hashable | Sequence[Hashable]]
        | Callable[..., Awaitable[Hashable | Sequence[Hashable]]]
        | Runnable[Any, Hashable | Sequence[Hashable]],
        path_map: dict[Hashable, str] | list[str] | None = None,
    ) -> Self:
        """그래프에 조건부 진입점을 설정합니다.

        Args:
            path: 다음 노드를 결정하는 호출 가능 객체입니다.
                `path_map`을 지정하지 않으면 하나 이상의 노드를 반환해야 합니다.
                END를 반환하면 그래프 실행이 중지됩니다.
            path_map: 경로를 노드 이름에 매핑하는 선택적 매핑입니다.
                생략하면 `path`가 반환하는 경로가 노드 이름이어야 합니다.

        Returns:
            Self: 메서드 체이닝을 허용하는 그래프의 인스턴스입니다.
        """
        return self.add_conditional_edges(START, path, path_map)

    def set_finish_point(self, key: str) -> Self:
        """노드를 그래프의 종료 지점으로 표시합니다.

        그래프가 이 노드에 도달하면 실행이 중지됩니다.

        Parameters:
            key (str): 종료 지점으로 설정할 노드의 키입니다.

        Returns:
            Self: 메서드 체이닝을 허용하는 그래프의 인스턴스입니다.
        """
        return self.add_edge(key, END)

    def validate(self, interrupt: Sequence[str] | None = None) -> Self:
        # assemble sources
        all_sources = {src for src, _ in self._all_edges}
        for start, branches in self.branches.items():
            all_sources.add(start)
        for name, spec in self.nodes.items():
            if spec.ends:
                all_sources.add(name)
        # validate sources
        for source in all_sources:
            if source not in self.nodes and source != START:
                raise ValueError(f"Found edge starting at unknown node '{source}'")

        if START not in all_sources:
            raise ValueError(
                "Graph must have an entrypoint: add at least one edge from START to another node"
            )

        # assemble targets
        all_targets = {end for _, end in self._all_edges}
        for start, branches in self.branches.items():
            for cond, branch in branches.items():
                if branch.ends is not None:
                    for end in branch.ends.values():
                        if end not in self.nodes and end != END:
                            raise ValueError(
                                f"At '{start}' node, '{cond}' branch found unknown target '{end}'"
                            )
                        all_targets.add(end)
                else:
                    all_targets.add(END)
                    for node in self.nodes:
                        if node != start:
                            all_targets.add(node)
        for name, spec in self.nodes.items():
            if spec.ends:
                all_targets.update(spec.ends)
        for target in all_targets:
            if target not in self.nodes and target != END:
                raise ValueError(f"Found edge ending at unknown node `{target}`")
        # validate interrupts
        if interrupt:
            for node in interrupt:
                if node not in self.nodes:
                    raise ValueError(f"Interrupt node `{node}` not found")

        self.compiled = True
        return self

    def compile(
        self,
        checkpointer: Checkpointer = None,
        *,
        cache: BaseCache | None = None,
        store: BaseStore | None = None,
        interrupt_before: All | list[str] | None = None,
        interrupt_after: All | list[str] | None = None,
        debug: bool = False,
        name: str | None = None,
    ) -> CompiledStateGraph[StateT, ContextT, InputT, OutputT]:
        """상태 그래프를 `CompiledStateGraph` 객체로 컴파일합니다.

        컴파일된 그래프는 `Runnable` 인터페이스를 구현하며 호출, 스트리밍,
        배치 처리 및 비동기 실행이 가능합니다.

        Args:
            checkpointer: 체크포인트 저장 객체 또는 플래그입니다.
                제공되면 이 `Checkpointer`는 그래프의 완전히 버전이 지정된 "단기 메모리" 역할을 하며,
                모든 지점에서 일시 중지, 재개 및 재생할 수 있습니다.
                `None`이면 서브그래프로 사용될 때 부모 그래프의 checkpointer를 상속할 수 있습니다.
                `False`이면 어떤 checkpointer도 사용하거나 상속하지 않습니다.
            interrupt_before: 이전에 중단할 노드 이름의 선택적 목록입니다.
            interrupt_after: 이후에 중단할 노드 이름의 선택적 목록입니다.
            debug: 디버그 모드를 활성화할지 여부를 나타내는 플래그입니다.
            name: 컴파일된 그래프에 사용할 이름입니다.

        Returns:
            CompiledStateGraph: 컴파일된 상태 그래프입니다.
        """
        # assign default values
        interrupt_before = interrupt_before or []
        interrupt_after = interrupt_after or []

        # validate the graph
        self.validate(
            interrupt=(
                (interrupt_before if interrupt_before != "*" else []) + interrupt_after
                if interrupt_after != "*"
                else []
            )
        )

        # prepare output channels
        output_channels = (
            "__root__"
            if len(self.schemas[self.output_schema]) == 1
            and "__root__" in self.schemas[self.output_schema]
            else [
                key
                for key, val in self.schemas[self.output_schema].items()
                if not is_managed_value(val)
            ]
        )
        stream_channels = (
            "__root__"
            if len(self.channels) == 1 and "__root__" in self.channels
            else [
                key for key, val in self.channels.items() if not is_managed_value(val)
            ]
        )

        compiled = CompiledStateGraph[StateT, ContextT, InputT, OutputT](
            builder=self,
            schema_to_mapper={},
            context_schema=self.context_schema,
            nodes={},
            channels={
                **self.channels,
                **self.managed,
                START: EphemeralValue(self.input_schema),
            },
            input_channels=START,
            stream_mode="updates",
            output_channels=output_channels,
            stream_channels=stream_channels,
            checkpointer=checkpointer,
            interrupt_before_nodes=interrupt_before,
            interrupt_after_nodes=interrupt_after,
            auto_validate=False,
            debug=debug,
            store=store,
            cache=cache,
            name=name or "LangGraph",
        )

        compiled.attach_node(START, None)
        for key, node in self.nodes.items():
            compiled.attach_node(key, node)

        for start, end in self.edges:
            compiled.attach_edge(start, end)

        for starts, end in self.waiting_edges:
            compiled.attach_edge(starts, end)

        for start, branches in self.branches.items():
            for name, branch in branches.items():
                compiled.attach_branch(start, name, branch)

        return compiled.validate()


class CompiledStateGraph(
    Pregel[StateT, ContextT, InputT, OutputT],
    Generic[StateT, ContextT, InputT, OutputT],
):
    builder: StateGraph[StateT, ContextT, InputT, OutputT]
    schema_to_mapper: dict[type[Any], Callable[[Any], Any] | None]

    def __init__(
        self,
        *,
        builder: StateGraph[StateT, ContextT, InputT, OutputT],
        schema_to_mapper: dict[type[Any], Callable[[Any], Any] | None],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.builder = builder
        self.schema_to_mapper = schema_to_mapper

    def get_input_jsonschema(
        self, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        return _get_json_schema(
            typ=self.builder.input_schema,
            schemas=self.builder.schemas,
            channels=self.builder.channels,
            name=self.get_name("Input"),
        )

    def get_output_jsonschema(
        self, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        return _get_json_schema(
            typ=self.builder.output_schema,
            schemas=self.builder.schemas,
            channels=self.builder.channels,
            name=self.get_name("Output"),
        )

    def attach_node(self, key: str, node: StateNodeSpec[Any, ContextT] | None) -> None:
        if key == START:
            output_keys = [
                k
                for k, v in self.builder.schemas[self.builder.input_schema].items()
                if not is_managed_value(v)
            ]
        else:
            output_keys = list(self.builder.channels) + [
                k for k, v in self.builder.managed.items()
            ]

        def _get_updates(
            input: None | dict | Any,
        ) -> Sequence[tuple[str, Any]] | None:
            if input is None:
                return None
            elif isinstance(input, dict):
                return [(k, v) for k, v in input.items() if k in output_keys]
            elif isinstance(input, Command):
                if input.graph == Command.PARENT:
                    return None
                return [
                    (k, v) for k, v in input._update_as_tuples() if k in output_keys
                ]
            elif (
                isinstance(input, (list, tuple))
                and input
                and any(isinstance(i, Command) for i in input)
            ):
                updates: list[tuple[str, Any]] = []
                for i in input:
                    if isinstance(i, Command):
                        if i.graph == Command.PARENT:
                            continue
                        updates.extend(
                            (k, v) for k, v in i._update_as_tuples() if k in output_keys
                        )
                    else:
                        updates.extend(_get_updates(i) or ())
                return updates
            elif (t := type(input)) and get_cached_annotated_keys(t):
                return get_update_as_tuples(input, output_keys)
            else:
                msg = create_error_message(
                    message=f"Expected dict, got {input}",
                    error_code=ErrorCode.INVALID_GRAPH_NODE_RETURN_VALUE,
                )
                raise InvalidUpdateError(msg)

        # state updaters
        write_entries: tuple[ChannelWriteEntry | ChannelWriteTupleEntry, ...] = (
            ChannelWriteTupleEntry(
                mapper=_get_root if output_keys == ["__root__"] else _get_updates
            ),
            ChannelWriteTupleEntry(
                mapper=_control_branch,
                static=_control_static(node.ends)
                if node is not None and node.ends is not None
                else None,
            ),
        )

        # add node and output channel
        if key == START:
            self.nodes[key] = PregelNode(
                tags=[TAG_HIDDEN],
                triggers=[START],
                channels=START,
                writers=[ChannelWrite(write_entries)],
            )
        elif node is not None:
            input_schema = node.input_schema if node else self.builder.state_schema
            input_channels = list(self.builder.schemas[input_schema])
            is_single_input = len(input_channels) == 1 and "__root__" in input_channels
            if input_schema in self.schema_to_mapper:
                mapper = self.schema_to_mapper[input_schema]
            else:
                mapper = _pick_mapper(input_channels, input_schema)
                self.schema_to_mapper[input_schema] = mapper

            branch_channel = _CHANNEL_BRANCH_TO.format(key)
            self.channels[branch_channel] = (
                LastValueAfterFinish(Any)
                if node.defer
                else EphemeralValue(Any, guard=False)
            )
            self.nodes[key] = PregelNode(
                triggers=[branch_channel],
                # read state keys and managed values
                channels=("__root__" if is_single_input else input_channels),
                # coerce state dict to schema class (eg. pydantic model)
                mapper=mapper,
                # publish to state keys
                writers=[ChannelWrite(write_entries)],
                metadata=node.metadata,
                retry_policy=node.retry_policy,
                cache_policy=node.cache_policy,
                bound=node.runnable,  # type: ignore[arg-type]
            )
        else:
            raise RuntimeError

    def attach_edge(self, starts: str | Sequence[str], end: str) -> None:
        if isinstance(starts, str):
            # subscribe to start channel
            if end != END:
                self.nodes[starts].writers.append(
                    ChannelWrite(
                        (ChannelWriteEntry(_CHANNEL_BRANCH_TO.format(end), None),)
                    )
                )
        elif end != END:
            channel_name = f"join:{'+'.join(starts)}:{end}"
            # register channel
            if self.builder.nodes[end].defer:
                self.channels[channel_name] = NamedBarrierValueAfterFinish(
                    str, set(starts)
                )
            else:
                self.channels[channel_name] = NamedBarrierValue(str, set(starts))
            # subscribe to channel
            self.nodes[end].triggers.append(channel_name)
            # publish to channel
            for start in starts:
                self.nodes[start].writers.append(
                    ChannelWrite((ChannelWriteEntry(channel_name, start),))
                )

    def attach_branch(
        self, start: str, name: str, branch: BranchSpec, *, with_reader: bool = True
    ) -> None:
        def get_writes(
            packets: Sequence[str | Send], static: bool = False
        ) -> Sequence[ChannelWriteEntry | Send]:
            writes = [
                (
                    ChannelWriteEntry(
                        p if p == END else _CHANNEL_BRANCH_TO.format(p), None
                    )
                    if not isinstance(p, Send)
                    else p
                )
                for p in packets
                if (True if static else p != END)
            ]
            if not writes:
                return []
            return writes

        if with_reader:
            # get schema
            schema = branch.input_schema or (
                self.builder.nodes[start].input_schema
                if start in self.builder.nodes
                else self.builder.state_schema
            )
            channels = list(self.builder.schemas[schema])
            # get mapper
            if schema in self.schema_to_mapper:
                mapper = self.schema_to_mapper[schema]
            else:
                mapper = _pick_mapper(channels, schema)
                self.schema_to_mapper[schema] = mapper
            # create reader
            reader: Callable[[RunnableConfig], Any] | None = partial(
                ChannelRead.do_read,
                select=channels[0] if channels == ["__root__"] else channels,
                fresh=True,
                # coerce state dict to schema class (eg. pydantic model)
                mapper=mapper,
            )
        else:
            reader = None

        # attach branch publisher
        self.nodes[start].writers.append(branch.run(get_writes, reader))

    def _migrate_checkpoint(self, checkpoint: Checkpoint) -> None:
        """체크포인트를 새 채널 레이아웃으로 마이그레이션합니다."""
        super()._migrate_checkpoint(checkpoint)

        values = checkpoint["channel_values"]
        versions = checkpoint["channel_versions"]
        seen = checkpoint["versions_seen"]

        # empty checkpoints do not need migration
        if not versions:
            return

        # current version
        if checkpoint["v"] >= 3:
            return

        # Migrate from start:node to branch:to:node
        for k in list(versions):
            if k.startswith("start:"):
                # confirm node is present
                node = k.split(":")[1]
                if node not in self.nodes:
                    continue
                # get next version
                new_k = f"branch:to:{node}"
                new_v = (
                    max(versions[new_k], versions.pop(k))
                    if new_k in versions
                    else versions.pop(k)
                )
                # update seen
                for ss in (seen.get(node, {}), seen.get(INTERRUPT, {})):
                    if k in ss:
                        s = ss.pop(k)
                        if new_k in ss:
                            ss[new_k] = max(s, ss[new_k])
                        else:
                            ss[new_k] = s
                # update value
                if new_k not in values and k in values:
                    values[new_k] = values.pop(k)
                # update version
                versions[new_k] = new_v

        # Migrate from branch:source:condition:node to branch:to:node
        for k in list(versions):
            if k.startswith("branch:") and k.count(":") == 3:
                # confirm node is present
                node = k.split(":")[-1]
                if node not in self.nodes:
                    continue
                # get next version
                new_k = f"branch:to:{node}"
                new_v = (
                    max(versions[new_k], versions.pop(k))
                    if new_k in versions
                    else versions.pop(k)
                )
                # update seen
                for ss in (seen.get(node, {}), seen.get(INTERRUPT, {})):
                    if k in ss:
                        s = ss.pop(k)
                        if new_k in ss:
                            ss[new_k] = max(s, ss[new_k])
                        else:
                            ss[new_k] = s
                # update value
                if new_k not in values and k in values:
                    values[new_k] = values.pop(k)
                # update version
                versions[new_k] = new_v

        if not set(self.nodes).isdisjoint(versions):
            # Migrate from "node" to "branch:to:node"
            source_to_target = defaultdict(list)
            for start, end in self.builder.edges:
                if start != START and end != END:
                    source_to_target[start].append(end)
            for k in list(versions):
                if k == START:
                    continue
                if k in self.nodes:
                    v = versions.pop(k)
                    c = values.pop(k, MISSING)
                    for end in source_to_target[k]:
                        # get next version
                        new_k = f"branch:to:{end}"
                        new_v = max(versions[new_k], v) if new_k in versions else v
                        # update seen
                        for ss in (seen.get(end, {}), seen.get(INTERRUPT, {})):
                            if k in ss:
                                s = ss.pop(k)
                                if new_k in ss:
                                    ss[new_k] = max(s, ss[new_k])
                                else:
                                    ss[new_k] = s
                        # update value
                        if new_k not in values and c is not MISSING:
                            values[new_k] = c
                        # update version
                        versions[new_k] = new_v
                    # pop interrupt seen
                    if INTERRUPT in seen:
                        seen[INTERRUPT].pop(k, MISSING)


def _pick_mapper(
    state_keys: Sequence[str], schema: type[Any]
) -> Callable[[Any], Any] | None:
    if state_keys == ["__root__"]:
        return None
    if isclass(schema) and issubclass(schema, dict):
        return None
    return partial(_coerce_state, schema)


def _coerce_state(schema: type[Any], input: dict[str, Any]) -> dict[str, Any]:
    return schema(**input)


def _control_branch(value: Any) -> Sequence[tuple[str, Any]]:
    if isinstance(value, Send):
        return ((TASKS, value),)
    commands: list[Command] = []
    if isinstance(value, Command):
        commands.append(value)
    elif isinstance(value, (list, tuple)):
        for cmd in value:
            if isinstance(cmd, Command):
                commands.append(cmd)
    rtn: list[tuple[str, Any]] = []
    for command in commands:
        if command.graph == Command.PARENT:
            raise ParentCommand(command)

        goto_targets = (
            [command.goto] if isinstance(command.goto, (Send, str)) else command.goto
        )

        for go in goto_targets:
            if isinstance(go, Send):
                rtn.append((TASKS, go))
            elif isinstance(go, str) and go != END:
                # END is a special case, it's not actually a node in a practical sense
                # but rather a special terminal node that we don't need to branch to
                rtn.append((_CHANNEL_BRANCH_TO.format(go), None))
    return rtn


def _control_static(
    ends: tuple[str, ...] | dict[str, str],
) -> Sequence[tuple[str, Any, str | None]]:
    if isinstance(ends, dict):
        return [
            (k if k == END else _CHANNEL_BRANCH_TO.format(k), None, label)
            for k, label in ends.items()
        ]
    else:
        return [
            (e if e == END else _CHANNEL_BRANCH_TO.format(e), None, None) for e in ends
        ]


def _get_root(input: Any) -> Sequence[tuple[str, Any]] | None:
    if isinstance(input, Command):
        if input.graph == Command.PARENT:
            return ()
        return input._update_as_tuples()
    elif (
        isinstance(input, (list, tuple))
        and input
        and any(isinstance(i, Command) for i in input)
    ):
        updates: list[tuple[str, Any]] = []
        for i in input:
            if isinstance(i, Command):
                if i.graph == Command.PARENT:
                    continue
                updates.extend(i._update_as_tuples())
            else:
                updates.append(("__root__", i))
        return updates
    elif input is not None:
        return [("__root__", input)]


def _get_channels(
    schema: type[dict],
) -> tuple[dict[str, BaseChannel], dict[str, ManagedValueSpec], dict[str, Any]]:
    if not hasattr(schema, "__annotations__"):
        return (
            {"__root__": _get_channel("__root__", schema, allow_managed=False)},
            {},
            {},
        )

    type_hints = get_type_hints(schema, include_extras=True)
    all_keys = {
        name: _get_channel(name, typ)
        for name, typ in type_hints.items()
        if name != "__slots__"
    }
    return (
        {k: v for k, v in all_keys.items() if isinstance(v, BaseChannel)},
        {k: v for k, v in all_keys.items() if is_managed_value(v)},
        type_hints,
    )


@overload
def _get_channel(
    name: str, annotation: Any, *, allow_managed: Literal[False]
) -> BaseChannel: ...


@overload
def _get_channel(
    name: str, annotation: Any, *, allow_managed: Literal[True] = True
) -> BaseChannel | ManagedValueSpec: ...


def _get_channel(
    name: str, annotation: Any, *, allow_managed: bool = True
) -> BaseChannel | ManagedValueSpec:
    # Strip out Required and NotRequired wrappers
    if hasattr(annotation, "__origin__") and annotation.__origin__ in (
        Required,
        NotRequired,
    ):
        annotation = annotation.__args__[0]
    if manager := _is_field_managed_value(name, annotation):
        if allow_managed:
            return manager
        else:
            raise ValueError(f"This {annotation} not allowed in this position")
    elif channel := _is_field_channel(annotation):
        channel.key = name
        return channel
    elif channel := _is_field_binop(annotation):
        channel.key = name
        return channel

    fallback: LastValue = LastValue(annotation)
    fallback.key = name
    return fallback


def _is_field_channel(typ: type[Any]) -> BaseChannel | None:
    if hasattr(typ, "__metadata__"):
        meta = typ.__metadata__
        # Search through all annotated medata to find channel annotations
        for item in meta:
            if isinstance(item, BaseChannel):
                return item
            elif isclass(item) and issubclass(item, BaseChannel):
                # ex, Annotated[int, EphemeralValue, SomeOtherAnnotation]
                # would return EphemeralValue(int)
                return item(typ.__origin__ if hasattr(typ, "__origin__") else typ)
    return None


def _is_field_binop(typ: type[Any]) -> BinaryOperatorAggregate | None:
    if hasattr(typ, "__metadata__"):
        meta = typ.__metadata__
        if len(meta) >= 1 and callable(meta[-1]):
            sig = signature(meta[-1])
            params = list(sig.parameters.values())
            if (
                sum(
                    p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                    for p in params
                )
                == 2
            ):
                return BinaryOperatorAggregate(typ, meta[-1])
            else:
                raise ValueError(
                    f"Invalid reducer signature. Expected (a, b) -> c. Got {sig}"
                )
    return None


def _is_field_managed_value(name: str, typ: type[Any]) -> ManagedValueSpec | None:
    if hasattr(typ, "__metadata__"):
        meta = typ.__metadata__
        if len(meta) >= 1:
            decoration = get_origin(meta[-1]) or meta[-1]
            if is_managed_value(decoration):
                return decoration

    # Handle Required, NotRequired, etc wrapped types by extracting the inner type
    if (
        get_origin(typ) is not None
        and (args := get_args(typ))
        and (inner_type := args[0])
    ):
        return _is_field_managed_value(name, inner_type)

    return None


def _get_json_schema(
    typ: type,
    schemas: dict,
    channels: dict,
    name: str,
) -> dict[str, Any]:
    if isclass(typ) and issubclass(typ, BaseModel):
        return typ.model_json_schema()
    elif is_typeddict(typ):
        return TypeAdapter(typ).json_schema()
    else:
        keys = list(schemas[typ].keys())
        if len(keys) == 1 and keys[0] == "__root__":
            return create_model(
                name,
                root=(channels[keys[0]].UpdateType, None),
            ).model_json_schema()
        else:
            return create_model(
                name,
                field_definitions={
                    k: (
                        channels[k].UpdateType,
                        (
                            get_field_default(
                                k,
                                channels[k].UpdateType,
                                typ,
                            )
                        ),
                    )
                    for k in schemas[typ]
                    if k in channels and isinstance(channels[k], BaseChannel)
                },
            ).model_json_schema()
