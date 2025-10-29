from __future__ import annotations

import asyncio
import concurrent
import concurrent.futures
import contextlib
import queue
import warnings
import weakref
from collections import defaultdict, deque
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import is_dataclass
from functools import partial
from inspect import isclass
from typing import (
    Any,
    Generic,
    cast,
    get_type_hints,
)
from uuid import UUID, uuid5

from langchain_core.globals import get_debug
from langchain_core.runnables import (
    RunnableSequence,
)
from langchain_core.runnables.base import Input, Output
from langchain_core.runnables.config import (
    RunnableConfig,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
)
from langchain_core.runnables.graph import Graph
from langgraph.cache.base import BaseCache
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointTuple,
)
from langgraph.store.base import BaseStore
from pydantic import BaseModel, TypeAdapter
from typing_extensions import Self, Unpack, deprecated, is_typeddict

from langgraph._internal._config import (
    ensure_config,
    merge_configs,
    patch_checkpoint_map,
    patch_config,
    patch_configurable,
    recast_checkpoint_ns,
)
from langgraph._internal._constants import (
    CACHE_NS_WRITES,
    CONF,
    CONFIG_KEY_CACHE,
    CONFIG_KEY_CHECKPOINT_ID,
    CONFIG_KEY_CHECKPOINT_NS,
    CONFIG_KEY_CHECKPOINTER,
    CONFIG_KEY_DURABILITY,
    CONFIG_KEY_NODE_FINISHED,
    CONFIG_KEY_READ,
    CONFIG_KEY_RUNNER_SUBMIT,
    CONFIG_KEY_RUNTIME,
    CONFIG_KEY_SEND,
    CONFIG_KEY_STREAM,
    CONFIG_KEY_TASK_ID,
    CONFIG_KEY_THREAD_ID,
    ERROR,
    INPUT,
    INTERRUPT,
    NS_END,
    NS_SEP,
    NULL_TASK_ID,
    PUSH,
    TASKS,
)
from langgraph._internal._pydantic import create_model
from langgraph._internal._queue import (  # type: ignore[attr-defined]
    AsyncQueue,
    SyncQueue,
)
from langgraph._internal._runnable import (
    Runnable,
    RunnableLike,
    RunnableSeq,
    coerce_to_runnable,
)
from langgraph._internal._typing import MISSING, DeprecatedKwargs
from langgraph.channels.base import BaseChannel
from langgraph.channels.topic import Topic
from langgraph.config import get_config
from langgraph.constants import END
from langgraph.errors import (
    ErrorCode,
    GraphRecursionError,
    InvalidUpdateError,
    create_error_message,
)
from langgraph.managed.base import ManagedValueSpec
from langgraph.pregel._algo import (
    PregelTaskWrites,
    _scratchpad,
    apply_writes,
    local_read,
    prepare_next_tasks,
)
from langgraph.pregel._call import identifier
from langgraph.pregel._checkpoint import (
    channels_from_checkpoint,
    copy_checkpoint,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.pregel._draw import draw_graph
from langgraph.pregel._io import map_input, read_channels
from langgraph.pregel._loop import AsyncPregelLoop, SyncPregelLoop
from langgraph.pregel._messages import StreamMessagesHandler
from langgraph.pregel._read import DEFAULT_BOUND, PregelNode
from langgraph.pregel._retry import RetryPolicy
from langgraph.pregel._runner import PregelRunner
from langgraph.pregel._utils import get_new_channel_versions
from langgraph.pregel._validate import validate_graph, validate_keys
from langgraph.pregel._write import ChannelWrite, ChannelWriteEntry
from langgraph.pregel.debug import get_bolded_text, get_colored_text, tasks_w_writes
from langgraph.pregel.protocol import PregelProtocol, StreamChunk, StreamProtocol
from langgraph.runtime import DEFAULT_RUNTIME, Runtime
from langgraph.types import (
    All,
    CachePolicy,
    Checkpointer,
    Command,
    Durability,
    Interrupt,
    Send,
    StateSnapshot,
    StateUpdate,
    StreamMode,
)
from langgraph.typing import ContextT, InputT, OutputT, StateT
from langgraph.warnings import LangGraphDeprecatedSinceV10

try:
    from langchain_core.tracers._streaming import _StreamingCallbackHandler
except ImportError:
    _StreamingCallbackHandler = None  # type: ignore

__all__ = ("NodeBuilder", "Pregel")

_WriteValue = Callable[[Input], Output] | Any


class NodeBuilder:
    __slots__ = (
        "_channels",
        "_triggers",
        "_tags",
        "_metadata",
        "_writes",
        "_bound",
        "_retry_policy",
        "_cache_policy",
    )

    _channels: str | list[str]
    _triggers: list[str]
    _tags: list[str]
    _metadata: dict[str, Any]
    _writes: list[ChannelWriteEntry]
    _bound: Runnable
    _retry_policy: list[RetryPolicy]
    _cache_policy: CachePolicy | None

    def __init__(
        self,
    ) -> None:
        self._channels = []
        self._triggers = []
        self._tags = []
        self._metadata = {}
        self._writes = []
        self._bound = DEFAULT_BOUND
        self._retry_policy = []
        self._cache_policy = None

    def subscribe_only(
        self,
        channel: str,
    ) -> Self:
        """Subscribe to a single channel."""
        if not self._channels:
            self._channels = channel
        else:
            raise ValueError(
                "Cannot subscribe to single channels when other channels are already subscribed to"
            )

        self._triggers.append(channel)

        return self

    def subscribe_to(
        self,
        *channels: str,
        read: bool = True,
    ) -> Self:
        """구독할 채널을 추가합니다. 이러한 채널 중 하나라도 업데이트되면
        노드가 호출되며, 채널 값의 dict가 입력으로 전달됩니다.

        Args:
            channels: 구독할 채널 이름
            read: `True`이면 채널이 노드의 입력에 포함됩니다.
                그렇지 않으면 입력으로 전송되지 않고 노드를 트리거합니다.

        Returns:
            체이닝을 위한 Self
        """
        if isinstance(self._channels, str):
            raise ValueError(
                "Cannot subscribe to channels when subscribed to a single channel"
            )
        if read:
            if not self._channels:
                self._channels = list(channels)
            else:
                self._channels.extend(channels)

        if isinstance(channels, str):
            self._triggers.append(channels)
        else:
            self._triggers.extend(channels)

        return self

    def read_from(
        self,
        *channels: str,
    ) -> Self:
        """구독하지 않고 읽을 지정된 채널을 추가합니다."""
        assert isinstance(self._channels, list), (
            "Cannot read additional channels when subscribed to single channels"
        )
        self._channels.extend(channels)
        return self

    def do(
        self,
        node: RunnableLike,
    ) -> Self:
        """지정된 노드를 추가합니다."""
        if self._bound is not DEFAULT_BOUND:
            self._bound = RunnableSeq(
                self._bound, coerce_to_runnable(node, name=None, trace=True)
            )
        else:
            self._bound = coerce_to_runnable(node, name=None, trace=True)
        return self

    def write_to(
        self,
        *channels: str | ChannelWriteEntry,
        **kwargs: _WriteValue,
    ) -> Self:
        """채널 쓰기를 추가합니다.

        Args:
            *channels: 쓸 채널 이름
            **kwargs: 채널 이름과 값 매핑

        Returns:
            체이닝을 위한 Self
        """
        self._writes.extend(
            ChannelWriteEntry(c) if isinstance(c, str) else c for c in channels
        )
        self._writes.extend(
            ChannelWriteEntry(k, mapper=v)
            if callable(v)
            else ChannelWriteEntry(k, value=v)
            for k, v in kwargs.items()
        )

        return self

    def meta(self, *tags: str, **metadata: Any) -> Self:
        """노드에 태그 또는 메타데이터를 추가합니다."""
        self._tags.extend(tags)
        self._metadata.update(metadata)
        return self

    def add_retry_policies(self, *policies: RetryPolicy) -> Self:
        """노드에 재시도 정책을 추가합니다."""
        self._retry_policy.extend(policies)
        return self

    def add_cache_policy(self, policy: CachePolicy) -> Self:
        """노드에 캐시 정책을 추가합니다."""
        self._cache_policy = policy
        return self

    def build(self) -> PregelNode:
        """노드를 빌드합니다."""
        return PregelNode(
            channels=self._channels,
            triggers=self._triggers,
            tags=self._tags,
            metadata=self._metadata,
            writers=[ChannelWrite(self._writes)],
            bound=self._bound,
            retry_policy=self._retry_policy,
            cache_policy=self._cache_policy,
        )


class Pregel(
    PregelProtocol[StateT, ContextT, InputT, OutputT],
    Generic[StateT, ContextT, InputT, OutputT],
):
    """Pregel은 LangGraph 애플리케이션의 런타임 동작을 관리합니다.

    ## 개요

    Pregel은 [**액터**](https://en.wikipedia.org/wiki/Actor_model)와
    **채널**을 단일 애플리케이션으로 결합합니다.
    **액터**는 채널에서 데이터를 읽고 채널에 데이터를 씁니다.
    Pregel은 **Pregel 알고리즘**/**Bulk Synchronous Parallel** 모델을 따라
    애플리케이션의 실행을 여러 단계로 구성합니다.

    각 단계는 세 가지 페이즈로 구성됩니다:

    - **계획(Plan)**: 이 단계에서 실행할 **액터**를 결정합니다. 예를 들어,
        첫 번째 단계에서는 특별한 **input** 채널을 구독하는 **액터**를 선택하고,
        이후 단계에서는 이전 단계에서 업데이트된 채널을 구독하는
        **액터**를 선택합니다.
    - **실행(Execution)**: 선택된 모든 **액터**를 병렬로 실행하며,
        모두 완료되거나, 하나가 실패하거나, 타임아웃에 도달할 때까지 실행합니다.
        이 페이즈 동안 채널 업데이트는 다음 단계까지 액터에게 보이지 않습니다.
    - **업데이트(Update)**: 이 단계에서 **액터**가 쓴 값으로
        채널을 업데이트합니다.

    실행할 **액터**가 선택되지 않거나 최대 단계 수에 도달할 때까지 반복합니다.

    ## 액터(Actors)

    **액터**는 `PregelNode`입니다.
    채널을 구독하고, 채널에서 데이터를 읽고, 채널에 데이터를 씁니다.
    Pregel 알고리즘의 **액터**로 생각할 수 있습니다.
    `PregelNodes`는 LangChain의
    Runnable 인터페이스를 구현합니다.

    ## 채널(Channels)

    채널은 액터(`PregelNodes`) 간의 통신에 사용됩니다.
    각 채널에는 값 타입, 업데이트 타입, 그리고 업데이트 함수가 있습니다 – 이 함수는
    업데이트 시퀀스를 받아서
    저장된 값을 수정합니다. 채널은 한 체인에서 다른 체인으로 데이터를 전송하거나,
    체인에서 미래 단계의 자신에게 데이터를 전송하는 데 사용할 수 있습니다. LangGraph는
    여러 가지 내장 채널을 제공합니다:

    ### 기본 채널: LastValue 및 Topic

    - `LastValue`: 기본 채널로, 채널에 전송된 마지막 값을 저장하며,
       입력 및 출력 값이나 한 단계에서 다음 단계로 데이터를 전송하는 데 유용합니다
    - `Topic`: 구성 가능한 PubSub 토픽으로, *액터* 간에 여러 값을 전송하거나
       출력을 누적하는 데 유용합니다. 값을 중복 제거하거나
       여러 단계에 걸쳐 값을 누적하도록 구성할 수 있습니다.

    ### 고급 채널: Context 및 BinaryOperatorAggregate

    - `Context`: 컨텍스트 관리자의 값을 노출하고 수명 주기를 관리합니다.
      설정 및/또는 해체가 필요한 외부 리소스에 액세스하는 데 유용합니다. 예:
      `client = Context(httpx.Client)`
    - `BinaryOperatorAggregate`: 현재 값과 채널로 전송된
       각 업데이트에 이진 연산자를 적용하여 업데이트되는
       영구 값을 저장하며, 여러 단계에 걸쳐 집계를 계산하는 데 유용합니다. 예:
      `total = BinaryOperatorAggregate(int, operator.add)`

    ## 예제

    대부분의 사용자는
    [StateGraph (Graph API)][langgraph.graph.StateGraph] 또는
    [entrypoint (Functional API)][langgraph.func.entrypoint]를 통해 Pregel과 상호 작용합니다.

    그러나 **고급** 사용 사례의 경우 Pregel을 직접 사용할 수 있습니다. Pregel을
    직접 사용해야 하는지 확실하지 않은 경우 답은 아마도 아니오일 것입니다
    - 대신 Graph API 또는 Functional API를 사용해야 합니다. 이것들은 더 높은 수준의
    인터페이스로, 내부적으로 Pregel로 컴파일됩니다.

    작동 방식을 이해하기 위한 몇 가지 예제입니다:

    Example: 단일 노드 애플리케이션
        ```python
        from langgraph.channels import EphemeralValue
        from langgraph.pregel import Pregel, NodeBuilder

        node1 = (
            NodeBuilder().subscribe_only("a")
            .do(lambda x: x + x)
            .write_to("b")
        )

        app = Pregel(
            nodes={"node1": node1},
            channels={
                "a": EphemeralValue(str),
                "b": EphemeralValue(str),
            },
            input_channels=["a"],
            output_channels=["b"],
        )

        app.invoke({"a": "foo"})
        ```

        ```con
        {'b': 'foofoo'}
        ```

    Example: 여러 노드 및 여러 출력 채널 사용하기
        ```python
        from langgraph.channels import LastValue, EphemeralValue
        from langgraph.pregel import Pregel, NodeBuilder

        node1 = (
            NodeBuilder().subscribe_only("a")
            .do(lambda x: x + x)
            .write_to("b")
        )

        node2 = (
            NodeBuilder().subscribe_to("b")
            .do(lambda x: x["b"] + x["b"])
            .write_to("c")
        )


        app = Pregel(
            nodes={"node1": node1, "node2": node2},
            channels={
                "a": EphemeralValue(str),
                "b": LastValue(str),
                "c": EphemeralValue(str),
            },
            input_channels=["a"],
            output_channels=["b", "c"],
        )

        app.invoke({"a": "foo"})
        ```

        ```con
        {'b': 'foofoo', 'c': 'foofoofoofoo'}
        ```

    Example: Topic 채널 사용하기
        ```python
        from langgraph.channels import LastValue, EphemeralValue, Topic
        from langgraph.pregel import Pregel, NodeBuilder

        node1 = (
            NodeBuilder().subscribe_only("a")
            .do(lambda x: x + x)
            .write_to("b", "c")
        )

        node2 = (
            NodeBuilder().subscribe_only("b")
            .do(lambda x: x + x)
            .write_to("c")
        )


        app = Pregel(
            nodes={"node1": node1, "node2": node2},
            channels={
                "a": EphemeralValue(str),
                "b": EphemeralValue(str),
                "c": Topic(str, accumulate=True),
            },
            input_channels=["a"],
            output_channels=["c"],
        )

        app.invoke({"a": "foo"})
        ```

        ```pycon
        {"c": ["foofoo", "foofoofoofoo"]}
        ```

    Example: BinaryOperatorAggregate 채널 사용하기
        ```python
        from langgraph.channels import EphemeralValue, BinaryOperatorAggregate
        from langgraph.pregel import Pregel, NodeBuilder


        node1 = (
            NodeBuilder().subscribe_only("a")
            .do(lambda x: x + x)
            .write_to("b", "c")
        )

        node2 = (
            NodeBuilder().subscribe_only("b")
            .do(lambda x: x + x)
            .write_to("c")
        )


        def reducer(current, update):
            if current:
                return current + " | " + update
            else:
                return update


        app = Pregel(
            nodes={"node1": node1, "node2": node2},
            channels={
                "a": EphemeralValue(str),
                "b": EphemeralValue(str),
                "c": BinaryOperatorAggregate(str, operator=reducer),
            },
            input_channels=["a"],
            output_channels=["c"],
        )

        app.invoke({"a": "foo"})
        ```

        ```con
        {'c': 'foofoo | foofoofoofoo'}
        ```

    Example: 사이클 도입하기
        이 예제는 체인이 구독하는 채널에 쓰기를 하도록 하여
        그래프에 사이클을 도입하는 방법을 보여줍니다. 실행은
        None 값이 채널에 기록될 때까지 계속됩니다.

        ```python
        from langgraph.channels import EphemeralValue
        from langgraph.pregel import Pregel, NodeBuilder, ChannelWriteEntry

        example_node = (
            NodeBuilder()
            .subscribe_only("value")
            .do(lambda x: x + x if len(x) < 10 else None)
            .write_to(ChannelWriteEntry(channel="value", skip_none=True))
        )

        app = Pregel(
            nodes={"example_node": example_node},
            channels={
                "value": EphemeralValue(str),
            },
            input_channels=["value"],
            output_channels=["value"],
        )

        app.invoke({"value": "a"})
        ```

        ```con
        {'value': 'aaaaaaaaaaaaaaaa'}
        ```
    """

    nodes: dict[str, PregelNode]

    channels: dict[str, BaseChannel | ManagedValueSpec]

    stream_mode: StreamMode = "values"
    """출력을 스트림하는 모드, 기본값은 'values'입니다."""

    stream_eager: bool = False
    """스트림 이벤트를 즉시 방출하도록 강제할지 여부, stream_mode가 "messages" 및
    "custom"인 경우 자동으로 켜집니다."""

    output_channels: str | Sequence[str]

    stream_channels: str | Sequence[str] | None = None
    """스트림할 채널, 기본값은 예약된 채널을 제외한 모든 채널입니다"""

    interrupt_after_nodes: All | Sequence[str]

    interrupt_before_nodes: All | Sequence[str]

    input_channels: str | Sequence[str]

    step_timeout: float | None = None
    """단계가 완료되기를 기다리는 최대 시간(초)입니다."""

    debug: bool
    """실행 중에 디버그 정보를 출력할지 여부입니다."""

    checkpointer: Checkpointer = None
    """그래프 상태를 저장하고 로드하는 데 사용되는 `Checkpointer`입니다."""

    store: BaseStore | None = None
    """SharedValues에 사용할 메모리 저장소입니다."""

    cache: BaseCache | None = None
    """노드 결과를 저장하는 데 사용할 캐시입니다."""

    retry_policy: Sequence[RetryPolicy] = ()
    """작업 실행 시 사용할 재시도 정책입니다. 빈 세트는 재시도를 비활성화합니다."""

    cache_policy: CachePolicy | None = None
    """모든 노드에 사용할 캐시 정책입니다. 개별 노드에서 재정의할 수 있습니다."""

    context_schema: type[ContextT] | None = None
    """워크플로에 전달될 컨텍스트 객체의 스키마를 지정합니다."""

    config: RunnableConfig | None = None

    name: str = "LangGraph"

    trigger_to_nodes: Mapping[str, Sequence[str]]

    def __init__(
        self,
        *,
        nodes: dict[str, PregelNode | NodeBuilder],
        channels: dict[str, BaseChannel | ManagedValueSpec] | None,
        auto_validate: bool = True,
        stream_mode: StreamMode = "values",
        stream_eager: bool = False,
        output_channels: str | Sequence[str],
        stream_channels: str | Sequence[str] | None = None,
        interrupt_after_nodes: All | Sequence[str] = (),
        interrupt_before_nodes: All | Sequence[str] = (),
        input_channels: str | Sequence[str],
        step_timeout: float | None = None,
        debug: bool | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        store: BaseStore | None = None,
        cache: BaseCache | None = None,
        retry_policy: RetryPolicy | Sequence[RetryPolicy] = (),
        cache_policy: CachePolicy | None = None,
        context_schema: type[ContextT] | None = None,
        config: RunnableConfig | None = None,
        trigger_to_nodes: Mapping[str, Sequence[str]] | None = None,
        name: str = "LangGraph",
        **deprecated_kwargs: Unpack[DeprecatedKwargs],
    ) -> None:
        if (
            config_type := deprecated_kwargs.get("config_type", MISSING)
        ) is not MISSING:
            warnings.warn(
                "`config_type` is deprecated and will be removed. Please use `context_schema` instead.",
                category=LangGraphDeprecatedSinceV10,
                stacklevel=2,
            )

            if context_schema is None:
                context_schema = cast(type[ContextT], config_type)

        self.nodes = {
            k: v.build() if isinstance(v, NodeBuilder) else v for k, v in nodes.items()
        }
        self.channels = channels or {}
        if TASKS in self.channels and not isinstance(self.channels[TASKS], Topic):
            raise ValueError(
                f"Channel '{TASKS}' is reserved and cannot be used in the graph."
            )
        else:
            self.channels[TASKS] = Topic(Send, accumulate=False)
        self.stream_mode = stream_mode
        self.stream_eager = stream_eager
        self.output_channels = output_channels
        self.stream_channels = stream_channels
        self.interrupt_after_nodes = interrupt_after_nodes
        self.interrupt_before_nodes = interrupt_before_nodes
        self.input_channels = input_channels
        self.step_timeout = step_timeout
        self.debug = debug if debug is not None else get_debug()
        self.checkpointer = checkpointer
        self.store = store
        self.cache = cache
        self.retry_policy = (
            (retry_policy,) if isinstance(retry_policy, RetryPolicy) else retry_policy
        )
        self.cache_policy = cache_policy
        self.context_schema = context_schema
        self.config = config
        self.trigger_to_nodes = trigger_to_nodes or {}
        self.name = name
        if auto_validate:
            self.validate()

    def get_graph(
        self, config: RunnableConfig | None = None, *, xray: int | bool = False
    ) -> Graph:
        """계산 그래프의 그릴 수 있는 표현을 반환합니다."""
        # 서브그래프 수집
        if xray:
            subgraphs = {
                k: v.get_graph(
                    config,
                    xray=xray if isinstance(xray, bool) or xray <= 0 else xray - 1,
                )
                for k, v in self.get_subgraphs()
            }
        else:
            subgraphs = {}

        return draw_graph(
            merge_configs(self.config, config),
            nodes=self.nodes,
            specs=self.channels,
            input_channels=self.input_channels,
            interrupt_after_nodes=self.interrupt_after_nodes,
            interrupt_before_nodes=self.interrupt_before_nodes,
            trigger_to_nodes=self.trigger_to_nodes,
            checkpointer=self.checkpointer,
            subgraphs=subgraphs,
        )

    async def aget_graph(
        self, config: RunnableConfig | None = None, *, xray: int | bool = False
    ) -> Graph:
        """계산 그래프의 그릴 수 있는 표현을 반환합니다."""

        # 서브그래프 수집
        if xray:
            subpregels: dict[str, PregelProtocol] = {
                k: v async for k, v in self.aget_subgraphs()
            }
            subgraphs = {
                k: v
                for k, v in zip(
                    subpregels,
                    await asyncio.gather(
                        *(
                            p.aget_graph(
                                config,
                                xray=xray
                                if isinstance(xray, bool) or xray <= 0
                                else xray - 1,
                            )
                            for p in subpregels.values()
                        )
                    ),
                )
            }
        else:
            subgraphs = {}

        return draw_graph(
            merge_configs(self.config, config),
            nodes=self.nodes,
            specs=self.channels,
            input_channels=self.input_channels,
            interrupt_after_nodes=self.interrupt_after_nodes,
            interrupt_before_nodes=self.interrupt_before_nodes,
            trigger_to_nodes=self.trigger_to_nodes,
            checkpointer=self.checkpointer,
            subgraphs=subgraphs,
        )

    def _repr_mimebundle_(self, **kwargs: Any) -> dict[str, Any]:
        """Jupyter가 그래프를 표시하는 데 사용하는 Mime 번들"""
        return {
            "text/plain": repr(self),
            "image/png": self.get_graph().draw_mermaid_png(),
        }

    def copy(self, update: dict[str, Any] | None = None) -> Self:
        attrs = {k: v for k, v in self.__dict__.items() if k != "__orig_class__"}
        attrs.update(update or {})
        return self.__class__(**attrs)

    def with_config(self, config: RunnableConfig | None = None, **kwargs: Any) -> Self:
        """업데이트된 구성으로 Pregel 객체의 복사본을 생성합니다."""
        return self.copy(
            {"config": merge_configs(self.config, config, cast(RunnableConfig, kwargs))}
        )

    def validate(self) -> Self:
        validate_graph(
            self.nodes,
            {k: v for k, v in self.channels.items() if isinstance(v, BaseChannel)},
            {k: v for k, v in self.channels.items() if not isinstance(v, BaseChannel)},
            self.input_channels,
            self.output_channels,
            self.stream_channels,
            self.interrupt_after_nodes,
            self.interrupt_before_nodes,
        )
        self.trigger_to_nodes = _trigger_to_nodes(self.nodes)
        return self

    @deprecated(
        "`config_schema` is deprecated. Use `get_context_jsonschema` for the relevant schema instead.",
        category=None,
    )
    def config_schema(self, *, include: Sequence[str] | None = None) -> type[BaseModel]:
        warnings.warn(
            "`config_schema` is deprecated. Use `get_context_jsonschema` for the relevant schema instead.",
            category=LangGraphDeprecatedSinceV10,
            stacklevel=2,
        )

        include = include or []
        fields = {
            **(
                {"configurable": (self.context_schema, None)}
                if self.context_schema
                else {}
            ),
            **{
                field_name: (field_type, None)
                for field_name, field_type in get_type_hints(RunnableConfig).items()
                if field_name in [i for i in include if i != "configurable"]
            },
        }
        return create_model(self.get_name("Config"), field_definitions=fields)

    @deprecated(
        "`get_config_jsonschema` is deprecated. Use `get_context_jsonschema` instead.",
        category=None,
    )
    def get_config_jsonschema(
        self, *, include: Sequence[str] | None = None
    ) -> dict[str, Any]:
        warnings.warn(
            "`get_config_jsonschema` is deprecated. Use `get_context_jsonschema` instead.",
            category=LangGraphDeprecatedSinceV10,
            stacklevel=2,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=LangGraphDeprecatedSinceV10)
            schema = self.config_schema(include=include)
        return schema.model_json_schema()

    def get_context_jsonschema(self) -> dict[str, Any] | None:
        if (context_schema := self.context_schema) is None:
            return None

        if isclass(context_schema) and issubclass(context_schema, BaseModel):
            return context_schema.model_json_schema()
        elif is_typeddict(context_schema) or is_dataclass(context_schema):
            return TypeAdapter(context_schema).json_schema()
        else:
            raise ValueError(
                f"Invalid context schema type: {context_schema}. Must be a BaseModel, TypedDict or dataclass."
            )

    @property
    def InputType(self) -> Any:
        if isinstance(self.input_channels, str):
            channel = self.channels[self.input_channels]
            if isinstance(channel, BaseChannel):
                return channel.UpdateType

    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        config = merge_configs(self.config, config)
        if isinstance(self.input_channels, str):
            return super().get_input_schema(config)
        else:
            return create_model(
                self.get_name("Input"),
                field_definitions={
                    k: (c.UpdateType, None)
                    for k in self.input_channels or self.channels.keys()
                    if (c := self.channels[k]) and isinstance(c, BaseChannel)
                },
            )

    def get_input_jsonschema(
        self, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        schema = self.get_input_schema(config)
        return schema.model_json_schema()

    @property
    def OutputType(self) -> Any:
        if isinstance(self.output_channels, str):
            channel = self.channels[self.output_channels]
            if isinstance(channel, BaseChannel):
                return channel.ValueType

    def get_output_schema(
        self, config: RunnableConfig | None = None
    ) -> type[BaseModel]:
        config = merge_configs(self.config, config)
        if isinstance(self.output_channels, str):
            return super().get_output_schema(config)
        else:
            return create_model(
                self.get_name("Output"),
                field_definitions={
                    k: (c.ValueType, None)
                    for k in self.output_channels
                    if (c := self.channels[k]) and isinstance(c, BaseChannel)
                },
            )

    def get_output_jsonschema(
        self, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        schema = self.get_output_schema(config)
        return schema.model_json_schema()

    @property
    def stream_channels_list(self) -> Sequence[str]:
        stream_channels = self.stream_channels_asis
        return (
            [stream_channels] if isinstance(stream_channels, str) else stream_channels
        )

    @property
    def stream_channels_asis(self) -> str | Sequence[str]:
        return self.stream_channels or [
            k for k in self.channels if isinstance(self.channels[k], BaseChannel)
        ]

    def get_subgraphs(
        self, *, namespace: str | None = None, recurse: bool = False
    ) -> Iterator[tuple[str, PregelProtocol]]:
        """그래프의 서브그래프를 가져옵니다.

        Args:
            namespace: 서브그래프를 필터링할 네임스페이스입니다.
            recurse: 서브그래프로 재귀할지 여부입니다.
                `False`인 경우 직접적인 서브그래프만 반환됩니다.

        Returns:
            `(namespace, subgraph)` 쌍의 이터레이터입니다.
        """
        for name, node in self.nodes.items():
            # 접두사로 필터링
            if namespace is not None:
                if not namespace.startswith(name):
                    continue

            # 서브그래프 찾기(있는 경우)
            graph = node.subgraphs[0] if node.subgraphs else None

            # 찾았으면 재귀적으로 yield
            if graph:
                if name == namespace:
                    yield name, graph
                    return  # 찾았으므로 검색 중지
                if namespace is None:
                    yield name, graph
                if recurse and isinstance(graph, Pregel):
                    if namespace is not None:
                        namespace = namespace[len(name) + 1 :]
                    yield from (
                        (f"{name}{NS_SEP}{n}", s)
                        for n, s in graph.get_subgraphs(
                            namespace=namespace, recurse=recurse
                        )
                    )

    async def aget_subgraphs(
        self, *, namespace: str | None = None, recurse: bool = False
    ) -> AsyncIterator[tuple[str, PregelProtocol]]:
        """그래프의 서브그래프를 가져옵니다.

        Args:
            namespace: 서브그래프를 필터링할 네임스페이스입니다.
            recurse: 서브그래프로 재귀할지 여부입니다.
                `False`인 경우 직접적인 서브그래프만 반환됩니다.

        Returns:
            `(namespace, subgraph)` 쌍의 이터레이터입니다.
        """
        for name, node in self.get_subgraphs(namespace=namespace, recurse=recurse):
            yield name, node

    def _migrate_checkpoint(self, checkpoint: Checkpoint) -> None:
        """저장된 체크포인트를 새 채널 레이아웃으로 마이그레이션합니다."""
        if checkpoint["v"] < 4 and checkpoint.get("pending_sends"):
            pending_sends: list[Send] = checkpoint.pop("pending_sends")
            checkpoint["channel_values"][TASKS] = pending_sends
            checkpoint["channel_versions"][TASKS] = max(
                checkpoint["channel_versions"].values()
            )

    def _prepare_state_snapshot(
        self,
        config: RunnableConfig,
        saved: CheckpointTuple | None,
        recurse: BaseCheckpointSaver | None = None,
        apply_pending_writes: bool = False,
    ) -> StateSnapshot:
        if not saved:
            return StateSnapshot(
                values={},
                next=(),
                config=config,
                metadata=None,
                created_at=None,
                parent_config=None,
                tasks=(),
                interrupts=(),
            )

        # 필요한 경우 체크포인트 마이그레이션
        self._migrate_checkpoint(saved.checkpoint)

        step = saved.metadata.get("step", -1) + 1
        stop = step + 2
        channels, managed = channels_from_checkpoint(
            self.channels,
            saved.checkpoint,
        )
        # 이 체크포인트에 대한 작업들
        next_tasks = prepare_next_tasks(
            saved.checkpoint,
            saved.pending_writes or [],
            self.nodes,
            channels,
            managed,
            saved.config,
            step,
            stop,
            for_execution=True,
            store=self.store,
            checkpointer=(
                self.checkpointer
                if isinstance(self.checkpointer, BaseCheckpointSaver)
                else None
            ),
            manager=None,
        )
        # 서브그래프 가져오기
        subgraphs = dict(self.get_subgraphs())
        parent_ns = saved.config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
        task_states: dict[str, RunnableConfig | StateSnapshot] = {}
        for task in next_tasks.values():
            if task.name not in subgraphs:
                continue
            # 이 작업에 대한 checkpoint_ns 조립
            task_ns = f"{task.name}{NS_END}{task.id}"
            if parent_ns:
                task_ns = f"{parent_ns}{NS_SEP}{task_ns}"
            if not recurse:
                # 서브그래프 체크포인트가 존재한다는 신호로 config 설정
                config = {
                    CONF: {
                        "thread_id": saved.config[CONF]["thread_id"],
                        CONFIG_KEY_CHECKPOINT_NS: task_ns,
                    }
                }
                task_states[task.id] = config
            else:
                # 서브그래프의 상태 가져오기
                config = {
                    CONF: {
                        CONFIG_KEY_CHECKPOINTER: recurse,
                        "thread_id": saved.config[CONF]["thread_id"],
                        CONFIG_KEY_CHECKPOINT_NS: task_ns,
                    }
                }
                task_states[task.id] = subgraphs[task.name].get_state(
                    config, subgraphs=True
                )
        # 대기 중인 쓰기 적용
        if null_writes := [
            w[1:] for w in saved.pending_writes or [] if w[0] == NULL_TASK_ID
        ]:
            apply_writes(
                saved.checkpoint,
                channels,
                [PregelTaskWrites((), INPUT, null_writes, [])],
                None,
                self.trigger_to_nodes,
            )
        if apply_pending_writes and saved.pending_writes:
            for tid, k, v in saved.pending_writes:
                if k in (ERROR, INTERRUPT):
                    continue
                if tid not in next_tasks:
                    continue
                next_tasks[tid].writes.append((k, v))
            if tasks := [t for t in next_tasks.values() if t.writes]:
                apply_writes(
                    saved.checkpoint, channels, tasks, None, self.trigger_to_nodes
                )
        tasks_with_writes = tasks_w_writes(
            next_tasks.values(),
            saved.pending_writes,
            task_states,
            self.stream_channels_asis,
        )
        # 상태 스냅샷 조립
        return StateSnapshot(
            read_channels(channels, self.stream_channels_asis),
            tuple(t.name for t in next_tasks.values() if not t.writes),
            patch_checkpoint_map(saved.config, saved.metadata),
            saved.metadata,
            saved.checkpoint["ts"],
            patch_checkpoint_map(saved.parent_config, saved.metadata),
            tasks_with_writes,
            tuple([i for task in tasks_with_writes for i in task.interrupts]),
        )

    async def _aprepare_state_snapshot(
        self,
        config: RunnableConfig,
        saved: CheckpointTuple | None,
        recurse: BaseCheckpointSaver | None = None,
        apply_pending_writes: bool = False,
    ) -> StateSnapshot:
        if not saved:
            return StateSnapshot(
                values={},
                next=(),
                config=config,
                metadata=None,
                created_at=None,
                parent_config=None,
                tasks=(),
                interrupts=(),
            )

        # 필요한 경우 체크포인트 마이그레이션
        self._migrate_checkpoint(saved.checkpoint)

        step = saved.metadata.get("step", -1) + 1
        stop = step + 2
        channels, managed = channels_from_checkpoint(
            self.channels,
            saved.checkpoint,
        )
        # 이 체크포인트에 대한 작업들
        next_tasks = prepare_next_tasks(
            saved.checkpoint,
            saved.pending_writes or [],
            self.nodes,
            channels,
            managed,
            saved.config,
            step,
            stop,
            for_execution=True,
            store=self.store,
            checkpointer=(
                self.checkpointer
                if isinstance(self.checkpointer, BaseCheckpointSaver)
                else None
            ),
            manager=None,
        )
        # 서브그래프 가져오기
        subgraphs = {n: g async for n, g in self.aget_subgraphs()}
        parent_ns = saved.config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
        task_states: dict[str, RunnableConfig | StateSnapshot] = {}
        for task in next_tasks.values():
            if task.name not in subgraphs:
                continue
            # 이 작업에 대한 checkpoint_ns 조립
            task_ns = f"{task.name}{NS_END}{task.id}"
            if parent_ns:
                task_ns = f"{parent_ns}{NS_SEP}{task_ns}"
            if not recurse:
                # 서브그래프 체크포인트가 존재한다는 신호로 config 설정
                config = {
                    CONF: {
                        "thread_id": saved.config[CONF]["thread_id"],
                        CONFIG_KEY_CHECKPOINT_NS: task_ns,
                    }
                }
                task_states[task.id] = config
            else:
                # 서브그래프의 상태 가져오기
                config = {
                    CONF: {
                        CONFIG_KEY_CHECKPOINTER: recurse,
                        "thread_id": saved.config[CONF]["thread_id"],
                        CONFIG_KEY_CHECKPOINT_NS: task_ns,
                    }
                }
                task_states[task.id] = await subgraphs[task.name].aget_state(
                    config, subgraphs=True
                )
        # 대기 중인 쓰기 적용
        if null_writes := [
            w[1:] for w in saved.pending_writes or [] if w[0] == NULL_TASK_ID
        ]:
            apply_writes(
                saved.checkpoint,
                channels,
                [PregelTaskWrites((), INPUT, null_writes, [])],
                None,
                self.trigger_to_nodes,
            )
        if apply_pending_writes and saved.pending_writes:
            for tid, k, v in saved.pending_writes:
                if k in (ERROR, INTERRUPT):
                    continue
                if tid not in next_tasks:
                    continue
                next_tasks[tid].writes.append((k, v))
            if tasks := [t for t in next_tasks.values() if t.writes]:
                apply_writes(
                    saved.checkpoint, channels, tasks, None, self.trigger_to_nodes
                )

        tasks_with_writes = tasks_w_writes(
            next_tasks.values(),
            saved.pending_writes,
            task_states,
            self.stream_channels_asis,
        )
        # 상태 스냅샷 조립
        return StateSnapshot(
            read_channels(channels, self.stream_channels_asis),
            tuple(t.name for t in next_tasks.values() if not t.writes),
            patch_checkpoint_map(saved.config, saved.metadata),
            saved.metadata,
            saved.checkpoint["ts"],
            patch_checkpoint_map(saved.parent_config, saved.metadata),
            tasks_with_writes,
            tuple([i for task in tasks_with_writes for i in task.interrupts]),
        )

    def get_state(
        self, config: RunnableConfig, *, subgraphs: bool = False
    ) -> StateSnapshot:
        """그래프의 현재 상태를 가져옵니다."""
        checkpointer: BaseCheckpointSaver | None = ensure_config(config)[CONF].get(
            CONFIG_KEY_CHECKPOINTER, self.checkpointer
        )
        if not checkpointer:
            raise ValueError("No checkpointer set")

        if (
            checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
        ) and CONFIG_KEY_CHECKPOINTER not in config[CONF]:
            # checkpoint_ns에서 task_ids 제거
            recast = recast_checkpoint_ns(checkpoint_ns)
            # 일치하는 이름을 가진 서브그래프 찾기
            for _, pregel in self.get_subgraphs(namespace=recast, recurse=True):
                return pregel.get_state(
                    patch_configurable(config, {CONFIG_KEY_CHECKPOINTER: checkpointer}),
                    subgraphs=subgraphs,
                )
            else:
                raise ValueError(f"Subgraph {recast} not found")

        config = merge_configs(self.config, config) if self.config else config
        if self.checkpointer is True:
            ns = cast(str, config[CONF][CONFIG_KEY_CHECKPOINT_NS])
            config = merge_configs(
                config, {CONF: {CONFIG_KEY_CHECKPOINT_NS: recast_checkpoint_ns(ns)}}
            )
        thread_id = config[CONF][CONFIG_KEY_THREAD_ID]
        if not isinstance(thread_id, str):
            config[CONF][CONFIG_KEY_THREAD_ID] = str(thread_id)

        saved = checkpointer.get_tuple(config)
        return self._prepare_state_snapshot(
            config,
            saved,
            recurse=checkpointer if subgraphs else None,
            apply_pending_writes=CONFIG_KEY_CHECKPOINT_ID not in config[CONF],
        )

    async def aget_state(
        self, config: RunnableConfig, *, subgraphs: bool = False
    ) -> StateSnapshot:
        """그래프의 현재 상태를 가져옵니다."""
        checkpointer: BaseCheckpointSaver | None = ensure_config(config)[CONF].get(
            CONFIG_KEY_CHECKPOINTER, self.checkpointer
        )
        if not checkpointer:
            raise ValueError("No checkpointer set")

        if (
            checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
        ) and CONFIG_KEY_CHECKPOINTER not in config[CONF]:
            # checkpoint_ns에서 task_ids 제거
            recast = recast_checkpoint_ns(checkpoint_ns)
            # 일치하는 이름을 가진 서브그래프 찾기
            async for _, pregel in self.aget_subgraphs(namespace=recast, recurse=True):
                return await pregel.aget_state(
                    patch_configurable(config, {CONFIG_KEY_CHECKPOINTER: checkpointer}),
                    subgraphs=subgraphs,
                )
            else:
                raise ValueError(f"Subgraph {recast} not found")

        config = merge_configs(self.config, config) if self.config else config
        if self.checkpointer is True:
            ns = cast(str, config[CONF][CONFIG_KEY_CHECKPOINT_NS])
            config = merge_configs(
                config, {CONF: {CONFIG_KEY_CHECKPOINT_NS: recast_checkpoint_ns(ns)}}
            )
        thread_id = config[CONF][CONFIG_KEY_THREAD_ID]
        if not isinstance(thread_id, str):
            config[CONF][CONFIG_KEY_THREAD_ID] = str(thread_id)

        saved = await checkpointer.aget_tuple(config)
        return await self._aprepare_state_snapshot(
            config,
            saved,
            recurse=checkpointer if subgraphs else None,
            apply_pending_writes=CONFIG_KEY_CHECKPOINT_ID not in config[CONF],
        )

    def get_state_history(
        self,
        config: RunnableConfig,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[StateSnapshot]:
        """그래프 상태의 히스토리를 가져옵니다."""
        config = ensure_config(config)
        checkpointer: BaseCheckpointSaver | None = config[CONF].get(
            CONFIG_KEY_CHECKPOINTER, self.checkpointer
        )
        if not checkpointer:
            raise ValueError("No checkpointer set")

        if (
            checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
        ) and CONFIG_KEY_CHECKPOINTER not in config[CONF]:
            # checkpoint_ns에서 task_ids 제거
            recast = recast_checkpoint_ns(checkpoint_ns)
            # 일치하는 이름을 가진 서브그래프 찾기
            for _, pregel in self.get_subgraphs(namespace=recast, recurse=True):
                yield from pregel.get_state_history(
                    patch_configurable(config, {CONFIG_KEY_CHECKPOINTER: checkpointer}),
                    filter=filter,
                    before=before,
                    limit=limit,
                )
                return
            else:
                raise ValueError(f"Subgraph {recast} not found")

        config = merge_configs(
            self.config,
            config,
            {
                CONF: {
                    CONFIG_KEY_CHECKPOINT_NS: checkpoint_ns,
                    CONFIG_KEY_THREAD_ID: str(config[CONF][CONFIG_KEY_THREAD_ID]),
                }
            },
        )
        # db 커서를 잡고 있지 않도록 list()를 즉시 소비
        for checkpoint_tuple in list(
            checkpointer.list(config, before=before, limit=limit, filter=filter)
        ):
            yield self._prepare_state_snapshot(
                checkpoint_tuple.config, checkpoint_tuple
            )

    async def aget_state_history(
        self,
        config: RunnableConfig,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[StateSnapshot]:
        """비동기적으로 그래프 상태의 히스토리를 가져옵니다."""
        config = ensure_config(config)
        checkpointer: BaseCheckpointSaver | None = ensure_config(config)[CONF].get(
            CONFIG_KEY_CHECKPOINTER, self.checkpointer
        )
        if not checkpointer:
            raise ValueError("No checkpointer set")

        if (
            checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
        ) and CONFIG_KEY_CHECKPOINTER not in config[CONF]:
            # checkpoint_ns에서 task_ids 제거
            recast = recast_checkpoint_ns(checkpoint_ns)
            # 일치하는 이름을 가진 서브그래프 찾기
            async for _, pregel in self.aget_subgraphs(namespace=recast, recurse=True):
                async for state in pregel.aget_state_history(
                    patch_configurable(config, {CONFIG_KEY_CHECKPOINTER: checkpointer}),
                    filter=filter,
                    before=before,
                    limit=limit,
                ):
                    yield state
                return
            else:
                raise ValueError(f"Subgraph {recast} not found")

        config = merge_configs(
            self.config,
            config,
            {
                CONF: {
                    CONFIG_KEY_CHECKPOINT_NS: checkpoint_ns,
                    CONFIG_KEY_THREAD_ID: str(config[CONF][CONFIG_KEY_THREAD_ID]),
                }
            },
        )
        # db 커서를 잡고 있지 않도록 list()를 즉시 소비
        for checkpoint_tuple in [
            c
            async for c in checkpointer.alist(
                config, before=before, limit=limit, filter=filter
            )
        ]:
            yield await self._aprepare_state_snapshot(
                checkpoint_tuple.config, checkpoint_tuple
            )

    def bulk_update_state(
        self,
        config: RunnableConfig,
        supersteps: Sequence[Sequence[StateUpdate]],
    ) -> RunnableConfig:
        """그래프 상태에 대한 업데이트를 대량으로 적용합니다. 체크포인터가 설정되어 있어야 합니다.

        Args:
            config: 업데이트를 적용할 config입니다.
            supersteps: 각각 그래프 상태에 순차적으로 적용할 업데이트 목록을 포함하는 슈퍼스텝 목록입니다.
                        각 업데이트는 `(values, as_node, task_id)` 형식의 튜플이며, `task_id`는 선택적입니다.

        Raises:
            ValueError: 체크포인터가 설정되지 않았거나 업데이트가 제공되지 않은 경우.
            InvalidUpdateError: 유효하지 않은 업데이트가 제공된 경우.

        Returns:
            RunnableConfig: 업데이트된 config입니다.
        """

        checkpointer: BaseCheckpointSaver | None = ensure_config(config)[CONF].get(
            CONFIG_KEY_CHECKPOINTER, self.checkpointer
        )
        if not checkpointer:
            raise ValueError("No checkpointer set")

        if len(supersteps) == 0:
            raise ValueError("No supersteps provided")

        if any(len(u) == 0 for u in supersteps):
            raise ValueError("No updates provided")

        # 서브그래프에 위임
        if (
            checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
        ) and CONFIG_KEY_CHECKPOINTER not in config[CONF]:
            # checkpoint_ns에서 task_ids 제거
            recast = recast_checkpoint_ns(checkpoint_ns)
            # 일치하는 이름을 가진 서브그래프 찾기
            for _, pregel in self.get_subgraphs(namespace=recast, recurse=True):
                return pregel.bulk_update_state(
                    patch_configurable(config, {CONFIG_KEY_CHECKPOINTER: checkpointer}),
                    supersteps,
                )
            else:
                raise ValueError(f"Subgraph {recast} not found")

        def perform_superstep(
            input_config: RunnableConfig, updates: Sequence[StateUpdate]
        ) -> RunnableConfig:
            # 마지막 체크포인트 가져오기
            config = ensure_config(self.config, input_config)
            saved = checkpointer.get_tuple(config)
            if saved is not None:
                self._migrate_checkpoint(saved.checkpoint)
            checkpoint = (
                copy_checkpoint(saved.checkpoint) if saved else empty_checkpoint()
            )
            checkpoint_previous_versions = (
                saved.checkpoint["channel_versions"].copy() if saved else {}
            )
            step = saved.metadata.get("step", -1) if saved else -1
            # 이전 체크포인트 config와 구성 가능한 필드 병합
            checkpoint_config = patch_configurable(
                config,
                {
                    CONFIG_KEY_CHECKPOINT_NS: config[CONF].get(
                        CONFIG_KEY_CHECKPOINT_NS, ""
                    )
                },
            )
            if saved:
                checkpoint_config = patch_configurable(config, saved.config[CONF])
            channels, managed = channels_from_checkpoint(
                self.channels,
                checkpoint,
            )
            values, as_node = updates[0][:2]

            # END로서 값이 없으면 모든 작업을 지웁니다
            if values is None and as_node == END:
                if len(updates) > 1:
                    raise InvalidUpdateError(
                        "Cannot apply multiple updates when clearing state"
                    )

                if saved is not None:
                    # 이 체크포인트에 대한 작업들
                    next_tasks = prepare_next_tasks(
                        checkpoint,
                        saved.pending_writes or [],
                        self.nodes,
                        channels,
                        managed,
                        saved.config,
                        step + 1,
                        step + 3,
                        for_execution=True,
                        store=self.store,
                        checkpointer=checkpointer,
                        manager=None,
                    )
                    # null 쓰기 적용
                    if null_writes := [
                        w[1:]
                        for w in saved.pending_writes or []
                        if w[0] == NULL_TASK_ID
                    ]:
                        apply_writes(
                            checkpoint,
                            channels,
                            [PregelTaskWrites((), INPUT, null_writes, [])],
                            checkpointer.get_next_version,
                            self.trigger_to_nodes,
                        )
                    # 이미 실행된 작업의 쓰기 적용
                    for tid, k, v in saved.pending_writes or []:
                        if k in (ERROR, INTERRUPT):
                            continue
                        if tid not in next_tasks:
                            continue
                        next_tasks[tid].writes.append((k, v))
                    # 모든 현재 작업 지우기
                    apply_writes(
                        checkpoint,
                        channels,
                        next_tasks.values(),
                        checkpointer.get_next_version,
                        self.trigger_to_nodes,
                    )
                # 체크포인트 저장
                next_config = checkpointer.put(
                    checkpoint_config,
                    create_checkpoint(checkpoint, channels, step),
                    {
                        "source": "update",
                        "step": step + 1,
                        "parents": saved.metadata.get("parents", {}) if saved else {},
                    },
                    get_new_channel_versions(
                        checkpoint_previous_versions,
                        checkpoint["channel_versions"],
                    ),
                )
                return patch_checkpoint_map(
                    next_config, saved.metadata if saved else None
                )

            # 입력으로 동작
            if as_node == INPUT:
                if len(updates) > 1:
                    raise InvalidUpdateError(
                        "Cannot apply multiple updates when updating as input"
                    )

                if input_writes := deque(map_input(self.input_channels, values)):
                    apply_writes(
                        checkpoint,
                        channels,
                        [PregelTaskWrites((), INPUT, input_writes, [])],
                        checkpointer.get_next_version,
                        self.trigger_to_nodes,
                    )

                    # 입력 쓰기를 채널에 적용
                    next_step = (
                        step + 1
                        if saved and saved.metadata.get("step") is not None
                        else -1
                    )
                    next_config = checkpointer.put(
                        checkpoint_config,
                        create_checkpoint(checkpoint, channels, next_step),
                        {
                            "source": "input",
                            "step": next_step,
                            "parents": saved.metadata.get("parents", {})
                            if saved
                            else {},
                        },
                        get_new_channel_versions(
                            checkpoint_previous_versions,
                            checkpoint["channel_versions"],
                        ),
                    )

                    # 쓰기를 저장
                    checkpointer.put_writes(
                        next_config,
                        input_writes,
                        str(uuid5(UUID(checkpoint["id"]), INPUT)),
                    )

                    return patch_checkpoint_map(
                        next_config, saved.metadata if saved else None
                    )
                else:
                    raise InvalidUpdateError(
                        f"Received no input writes for {self.input_channels}"
                    )

            # copy checkpoint
            if as_node == "__copy__":
                if len(updates) > 1:
                    raise InvalidUpdateError(
                        "Cannot copy checkpoint with multiple updates"
                    )

                if saved is None:
                    raise InvalidUpdateError("Cannot copy a non-existent checkpoint")

                next_checkpoint = create_checkpoint(checkpoint, None, step)

                # 체크포인트 복사
                next_config = checkpointer.put(
                    saved.parent_config
                    or patch_configurable(
                        saved.config, {CONFIG_KEY_CHECKPOINT_ID: None}
                    ),
                    next_checkpoint,
                    {
                        "source": "fork",
                        "step": step + 1,
                        "parents": saved.metadata.get("parents", {}),
                    },
                    {},
                )

                # 한 번에 체크포인트를 복제하고 상태를 업데이트하려고 합니다.
                # 가능한 경우 동일한 task ID를 재사용합니다.
                if isinstance(values, list) and len(values) > 0:
                    # 다음 업데이트 체크포인트를 위한 task ID를 파악합니다
                    next_tasks = prepare_next_tasks(
                        next_checkpoint,
                        saved.pending_writes or [],
                        self.nodes,
                        channels,
                        managed,
                        next_config,
                        step + 2,
                        step + 4,
                        for_execution=True,
                        store=self.store,
                        checkpointer=checkpointer,
                        manager=None,
                    )

                    tasks_group_by = defaultdict(list)
                    user_group_by: dict[str, list[StateUpdate]] = defaultdict(list)

                    for task in next_tasks.values():
                        tasks_group_by[task.name].append(task.id)

                    for item in values:
                        if not isinstance(item, Sequence):
                            raise InvalidUpdateError(
                                f"Invalid update item: {item} when copying checkpoint"
                            )

                        values, as_node = item[:2]

                        user_group = user_group_by[as_node]
                        tasks_group = tasks_group_by[as_node]

                        target_idx = len(user_group)
                        task_id = (
                            tasks_group[target_idx]
                            if target_idx < len(tasks_group)
                            else None
                        )

                        user_group_by[as_node].append(
                            StateUpdate(values=values, as_node=as_node, task_id=task_id)
                        )

                    return perform_superstep(
                        patch_checkpoint_map(next_config, saved.metadata),
                        [item for lst in user_group_by.values() for item in lst],
                    )

                return patch_checkpoint_map(next_config, saved.metadata)

            # task id는 StateUpdate에 제공될 수 있지만, 제공되지 않으면
            # prepare_next_tasks에서 생성된 task id를 사용합니다
            node_to_task_ids: dict[str, deque[str]] = defaultdict(deque)
            if saved is not None and saved.pending_writes is not None:
                # 이 체크포인트에 대한 작업들
                next_tasks = prepare_next_tasks(
                    checkpoint,
                    saved.pending_writes,
                    self.nodes,
                    channels,
                    managed,
                    saved.config,
                    step + 1,
                    step + 3,
                    for_execution=True,
                    store=self.store,
                    checkpointer=checkpointer,
                    manager=None,
                )
                # 작업 결과를 올바르게 연결할 수 있도록 재사용할 task id를 수집합니다
                for t in next_tasks.values():
                    node_to_task_ids[t.name].append(t.id)

                # null 쓰기 적용
                if null_writes := [
                    w[1:] for w in saved.pending_writes or [] if w[0] == NULL_TASK_ID
                ]:
                    apply_writes(
                        checkpoint,
                        channels,
                        [PregelTaskWrites((), INPUT, null_writes, [])],
                        checkpointer.get_next_version,
                        self.trigger_to_nodes,
                    )
                # 쓰기 적용
                for tid, k, v in saved.pending_writes:
                    if k in (ERROR, INTERRUPT):
                        continue
                    if tid not in next_tasks:
                        continue
                    next_tasks[tid].writes.append((k, v))
                if tasks := [t for t in next_tasks.values() if t.writes]:
                    apply_writes(
                        checkpoint,
                        channels,
                        tasks,
                        checkpointer.get_next_version,
                        self.trigger_to_nodes,
                    )
            valid_updates: list[tuple[str, dict[str, Any] | None, str | None]] = []
            if len(updates) == 1:
                values, as_node, task_id = updates[0]
                # 제공되지 않은 경우 상태를 업데이트한 마지막 노드를 찾습니다
                if as_node is None and len(self.nodes) == 1:
                    as_node = tuple(self.nodes)[0]
                elif as_node is None and not any(
                    v
                    for vv in checkpoint["versions_seen"].values()
                    for v in vv.values()
                ):
                    if (
                        isinstance(self.input_channels, str)
                        and self.input_channels in self.nodes
                    ):
                        as_node = self.input_channels
                elif as_node is None:
                    last_seen_by_node = sorted(
                        (v, n)
                        for n, seen in checkpoint["versions_seen"].items()
                        if n in self.nodes
                        for v in seen.values()
                    )
                    # 두 노드가 동시에 상태를 업데이트한 경우, 모호합니다
                    if last_seen_by_node:
                        if len(last_seen_by_node) == 1:
                            as_node = last_seen_by_node[0][1]
                        elif last_seen_by_node[-1][0] != last_seen_by_node[-2][0]:
                            as_node = last_seen_by_node[-1][1]
                if as_node is None:
                    raise InvalidUpdateError("Ambiguous update, specify as_node")
                if as_node not in self.nodes:
                    raise InvalidUpdateError(f"Node {as_node} does not exist")
                valid_updates.append((as_node, values, task_id))
            else:
                for values, as_node, task_id in updates:
                    if as_node is None:
                        raise InvalidUpdateError(
                            "as_node is required when applying multiple updates"
                        )
                    if as_node not in self.nodes:
                        raise InvalidUpdateError(f"Node {as_node} does not exist")

                    valid_updates.append((as_node, values, task_id))

            run_tasks: list[PregelTaskWrites] = []
            run_task_ids: list[str] = []

            for as_node, values, provided_task_id in valid_updates:
                # 선택된 노드의 모든 writer를 실행할 작업을 생성합니다
                writers = self.nodes[as_node].flat_writers
                if not writers:
                    raise InvalidUpdateError(f"Node {as_node} has no writers")
                writes: deque[tuple[str, Any]] = deque()
                task = PregelTaskWrites((), as_node, writes, [INTERRUPT])
                # 이 노드를 위해 준비된 task id를 가져옵니다
                # StateUpdate에 task id가 제공된 경우 이를 사용합니다
                # 그렇지 않으면 다음으로 사용 가능한 task id를 사용합니다
                prepared_task_ids = node_to_task_ids.get(as_node, deque())
                task_id = provided_task_id or (
                    prepared_task_ids.popleft()
                    if prepared_task_ids
                    else str(uuid5(UUID(checkpoint["id"]), INTERRUPT))
                )
                run_tasks.append(task)
                run_task_ids.append(task_id)
                run = RunnableSequence(*writers) if len(writers) > 1 else writers[0]
                # 작업 실행
                run.invoke(
                    values,
                    patch_config(
                        config,
                        run_name=self.name + "UpdateState",
                        configurable={
                            # deque.extend는 스레드 세이프합니다
                            CONFIG_KEY_SEND: writes.extend,
                            CONFIG_KEY_TASK_ID: task_id,
                            CONFIG_KEY_READ: partial(
                                local_read,
                                _scratchpad(
                                    None,
                                    [],
                                    task_id,
                                    "",
                                    None,
                                    step,
                                    step + 2,
                                ),
                                channels,
                                managed,
                                task,
                            ),
                        },
                    ),
                )
            # 작업 쓰기 저장
            for task_id, task in zip(run_task_ids, run_tasks):
                # 채널 쓰기는 현재 체크포인트에 저장됩니다
                channel_writes = [w for w in task.writes if w[0] != PUSH]
                if saved and channel_writes:
                    checkpointer.put_writes(checkpoint_config, channel_writes, task_id)
            # 체크포인트에 적용하고 저장
            apply_writes(
                checkpoint,
                channels,
                run_tasks,
                checkpointer.get_next_version,
                self.trigger_to_nodes,
            )
            checkpoint = create_checkpoint(checkpoint, channels, step + 1)
            next_config = checkpointer.put(
                checkpoint_config,
                checkpoint,
                {
                    "source": "update",
                    "step": step + 1,
                    "parents": saved.metadata.get("parents", {}) if saved else {},
                },
                get_new_channel_versions(
                    checkpoint_previous_versions, checkpoint["channel_versions"]
                ),
            )
            for task_id, task in zip(run_task_ids, run_tasks):
                # push 쓰기 저장
                if push_writes := [w for w in task.writes if w[0] == PUSH]:
                    checkpointer.put_writes(next_config, push_writes, task_id)

            return patch_checkpoint_map(next_config, saved.metadata if saved else None)

        current_config = patch_configurable(
            config, {CONFIG_KEY_THREAD_ID: str(config[CONF][CONFIG_KEY_THREAD_ID])}
        )
        for superstep in supersteps:
            current_config = perform_superstep(current_config, superstep)
        return current_config

    async def abulk_update_state(
        self,
        config: RunnableConfig,
        supersteps: Sequence[Sequence[StateUpdate]],
    ) -> RunnableConfig:
        """그래프 상태에 대한 업데이트를 대량으로 비동기적으로 적용합니다. 체크포인터가 설정되어 있어야 합니다.

        Args:
            config: 업데이트를 적용할 config입니다.
            supersteps: 각 superstep에는 그래프 상태에 순차적으로 적용할 업데이트 목록이 포함됩니다.
                        각 업데이트는 `(values, as_node, task_id)` 형식의 튜플이며, 여기서 `task_id`는 선택 사항입니다.

        Raises:
            ValueError: 체크포인터가 설정되지 않았거나 업데이트가 제공되지 않은 경우.
            InvalidUpdateError: 잘못된 업데이트가 제공된 경우.

        Returns:
            RunnableConfig: 업데이트된 config입니다.
        """

        checkpointer: BaseCheckpointSaver | None = ensure_config(config)[CONF].get(
            CONFIG_KEY_CHECKPOINTER, self.checkpointer
        )
        if not checkpointer:
            raise ValueError("No checkpointer set")

        if len(supersteps) == 0:
            raise ValueError("No supersteps provided")

        if any(len(u) == 0 for u in supersteps):
            raise ValueError("No updates provided")

        # 서브그래프에 위임
        if (
            checkpoint_ns := config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, "")
        ) and CONFIG_KEY_CHECKPOINTER not in config[CONF]:
            # checkpoint_ns에서 task_ids 제거
            recast = recast_checkpoint_ns(checkpoint_ns)
            # 일치하는 이름을 가진 서브그래프 찾기
            async for _, pregel in self.aget_subgraphs(namespace=recast, recurse=True):
                return await pregel.abulk_update_state(
                    patch_configurable(config, {CONFIG_KEY_CHECKPOINTER: checkpointer}),
                    supersteps,
                )
            else:
                raise ValueError(f"Subgraph {recast} not found")

        async def aperform_superstep(
            input_config: RunnableConfig, updates: Sequence[StateUpdate]
        ) -> RunnableConfig:
            # 마지막 체크포인트 가져오기
            config = ensure_config(self.config, input_config)
            saved = await checkpointer.aget_tuple(config)
            if saved is not None:
                self._migrate_checkpoint(saved.checkpoint)
            checkpoint = (
                copy_checkpoint(saved.checkpoint) if saved else empty_checkpoint()
            )
            checkpoint_previous_versions = (
                saved.checkpoint["channel_versions"].copy() if saved else {}
            )
            step = saved.metadata.get("step", -1) if saved else -1
            # 이전 체크포인트 config와 구성 가능한 필드 병합
            checkpoint_config = patch_configurable(
                config,
                {
                    CONFIG_KEY_CHECKPOINT_NS: config[CONF].get(
                        CONFIG_KEY_CHECKPOINT_NS, ""
                    )
                },
            )
            if saved:
                checkpoint_config = patch_configurable(config, saved.config[CONF])
            channels, managed = channels_from_checkpoint(
                self.channels,
                checkpoint,
            )
            values, as_node = updates[0][:2]
            # 값이 없으면 모든 작업을 지웁니다
            if values is None and as_node == END:
                if len(updates) > 1:
                    raise InvalidUpdateError(
                        "Cannot apply multiple updates when clearing state"
                    )
                if saved is not None:
                    # 이 체크포인트에 대한 작업들
                    next_tasks = prepare_next_tasks(
                        checkpoint,
                        saved.pending_writes or [],
                        self.nodes,
                        channels,
                        managed,
                        saved.config,
                        step + 1,
                        step + 3,
                        for_execution=True,
                        store=self.store,
                        checkpointer=checkpointer,
                        manager=None,
                    )
                    # null 쓰기 적용
                    if null_writes := [
                        w[1:]
                        for w in saved.pending_writes or []
                        if w[0] == NULL_TASK_ID
                    ]:
                        apply_writes(
                            checkpoint,
                            channels,
                            [PregelTaskWrites((), INPUT, null_writes, [])],
                            checkpointer.get_next_version,
                            self.trigger_to_nodes,
                        )
                    # 이미 실행된 작업의 쓰기 적용
                    for tid, k, v in saved.pending_writes or []:
                        if k in (ERROR, INTERRUPT):
                            continue
                        if tid not in next_tasks:
                            continue
                        next_tasks[tid].writes.append((k, v))
                    # 모든 현재 작업 지우기
                    apply_writes(
                        checkpoint,
                        channels,
                        next_tasks.values(),
                        checkpointer.get_next_version,
                        self.trigger_to_nodes,
                    )
                # 체크포인트 저장
                next_config = await checkpointer.aput(
                    checkpoint_config,
                    create_checkpoint(checkpoint, channels, step),
                    {
                        "source": "update",
                        "step": step + 1,
                        "parents": saved.metadata.get("parents", {}) if saved else {},
                    },
                    get_new_channel_versions(
                        checkpoint_previous_versions, checkpoint["channel_versions"]
                    ),
                )
                return patch_checkpoint_map(
                    next_config, saved.metadata if saved else None
                )

            # 입력으로 동작
            if as_node == INPUT:
                if len(updates) > 1:
                    raise InvalidUpdateError(
                        "Cannot apply multiple updates when updating as input"
                    )

                if input_writes := deque(map_input(self.input_channels, values)):
                    apply_writes(
                        checkpoint,
                        channels,
                        [PregelTaskWrites((), INPUT, input_writes, [])],
                        checkpointer.get_next_version,
                        self.trigger_to_nodes,
                    )

                    # 입력 쓰기를 채널에 적용
                    next_step = (
                        step + 1
                        if saved and saved.metadata.get("step") is not None
                        else -1
                    )
                    next_config = await checkpointer.aput(
                        checkpoint_config,
                        create_checkpoint(checkpoint, channels, next_step),
                        {
                            "source": "input",
                            "step": next_step,
                            "parents": saved.metadata.get("parents", {})
                            if saved
                            else {},
                        },
                        get_new_channel_versions(
                            checkpoint_previous_versions,
                            checkpoint["channel_versions"],
                        ),
                    )

                    # 쓰기를 저장
                    await checkpointer.aput_writes(
                        next_config,
                        input_writes,
                        str(uuid5(UUID(checkpoint["id"]), INPUT)),
                    )

                    return patch_checkpoint_map(
                        next_config, saved.metadata if saved else None
                    )
                else:
                    raise InvalidUpdateError(
                        f"Received no input writes for {self.input_channels}"
                    )

            # 값이 없으면 체크포인트를 복사
            if as_node == "__copy__":
                if len(updates) > 1:
                    raise InvalidUpdateError(
                        "Cannot copy checkpoint with multiple updates"
                    )

                if saved is None:
                    raise InvalidUpdateError("Cannot copy a non-existent checkpoint")

                next_checkpoint = create_checkpoint(checkpoint, None, step)

                # 체크포인트 복사
                next_config = await checkpointer.aput(
                    saved.parent_config
                    or patch_configurable(
                        saved.config, {CONFIG_KEY_CHECKPOINT_ID: None}
                    ),
                    next_checkpoint,
                    {
                        "source": "fork",
                        "step": step + 1,
                        "parents": saved.metadata.get("parents", {}),
                    },
                    {},
                )

                # 한 번에 체크포인트를 복제하고 상태를 업데이트하려고 합니다.
                # 가능한 경우 동일한 task ID를 재사용합니다.
                if isinstance(values, list) and len(values) > 0:
                    # 다음 업데이트 체크포인트를 위한 task ID를 파악합니다
                    next_tasks = prepare_next_tasks(
                        next_checkpoint,
                        saved.pending_writes or [],
                        self.nodes,
                        channels,
                        managed,
                        next_config,
                        step + 2,
                        step + 4,
                        for_execution=True,
                        store=self.store,
                        checkpointer=checkpointer,
                        manager=None,
                    )

                    tasks_group_by = defaultdict(list)
                    user_group_by: dict[str, list[StateUpdate]] = defaultdict(list)

                    for task in next_tasks.values():
                        tasks_group_by[task.name].append(task.id)

                    for item in values:
                        if not isinstance(item, Sequence):
                            raise InvalidUpdateError(
                                f"Invalid update item: {item} when copying checkpoint"
                            )

                        values, as_node = item[:2]
                        user_group = user_group_by[as_node]
                        tasks_group = tasks_group_by[as_node]

                        target_idx = len(user_group)
                        task_id = (
                            tasks_group[target_idx]
                            if target_idx < len(tasks_group)
                            else None
                        )

                        user_group_by[as_node].append(
                            StateUpdate(values=values, as_node=as_node, task_id=task_id)
                        )

                    return await aperform_superstep(
                        patch_checkpoint_map(next_config, saved.metadata),
                        [item for lst in user_group_by.values() for item in lst],
                    )

                return patch_checkpoint_map(
                    next_config, saved.metadata if saved else None
                )

            # task id는 StateUpdate에 제공될 수 있지만, 제공되지 않으면
            # prepare_next_tasks에서 생성된 task id를 사용합니다
            node_to_task_ids: dict[str, deque[str]] = defaultdict(deque)
            if saved is not None and saved.pending_writes is not None:
                # 이 체크포인트에 대한 작업들
                next_tasks = prepare_next_tasks(
                    checkpoint,
                    saved.pending_writes,
                    self.nodes,
                    channels,
                    managed,
                    saved.config,
                    step + 1,
                    step + 3,
                    for_execution=True,
                    store=self.store,
                    checkpointer=checkpointer,
                    manager=None,
                )
                # 작업 결과를 올바르게 연결할 수 있도록 재사용할 task id를 수집합니다
                for t in next_tasks.values():
                    node_to_task_ids[t.name].append(t.id)

                # null 쓰기 적용
                if null_writes := [
                    w[1:] for w in saved.pending_writes or [] if w[0] == NULL_TASK_ID
                ]:
                    apply_writes(
                        checkpoint,
                        channels,
                        [PregelTaskWrites((), INPUT, null_writes, [])],
                        checkpointer.get_next_version,
                        self.trigger_to_nodes,
                    )
                for tid, k, v in saved.pending_writes:
                    if k in (ERROR, INTERRUPT):
                        continue
                    if tid not in next_tasks:
                        continue
                    next_tasks[tid].writes.append((k, v))
                if tasks := [t for t in next_tasks.values() if t.writes]:
                    apply_writes(
                        checkpoint,
                        channels,
                        tasks,
                        checkpointer.get_next_version,
                        self.trigger_to_nodes,
                    )
            valid_updates: list[tuple[str, dict[str, Any] | None, str | None]] = []
            if len(updates) == 1:
                values, as_node, task_id = updates[0]
                # 제공되지 않은 경우 상태를 업데이트한 마지막 노드를 찾습니다
                if as_node is None and len(self.nodes) == 1:
                    as_node = tuple(self.nodes)[0]
                elif as_node is None and not saved:
                    if (
                        isinstance(self.input_channels, str)
                        and self.input_channels in self.nodes
                    ):
                        as_node = self.input_channels
                elif as_node is None:
                    last_seen_by_node = sorted(
                        (v, n)
                        for n, seen in checkpoint["versions_seen"].items()
                        if n in self.nodes
                        for v in seen.values()
                    )
                    # 두 노드가 동시에 상태를 업데이트한 경우, 모호합니다
                    if last_seen_by_node:
                        if len(last_seen_by_node) == 1:
                            as_node = last_seen_by_node[0][1]
                        elif last_seen_by_node[-1][0] != last_seen_by_node[-2][0]:
                            as_node = last_seen_by_node[-1][1]
                if as_node is None:
                    raise InvalidUpdateError("Ambiguous update, specify as_node")
                if as_node not in self.nodes:
                    raise InvalidUpdateError(f"Node {as_node} does not exist")
                valid_updates.append((as_node, values, task_id))
            else:
                for values, as_node, task_id in updates:
                    if as_node is None:
                        raise InvalidUpdateError(
                            "as_node is required when applying multiple updates"
                        )
                    if as_node not in self.nodes:
                        raise InvalidUpdateError(f"Node {as_node} does not exist")

                    valid_updates.append((as_node, values, task_id))

            run_tasks: list[PregelTaskWrites] = []
            run_task_ids: list[str] = []

            for as_node, values, provided_task_id in valid_updates:
                # 선택된 노드의 모든 writer를 실행할 작업을 생성합니다
                writers = self.nodes[as_node].flat_writers
                if not writers:
                    raise InvalidUpdateError(f"Node {as_node} has no writers")
                writes: deque[tuple[str, Any]] = deque()
                task = PregelTaskWrites((), as_node, writes, [INTERRUPT])
                # 이 노드를 위해 준비된 task id를 가져옵니다
                # StateUpdate에 task id가 제공된 경우 이를 사용합니다
                # 그렇지 않으면 다음으로 사용 가능한 task id를 사용합니다
                prepared_task_ids = node_to_task_ids.get(as_node, deque())
                task_id = provided_task_id or (
                    prepared_task_ids.popleft()
                    if prepared_task_ids
                    else str(uuid5(UUID(checkpoint["id"]), INTERRUPT))
                )
                run_tasks.append(task)
                run_task_ids.append(task_id)
                run = RunnableSequence(*writers) if len(writers) > 1 else writers[0]
                # 작업 실행
                await run.ainvoke(
                    values,
                    patch_config(
                        config,
                        run_name=self.name + "UpdateState",
                        configurable={
                            # deque.extend는 스레드 세이프합니다
                            CONFIG_KEY_SEND: writes.extend,
                            CONFIG_KEY_TASK_ID: task_id,
                            CONFIG_KEY_READ: partial(
                                local_read,
                                _scratchpad(
                                    None,
                                    [],
                                    task_id,
                                    "",
                                    None,
                                    step,
                                    step + 2,
                                ),
                                channels,
                                managed,
                                task,
                            ),
                        },
                    ),
                )
            # 작업 쓰기 저장
            for task_id, task in zip(run_task_ids, run_tasks):
                # 채널 쓰기는 현재 체크포인트에 저장됩니다
                channel_writes = [w for w in task.writes if w[0] != PUSH]
                if saved and channel_writes:
                    await checkpointer.aput_writes(
                        checkpoint_config, channel_writes, task_id
                    )
            # 체크포인트에 적용하고 저장
            apply_writes(
                checkpoint,
                channels,
                run_tasks,
                checkpointer.get_next_version,
                self.trigger_to_nodes,
            )
            checkpoint = create_checkpoint(checkpoint, channels, step + 1)
            # 쓰기 적용 후 체크포인트 저장
            next_config = await checkpointer.aput(
                checkpoint_config,
                checkpoint,
                {
                    "source": "update",
                    "step": step + 1,
                    "parents": saved.metadata.get("parents", {}) if saved else {},
                },
                get_new_channel_versions(
                    checkpoint_previous_versions, checkpoint["channel_versions"]
                ),
            )
            for task_id, task in zip(run_task_ids, run_tasks):
                # push 쓰기 저장
                if push_writes := [w for w in task.writes if w[0] == PUSH]:
                    await checkpointer.aput_writes(next_config, push_writes, task_id)
            return patch_checkpoint_map(next_config, saved.metadata if saved else None)

        current_config = patch_configurable(
            config, {CONFIG_KEY_THREAD_ID: str(config[CONF][CONFIG_KEY_THREAD_ID])}
        )
        for superstep in supersteps:
            current_config = await aperform_superstep(current_config, superstep)
        return current_config

    def update_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any] | Any | None,
        as_node: str | None = None,
        task_id: str | None = None,
    ) -> RunnableConfig:
        """주어진 값으로 그래프 상태를 업데이트합니다. 마치 `as_node` 노드에서 온 것처럼 동작합니다.
        `as_node`가 제공되지 않으면 모호하지 않은 경우 상태를 업데이트한 마지막 노드로 설정됩니다.
        """
        return self.bulk_update_state(config, [[StateUpdate(values, as_node, task_id)]])

    async def aupdate_state(
        self,
        config: RunnableConfig,
        values: dict[str, Any] | Any,
        as_node: str | None = None,
        task_id: str | None = None,
    ) -> RunnableConfig:
        """주어진 값으로 그래프 상태를 비동기적으로 업데이트합니다. 마치 `as_node` 노드에서 온 것처럼 동작합니다.
        `as_node`가 제공되지 않으면 모호하지 않은 경우 상태를 업데이트한 마지막 노드로 설정됩니다.
        """
        return await self.abulk_update_state(
            config, [[StateUpdate(values, as_node, task_id)]]
        )

    def _defaults(
        self,
        config: RunnableConfig,
        *,
        stream_mode: StreamMode | Sequence[StreamMode],
        print_mode: StreamMode | Sequence[StreamMode],
        output_keys: str | Sequence[str] | None,
        interrupt_before: All | Sequence[str] | None,
        interrupt_after: All | Sequence[str] | None,
        durability: Durability | None = None,
    ) -> tuple[
        set[StreamMode],
        str | Sequence[str],
        All | Sequence[str],
        All | Sequence[str],
        BaseCheckpointSaver | None,
        BaseStore | None,
        BaseCache | None,
        Durability,
    ]:
        if config["recursion_limit"] < 1:
            raise ValueError("recursion_limit must be at least 1")
        if output_keys is None:
            output_keys = self.stream_channels_asis
        else:
            validate_keys(output_keys, self.channels)
        interrupt_before = interrupt_before or self.interrupt_before_nodes
        interrupt_after = interrupt_after or self.interrupt_after_nodes
        if not isinstance(stream_mode, list):
            stream_modes = {stream_mode}
        else:
            stream_modes = set(stream_mode)
        if isinstance(print_mode, str):
            stream_modes.add(print_mode)
        else:
            stream_modes.update(print_mode)
        if self.checkpointer is False:
            checkpointer: BaseCheckpointSaver | None = None
        elif CONFIG_KEY_CHECKPOINTER in config.get(CONF, {}):
            checkpointer = config[CONF][CONFIG_KEY_CHECKPOINTER]
        elif self.checkpointer is True:
            raise RuntimeError("checkpointer=True cannot be used for root graphs.")
        else:
            checkpointer = self.checkpointer
        if checkpointer and not config.get(CONF):
            raise ValueError(
                "Checkpointer requires one or more of the following 'configurable' "
                "keys: thread_id, checkpoint_ns, checkpoint_id"
            )
        if CONFIG_KEY_RUNTIME in config.get(CONF, {}):
            store: BaseStore | None = config[CONF][CONFIG_KEY_RUNTIME].store
        else:
            store = self.store
        if CONFIG_KEY_CACHE in config.get(CONF, {}):
            cache: BaseCache | None = config[CONF][CONFIG_KEY_CACHE]
        else:
            cache = self.cache
        if durability is None:
            durability = config.get(CONF, {}).get(CONFIG_KEY_DURABILITY, "async")
        return (
            stream_modes,
            output_keys,
            interrupt_before,
            interrupt_after,
            checkpointer,
            store,
            cache,
            durability,
        )

    def stream(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        *,
        context: ContextT | None = None,
        stream_mode: StreamMode | Sequence[StreamMode] | None = None,
        print_mode: StreamMode | Sequence[StreamMode] = (),
        output_keys: str | Sequence[str] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        durability: Durability | None = None,
        subgraphs: bool = False,
        debug: bool | None = None,
        **kwargs: Unpack[DeprecatedKwargs],
    ) -> Iterator[dict[str, Any] | Any]:
        """단일 입력에 대한 그래프 단계를 스트리밍합니다.

        Args:
            input: 그래프에 대한 입력입니다.
            config: 실행에 사용할 구성입니다.
            context: 실행에 사용할 정적 컨텍스트입니다.
                !!! version-added "버전 0.6.0에서 추가됨"
            stream_mode: 출력을 스트리밍하는 모드, 기본값은 `self.stream_mode`입니다.
                옵션은 다음과 같습니다:

                - `"values"`: 인터럽트를 포함하여 각 단계 후 상태의 모든 값을 방출합니다.
                    함수형 API와 함께 사용하면 워크플로우 끝에서 한 번 값이 방출됩니다.
                - `"updates"`: 각 단계 후 노드 또는 작업 이름과 노드 또는 작업에서 반환된 업데이트만 방출합니다.
                    동일한 단계에서 여러 업데이트가 수행되는 경우(예: 여러 노드가 실행됨) 해당 업데이트는 별도로 방출됩니다.
                - `"custom"`: `StreamWriter`를 사용하여 노드 또는 작업 내부에서 커스텀 데이터를 방출합니다.
                - `"messages"`: 노드 또는 작업 내부의 모든 LLM 호출에 대한 메타데이터와 함께 LLM 메시지를 토큰 단위로 방출합니다.
                    2-튜플 `(LLM token, metadata)` 형식으로 방출됩니다.
                - `"checkpoints"`: 체크포인트가 생성될 때 이벤트를 방출하며, `get_state()`에서 반환하는 것과 동일한 형식입니다.
                - `"tasks"`: 작업이 시작되고 완료될 때 결과와 오류를 포함하여 이벤트를 방출합니다.

                `stream_mode` 매개변수에 리스트를 전달하여 여러 모드를 한 번에 스트리밍할 수 있습니다.
                스트리밍된 출력은 `(mode, data)` 튜플이 됩니다.

                자세한 내용은 [LangGraph 스트리밍 가이드](https://langchain-ai.github.io/langgraph/how-tos/streaming/)를 참조하세요.
            print_mode: `stream_mode`와 동일한 값을 허용하지만, 디버깅 목적으로 콘솔에 출력만 인쇄합니다. 그래프의 출력에는 어떤 식으로도 영향을 주지 않습니다.
            output_keys: 스트리밍할 키, 기본값은 모든 비컨텍스트 채널입니다.
            interrupt_before: 인터럽트할 노드(이전), 기본값은 그래프의 모든 노드입니다.
            interrupt_after: 인터럽트할 노드(이후), 기본값은 그래프의 모든 노드입니다.
            durability: 그래프 실행에 대한 내구성 모드, 기본값은 `"async"`입니다.
                옵션은 다음과 같습니다:

                - `"sync"`: 다음 단계가 시작되기 전에 변경 사항이 동기적으로 유지됩니다.
                - `"async"`: 다음 단계가 실행되는 동안 변경 사항이 비동기적으로 유지됩니다.
                - `"exit"`: 그래프가 종료될 때만 변경 사항이 유지됩니다.
            subgraphs: 서브그래프 내부에서 이벤트를 스트리밍할지 여부, 기본값은 False입니다.
                `True`이면 이벤트는 `(namespace, data)` 튜플로 방출되며,
                `stream_mode`가 리스트인 경우 `(namespace, mode, data)`로 방출됩니다.
                여기서 `namespace`는 서브그래프가 호출되는 노드의 경로를 가진 튜플입니다.
                예: `("parent_node:<task_id>", "child_node:<task_id>")`.

        Yields:
            그래프의 각 단계 출력입니다. 출력 형태는 `stream_mode`에 따라 달라집니다.
        """
        if (checkpoint_during := kwargs.get("checkpoint_during")) is not None:
            warnings.warn(
                "`checkpoint_during` is deprecated and will be removed. Please use `durability` instead.",
                category=LangGraphDeprecatedSinceV10,
                stacklevel=2,
            )
            if durability is not None:
                raise ValueError(
                    "Cannot use both `checkpoint_during` and `durability` parameters. Please use `durability` instead."
                )
            durability = "async" if checkpoint_during else "exit"

        if stream_mode is None:
            # 다른 그래프의 노드로 호출되는 경우 values 모드를 기본값으로 사용합니다
            # 하지만 stream_mode 인수가 제공된 경우 덮어쓰지 않습니다
            stream_mode = (
                "values"
                if config is not None and CONFIG_KEY_TASK_ID in config.get(CONF, {})
                else self.stream_mode
            )
        if debug or self.debug:
            print_mode = ["updates", "values"]

        stream = SyncQueue()

        config = ensure_config(self.config, config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name", self.get_name()),
            run_id=config.get("run_id"),
        )
        try:
            # 기본값 할당
            (
                stream_modes,
                output_keys,
                interrupt_before_,
                interrupt_after_,
                checkpointer,
                store,
                cache,
                durability_,
            ) = self._defaults(
                config,
                stream_mode=stream_mode,
                print_mode=print_mode,
                output_keys=output_keys,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                durability=durability,
            )
            if checkpointer is None and durability is not None:
                warnings.warn(
                    "`durability` has no effect when no checkpointer is present.",
                )
            # 서브그래프 체크포인팅 설정
            if self.checkpointer is True:
                ns = cast(str, config[CONF][CONFIG_KEY_CHECKPOINT_NS])
                config[CONF][CONFIG_KEY_CHECKPOINT_NS] = recast_checkpoint_ns(ns)
            # messages 스트림 모드 설정
            if "messages" in stream_modes:
                ns_ = cast(str | None, config[CONF].get(CONFIG_KEY_CHECKPOINT_NS))
                run_manager.inheritable_handlers.append(
                    StreamMessagesHandler(
                        stream.put,
                        subgraphs,
                        parent_ns=tuple(ns_.split(NS_SEP)) if ns_ else None,
                    )
                )

            # custom 스트림 모드 설정
            if "custom" in stream_modes:

                def stream_writer(c: Any) -> None:
                    stream.put(
                        (
                            tuple(
                                get_config()[CONF][CONFIG_KEY_CHECKPOINT_NS].split(
                                    NS_SEP
                                )[:-1]
                            ),
                            "custom",
                            c,
                        )
                    )
            elif CONFIG_KEY_STREAM in config[CONF]:
                stream_writer = config[CONF][CONFIG_KEY_RUNTIME].stream_writer
            else:

                def stream_writer(c: Any) -> None:
                    pass

            # 서브그래프에 대한 내구성 모드 설정
            if durability is not None:
                config[CONF][CONFIG_KEY_DURABILITY] = durability_

            runtime = Runtime(
                context=_coerce_context(self.context_schema, context),
                store=store,
                stream_writer=stream_writer,
                previous=None,
            )
            parent_runtime = config[CONF].get(CONFIG_KEY_RUNTIME, DEFAULT_RUNTIME)
            runtime = parent_runtime.merge(runtime)
            config[CONF][CONFIG_KEY_RUNTIME] = runtime

            with SyncPregelLoop(
                input,
                stream=StreamProtocol(stream.put, stream_modes),
                config=config,
                store=store,
                cache=cache,
                checkpointer=checkpointer,
                nodes=self.nodes,
                specs=self.channels,
                output_keys=output_keys,
                input_keys=self.input_channels,
                stream_keys=self.stream_channels_asis,
                interrupt_before=interrupt_before_,
                interrupt_after=interrupt_after_,
                manager=run_manager,
                durability=durability_,
                trigger_to_nodes=self.trigger_to_nodes,
                migrate_checkpoint=self._migrate_checkpoint,
                retry_policy=self.retry_policy,
                cache_policy=self.cache_policy,
            ) as loop:
                # runner 생성
                runner = PregelRunner(
                    submit=config[CONF].get(
                        CONFIG_KEY_RUNNER_SUBMIT, weakref.WeakMethod(loop.submit)
                    ),
                    put_writes=weakref.WeakMethod(loop.put_writes),
                    node_finished=config[CONF].get(CONFIG_KEY_NODE_FINISHED),
                )
                # 서브그래프 스트리밍 활성화
                if subgraphs:
                    loop.config[CONF][CONFIG_KEY_STREAM] = loop.stream
                # 동시 스트리밍 활성화
                get_waiter: Callable[[], concurrent.futures.Future[None]] | None = None
                if (
                    self.stream_eager
                    or subgraphs
                    or "messages" in stream_modes
                    or "custom" in stream_modes
                ):
                    # 한 번에 하나의 waiter만 유지하도록 주의합니다
                    # 종료 시 세마포어 카운트를 정확히 1만큼 증가시키기 때문입니다
                    waiter: concurrent.futures.Future | None = None
                    # 동기 futures는 취소할 수 없으므로 대신
                    # 종료 시 스트림 세마포어를 해제하여
                    # 대기 중인 waiter가 즉시 반환되도록 합니다
                    loop.stack.callback(stream._count.release)

                    def get_waiter() -> concurrent.futures.Future[None]:
                        nonlocal waiter
                        if waiter is None or waiter.done():
                            waiter = loop.submit(stream.wait)
                            return waiter
                        else:
                            return waiter

                # Bulk Synchronous Parallel / Pregel 모델과 유사하게
                # 채널 업데이트가 있는 동안 계산이 단계별로 진행됩니다.
                # 단계 N의 채널 업데이트는 단계 N+1에서만 볼 수 있습니다
                # 채널은 단계 동안 불변임이 보장되며,
                # 채널 업데이트는 단계 간 전환 시에만 적용됩니다.
                while loop.tick():
                    for task in loop.match_cached_writes():
                        loop.output_writes(task.id, task.writes, cached=True)
                    for _ in runner.tick(
                        [t for t in loop.tasks.values() if not t.writes],
                        timeout=self.step_timeout,
                        get_waiter=get_waiter,
                        schedule_task=loop.accept_push,
                    ):
                        # 출력 방출
                        yield from _output(
                            stream_mode, print_mode, subgraphs, stream.get, queue.Empty
                        )
                    loop.after_tick()
                    # 체크포인트 대기
                    if durability_ == "sync":
                        loop._put_checkpoint_fut.result()
            # 출력 방출
            yield from _output(
                stream_mode, print_mode, subgraphs, stream.get, queue.Empty
            )
            # 종료 처리
            if loop.status == "out_of_steps":
                msg = create_error_message(
                    message=(
                        f"Recursion limit of {config['recursion_limit']} reached "
                        "without hitting a stop condition. You can increase the "
                        "limit by setting the `recursion_limit` config key."
                    ),
                    error_code=ErrorCode.GRAPH_RECURSION_LIMIT,
                )
                raise GraphRecursionError(msg)
            # 최종 채널 값을 실행 출력으로 설정
            run_manager.on_chain_end(loop.output)
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise

    async def astream(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        *,
        context: ContextT | None = None,
        stream_mode: StreamMode | Sequence[StreamMode] | None = None,
        print_mode: StreamMode | Sequence[StreamMode] = (),
        output_keys: str | Sequence[str] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        durability: Durability | None = None,
        subgraphs: bool = False,
        debug: bool | None = None,
        **kwargs: Unpack[DeprecatedKwargs],
    ) -> AsyncIterator[dict[str, Any] | Any]:
        """단일 입력에 대한 그래프 단계를 비동기적으로 스트리밍합니다.

        Args:
            input: 그래프에 대한 입력입니다.
            config: 실행에 사용할 구성입니다.
            context: 실행에 사용할 정적 컨텍스트입니다.
                !!! version-added "버전 0.6.0에서 추가됨"
            stream_mode: 출력을 스트리밍하는 모드, 기본값은 `self.stream_mode`입니다.
                옵션은 다음과 같습니다:

                - `"values"`: 인터럽트를 포함하여 각 단계 후 상태의 모든 값을 방출합니다.
                    함수형 API와 함께 사용하면 워크플로우 끝에서 한 번 값이 방출됩니다.
                - `"updates"`: 각 단계 후 노드 또는 작업 이름과 노드 또는 작업에서 반환된 업데이트만 방출합니다.
                    동일한 단계에서 여러 업데이트가 수행되는 경우(예: 여러 노드가 실행됨) 해당 업데이트는 별도로 방출됩니다.
                - `"custom"`: `StreamWriter`를 사용하여 노드 또는 작업 내부에서 커스텀 데이터를 방출합니다.
                - `"messages"`: 노드 또는 작업 내부의 모든 LLM 호출에 대한 메타데이터와 함께 LLM 메시지를 토큰 단위로 방출합니다.
                    2-튜플 `(LLM token, metadata)` 형식으로 방출됩니다.
                - `"debug"`: 각 단계에 대해 가능한 한 많은 정보를 포함하는 디버그 이벤트를 방출합니다.

                `stream_mode` 매개변수에 리스트를 전달하여 여러 모드를 한 번에 스트리밍할 수 있습니다.
                스트리밍된 출력은 `(mode, data)` 튜플이 됩니다.

                자세한 내용은 [LangGraph 스트리밍 가이드](https://langchain-ai.github.io/langgraph/how-tos/streaming/)를 참조하세요.
            print_mode: `stream_mode`와 동일한 값을 허용하지만, 디버깅 목적으로 콘솔에 출력만 인쇄합니다. 그래프의 출력에는 어떤 식으로도 영향을 주지 않습니다.
            output_keys: 스트리밍할 키, 기본값은 모든 비컨텍스트 채널입니다.
            interrupt_before: 인터럽트할 노드(이전), 기본값은 그래프의 모든 노드입니다.
            interrupt_after: 인터럽트할 노드(이후), 기본값은 그래프의 모든 노드입니다.
            durability: 그래프 실행에 대한 내구성 모드, 기본값은 `"async"`입니다.
                옵션은 다음과 같습니다:

                - `"sync"`: 다음 단계가 시작되기 전에 변경 사항이 동기적으로 유지됩니다.
                - `"async"`: 다음 단계가 실행되는 동안 변경 사항이 비동기적으로 유지됩니다.
                - `"exit"`: 그래프가 종료될 때만 변경 사항이 유지됩니다.
            subgraphs: 서브그래프 내부에서 이벤트를 스트리밍할지 여부, 기본값은 False입니다.
                `True`이면 이벤트는 `(namespace, data)` 튜플로 방출되며,
                `stream_mode`가 리스트인 경우 `(namespace, mode, data)`로 방출됩니다.
                여기서 `namespace`는 서브그래프가 호출되는 노드의 경로를 가진 튜플입니다.
                예: `("parent_node:<task_id>", "child_node:<task_id>")`.

        Yields:
            그래프의 각 단계 출력입니다. 출력 형태는 `stream_mode`에 따라 달라집니다.
        """
        if (checkpoint_during := kwargs.get("checkpoint_during")) is not None:
            warnings.warn(
                "`checkpoint_during` is deprecated and will be removed. Please use `durability` instead.",
                category=LangGraphDeprecatedSinceV10,
                stacklevel=2,
            )
            if durability is not None:
                raise ValueError(
                    "Cannot use both `checkpoint_during` and `durability` parameters. Please use `durability` instead."
                )
            durability = "async" if checkpoint_during else "exit"

        if stream_mode is None:
            # 다른 그래프의 노드로 호출되는 경우 values 모드를 기본값으로 사용합니다
            # 하지만 stream_mode 인수가 제공된 경우 덮어쓰지 않습니다
            stream_mode = (
                "values"
                if config is not None and CONFIG_KEY_TASK_ID in config.get(CONF, {})
                else self.stream_mode
            )
        if debug or self.debug:
            print_mode = ["updates", "values"]

        stream = AsyncQueue()
        aioloop = asyncio.get_running_loop()
        stream_put = cast(
            Callable[[StreamChunk], None],
            partial(aioloop.call_soon_threadsafe, stream.put_nowait),
        )

        config = ensure_config(self.config, config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name", self.get_name()),
            run_id=config.get("run_id"),
        )
        # astream_log()에서 실행 중인 경우 각 프로세스를 스트리밍과 함께 실행합니다
        do_stream = (
            next(
                (
                    True
                    for h in run_manager.handlers
                    if isinstance(h, _StreamingCallbackHandler)
                    and not isinstance(h, StreamMessagesHandler)
                ),
                False,
            )
            if _StreamingCallbackHandler is not None
            else False
        )
        try:
            # 기본값 할당
            (
                stream_modes,
                output_keys,
                interrupt_before_,
                interrupt_after_,
                checkpointer,
                store,
                cache,
                durability_,
            ) = self._defaults(
                config,
                stream_mode=stream_mode,
                print_mode=print_mode,
                output_keys=output_keys,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                durability=durability,
            )
            if checkpointer is None and durability is not None:
                warnings.warn(
                    "`durability` has no effect when no checkpointer is present.",
                )
            # 서브그래프 체크포인팅 설정
            if self.checkpointer is True:
                ns = cast(str, config[CONF][CONFIG_KEY_CHECKPOINT_NS])
                config[CONF][CONFIG_KEY_CHECKPOINT_NS] = recast_checkpoint_ns(ns)
            # messages 스트림 모드 설정
            if "messages" in stream_modes:
                # 루트 레벨 그래프에서 namespace는 None일 수 있습니다
                ns_ = cast(str | None, config[CONF].get(CONFIG_KEY_CHECKPOINT_NS))
                run_manager.inheritable_handlers.append(
                    StreamMessagesHandler(
                        stream_put,
                        subgraphs,
                        parent_ns=tuple(ns_.split(NS_SEP)) if ns_ else None,
                    )
                )

            # custom 스트림 모드 설정
            def stream_writer(c: Any) -> None:
                aioloop.call_soon_threadsafe(
                    stream.put_nowait,
                    (
                        tuple(
                            get_config()[CONF][CONFIG_KEY_CHECKPOINT_NS].split(NS_SEP)[
                                :-1
                            ]
                        ),
                        "custom",
                        c,
                    ),
                )

            if "custom" in stream_modes:

                def stream_writer(c: Any) -> None:
                    aioloop.call_soon_threadsafe(
                        stream.put_nowait,
                        (
                            tuple(
                                get_config()[CONF][CONFIG_KEY_CHECKPOINT_NS].split(
                                    NS_SEP
                                )[:-1]
                            ),
                            "custom",
                            c,
                        ),
                    )
            elif CONFIG_KEY_STREAM in config[CONF]:
                stream_writer = config[CONF][CONFIG_KEY_RUNTIME].stream_writer
            else:

                def stream_writer(c: Any) -> None:
                    pass

            # 서브그래프에 대한 내구성 모드 설정
            if durability is not None:
                config[CONF][CONFIG_KEY_DURABILITY] = durability_

            runtime = Runtime(
                context=_coerce_context(self.context_schema, context),
                store=store,
                stream_writer=stream_writer,
                previous=None,
            )
            parent_runtime = config[CONF].get(CONFIG_KEY_RUNTIME, DEFAULT_RUNTIME)
            runtime = parent_runtime.merge(runtime)
            config[CONF][CONFIG_KEY_RUNTIME] = runtime

            async with AsyncPregelLoop(
                input,
                stream=StreamProtocol(stream.put_nowait, stream_modes),
                config=config,
                store=store,
                cache=cache,
                checkpointer=checkpointer,
                nodes=self.nodes,
                specs=self.channels,
                output_keys=output_keys,
                input_keys=self.input_channels,
                stream_keys=self.stream_channels_asis,
                interrupt_before=interrupt_before_,
                interrupt_after=interrupt_after_,
                manager=run_manager,
                durability=durability_,
                trigger_to_nodes=self.trigger_to_nodes,
                migrate_checkpoint=self._migrate_checkpoint,
                retry_policy=self.retry_policy,
                cache_policy=self.cache_policy,
            ) as loop:
                # runner 생성
                runner = PregelRunner(
                    submit=config[CONF].get(
                        CONFIG_KEY_RUNNER_SUBMIT, weakref.WeakMethod(loop.submit)
                    ),
                    put_writes=weakref.WeakMethod(loop.put_writes),
                    use_astream=do_stream,
                    node_finished=config[CONF].get(CONFIG_KEY_NODE_FINISHED),
                )
                # 서브그래프 스트리밍 활성화
                if subgraphs:
                    loop.config[CONF][CONFIG_KEY_STREAM] = StreamProtocol(
                        stream_put, stream_modes
                    )
                # 동시 스트리밍 활성화
                get_waiter: Callable[[], asyncio.Task[None]] | None = None
                _cleanup_waiter: Callable[[], Awaitable[None]] | None = None
                if (
                    self.stream_eager
                    or subgraphs
                    or "messages" in stream_modes
                    or "custom" in stream_modes
                ):
                    # 단일 waiter 작업을 유지하고 종료 시 정리를 보장합니다.
                    waiter: asyncio.Task[None] | None = None

                    def get_waiter() -> asyncio.Task[None]:
                        nonlocal waiter
                        if waiter is None or waiter.done():
                            waiter = aioloop.create_task(stream.wait())

                            def _clear(t: asyncio.Task[None]) -> None:
                                nonlocal waiter
                                if waiter is t:
                                    waiter = None

                            waiter.add_done_callback(_clear)
                        return waiter

                    async def _cleanup_waiter() -> None:
                        """대기 중인 waiter를 깨우거나 취소 및 대기하여 대기 중인 작업을 방지합니다."""
                        nonlocal waiter
                        # SyncPregelLoop처럼 세마포어를 통해 깨우기를 시도합니다
                        with contextlib.suppress(Exception):
                            if hasattr(stream, "_count"):
                                stream._count.release()
                        t = waiter
                        waiter = None
                        if t is not None and not t.done():
                            t.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await t

                # Bulk Synchronous Parallel / Pregel 모델과 유사하게
                # 채널 업데이트가 있는 동안 계산이 단계별로 진행됩니다
                # 단계 N의 채널 업데이트는 단계 N+1에서만 볼 수 있습니다
                # 채널은 단계 동안 불변임이 보장되며,
                # 채널 업데이트는 단계 간 전환 시에만 적용됩니다
                try:
                    while loop.tick():
                        for task in await loop.amatch_cached_writes():
                            loop.output_writes(task.id, task.writes, cached=True)
                        async for _ in runner.atick(
                            [t for t in loop.tasks.values() if not t.writes],
                            timeout=self.step_timeout,
                            get_waiter=get_waiter,
                            schedule_task=loop.aaccept_push,
                        ):
                            # 출력 방출
                            for o in _output(
                                stream_mode,
                                print_mode,
                                subgraphs,
                                stream.get_nowait,
                                asyncio.QueueEmpty,
                            ):
                                yield o
                        loop.after_tick()
                        # 체크포인트 대기
                        if durability_ == "sync":
                            await cast(asyncio.Future, loop._put_checkpoint_fut)
                finally:
                    # 취소/종료 시 waiter가 대기 중으로 남지 않도록 보장합니다
                    if _cleanup_waiter is not None:
                        await _cleanup_waiter()

            # 출력 방출
            for o in _output(
                stream_mode,
                print_mode,
                subgraphs,
                stream.get_nowait,
                asyncio.QueueEmpty,
            ):
                yield o
            # 종료 처리
            if loop.status == "out_of_steps":
                msg = create_error_message(
                    message=(
                        f"Recursion limit of {config['recursion_limit']} reached "
                        "without hitting a stop condition. You can increase the "
                        "limit by setting the `recursion_limit` config key."
                    ),
                    error_code=ErrorCode.GRAPH_RECURSION_LIMIT,
                )
                raise GraphRecursionError(msg)
            # 최종 채널 값을 실행 출력으로 설정
            await run_manager.on_chain_end(loop.output)
        except BaseException as e:
            await asyncio.shield(run_manager.on_chain_error(e))
            raise

    def invoke(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        *,
        context: ContextT | None = None,
        stream_mode: StreamMode = "values",
        print_mode: StreamMode | Sequence[StreamMode] = (),
        output_keys: str | Sequence[str] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        durability: Durability | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | Any:
        """단일 입력과 config로 그래프를 실행합니다.

        Args:
            input: 그래프의 입력 데이터입니다. 딕셔너리 또는 다른 타입일 수 있습니다.
            config: 그래프 실행을 위한 구성입니다.
            context: 실행에 사용할 정적 컨텍스트입니다.
                !!! version-added "버전 0.6.0에서 추가됨"
            stream_mode: 그래프 실행을 위한 스트림 모드입니다.
            print_mode: `stream_mode`와 동일한 값을 허용하지만, 디버깅 목적으로 콘솔에 출력만 인쇄합니다. 그래프의 출력에는 어떤 식으로도 영향을 주지 않습니다.
            output_keys: 그래프 실행에서 검색할 출력 키입니다.
            interrupt_before: 그래프 실행을 중단할 노드(이전)입니다.
            interrupt_after: 그래프 실행을 중단할 노드(이후)입니다.
            durability: 그래프 실행에 대한 내구성 모드, 기본값은 `"async"`입니다.
                옵션은 다음과 같습니다:

                - `"sync"`: 다음 단계가 시작되기 전에 변경 사항이 동기적으로 유지됩니다.
                - `"async"`: 다음 단계가 실행되는 동안 변경 사항이 비동기적으로 유지됩니다.
                - `"exit"`: 그래프가 종료될 때만 변경 사항이 유지됩니다.
            **kwargs: 그래프 실행에 전달할 추가 키워드 인수입니다.

        Returns:
            그래프 실행의 출력입니다. `stream_mode`가 `"values"`인 경우 최신 출력을 반환합니다.
            `stream_mode`가 `"values"`가 아닌 경우 출력 청크 목록을 반환합니다.
        """
        output_keys = output_keys if output_keys is not None else self.output_channels

        latest: dict[str, Any] | Any = None
        chunks: list[dict[str, Any] | Any] = []
        interrupts: list[Interrupt] = []

        for chunk in self.stream(
            input,
            config,
            context=context,
            stream_mode=["updates", "values"]
            if stream_mode == "values"
            else stream_mode,
            print_mode=print_mode,
            output_keys=output_keys,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            durability=durability,
            **kwargs,
        ):
            if stream_mode == "values":
                if len(chunk) == 2:
                    mode, payload = cast(tuple[StreamMode, Any], chunk)
                else:
                    _, mode, payload = cast(
                        tuple[tuple[str, ...], StreamMode, Any], chunk
                    )
                if (
                    mode == "updates"
                    and isinstance(payload, dict)
                    and (ints := payload.get(INTERRUPT)) is not None
                ):
                    interrupts.extend(ints)
                elif mode == "values":
                    latest = payload
            else:
                chunks.append(chunk)

        if stream_mode == "values":
            if interrupts:
                return (
                    {**latest, INTERRUPT: interrupts}
                    if isinstance(latest, dict)
                    else {INTERRUPT: interrupts}
                )
            return latest
        else:
            return chunks

    async def ainvoke(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        *,
        context: ContextT | None = None,
        stream_mode: StreamMode = "values",
        print_mode: StreamMode | Sequence[StreamMode] = (),
        output_keys: str | Sequence[str] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        durability: Durability | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | Any:
        """단일 입력에 대해 그래프를 비동기적으로 호출합니다.

        Args:
            input: 계산을 위한 입력 데이터입니다. 딕셔너리 또는 다른 타입일 수 있습니다.
            config: 계산을 위한 구성입니다.
            context: 실행에 사용할 정적 컨텍스트입니다.
                !!! version-added "버전 0.6.0에서 추가됨"
            stream_mode: 계산을 위한 스트림 모드입니다.
            print_mode: `stream_mode`와 동일한 값을 허용하지만, 디버깅 목적으로 콘솔에 출력만 인쇄합니다. 그래프의 출력에는 어떤 식으로도 영향을 주지 않습니다.
            output_keys: 결과에 포함할 출력 키입니다.
            interrupt_before: 인터럽트할 노드(이전)입니다.
            interrupt_after: 인터럽트할 노드(이후)입니다.
            durability: 그래프 실행에 대한 내구성 모드, 기본값은 `"async"`입니다.
                옵션은 다음과 같습니다:

                - `"sync"`: 다음 단계가 시작되기 전에 변경 사항이 동기적으로 유지됩니다.
                - `"async"`: 다음 단계가 실행되는 동안 변경 사항이 비동기적으로 유지됩니다.
                - `"exit"`: 그래프가 종료될 때만 변경 사항이 유지됩니다.
            **kwargs: 추가 키워드 인수입니다.

        Returns:
            계산 결과입니다. `stream_mode`가 `"values"`인 경우 최신 값을 반환합니다.
            `stream_mode`가 `"chunks"`인 경우 청크 목록을 반환합니다.
        """

        output_keys = output_keys if output_keys is not None else self.output_channels

        latest: dict[str, Any] | Any = None
        chunks: list[dict[str, Any] | Any] = []
        interrupts: list[Interrupt] = []

        async for chunk in self.astream(
            input,
            config,
            context=context,
            stream_mode=["updates", "values"]
            if stream_mode == "values"
            else stream_mode,
            print_mode=print_mode,
            output_keys=output_keys,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            durability=durability,
            **kwargs,
        ):
            if stream_mode == "values":
                if len(chunk) == 2:
                    mode, payload = cast(tuple[StreamMode, Any], chunk)
                else:
                    _, mode, payload = cast(
                        tuple[tuple[str, ...], StreamMode, Any], chunk
                    )
                if (
                    mode == "updates"
                    and isinstance(payload, dict)
                    and (ints := payload.get(INTERRUPT)) is not None
                ):
                    interrupts.extend(ints)
                elif mode == "values":
                    latest = payload
            else:
                chunks.append(chunk)

        if stream_mode == "values":
            if interrupts:
                return (
                    {**latest, INTERRUPT: interrupts}
                    if isinstance(latest, dict)
                    else {INTERRUPT: interrupts}
                )
            return latest
        else:
            return chunks

    def clear_cache(self, nodes: Sequence[str] | None = None) -> None:
        """주어진 노드에 대한 캐시를 지웁니다."""
        if not self.cache:
            raise ValueError("No cache is set for this graph. Cannot clear cache.")
        nodes = nodes or self.nodes.keys()
        # 지울 네임스페이스 수집
        namespaces: list[tuple[str, ...]] = []
        for node in nodes:
            if node in self.nodes:
                namespaces.append(
                    (
                        CACHE_NS_WRITES,
                        (identifier(self.nodes[node]) or "__dynamic__"),
                        node,
                    ),
                )
        # 캐시 지우기
        self.cache.clear(namespaces)

    async def aclear_cache(self, nodes: Sequence[str] | None = None) -> None:
        """주어진 노드에 대한 캐시를 비동기적으로 지웁니다."""
        if not self.cache:
            raise ValueError("No cache is set for this graph. Cannot clear cache.")
        nodes = nodes or self.nodes.keys()
        # 지울 네임스페이스 수집
        namespaces: list[tuple[str, ...]] = []
        for node in nodes:
            if node in self.nodes:
                namespaces.append(
                    (
                        CACHE_NS_WRITES,
                        (identifier(self.nodes[node]) or "__dynamic__"),
                        node,
                    ),
                )
        # 캐시 지우기
        await self.cache.aclear(namespaces)


def _trigger_to_nodes(nodes: dict[str, PregelNode]) -> Mapping[str, Sequence[str]]:
    """트리거에서 해당 트리거에 의존하는 노드로의 인덱스입니다."""
    trigger_to_nodes: defaultdict[str, list[str]] = defaultdict(list)
    for name, node in nodes.items():
        for trigger in node.triggers:
            trigger_to_nodes[trigger].append(name)
    return dict(trigger_to_nodes)


def _output(
    stream_mode: StreamMode | Sequence[StreamMode],
    print_mode: StreamMode | Sequence[StreamMode],
    stream_subgraphs: bool,
    getter: Callable[[], tuple[tuple[str, ...], str, Any]],
    empty_exc: type[Exception],
) -> Iterator:
    while True:
        try:
            ns, mode, payload = getter()
        except empty_exc:
            break
        if mode in print_mode:
            if stream_subgraphs and ns:
                print(
                    " ".join(
                        (
                            get_bolded_text(f"[{mode}]"),
                            get_colored_text(f"[graph={ns}]", color="yellow"),
                            repr(payload),
                        )
                    )
                )
            else:
                print(
                    " ".join(
                        (
                            get_bolded_text(f"[{mode}]"),
                            repr(payload),
                        )
                    )
                )
        if mode in stream_mode:
            if stream_subgraphs and isinstance(stream_mode, list):
                yield (ns, mode, payload)
            elif isinstance(stream_mode, list):
                yield (mode, payload)
            elif stream_subgraphs:
                yield (ns, payload)
            else:
                yield payload


def _coerce_context(
    context_schema: type[ContextT] | None, context: Any
) -> ContextT | None:
    """컨텍스트 입력을 적절한 스키마 타입으로 강제 변환합니다.

    context가 딕셔너리이고 context_schema가 dataclass 또는 pydantic 모델인 경우 강제 변환합니다.
    그렇지 않으면 context를 그대로 반환합니다.

    Args:
        context_schema: 강제 변환할 스키마 타입입니다 (BaseModel, dataclass, 또는 TypedDict).
        context: 강제 변환할 컨텍스트 값입니다.

    Returns:
        강제 변환된 컨텍스트 값 또는 context가 None인 경우 None입니다.
    """
    if context is None:
        return None

    if context_schema is None:
        return context

    schema_is_class = issubclass(context_schema, BaseModel) or is_dataclass(
        context_schema
    )
    if isinstance(context, dict) and schema_is_class:
        return context_schema(**context)  # type: ignore[misc]

    return cast(ContextT, context)
