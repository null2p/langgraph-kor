from __future__ import annotations

import binascii
import itertools
import sys
import threading
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import copy
from functools import partial
from hashlib import sha1
from typing import (
    Any,
    Literal,
    NamedTuple,
    Protocol,
    cast,
    overload,
)

from langchain_core.callbacks import Callbacks
from langchain_core.callbacks.manager import AsyncParentRunManager, ParentRunManager
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    PendingWrite,
    V,
)
from langgraph.store.base import BaseStore
from xxhash import xxh3_128_hexdigest

from langgraph._internal._config import merge_configs, patch_config
from langgraph._internal._constants import (
    CACHE_NS_WRITES,
    CONF,
    CONFIG_KEY_CHECKPOINT_ID,
    CONFIG_KEY_CHECKPOINT_MAP,
    CONFIG_KEY_CHECKPOINT_NS,
    CONFIG_KEY_CHECKPOINTER,
    CONFIG_KEY_READ,
    CONFIG_KEY_RESUME_MAP,
    CONFIG_KEY_RUNTIME,
    CONFIG_KEY_SCRATCHPAD,
    CONFIG_KEY_SEND,
    CONFIG_KEY_TASK_ID,
    ERROR,
    INTERRUPT,
    NO_WRITES,
    NS_END,
    NS_SEP,
    NULL_TASK_ID,
    PREVIOUS,
    PULL,
    PUSH,
    RESERVED,
    RESUME,
    RETURN,
    TASKS,
)
from langgraph._internal._scratchpad import PregelScratchpad
from langgraph._internal._typing import EMPTY_SEQ, MISSING
from langgraph.channels.base import BaseChannel
from langgraph.channels.topic import Topic
from langgraph.constants import TAG_HIDDEN
from langgraph.managed.base import ManagedValueMapping
from langgraph.pregel._call import get_runnable_for_task, identifier
from langgraph.pregel._io import read_channels
from langgraph.pregel._log import logger
from langgraph.pregel._read import INPUT_CACHE_KEY_TYPE, PregelNode
from langgraph.runtime import DEFAULT_RUNTIME, Runtime
from langgraph.types import (
    All,
    CacheKey,
    CachePolicy,
    PregelExecutableTask,
    PregelTask,
    RetryPolicy,
    Send,
)

GetNextVersion = Callable[[V | None, None], V]
SUPPORTS_EXC_NOTES = sys.version_info >= (3, 11)


class WritesProtocol(Protocol):
    """체크포인트에 적용할 쓰기를 포함하는 객체의 프로토콜입니다.
    PregelTaskWrites 및 PregelExecutableTask에 의해 구현됩니다."""

    @property
    def path(self) -> tuple[str | int | tuple, ...]: ...

    @property
    def name(self) -> str: ...

    @property
    def writes(self) -> Sequence[tuple[str, Any]]: ...

    @property
    def triggers(self) -> Sequence[str]: ...


class PregelTaskWrites(NamedTuple):
    """WritesProtocol의 가장 간단한 구현으로, runnable 태스크에서 유래하지 않은 쓰기
    (예: 그래프 입력, update_state 등)와 함께 사용하기 위한 것입니다."""

    path: tuple[str | int | tuple, ...]
    name: str
    writes: Sequence[tuple[str, Any]]
    triggers: Sequence[str]


class Call:
    __slots__ = ("func", "input", "retry_policy", "cache_policy", "callbacks")

    func: Callable
    input: tuple[tuple[Any, ...], dict[str, Any]]
    retry_policy: Sequence[RetryPolicy] | None
    cache_policy: CachePolicy | None
    callbacks: Callbacks

    def __init__(
        self,
        func: Callable,
        input: tuple[tuple[Any, ...], dict[str, Any]],
        *,
        retry_policy: Sequence[RetryPolicy] | None,
        cache_policy: CachePolicy | None,
        callbacks: Callbacks,
    ) -> None:
        self.func = func
        self.input = input
        self.retry_policy = retry_policy
        self.cache_policy = cache_policy
        self.callbacks = callbacks


def should_interrupt(
    checkpoint: Checkpoint,
    interrupt_nodes: All | Sequence[str],
    tasks: Iterable[PregelExecutableTask],
) -> list[PregelExecutableTask]:
    """현재 상태를 기반으로 그래프를 중단해야 하는지 확인합니다."""
    version_type = type(next(iter(checkpoint["channel_versions"].values()), None))
    null_version = version_type()  # type: ignore[misc]
    seen = checkpoint["versions_seen"].get(INTERRUPT, {})
    # 마지막 인터럽트 이후 채널이 업데이트된 경우 인터럽트합니다
    any_updates_since_prev_interrupt = any(
        version > seen.get(chan, null_version)  # type: ignore[operator]
        for chan, version in checkpoint["channel_versions"].items()
    )
    # 그리고 트리거된 노드가 interrupt_nodes 목록에 있는 경우
    return (
        [
            task
            for task in tasks
            if (
                (
                    not task.config
                    or TAG_HIDDEN not in task.config.get("tags", EMPTY_SEQ)
                )
                if interrupt_nodes == "*"
                else task.name in interrupt_nodes
            )
        ]
        if any_updates_since_prev_interrupt
        else []
    )


def local_read(
    scratchpad: PregelScratchpad,
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    task: WritesProtocol,
    select: list[str] | str,
    fresh: bool = False,
) -> dict[str, Any] | Any:
    """태스크 config의 CONFIG_KEY_READ 아래에 주입되어 현재 상태를 읽는 함수입니다.
    조건부 엣지가 해당 노드의 쓰기만 반영된 상태의 복사본을 읽을 때 사용됩니다."""
    updated: dict[str, list[Any]] = defaultdict(list)
    if isinstance(select, str):
        managed_keys = []
        for c, v in task.writes:
            if c == select:
                updated[c].append(v)
    else:
        managed_keys = [k for k in select if k in managed]
        select = [k for k in select if k not in managed]
        for c, v in task.writes:
            if c in select:
                updated[c].append(v)
    if fresh:
        # 쓰기를 적용합니다
        local_channels: dict[str, BaseChannel] = {}
        for k in channels:
            cc = channels[k].copy()
            cc.update(updated[k])
            local_channels[k] = cc
        # 최신 값을 읽습니다
        values = read_channels(local_channels, select)
    else:
        values = read_channels(channels, select)
    if managed_keys:
        values.update({k: managed[k].get(scratchpad) for k in managed_keys})
    return values


def increment(current: int | None, channel: None) -> int:
    """기본 채널 버저닝 함수로, 현재 int 버전을 증가시킵니다."""
    return current + 1 if current is not None else 1


def apply_writes(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel],
    tasks: Iterable[WritesProtocol],
    get_next_version: GetNextVersion | None,
    trigger_to_nodes: Mapping[str, Sequence[str]],
) -> set[str]:
    """태스크 집합(보통 Pregel 단계의 태스크)의 쓰기를 체크포인트 및 채널에 적용하고,
    외부적으로 적용할 관리 값 쓰기를 반환합니다.

    Args:
        checkpoint: 업데이트할 체크포인트입니다.
        channels: 업데이트할 채널입니다.
        tasks: 쓰기를 적용할 태스크입니다.
        get_next_version: 채널의 다음 버전을 결정하는 선택적 함수입니다.
        trigger_to_nodes: 채널 이름을 해당 채널에 대한 업데이트에 의해 트리거될 수 있는 노드 집합에 매핑합니다.

    Returns:
        이 단계에서 업데이트된 채널의 집합입니다.
    """
    # 업데이트 적용 순서가 결정적이도록 경로별로 태스크를 정렬합니다
    # 세 번째 이후의 경로 부분은 정렬에서 무시됩니다
    # (예: 정렬에 적합하지 않은 태스크 ID에 사용합니다)
    tasks = sorted(tasks, key=lambda t: task_path_str(t.path[:3]))
    # 트리거가 있는 태스크가 없으면 null 태스크의 쓰기만 적용하는 것이므로
    # 쓰기된 채널을 업데이트하는 것 외에는 아무것도 하지 않습니다
    bump_step = any(t.triggers for t in tasks)

    # 확인된 버전을 업데이트합니다
    for task in tasks:
        checkpoint["versions_seen"].setdefault(task.name, {}).update(
            {
                chan: checkpoint["channel_versions"][chan]
                for chan in task.triggers
                if chan in checkpoint["channel_versions"]
            }
        )

    # 모든 채널의 가장 높은 버전을 찾습니다
    if get_next_version is None:
        next_version = None
    else:
        next_version = get_next_version(
            max(checkpoint["channel_versions"].values())
            if checkpoint["channel_versions"]
            else None,
            None,
        )

    # Consume all channels that were read
    for chan in {
        chan
        for task in tasks
        for chan in task.triggers
        if chan not in RESERVED and chan in channels
    }:
        if channels[chan].consume() and next_version is not None:
            checkpoint["channel_versions"][chan] = next_version

    # Group writes by channel
    pending_writes_by_channel: dict[str, list[Any]] = defaultdict(list)
    for task in tasks:
        for chan, val in task.writes:
            if chan in (NO_WRITES, PUSH, RESUME, INTERRUPT, RETURN, ERROR):
                pass
            elif chan in channels:
                pending_writes_by_channel[chan].append(val)
            else:
                logger.warning(
                    f"Task {task.name} with path {task.path} wrote to unknown channel {chan}, ignoring it."
                )

    # Apply writes to channels
    updated_channels: set[str] = set()
    for chan, vals in pending_writes_by_channel.items():
        if chan in channels:
            if channels[chan].update(vals) and next_version is not None:
                checkpoint["channel_versions"][chan] = next_version
                # unavailable channels can't trigger tasks, so don't add them
                if channels[chan].is_available():
                    updated_channels.add(chan)

    # Channels that weren't updated in this step are notified of a new step
    if bump_step:
        for chan in channels:
            if channels[chan].is_available() and chan not in updated_channels:
                if channels[chan].update(EMPTY_SEQ) and next_version is not None:
                    checkpoint["channel_versions"][chan] = next_version
                    # unavailable channels can't trigger tasks, so don't add them
                    if channels[chan].is_available():
                        updated_channels.add(chan)

    # If this is (tentatively) the last superstep, notify all channels of finish
    if bump_step and updated_channels.isdisjoint(trigger_to_nodes):
        for chan in channels:
            if channels[chan].finish() and next_version is not None:
                checkpoint["channel_versions"][chan] = next_version
                # unavailable channels can't trigger tasks, so don't add them
                if channels[chan].is_available():
                    updated_channels.add(chan)

    # Return managed values writes to be applied externally
    return updated_channels


@overload
def prepare_next_tasks(
    checkpoint: Checkpoint,
    pending_writes: list[PendingWrite],
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    config: RunnableConfig,
    step: int,
    stop: int,
    *,
    for_execution: Literal[False],
    store: Literal[None] = None,
    checkpointer: Literal[None] = None,
    manager: Literal[None] = None,
    trigger_to_nodes: Mapping[str, Sequence[str]] | None = None,
    updated_channels: set[str] | None = None,
    retry_policy: Sequence[RetryPolicy] = (),
    cache_policy: Literal[None] = None,
) -> dict[str, PregelTask]: ...


@overload
def prepare_next_tasks(
    checkpoint: Checkpoint,
    pending_writes: list[PendingWrite],
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    config: RunnableConfig,
    step: int,
    stop: int,
    *,
    for_execution: Literal[True],
    store: BaseStore | None,
    checkpointer: BaseCheckpointSaver | None,
    manager: None | ParentRunManager | AsyncParentRunManager,
    trigger_to_nodes: Mapping[str, Sequence[str]] | None = None,
    updated_channels: set[str] | None = None,
    retry_policy: Sequence[RetryPolicy] = (),
    cache_policy: CachePolicy | None = None,
) -> dict[str, PregelExecutableTask]: ...


def prepare_next_tasks(
    checkpoint: Checkpoint,
    pending_writes: list[PendingWrite],
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    config: RunnableConfig,
    step: int,
    stop: int,
    *,
    for_execution: bool,
    store: BaseStore | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    manager: None | ParentRunManager | AsyncParentRunManager = None,
    trigger_to_nodes: Mapping[str, Sequence[str]] | None = None,
    updated_channels: set[str] | None = None,
    retry_policy: Sequence[RetryPolicy] = (),
    cache_policy: CachePolicy | None = None,
) -> dict[str, PregelTask] | dict[str, PregelExecutableTask]:
    """다음 Pregel 단계를 구성할 태스크 집합을 준비합니다.

    Args:
        checkpoint: 현재 체크포인트입니다.
        pending_writes: 보류 중인 쓰기 목록입니다.
        processes: 프로세스 이름을 PregelNode 인스턴스에 매핑합니다.
        channels: 채널 이름을 BaseChannel 인스턴스에 매핑합니다.
        managed: 관리 값 이름을 함수에 매핑합니다.
        config: `Runnable` 구성입니다.
        step: 현재 단계입니다.
        for_execution: 태스크가 실행을 위해 준비되고 있는지 여부입니다.
        store: 태스크 내에서 사용할 수 있도록 하는 BaseStore의 인스턴스입니다.
        checkpointer: 체크포인트 저장에 사용되는 `Checkpointer` 인스턴스입니다.
        manager: 태스크에 사용할 부모 run manager입니다.
        trigger_to_nodes: 선택사항: 채널 이름을 해당 채널에 의해 트리거될 수 있는
            노드 집합에 매핑합니다.
        updated_channels: 선택사항. 이전 단계에서 업데이트된 채널 이름 집합입니다.
            trigger_to_nodes와 함께 사용하여 다음 단계에서 트리거되어야 하는
            노드를 결정하는 프로세스를 가속화합니다.

    Returns:
        실행할 태스크의 딕셔너리입니다. 키는 태스크 ID이고 값은
        태스크 자체입니다. 이것은 모든 PUSH 태스크(Send)와
        PULL 태스크(엣지에 의해 트리거된 노드)의 합집합입니다.
    """
    input_cache: dict[INPUT_CACHE_KEY_TYPE, Any] = {}
    checkpoint_id_bytes = binascii.unhexlify(checkpoint["id"].replace("-", ""))
    null_version = checkpoint_null_version(checkpoint)
    tasks: list[PregelTask | PregelExecutableTask] = []
    # 보류 중인 태스크를 소비합니다
    tasks_channel = cast(Topic[Send] | None, channels.get(TASKS))
    if tasks_channel and tasks_channel.is_available():
        for idx, _ in enumerate(tasks_channel.get()):
            if task := prepare_single_task(
                (PUSH, idx),
                None,
                checkpoint=checkpoint,
                checkpoint_id_bytes=checkpoint_id_bytes,
                checkpoint_null_version=null_version,
                pending_writes=pending_writes,
                processes=processes,
                channels=channels,
                managed=managed,
                config=config,
                step=step,
                stop=stop,
                for_execution=for_execution,
                store=store,
                checkpointer=checkpointer,
                manager=manager,
                input_cache=input_cache,
                cache_policy=cache_policy,
                retry_policy=retry_policy,
            ):
                tasks.append(task)

    # This section is an optimization that allows which nodes will be active
    # during the next step.
    # When there's information about:
    # 1. Which channels were updated in the previous step
    # 2. Which nodes are triggered by which channels
    # Then we can determine which nodes should be triggered in the next step
    # without having to cycle through all nodes.
    if updated_channels and trigger_to_nodes:
        triggered_nodes: set[str] = set()
        # Get all nodes that have triggers associated with an updated channel
        for channel in updated_channels:
            if node_ids := trigger_to_nodes.get(channel):
                triggered_nodes.update(node_ids)
        # Sort the nodes to ensure deterministic order
        candidate_nodes: Iterable[str] = sorted(triggered_nodes)
    elif not checkpoint["channel_versions"]:
        candidate_nodes = ()
    else:
        candidate_nodes = processes.keys()

    # Check if any processes should be run in next step
    # If so, prepare the values to be passed to them
    for name in candidate_nodes:
        if task := prepare_single_task(
            (PULL, name),
            None,
            checkpoint=checkpoint,
            checkpoint_id_bytes=checkpoint_id_bytes,
            checkpoint_null_version=null_version,
            pending_writes=pending_writes,
            processes=processes,
            channels=channels,
            managed=managed,
            config=config,
            step=step,
            stop=stop,
            for_execution=for_execution,
            store=store,
            checkpointer=checkpointer,
            manager=manager,
            input_cache=input_cache,
            cache_policy=cache_policy,
            retry_policy=retry_policy,
        ):
            tasks.append(task)
    return {t.id: t for t in tasks}


PUSH_TRIGGER = (PUSH,)


def prepare_single_task(
    task_path: tuple[Any, ...],
    task_id_checksum: str | None,
    *,
    checkpoint: Checkpoint,
    checkpoint_id_bytes: bytes,
    checkpoint_null_version: V | None,
    pending_writes: list[PendingWrite],
    processes: Mapping[str, PregelNode],
    channels: Mapping[str, BaseChannel],
    managed: ManagedValueMapping,
    config: RunnableConfig,
    step: int,
    stop: int,
    for_execution: bool,
    store: BaseStore | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    manager: None | ParentRunManager | AsyncParentRunManager = None,
    input_cache: dict[INPUT_CACHE_KEY_TYPE, Any] | None = None,
    cache_policy: CachePolicy | None = None,
    retry_policy: Sequence[RetryPolicy] = (),
) -> None | PregelTask | PregelExecutableTask:
    """그래프 내에서 PUSH 또는 PULL 태스크를 고유하게 식별하는 태스크 경로가 주어지면
    다음 Pregel 단계를 위한 단일 태스크를 준비합니다."""
    configurable = config.get(CONF, {})
    parent_ns = configurable.get(CONFIG_KEY_CHECKPOINT_NS, "")
    task_id_func = _xxhash_str if checkpoint["v"] > 1 else _uuid5_str

    if task_path[0] == PUSH and isinstance(task_path[-1], Call):
        # (PUSH, parent task path, idx of PUSH write, id of parent task, Call)
        task_path_t = cast(tuple[str, tuple, int, str, Call], task_path)
        call = task_path_t[-1]
        proc_ = get_runnable_for_task(call.func)
        name = proc_.name
        if name is None:
            raise ValueError("`call` functions must have a `__name__` attribute")
        # create task id
        triggers: Sequence[str] = PUSH_TRIGGER
        checkpoint_ns = f"{parent_ns}{NS_SEP}{name}" if parent_ns else name
        task_id = task_id_func(
            checkpoint_id_bytes,
            checkpoint_ns,
            str(step),
            name,
            PUSH,
            task_path_str(task_path[1]),
            str(task_path[2]),
        )
        task_checkpoint_ns = f"{checkpoint_ns}:{task_id}"
        # we append True to the task path to indicate that a call is being
        # made, so we should not return interrupts from this task (responsibility lies with the parent)
        task_path = (*task_path[:3], True)
        metadata = {
            "langgraph_step": step,
            "langgraph_node": name,
            "langgraph_triggers": triggers,
            "langgraph_path": task_path,
            "langgraph_checkpoint_ns": task_checkpoint_ns,
        }
        if task_id_checksum is not None:
            assert task_id == task_id_checksum, f"{task_id} != {task_id_checksum}"
        if for_execution:
            writes: deque[tuple[str, Any]] = deque()
            cache_policy = call.cache_policy or cache_policy
            if cache_policy:
                args_key = cache_policy.key_func(*call.input[0], **call.input[1])
                cache_key: CacheKey | None = CacheKey(
                    (
                        CACHE_NS_WRITES,
                        (identifier(call.func) or "__dynamic__"),
                    ),
                    xxh3_128_hexdigest(
                        args_key.encode() if isinstance(args_key, str) else args_key,
                    ),
                    cache_policy.ttl,
                )
            else:
                cache_key = None
            scratchpad = _scratchpad(
                config[CONF].get(CONFIG_KEY_SCRATCHPAD),
                pending_writes,
                task_id,
                xxh3_128_hexdigest(task_checkpoint_ns.encode()),
                config[CONF].get(CONFIG_KEY_RESUME_MAP),
                step,
                stop,
            )
            runtime = cast(
                Runtime, configurable.get(CONFIG_KEY_RUNTIME, DEFAULT_RUNTIME)
            )
            runtime = runtime.override(store=store)
            return PregelExecutableTask(
                name,
                call.input,
                proc_,
                writes,
                patch_config(
                    merge_configs(config, {"metadata": metadata}),
                    run_name=name,
                    callbacks=call.callbacks
                    or (manager.get_child(f"graph:step:{step}") if manager else None),
                    configurable={
                        CONFIG_KEY_TASK_ID: task_id,
                        # deque.extend is thread-safe
                        CONFIG_KEY_SEND: writes.extend,
                        CONFIG_KEY_READ: partial(
                            local_read,
                            scratchpad,
                            channels,
                            managed,
                            PregelTaskWrites(task_path, name, writes, triggers),
                        ),
                        CONFIG_KEY_CHECKPOINTER: (
                            checkpointer or configurable.get(CONFIG_KEY_CHECKPOINTER)
                        ),
                        CONFIG_KEY_CHECKPOINT_MAP: {
                            **configurable.get(CONFIG_KEY_CHECKPOINT_MAP, {}),
                            parent_ns: checkpoint["id"],
                        },
                        CONFIG_KEY_CHECKPOINT_ID: None,
                        CONFIG_KEY_CHECKPOINT_NS: task_checkpoint_ns,
                        CONFIG_KEY_SCRATCHPAD: scratchpad,
                        CONFIG_KEY_RUNTIME: runtime,
                    },
                ),
                triggers,
                call.retry_policy or retry_policy,
                cache_key,
                task_id,
                task_path,
            )
        else:
            return PregelTask(task_id, name, task_path)
    elif task_path[0] == PUSH:
        if len(task_path) == 2:
            # SEND tasks, executed in superstep n+1
            # (PUSH, idx of pending send)
            idx = cast(int, task_path[1])
            if not channels[TASKS].is_available():
                return
            sends: Sequence[Send] = channels[TASKS].get()
            if idx < 0 or idx >= len(sends):
                return
            packet = sends[idx]
            if not isinstance(packet, Send):
                logger.warning(
                    f"Ignoring invalid packet type {type(packet)} in pending sends"
                )
                return
            if packet.node not in processes:
                logger.warning(
                    f"Ignoring unknown node name {packet.node} in pending sends"
                )
                return
            # find process
            proc = processes[packet.node]
            proc_node = proc.node
            if proc_node is None:
                return
            # create task id
            triggers = PUSH_TRIGGER
            checkpoint_ns = (
                f"{parent_ns}{NS_SEP}{packet.node}" if parent_ns else packet.node
            )
            task_id = task_id_func(
                checkpoint_id_bytes,
                checkpoint_ns,
                str(step),
                packet.node,
                PUSH,
                str(idx),
            )
        else:
            logger.warning(f"Ignoring invalid PUSH task path {task_path}")
            return
        task_checkpoint_ns = f"{checkpoint_ns}:{task_id}"
        # we append False to the task path to indicate that a call is not being made
        # so we should return interrupts from this task
        task_path = (*task_path[:3], False)
        metadata = {
            "langgraph_step": step,
            "langgraph_node": packet.node,
            "langgraph_triggers": triggers,
            "langgraph_path": task_path,
            "langgraph_checkpoint_ns": task_checkpoint_ns,
        }
        if task_id_checksum is not None:
            assert task_id == task_id_checksum, f"{task_id} != {task_id_checksum}"
        if for_execution:
            if proc.metadata:
                metadata.update(proc.metadata)
            writes = deque()
            cache_policy = proc.cache_policy or cache_policy
            if cache_policy:
                args_key = cache_policy.key_func(packet.arg)
                cache_key = CacheKey(
                    (
                        CACHE_NS_WRITES,
                        (identifier(proc) or "__dynamic__"),
                        packet.node,
                    ),
                    xxh3_128_hexdigest(
                        args_key.encode() if isinstance(args_key, str) else args_key,
                    ),
                    cache_policy.ttl,
                )
            else:
                cache_key = None
            scratchpad = _scratchpad(
                config[CONF].get(CONFIG_KEY_SCRATCHPAD),
                pending_writes,
                task_id,
                xxh3_128_hexdigest(task_checkpoint_ns.encode()),
                config[CONF].get(CONFIG_KEY_RESUME_MAP),
                step,
                stop,
            )
            runtime = cast(
                Runtime, configurable.get(CONFIG_KEY_RUNTIME, DEFAULT_RUNTIME)
            )
            runtime = runtime.override(
                store=store, previous=checkpoint["channel_values"].get(PREVIOUS, None)
            )
            additional_config: RunnableConfig = {
                "metadata": metadata,
                "tags": proc.tags,
            }
            return PregelExecutableTask(
                packet.node,
                packet.arg,
                proc_node,
                writes,
                patch_config(
                    merge_configs(config, additional_config),
                    run_name=packet.node,
                    callbacks=(
                        manager.get_child(f"graph:step:{step}") if manager else None
                    ),
                    configurable={
                        CONFIG_KEY_TASK_ID: task_id,
                        # deque.extend is thread-safe
                        CONFIG_KEY_SEND: writes.extend,
                        CONFIG_KEY_READ: partial(
                            local_read,
                            scratchpad,
                            channels,
                            managed,
                            PregelTaskWrites(task_path, packet.node, writes, triggers),
                        ),
                        CONFIG_KEY_CHECKPOINTER: (
                            checkpointer or configurable.get(CONFIG_KEY_CHECKPOINTER)
                        ),
                        CONFIG_KEY_CHECKPOINT_MAP: {
                            **configurable.get(CONFIG_KEY_CHECKPOINT_MAP, {}),
                            parent_ns: checkpoint["id"],
                        },
                        CONFIG_KEY_CHECKPOINT_ID: None,
                        CONFIG_KEY_CHECKPOINT_NS: task_checkpoint_ns,
                        CONFIG_KEY_SCRATCHPAD: scratchpad,
                        CONFIG_KEY_RUNTIME: runtime,
                    },
                ),
                triggers,
                proc.retry_policy or retry_policy,
                cache_key,
                task_id,
                task_path,
                writers=proc.flat_writers,
                subgraphs=proc.subgraphs,
            )
        else:
            return PregelTask(task_id, packet.node, task_path)
    elif task_path[0] == PULL:
        # (PULL, node name)
        name = cast(str, task_path[1])
        if name not in processes:
            return
        proc = processes[name]
        if checkpoint_null_version is None:
            return
        # If any of the channels read by this process were updated
        if _triggers(
            channels,
            checkpoint["channel_versions"],
            checkpoint["versions_seen"].get(name),
            checkpoint_null_version,
            proc,
        ):
            triggers = tuple(sorted(proc.triggers))
            # create task id
            checkpoint_ns = f"{parent_ns}{NS_SEP}{name}" if parent_ns else name
            task_id = task_id_func(
                checkpoint_id_bytes,
                checkpoint_ns,
                str(step),
                name,
                PULL,
                *triggers,
            )
            task_checkpoint_ns = f"{checkpoint_ns}{NS_END}{task_id}"
            # create scratchpad
            scratchpad = _scratchpad(
                config[CONF].get(CONFIG_KEY_SCRATCHPAD),
                pending_writes,
                task_id,
                xxh3_128_hexdigest(task_checkpoint_ns.encode()),
                config[CONF].get(CONFIG_KEY_RESUME_MAP),
                step,
                stop,
            )
            # create task input
            try:
                val = _proc_input(
                    proc,
                    managed,
                    channels,
                    for_execution=for_execution,
                    input_cache=input_cache,
                    scratchpad=scratchpad,
                )
                if val is MISSING:
                    return
            except Exception as exc:
                if SUPPORTS_EXC_NOTES:
                    exc.add_note(
                        f"Before task with name '{name}' and path '{task_path[:3]}'"
                    )
                raise

            metadata = {
                "langgraph_step": step,
                "langgraph_node": name,
                "langgraph_triggers": triggers,
                "langgraph_path": task_path[:3],
                "langgraph_checkpoint_ns": task_checkpoint_ns,
            }
            if task_id_checksum is not None:
                assert task_id == task_id_checksum, f"{task_id} != {task_id_checksum}"
            if for_execution:
                if node := proc.node:
                    if proc.metadata:
                        metadata.update(proc.metadata)
                    writes = deque()
                    cache_policy = proc.cache_policy or cache_policy
                    if cache_policy:
                        args_key = cache_policy.key_func(val)
                        cache_key = CacheKey(
                            (
                                CACHE_NS_WRITES,
                                (identifier(proc) or "__dynamic__"),
                                name,
                            ),
                            xxh3_128_hexdigest(
                                args_key.encode()
                                if isinstance(args_key, str)
                                else args_key,
                            ),
                            cache_policy.ttl,
                        )
                    else:
                        cache_key = None
                    runtime = cast(
                        Runtime, configurable.get(CONFIG_KEY_RUNTIME, DEFAULT_RUNTIME)
                    )
                    runtime = runtime.override(
                        previous=checkpoint["channel_values"].get(PREVIOUS, None),
                        store=store,
                    )
                    additional_config = {
                        "metadata": metadata,
                        "tags": proc.tags,
                    }
                    return PregelExecutableTask(
                        name,
                        val,
                        node,
                        writes,
                        patch_config(
                            merge_configs(config, additional_config),
                            run_name=name,
                            callbacks=(
                                manager.get_child(f"graph:step:{step}")
                                if manager
                                else None
                            ),
                            configurable={
                                CONFIG_KEY_TASK_ID: task_id,
                                # deque.extend is thread-safe
                                CONFIG_KEY_SEND: writes.extend,
                                CONFIG_KEY_READ: partial(
                                    local_read,
                                    scratchpad,
                                    channels,
                                    managed,
                                    PregelTaskWrites(
                                        task_path[:3],
                                        name,
                                        writes,
                                        triggers,
                                    ),
                                ),
                                CONFIG_KEY_CHECKPOINTER: (
                                    checkpointer
                                    or configurable.get(CONFIG_KEY_CHECKPOINTER)
                                ),
                                CONFIG_KEY_CHECKPOINT_MAP: {
                                    **configurable.get(CONFIG_KEY_CHECKPOINT_MAP, {}),
                                    parent_ns: checkpoint["id"],
                                },
                                CONFIG_KEY_CHECKPOINT_ID: None,
                                CONFIG_KEY_CHECKPOINT_NS: task_checkpoint_ns,
                                CONFIG_KEY_SCRATCHPAD: scratchpad,
                                CONFIG_KEY_RUNTIME: runtime,
                            },
                        ),
                        triggers,
                        proc.retry_policy or retry_policy,
                        cache_key,
                        task_id,
                        task_path[:3],
                        writers=proc.flat_writers,
                        subgraphs=proc.subgraphs,
                    )
            else:
                return PregelTask(task_id, name, task_path[:3])


def checkpoint_null_version(
    checkpoint: Checkpoint,
) -> V | None:
    """Get the null version for the checkpoint, if available."""
    for version in checkpoint["channel_versions"].values():
        return type(version)()
    return None


def _triggers(
    channels: Mapping[str, BaseChannel],
    versions: ChannelVersions,
    seen: ChannelVersions | None,
    null_version: V,
    proc: PregelNode,
) -> bool:
    if seen is None:
        for chan in proc.triggers:
            if channels[chan].is_available():
                return True
    else:
        for chan in proc.triggers:
            if channels[chan].is_available() and versions.get(  # type: ignore[operator]
                chan, null_version
            ) > seen.get(chan, null_version):
                return True
    return False


def _scratchpad(
    parent_scratchpad: PregelScratchpad | None,
    pending_writes: list[PendingWrite],
    task_id: str,
    namespace_hash: str,
    resume_map: dict[str, Any] | None,
    step: int,
    stop: int,
) -> PregelScratchpad:
    if len(pending_writes) > 0:
        # find global resume value
        for w in pending_writes:
            if w[0] == NULL_TASK_ID and w[1] == RESUME:
                null_resume_write = w
                break
        else:
            # None cannot be used as a resume value, because it would be difficult to
            # distinguish from missing when used over http
            null_resume_write = None

        # find task-specific resume value
        for w in pending_writes:
            if w[0] == task_id and w[1] == RESUME:
                task_resume_write = w[2]
                if not isinstance(task_resume_write, list):
                    task_resume_write = [task_resume_write]
                break
        else:
            task_resume_write = []
        del w

        # find namespace and task-specific resume value
        if resume_map and namespace_hash in resume_map:
            mapped_resume_write = resume_map[namespace_hash]
            task_resume_write.append(mapped_resume_write)

    else:
        null_resume_write = None
        task_resume_write = []

    def get_null_resume(consume: bool = False) -> Any:
        if null_resume_write is None:
            if parent_scratchpad is not None:
                return parent_scratchpad.get_null_resume(consume)
            return None
        if consume:
            try:
                pending_writes.remove(null_resume_write)
                return null_resume_write[2]
            except ValueError:
                return None
        return null_resume_write[2]

    # using itertools.count as an atomic counter (+= 1 is not thread-safe)
    return PregelScratchpad(
        step=step,
        stop=stop,
        # call
        call_counter=LazyAtomicCounter(),
        # interrupt
        interrupt_counter=LazyAtomicCounter(),
        resume=task_resume_write,
        get_null_resume=get_null_resume,
        # subgraph
        subgraph_counter=LazyAtomicCounter(),
    )


def _proc_input(
    proc: PregelNode,
    managed: ManagedValueMapping,
    channels: Mapping[str, BaseChannel],
    *,
    for_execution: bool,
    scratchpad: PregelScratchpad,
    input_cache: dict[INPUT_CACHE_KEY_TYPE, Any] | None,
) -> Any:
    """Prepare input for a PULL task, based on the process's channels and triggers."""
    # if in cache return shallow copy
    if input_cache is not None and proc.input_cache_key in input_cache:
        return copy(input_cache[proc.input_cache_key])
    # If all trigger channels subscribed by this process are not empty
    # then invoke the process with the values of all non-empty channels
    if isinstance(proc.channels, list):
        val: dict[str, Any] = {}
        for chan in proc.channels:
            if chan in channels:
                if channels[chan].is_available():
                    val[chan] = channels[chan].get()
            else:
                val[chan] = managed[chan].get(scratchpad)
    elif isinstance(proc.channels, str):
        if proc.channels in channels:
            if channels[proc.channels].is_available():
                val = channels[proc.channels].get()
            else:
                return MISSING
        else:
            return MISSING
    else:
        raise RuntimeError(
            f"Invalid channels type, expected list or dict, got {proc.channels}"
        )

    # If the process has a mapper, apply it to the value
    if for_execution and proc.mapper is not None:
        val = proc.mapper(val)

    # Cache the input value
    if input_cache is not None:
        input_cache[proc.input_cache_key] = val

    return val


def _uuid5_str(namespace: bytes, *parts: str | bytes) -> str:
    """Generate a UUID from the SHA-1 hash of a namespace and str parts."""

    sha = sha1(namespace, usedforsecurity=False)
    sha.update(b"".join(p.encode() if isinstance(p, str) else p for p in parts))
    hex = sha.hexdigest()
    return f"{hex[:8]}-{hex[8:12]}-{hex[12:16]}-{hex[16:20]}-{hex[20:32]}"


def _xxhash_str(namespace: bytes, *parts: str | bytes) -> str:
    """Generate a UUID from the XXH3 hash of a namespace and str parts."""
    hex = xxh3_128_hexdigest(
        namespace + b"".join(p.encode() if isinstance(p, str) else p for p in parts)
    )
    return f"{hex[:8]}-{hex[8:12]}-{hex[12:16]}-{hex[16:20]}-{hex[20:32]}"


def task_path_str(tup: str | int | tuple) -> str:
    """Generate a string representation of the task path."""
    return (
        f"~{', '.join(task_path_str(x) for x in tup)}"
        if isinstance(tup, (tuple, list))
        else f"{tup:010d}"
        if isinstance(tup, int)
        else str(tup)
    )


LAZY_ATOMIC_COUNTER_LOCK = threading.Lock()


class LazyAtomicCounter:
    __slots__ = ("_counter",)

    _counter: Callable[[], int] | None

    def __init__(self) -> None:
        self._counter = None

    def __call__(self) -> int:
        if self._counter is None:
            with LAZY_ATOMIC_COUNTER_LOCK:
                if self._counter is None:
                    self._counter = itertools.count(0).__next__
        return self._counter()
