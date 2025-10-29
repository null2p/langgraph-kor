"""LangGraph API와 상호작용하기 위한 데이터 모델입니다."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import (
    Any,
    Literal,
    NamedTuple,
    TypeAlias,
    TypedDict,
)

Json = dict[str, Any] | None
"""None이거나 문자열 키와 임의 값을 가진 딕셔너리일 수 있는 JSON과 유사한 구조를 나타냅니다."""

RunStatus = Literal["pending", "running", "error", "success", "timeout", "interrupted"]
"""
실행의 상태를 나타냅니다:
- "pending": 실행이 시작을 대기 중입니다.
- "running": 실행이 현재 실행 중입니다.
- "error": 실행이 오류를 만나 중지되었습니다.
- "success": 실행이 성공적으로 완료되었습니다.
- "timeout": 실행이 시간 제한을 초과했습니다.
- "interrupted": 실행이 수동으로 중지되거나 중단되었습니다.
"""

ThreadStatus = Literal["idle", "busy", "interrupted", "error"]
"""
스레드의 상태를 나타냅니다:
- "idle": 스레드가 현재 작업을 처리하지 않습니다.
- "busy": 스레드가 작업을 활발히 처리 중입니다.
- "interrupted": 스레드의 실행이 중단되었습니다.
- "error": 작업 처리 중 예외가 발생했습니다.
"""

ThreadStreamMode = Literal["run_modes", "lifecycle", "state_update"]
"""
스트리밍 모드를 정의합니다:
- "run_modes": 스레드의 실행과 동일한 이벤트 및 run_done 이벤트를 스트림합니다.
- "lifecycle": 실행 시작/종료 이벤트만 스트림합니다.
- "state_update": 스레드의 상태 업데이트를 스트림합니다.
"""

StreamMode = Literal[
    "values",
    "messages",
    "updates",
    "events",
    "tasks",
    "checkpoints",
    "debug",
    "custom",
    "messages-tuple",
]
"""
스트리밍 모드를 정의합니다:
- "values": 값만 스트림합니다.
- "messages": 완전한 메시지를 스트림합니다.
- "updates": 상태에 대한 업데이트를 스트림합니다.
- "events": 실행 중 발생하는 이벤트를 스트림합니다.
- "checkpoints": 생성되는 체크포인트를 스트림합니다.
- "tasks": 작업 시작 및 완료 이벤트를 스트림합니다.
- "debug": 상세한 디버그 정보를 스트림합니다.
- "custom": 커스텀 이벤트를 스트림합니다.
"""

DisconnectMode = Literal["cancel", "continue"]
"""
연결 해제 시 동작을 지정합니다:
- "cancel": 연결 해제 시 작업을 취소합니다.
- "continue": 연결이 해제되어도 작업을 계속합니다.
"""

MultitaskStrategy = Literal["reject", "interrupt", "rollback", "enqueue"]
"""
여러 작업을 처리하는 방법을 정의합니다:
- "reject": 사용 중일 때 새 작업을 거부합니다.
- "interrupt": 새 작업을 위해 현재 작업을 중단합니다.
- "rollback": 현재 작업을 롤백하고 새 작업을 시작합니다.
- "enqueue": 나중에 실행하기 위해 새 작업을 대기열에 추가합니다.
"""

OnConflictBehavior = Literal["raise", "do_nothing"]
"""
충돌 시 동작을 지정합니다:
- "raise": 충돌이 발생하면 예외를 발생시킵니다.
- "do_nothing": 충돌을 무시하고 계속합니다.
"""

OnCompletionBehavior = Literal["delete", "keep"]
"""
완료 후 액션을 정의합니다:
- "delete": 완료 후 리소스를 삭제합니다.
- "keep": 완료 후 리소스를 유지합니다.
"""

Durability = Literal["sync", "async", "exit"]
"""그래프 실행을 위한 내구성 모드입니다.
- `"sync"`: 다음 단계가 시작되기 전에 변경 사항이 동기적으로 유지됩니다.
- `"async"`: 다음 단계가 실행되는 동안 변경 사항이 비동기적으로 유지됩니다.
- `"exit"`: 그래프가 종료될 때만 변경 사항이 유지됩니다."""

All = Literal["*"]
"""와일드카드 또는 '모두' 선택자를 나타냅니다."""

IfNotExists = Literal["create", "reject"]
"""
Specifies behavior if the thread doesn't exist:
- "create": Create a new thread if it doesn't exist.
- "reject": Reject the operation if the thread doesn't exist.
"""

CancelAction = Literal["interrupt", "rollback"]
"""
Action to take when cancelling the run.
- "interrupt": Simply cancel the run.
- "rollback": Cancel the run. Then delete the run and associated checkpoints.
"""

AssistantSortBy = Literal[
    "assistant_id", "graph_id", "name", "created_at", "updated_at"
]
"""
The field to sort by.
"""

ThreadSortBy = Literal["thread_id", "status", "created_at", "updated_at"]
"""
The field to sort by.
"""

CronSortBy = Literal[
    "cron_id", "assistant_id", "thread_id", "created_at", "updated_at", "next_run_date"
]
"""
The field to sort by.
"""

SortOrder = Literal["asc", "desc"]
"""
The order to sort by.
"""

Context: TypeAlias = dict[str, Any]


class Config(TypedDict, total=False):
    """Configuration options for a call."""

    tags: list[str]
    """
    Tags for this call and any sub-calls (eg. a Chain calling an LLM).
    You can use these to filter calls.
    """

    recursion_limit: int
    """
    Maximum number of times a call can recurse. If not provided, defaults to 25.
    """

    configurable: dict[str, Any]
    """
    Runtime values for attributes previously made configurable on this Runnable,
    or sub-Runnables, through .configurable_fields() or .configurable_alternatives().
    Check .output_schema() for a description of the attributes that have been made 
    configurable.
    """


class Checkpoint(TypedDict):
    """Represents a checkpoint in the execution process."""

    thread_id: str
    """Unique identifier for the thread associated with this checkpoint."""
    checkpoint_ns: str
    """Namespace for the checkpoint; used internally to manage subgraph state."""
    checkpoint_id: str | None
    """Optional unique identifier for the checkpoint itself."""
    checkpoint_map: dict[str, Any] | None
    """Optional dictionary containing checkpoint-specific data."""


class GraphSchema(TypedDict):
    """Defines the structure and properties of a graph."""

    graph_id: str
    """The ID of the graph."""
    input_schema: dict | None
    """The schema for the graph input.
    Missing if unable to generate JSON schema from graph."""
    output_schema: dict | None
    """The schema for the graph output.
    Missing if unable to generate JSON schema from graph."""
    state_schema: dict | None
    """The schema for the graph state.
    Missing if unable to generate JSON schema from graph."""
    config_schema: dict | None
    """The schema for the graph config.
    Missing if unable to generate JSON schema from graph."""
    context_schema: dict | None
    """The schema for the graph context.
    Missing if unable to generate JSON schema from graph."""


Subgraphs = dict[str, GraphSchema]


class AssistantBase(TypedDict):
    """Base model for an assistant."""

    assistant_id: str
    """The ID of the assistant."""
    graph_id: str
    """The ID of the graph."""
    config: Config
    """The assistant config."""
    context: Context
    """The static context of the assistant."""
    created_at: datetime
    """The time the assistant was created."""
    metadata: Json
    """The assistant metadata."""
    version: int
    """The version of the assistant"""
    name: str
    """The name of the assistant"""
    description: str | None
    """The description of the assistant"""


class AssistantVersion(AssistantBase):
    """Represents a specific version of an assistant."""

    pass


class Assistant(AssistantBase):
    """Represents an assistant with additional properties."""

    updated_at: datetime
    """The last time the assistant was updated."""


class Interrupt(TypedDict):
    """Represents an interruption in the execution flow."""

    value: Any
    """The value associated with the interrupt."""
    id: str
    """The ID of the interrupt. Can be used to resume the interrupt."""


class Thread(TypedDict):
    """Represents a conversation thread."""

    thread_id: str
    """The ID of the thread."""
    created_at: datetime
    """The time the thread was created."""
    updated_at: datetime
    """The last time the thread was updated."""
    metadata: Json
    """The thread metadata."""
    status: ThreadStatus
    """The status of the thread, one of 'idle', 'busy', 'interrupted'."""
    values: Json
    """The current state of the thread."""
    interrupts: dict[str, list[Interrupt]]
    """Mapping of task ids to interrupts that were raised in that task."""


class ThreadTask(TypedDict):
    """Represents a task within a thread."""

    id: str
    name: str
    error: str | None
    interrupts: list[Interrupt]
    checkpoint: Checkpoint | None
    state: ThreadState | None
    result: dict[str, Any] | None


class ThreadState(TypedDict):
    """Represents the state of a thread."""

    values: list[dict] | dict[str, Any]
    """The state values."""
    next: Sequence[str]
    """The next nodes to execute. If empty, the thread is done until new input is 
    received."""
    checkpoint: Checkpoint
    """The ID of the checkpoint."""
    metadata: Json
    """Metadata for this state"""
    created_at: str | None
    """Timestamp of state creation"""
    parent_checkpoint: Checkpoint | None
    """The ID of the parent checkpoint. If missing, this is the root checkpoint."""
    tasks: Sequence[ThreadTask]
    """Tasks to execute in this step. If already attempted, may contain an error."""
    interrupts: list[Interrupt]
    """Interrupts which were thrown in this thread."""


class ThreadUpdateStateResponse(TypedDict):
    """Represents the response from updating a thread's state."""

    checkpoint: Checkpoint
    """Checkpoint of the latest state."""


class Run(TypedDict):
    """Represents a single execution run."""

    run_id: str
    """The ID of the run."""
    thread_id: str
    """The ID of the thread."""
    assistant_id: str
    """The assistant that was used for this run."""
    created_at: datetime
    """The time the run was created."""
    updated_at: datetime
    """The last time the run was updated."""
    status: RunStatus
    """The status of the run. One of 'pending', 'running', "error", 'success', "timeout", "interrupted"."""
    metadata: Json
    """The run metadata."""
    multitask_strategy: MultitaskStrategy
    """Strategy to handle concurrent runs on the same thread."""


class Cron(TypedDict):
    """Represents a scheduled task."""

    cron_id: str
    """The ID of the cron."""
    assistant_id: str
    """The ID of the assistant."""
    thread_id: str | None
    """The ID of the thread."""
    end_time: datetime | None
    """The end date to stop running the cron."""
    schedule: str
    """The schedule to run, cron format."""
    created_at: datetime
    """The time the cron was created."""
    updated_at: datetime
    """The last time the cron was updated."""
    payload: dict
    """The run payload to use for creating new run."""
    user_id: str | None
    """The user ID of the cron."""
    next_run_date: datetime | None
    """The next run date of the cron."""
    metadata: dict
    """The metadata of the cron."""


# Select field aliases for client-side typing of `select` parameters.
# These mirror the server's allowed field sets.

AssistantSelectField = Literal[
    "assistant_id",
    "graph_id",
    "name",
    "description",
    "config",
    "context",
    "created_at",
    "updated_at",
    "metadata",
    "version",
]

ThreadSelectField = Literal[
    "thread_id",
    "created_at",
    "updated_at",
    "metadata",
    "config",
    "context",
    "status",
    "values",
    "interrupts",
]

RunSelectField = Literal[
    "run_id",
    "thread_id",
    "assistant_id",
    "created_at",
    "updated_at",
    "status",
    "metadata",
    "kwargs",
    "multitask_strategy",
]

CronSelectField = Literal[
    "cron_id",
    "assistant_id",
    "thread_id",
    "end_time",
    "schedule",
    "created_at",
    "updated_at",
    "user_id",
    "payload",
    "next_run_date",
    "metadata",
    "now",
]

PrimitiveData = str | int | float | bool | None

QueryParamTypes = (
    Mapping[str, PrimitiveData | Sequence[PrimitiveData]]
    | list[tuple[str, PrimitiveData]]
    | tuple[tuple[str, PrimitiveData], ...]
    | str
    | bytes
)


class RunCreate(TypedDict):
    """Defines the parameters for initiating a background run."""

    thread_id: str | None
    """The identifier of the thread to run. If not provided, the run is stateless."""
    assistant_id: str
    """The identifier of the assistant to use for this run."""
    input: dict | None
    """Initial input data for the run."""
    metadata: dict | None
    """Additional metadata to associate with the run."""
    config: Config | None
    """Configuration options for the run."""
    context: Context | None
    """The static context of the run."""
    checkpoint_id: str | None
    """The identifier of a checkpoint to resume from."""
    interrupt_before: list[str] | None
    """List of node names to interrupt execution before."""
    interrupt_after: list[str] | None
    """List of node names to interrupt execution after."""
    webhook: str | None
    """URL to send webhook notifications about the run's progress."""
    multitask_strategy: MultitaskStrategy | None
    """Strategy for handling concurrent runs on the same thread."""


class Item(TypedDict):
    """Represents a single document or data entry in the graph's Store.

    Items are used to store cross-thread memories.
    """

    namespace: list[str]
    """The namespace of the item. A namespace is analogous to a document's directory."""
    key: str
    """The unique identifier of the item within its namespace.
    
    In general, keys needn't be globally unique.
    """
    value: dict[str, Any]
    """The value stored in the item. This is the document itself."""
    created_at: datetime
    """The timestamp when the item was created."""
    updated_at: datetime
    """The timestamp when the item was last updated."""


class ListNamespaceResponse(TypedDict):
    """Response structure for listing namespaces."""

    namespaces: list[list[str]]
    """A list of namespace paths, where each path is a list of strings."""


class SearchItem(Item, total=False):
    """Item with an optional relevance score from search operations.

    Attributes:
        score (Optional[float]): Relevance/similarity score. Included when
            searching a compatible store with a natural language query.
    """

    score: float | None


class SearchItemsResponse(TypedDict):
    """Response structure for searching items."""

    items: list[SearchItem]
    """A list of items matching the search criteria."""


class StreamPart(NamedTuple):
    """Represents a part of a stream response."""

    event: str
    """The type of event for this stream part."""
    data: dict
    """The data payload associated with the event."""


class Send(TypedDict):
    """Represents a message to be sent to a specific node in the graph.

    This type is used to explicitly send messages to nodes in the graph, typically
    used within Command objects to control graph execution flow.
    """

    node: str
    """The name of the target node to send the message to."""
    input: dict[str, Any] | None
    """Optional dictionary containing the input data to be passed to the node.

    If None, the node will be called with no input."""


class Command(TypedDict, total=False):
    """Represents one or more commands to control graph execution flow and state.

    This type defines the control commands that can be returned by nodes to influence
    graph execution. It lets you navigate to other nodes, update graph state,
    and resume from interruptions.
    """

    goto: Send | str | Sequence[Send | str]
    """Specifies where execution should continue. Can be:

        - A string node name to navigate to
        - A Send object to execute a node with specific input
        - A sequence of node names or Send objects to execute in order
    """
    update: dict[str, Any] | Sequence[tuple[str, Any]]
    """Updates to apply to the graph's state. Can be:

        - A dictionary of state updates to merge
        - A sequence of (key, value) tuples for ordered updates
    """
    resume: Any
    """Value to resume execution with after an interruption.
       Used in conjunction with interrupt() to implement control flow.
    """


class RunCreateMetadata(TypedDict):
    """Metadata for a run creation request."""

    run_id: str
    """The ID of the run."""

    thread_id: str | None
    """The ID of the thread."""
