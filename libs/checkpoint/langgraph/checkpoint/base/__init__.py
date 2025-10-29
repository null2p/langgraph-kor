from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import (  # noqa: UP035
    Any,
    Generic,
    Literal,
    NamedTuple,
    TypedDict,
    TypeVar,
)

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.serde.base import SerializerProtocol, maybe_add_typed_methods
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import (
    ERROR,
    INTERRUPT,
    RESUME,
    SCHEDULED,
    ChannelProtocol,
)

V = TypeVar("V", int, float, str)
PendingWrite = tuple[str, str, Any]


# 향후 확장을 허용하기 위해 total=False로 표시됩니다.
class CheckpointMetadata(TypedDict, total=False):
    """체크포인트와 관련된 메타데이터입니다."""

    source: Literal["input", "loop", "update", "fork"]
    """체크포인트의 소스입니다.

    - `"input"`: invoke/stream/batch의 입력에서 생성된 체크포인트입니다.
    - `"loop"`: pregel 루프 내부에서 생성된 체크포인트입니다.
    - `"update"`: 수동 상태 업데이트에서 생성된 체크포인트입니다.
    - `"fork"`: 다른 체크포인트의 복사본으로 생성된 체크포인트입니다.
    """
    step: int
    """체크포인트의 단계 번호입니다.

    첫 번째 `"input"` 체크포인트는 `-1`입니다.
    첫 번째 `"loop"` 체크포인트는 `0`입니다.
    이후 `n번째` 체크포인트는 `...`입니다.
    """
    parents: dict[str, str]
    """부모 체크포인트의 ID입니다.

    체크포인트 네임스페이스에서 체크포인트 ID로의 매핑입니다.
    """


ChannelVersions = dict[str, str | int | float]


class Checkpoint(TypedDict):
    """특정 시점의 상태 스냅샷입니다."""

    v: int
    """체크포인트 형식의 버전입니다. 현재는 1입니다."""
    id: str
    """체크포인트의 ID입니다. 이것은 고유하며 단조롭게 증가하므로
    체크포인트를 처음부터 끝까지 정렬하는 데 사용할 수 있습니다."""
    ts: str
    """ISO 8601 형식의 체크포인트 타임스탬프입니다."""
    channel_values: dict[str, Any]
    """체크포인트 시점의 채널 값입니다.
    채널 이름에서 역직렬화된 채널 스냅샷 값으로의 매핑입니다.
    """
    channel_versions: ChannelVersions
    """체크포인트 시점의 채널 버전입니다.
    키는 채널 이름이고 값은 각 채널에 대해 단조롭게 증가하는
    버전 문자열입니다.
    """
    versions_seen: dict[str, ChannelVersions]
    """노드 ID에서 채널 이름에서 본 버전으로의 맵입니다.
    이것은 각 노드가 본 채널의 버전을 추적합니다.
    다음에 실행할 노드를 결정하는 데 사용됩니다.
    """
    updated_channels: list[str] | None
    """이 체크포인트에서 업데이트된 채널입니다.
    """


def copy_checkpoint(checkpoint: Checkpoint) -> Checkpoint:
    return Checkpoint(
        v=checkpoint["v"],
        ts=checkpoint["ts"],
        id=checkpoint["id"],
        channel_values=checkpoint["channel_values"].copy(),
        channel_versions=checkpoint["channel_versions"].copy(),
        versions_seen={k: v.copy() for k, v in checkpoint["versions_seen"].items()},
        pending_sends=checkpoint.get("pending_sends", []).copy(),
        updated_channels=checkpoint.get("updated_channels", None),
    )


class CheckpointTuple(NamedTuple):
    """체크포인트와 관련 데이터를 포함하는 튜플입니다."""

    config: RunnableConfig
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    parent_config: RunnableConfig | None = None
    pending_writes: list[PendingWrite] | None = None


class BaseCheckpointSaver(Generic[V]):
    """그래프 체크포인터를 생성하기 위한 기본 클래스입니다.

    체크포인터는 LangGraph 에이전트가 여러 상호작용 내에서 그리고
    여러 상호작용에 걸쳐 상태를 유지할 수 있도록 합니다.

    Attributes:
        serde (SerializerProtocol): 체크포인트 인코딩/디코딩을 위한 시리얼라이저입니다.

    Note:
        사용자 정의 체크포인트 세이버를 생성할 때 메인 스레드 차단을 피하기 위해
        비동기 버전 구현을 고려하세요.
    """

    serde: SerializerProtocol = JsonPlusSerializer()

    def __init__(
        self,
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        self.serde = maybe_add_typed_methods(serde or self.serde)

    @property
    def config_specs(self) -> list:
        """체크포인트 세이버의 구성 옵션을 정의합니다.

        Returns:
            list: 구성 필드 사양 목록입니다.
        """
        return []

    def get(self, config: RunnableConfig) -> Checkpoint | None:
        """주어진 구성을 사용하여 체크포인트를 가져옵니다.

        Args:
            config: 가져올 체크포인트를 지정하는 구성입니다.

        Returns:
            요청한 체크포인트 또는 찾을 수 없으면 `None`입니다.
        """
        if value := self.get_tuple(config):
            return value.checkpoint

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """주어진 구성을 사용하여 체크포인트 튜플을 가져옵니다.

        Args:
            config: 가져올 체크포인트를 지정하는 구성입니다.

        Returns:
            요청한 체크포인트 튜플 또는 찾을 수 없으면 `None`입니다.

        Raises:
            NotImplementedError: 사용자 정의 체크포인트 세이버에서 이 메서드를 구현하세요.
        """
        raise NotImplementedError

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """주어진 기준과 일치하는 체크포인트를 나열합니다.

        Args:
            config: 체크포인트 필터링을 위한 기본 구성입니다.
            filter: 추가 필터링 기준입니다.
            before: 이 구성 이전에 생성된 체크포인트를 나열합니다.
            limit: 반환할 최대 체크포인트 수입니다.

        Returns:
            일치하는 체크포인트 튜플의 반복자입니다.

        Raises:
            NotImplementedError: 사용자 정의 체크포인트 세이버에서 이 메서드를 구현하세요.
        """
        raise NotImplementedError

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """체크포인트를 구성 및 메타데이터와 함께 저장합니다.

        Args:
            config: 체크포인트에 대한 구성입니다.
            checkpoint: 저장할 체크포인트입니다.
            metadata: 체크포인트에 대한 추가 메타데이터입니다.
            new_versions: 이 쓰기 시점의 새로운 채널 버전입니다.

        Returns:
            RunnableConfig: 체크포인트 저장 후 업데이트된 구성입니다.

        Raises:
            NotImplementedError: 사용자 정의 체크포인트 세이버에서 이 메서드를 구현하세요.
        """
        raise NotImplementedError

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """체크포인트에 연결된 중간 쓰기 작업을 저장합니다.

        Args:
            config: 관련 체크포인트의 구성입니다.
            writes: 저장할 쓰기 작업 목록입니다.
            task_id: 쓰기 작업을 생성하는 작업의 식별자입니다.
            task_path: 쓰기 작업을 생성하는 작업의 경로입니다.

        Raises:
            NotImplementedError: 사용자 정의 체크포인트 세이버에서 이 메서드를 구현하세요.
        """
        raise NotImplementedError

    def delete_thread(
        self,
        thread_id: str,
    ) -> None:
        """특정 스레드 ID와 연결된 모든 체크포인트 및 쓰기 작업을 삭제합니다.

        Args:
            thread_id: 체크포인트를 삭제할 스레드 ID입니다.
        """
        raise NotImplementedError

    async def aget(self, config: RunnableConfig) -> Checkpoint | None:
        """주어진 구성을 사용하여 체크포인트를 비동기적으로 가져옵니다.

        Args:
            config: 가져올 체크포인트를 지정하는 구성입니다.

        Returns:
            요청한 체크포인트 또는 찾을 수 없으면 `None`입니다.
        """
        if value := await self.aget_tuple(config):
            return value.checkpoint

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """주어진 구성을 사용하여 체크포인트 튜플을 비동기적으로 가져옵니다.

        Args:
            config: 가져올 체크포인트를 지정하는 구성입니다.

        Returns:
            요청한 체크포인트 튜플 또는 찾을 수 없으면 `None`입니다.

        Raises:
            NotImplementedError: 사용자 정의 체크포인트 세이버에서 이 메서드를 구현하세요.
        """
        raise NotImplementedError

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """주어진 기준과 일치하는 체크포인트를 비동기적으로 나열합니다.

        Args:
            config: 체크포인트 필터링을 위한 기본 구성입니다.
            filter: 메타데이터에 대한 추가 필터링 기준입니다.
            before: 이 구성 이전에 생성된 체크포인트를 나열합니다.
            limit: 반환할 최대 체크포인트 수입니다.

        Returns:
            일치하는 체크포인트 튜플의 비동기 반복자입니다.

        Raises:
            NotImplementedError: 사용자 정의 체크포인트 세이버에서 이 메서드를 구현하세요.
        """
        raise NotImplementedError
        yield

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """체크포인트를 구성 및 메타데이터와 함께 비동기적으로 저장합니다.

        Args:
            config: 체크포인트에 대한 구성입니다.
            checkpoint: 저장할 체크포인트입니다.
            metadata: 체크포인트에 대한 추가 메타데이터입니다.
            new_versions: 이 쓰기 시점의 새로운 채널 버전입니다.

        Returns:
            RunnableConfig: 체크포인트 저장 후 업데이트된 구성입니다.

        Raises:
            NotImplementedError: 사용자 정의 체크포인트 세이버에서 이 메서드를 구현하세요.
        """
        raise NotImplementedError

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """체크포인트에 연결된 중간 쓰기 작업을 비동기적으로 저장합니다.

        Args:
            config: 관련 체크포인트의 구성입니다.
            writes: 저장할 쓰기 작업 목록입니다.
            task_id: 쓰기 작업을 생성하는 작업의 식별자입니다.
            task_path: 쓰기 작업을 생성하는 작업의 경로입니다.

        Raises:
            NotImplementedError: 사용자 정의 체크포인트 세이버에서 이 메서드를 구현하세요.
        """
        raise NotImplementedError

    async def adelete_thread(
        self,
        thread_id: str,
    ) -> None:
        """특정 스레드 ID와 연결된 모든 체크포인트 및 쓰기 작업을 삭제합니다.

        Args:
            thread_id: 체크포인트를 삭제할 스레드 ID입니다.
        """
        raise NotImplementedError

    def get_next_version(self, current: V | None, channel: None) -> V:
        """채널의 다음 버전 ID를 생성합니다.

        기본값은 정수 버전을 사용하여 `1`씩 증가합니다. 오버라이드하는 경우
        단조롭게 증가하는 한 `str`/`int`/`float` 버전을 사용할 수 있습니다.

        Args:
            current: 현재 버전 식별자(`int`, `float` 또는 `str`)입니다.
            channel: 더 이상 사용되지 않는 인수로, 하위 호환성을 위해 유지됩니다.

        Returns:
            V: 다음 버전 식별자로, 증가해야 합니다.
        """
        if isinstance(current, str):
            raise NotImplementedError
        elif current is None:
            return 1
        else:
            return current + 1


class EmptyChannelError(Exception):
    """아직 처음으로 업데이트되지 않은 채널의 값을 가져오려고 할 때 발생합니다."""

    pass


def get_checkpoint_id(config: RunnableConfig) -> str | None:
    """체크포인트 ID를 가져옵니다."""
    return config["configurable"].get("checkpoint_id")


def get_checkpoint_metadata(
    config: RunnableConfig, metadata: CheckpointMetadata
) -> CheckpointMetadata:
    """하위 호환되는 방식으로 체크포인트 메타데이터를 가져옵니다."""
    metadata = {
        k: v.replace("\u0000", "") if isinstance(v, str) else v
        for k, v in metadata.items()
    }
    for obj in (config.get("metadata"), config.get("configurable")):
        if not obj:
            continue
        for key, v in obj.items():
            if key in metadata or key in EXCLUDED_METADATA_KEYS or key.startswith("__"):
                continue
            elif isinstance(v, str):
                metadata[key] = v.replace("\u0000", "")
            elif isinstance(v, (int, bool, float)):
                metadata[key] = v
    return metadata


def get_serializable_checkpoint_metadata(
    config: RunnableConfig, metadata: CheckpointMetadata
) -> CheckpointMetadata:
    """하위 호환되는 방식으로 체크포인트 메타데이터를 가져옵니다."""
    checkpoint_metadata = get_checkpoint_metadata(config, metadata)
    if "writes" in checkpoint_metadata:
        checkpoint_metadata.pop("writes")
    return checkpoint_metadata


"""
오류 타입에서 오류 인덱스로의 매핑입니다.
일반 쓰기는 저장되는 쓰기 목록의 인덱스에 매핑됩니다.
특수 쓰기(예: 오류)는 일반 쓰기와 충돌하지 않도록 음수 인덱스에 매핑됩니다.
각 Checkpointer 구현은 put_writes에서 이 매핑을 사용해야 합니다.
"""
WRITES_IDX_MAP = {ERROR: -1, SCHEDULED: -2, INTERRUPT: -3, RESUME: -4}

EXCLUDED_METADATA_KEYS = {
    "thread_id",
    "checkpoint_id",
    "checkpoint_ns",
    "checkpoint_map",
    "langgraph_step",
    "langgraph_node",
    "langgraph_triggers",
    "langgraph_path",
    "langgraph_checkpoint_ns",
}

# --- 아래는 과거 LangGraph 버전에서 사용된 더 이상 사용되지 않는 유틸리티입니다 ---

LATEST_VERSION = 2


def empty_checkpoint() -> Checkpoint:
    from datetime import datetime, timezone

    return Checkpoint(
        v=LATEST_VERSION,
        id=str(uuid6(clock_seq=-2)),
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values={},
        channel_versions={},
        versions_seen={},
        pending_sends=[],
        updated_channels=None,
    )


def create_checkpoint(
    checkpoint: Checkpoint,
    channels: Mapping[str, ChannelProtocol] | None,
    step: int,
    *,
    id: str | None = None,
) -> Checkpoint:
    """주어진 채널에 대한 체크포인트를 생성합니다."""
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).isoformat()
    if channels is None:
        values = checkpoint["channel_values"]
    else:
        values = {}
        for k, v in channels.items():
            if k not in checkpoint["channel_versions"]:
                continue
            try:
                values[k] = v.checkpoint()
            except EmptyChannelError:
                pass
    return Checkpoint(
        v=LATEST_VERSION,
        ts=ts,
        id=id or str(uuid6(clock_seq=step)),
        channel_values=values,
        channel_versions=checkpoint["channel_versions"],
        versions_seen=checkpoint["versions_seen"],
        pending_sends=checkpoint.get("pending_sends", []),
        updated_channels=None,
    )
