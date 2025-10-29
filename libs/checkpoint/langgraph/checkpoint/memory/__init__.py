from __future__ import annotations

import logging
import os
import pickle
import random
import shutil
from collections import defaultdict
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager, ExitStack
from types import TracebackType
from typing import Any

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

logger = logging.getLogger(__name__)


class InMemorySaver(
    BaseCheckpointSaver[str], AbstractContextManager, AbstractAsyncContextManager
):
    """메모리 내 체크포인트 세이버입니다.

    이 체크포인트 세이버는 defaultdict를 사용하여 메모리에 체크포인트를 저장합니다.

    Note:
        디버깅이나 테스트 목적으로만 `InMemorySaver`를 사용하세요.
        프로덕션 사용 사례의 경우 [langgraph-checkpoint-postgres](https://pypi.org/project/langgraph-checkpoint-postgres/)를 설치하고 `PostgresSaver` / `AsyncPostgresSaver`를 사용하는 것이 좋습니다.

        LangSmith Deployment를 사용하는 경우 체크포인터를 지정할 필요가 없습니다. 올바른 관리형 체크포인터가 자동으로 사용됩니다.

    Args:
        serde: 체크포인트 직렬화 및 역직렬화에 사용할 시리얼라이저입니다.

    Examples:

            import asyncio

            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.graph import StateGraph

            builder = StateGraph(int)
            builder.add_node("add_one", lambda x: x + 1)
            builder.set_entry_point("add_one")
            builder.set_finish_point("add_one")

            memory = InMemorySaver()
            graph = builder.compile(checkpointer=memory)
            coro = graph.ainvoke(1, {"configurable": {"thread_id": "thread-1"}})
            asyncio.run(coro)  # Output: 2
    """

    # thread ID -> checkpoint NS -> checkpoint ID -> checkpoint 매핑
    storage: defaultdict[
        str,
        dict[str, dict[str, tuple[tuple[str, bytes], tuple[str, bytes], str | None]]],
    ]
    # (thread ID, checkpoint NS, checkpoint ID) -> (task ID, write idx)
    writes: defaultdict[
        tuple[str, str, str],
        dict[tuple[str, int], tuple[str, str, tuple[str, bytes], str]],
    ]
    blobs: dict[
        tuple[
            str, str, str, str | int | float
        ],  # thread id, checkpoint ns, channel, version
        tuple[str, bytes],
    ]

    def __init__(
        self,
        *,
        serde: SerializerProtocol | None = None,
        factory: type[defaultdict] = defaultdict,
    ) -> None:
        super().__init__(serde=serde)
        self.storage = factory(lambda: defaultdict(dict))
        self.writes = factory(dict)
        self.blobs = factory()
        self.stack = ExitStack()
        if factory is not defaultdict:
            self.stack.enter_context(self.storage)  # type: ignore[arg-type]
            self.stack.enter_context(self.writes)  # type: ignore[arg-type]
            self.stack.enter_context(self.blobs)  # type: ignore[arg-type]

    def __enter__(self) -> InMemorySaver:
        return self.stack.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return self.stack.__exit__(exc_type, exc_value, traceback)

    async def __aenter__(self) -> InMemorySaver:
        return self.stack.__enter__()

    async def __aexit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> bool | None:
        return self.stack.__exit__(__exc_type, __exc_value, __traceback)

    def _load_blobs(
        self, thread_id: str, checkpoint_ns: str, versions: ChannelVersions
    ) -> dict[str, Any]:
        channel_values: dict[str, Any] = {}
        for k, v in versions.items():
            kk = (thread_id, checkpoint_ns, k, v)
            if kk in self.blobs:
                vv = self.blobs[kk]
                if vv[0] != "empty":
                    channel_values[k] = self.serde.loads_typed(vv)
        return channel_values

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """메모리 내 저장소에서 체크포인트 튜플을 가져옵니다.

        이 메서드는 제공된 config를 기반으로 메모리 내 저장소에서 체크포인트 튜플을 검색합니다.
        config에 `checkpoint_id` 키가 포함된 경우 일치하는 thread ID와 타임스탬프를 가진
        체크포인트가 검색됩니다. 그렇지 않으면 주어진 thread ID에 대한 최신 체크포인트가 검색됩니다.

        Args:
            config: 체크포인트 검색에 사용할 config입니다.

        Returns:
            검색된 체크포인트 튜플 또는 일치하는 체크포인트가 없으면 None입니다.
        """
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id := get_checkpoint_id(config):
            if saved := self.storage[thread_id][checkpoint_ns].get(checkpoint_id):
                checkpoint, metadata, parent_checkpoint_id = saved
                writes = self.writes[(thread_id, checkpoint_ns, checkpoint_id)].values()
                checkpoint_: Checkpoint = self.serde.loads_typed(checkpoint)
                return CheckpointTuple(
                    config=config,
                    checkpoint={
                        **checkpoint_,
                        "channel_values": self._load_blobs(
                            thread_id, checkpoint_ns, checkpoint_["channel_versions"]
                        ),
                    },
                    metadata=self.serde.loads_typed(metadata),
                    pending_writes=[
                        (id, c, self.serde.loads_typed(v)) for id, c, v, _ in writes
                    ],
                    parent_config=(
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": parent_checkpoint_id,
                            }
                        }
                        if parent_checkpoint_id
                        else None
                    ),
                )
        else:
            if checkpoints := self.storage[thread_id][checkpoint_ns]:
                checkpoint_id = max(checkpoints.keys())
                checkpoint, metadata, parent_checkpoint_id = checkpoints[checkpoint_id]
                writes = self.writes[(thread_id, checkpoint_ns, checkpoint_id)].values()
                checkpoint_ = self.serde.loads_typed(checkpoint)
                return CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        }
                    },
                    checkpoint={
                        **checkpoint_,
                        "channel_values": self._load_blobs(
                            thread_id, checkpoint_ns, checkpoint_["channel_versions"]
                        ),
                    },
                    metadata=self.serde.loads_typed(metadata),
                    pending_writes=[
                        (id, c, self.serde.loads_typed(v)) for id, c, v, _ in writes
                    ],
                    parent_config=(
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": parent_checkpoint_id,
                            }
                        }
                        if parent_checkpoint_id
                        else None
                    ),
                )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """메모리 내 저장소에서 체크포인트를 나열합니다.

        이 메서드는 제공된 기준을 기반으로 메모리 내 저장소에서 체크포인트 튜플 목록을 검색합니다.

        Args:
            config: 체크포인트 필터링을 위한 기본 구성입니다.
            filter: 메타데이터에 대한 추가 필터링 기준입니다.
            before: 이 구성 이전에 생성된 체크포인트를 나열합니다.
            limit: 반환할 최대 체크포인트 수입니다.

        Yields:
            일치하는 체크포인트 튜플의 반복자입니다.
        """
        thread_ids = (config["configurable"]["thread_id"],) if config else self.storage
        config_checkpoint_ns = (
            config["configurable"].get("checkpoint_ns") if config else None
        )
        config_checkpoint_id = get_checkpoint_id(config) if config else None
        for thread_id in thread_ids:
            for checkpoint_ns in self.storage[thread_id].keys():
                if (
                    config_checkpoint_ns is not None
                    and checkpoint_ns != config_checkpoint_ns
                ):
                    continue

                for checkpoint_id, (
                    checkpoint,
                    metadata_b,
                    parent_checkpoint_id,
                ) in sorted(
                    self.storage[thread_id][checkpoint_ns].items(),
                    key=lambda x: x[0],
                    reverse=True,
                ):
                    # config의 checkpoint ID로 필터링
                    if config_checkpoint_id and checkpoint_id != config_checkpoint_id:
                        continue

                    # `before` config의 checkpoint ID로 필터링
                    if (
                        before
                        and (before_checkpoint_id := get_checkpoint_id(before))
                        and checkpoint_id >= before_checkpoint_id
                    ):
                        continue

                    # 메타데이터로 필터링
                    metadata = self.serde.loads_typed(metadata_b)
                    if filter and not all(
                        query_value == metadata.get(query_key)
                        for query_key, query_value in filter.items()
                    ):
                        continue

                    # 검색 결과 제한
                    if limit is not None and limit <= 0:
                        break
                    elif limit is not None:
                        limit -= 1

                    writes = self.writes[
                        (thread_id, checkpoint_ns, checkpoint_id)
                    ].values()

                    checkpoint_: Checkpoint = self.serde.loads_typed(checkpoint)

                    yield CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": checkpoint_id,
                            }
                        },
                        checkpoint={
                            **checkpoint_,
                            "channel_values": self._load_blobs(
                                thread_id,
                                checkpoint_ns,
                                checkpoint_["channel_versions"],
                            ),
                        },
                        metadata=metadata,
                        parent_config=(
                            {
                                "configurable": {
                                    "thread_id": thread_id,
                                    "checkpoint_ns": checkpoint_ns,
                                    "checkpoint_id": parent_checkpoint_id,
                                }
                            }
                            if parent_checkpoint_id
                            else None
                        ),
                        pending_writes=[
                            (id, c, self.serde.loads_typed(v)) for id, c, v, _ in writes
                        ],
                    )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """메모리 내 저장소에 체크포인트를 저장합니다.

        이 메서드는 메모리 내 저장소에 체크포인트를 저장합니다. 체크포인트는
        제공된 config와 연결됩니다.

        Args:
            config: 체크포인트와 연결할 config입니다.
            checkpoint: 저장할 체크포인트입니다.
            metadata: 체크포인트와 함께 저장할 추가 메타데이터입니다.
            new_versions: 이 쓰기 시점의 새로운 버전입니다.

        Returns:
            RunnableConfig: 저장된 체크포인트의 타임스탬프를 포함하는 업데이트된 config입니다.
        """
        c = checkpoint.copy()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        values: dict[str, Any] = c.pop("channel_values")  # type: ignore[misc]
        for k, v in new_versions.items():
            self.blobs[(thread_id, checkpoint_ns, k, v)] = (
                self.serde.dumps_typed(values[k]) if k in values else ("empty", b"")
            )
        self.storage[thread_id][checkpoint_ns].update(
            {
                checkpoint["id"]: (
                    self.serde.dumps_typed(c),
                    self.serde.dumps_typed(get_checkpoint_metadata(config, metadata)),
                    config["configurable"].get("checkpoint_id"),  # parent
                )
            }
        )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """메모리 내 저장소에 쓰기 작업 목록을 저장합니다.

        이 메서드는 메모리 내 저장소에 쓰기 작업 목록을 저장합니다. 쓰기 작업은
        제공된 config와 연결됩니다.

        Args:
            config: 쓰기 작업과 연결할 config입니다.
            writes: 저장할 쓰기 작업입니다.
            task_id: 쓰기 작업을 생성하는 작업의 식별자입니다.
            task_path: 쓰기 작업을 생성하는 작업의 경로입니다.

        Returns:
            RunnableConfig: 저장된 쓰기 작업의 타임스탬프를 포함하는 업데이트된 config입니다.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        outer_key = (thread_id, checkpoint_ns, checkpoint_id)
        outer_writes_ = self.writes.get(outer_key)
        for idx, (c, v) in enumerate(writes):
            inner_key = (task_id, WRITES_IDX_MAP.get(c, idx))
            if inner_key[1] >= 0 and outer_writes_ and inner_key in outer_writes_:
                continue

            self.writes[outer_key][inner_key] = (
                task_id,
                c,
                self.serde.dumps_typed(v),
                task_path,
            )

    def delete_thread(self, thread_id: str) -> None:
        """thread ID와 연결된 모든 체크포인트 및 쓰기 작업을 삭제합니다.

        Args:
            thread_id: 삭제할 thread ID입니다.

        Returns:
            None
        """
        if thread_id in self.storage:
            del self.storage[thread_id]
        for k in list(self.writes.keys()):
            if k[0] == thread_id:
                del self.writes[k]
        for k in list(self.blobs.keys()):
            if k[0] == thread_id:
                del self.blobs[k]

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """비동기 버전의 `get_tuple`입니다.

        이 메서드는 asyncio를 사용하여 별도의 스레드에서 동기 메서드를 실행하는
        `get_tuple`에 대한 비동기 래퍼입니다.

        Args:
            config: 체크포인트 검색에 사용할 config입니다.

        Returns:
            검색된 체크포인트 튜플 또는 일치하는 체크포인트가 없으면 None입니다.
        """
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """비동기 버전의 `list`입니다.

        이 메서드는 asyncio를 사용하여 별도의 스레드에서 동기 메서드를 실행하는
        `list`에 대한 비동기 래퍼입니다.

        Args:
            config: 체크포인트 나열에 사용할 config입니다.

        Yields:
            체크포인트 튜플의 비동기 반복자입니다.
        """
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """비동기 버전의 `put`입니다.

        Args:
            config: 체크포인트와 연결할 config입니다.
            checkpoint: 저장할 체크포인트입니다.
            metadata: 체크포인트와 함께 저장할 추가 메타데이터입니다.
            new_versions: 이 쓰기 시점의 새로운 버전입니다.

        Returns:
            RunnableConfig: 저장된 체크포인트의 타임스탬프를 포함하는 업데이트된 config입니다.
        """
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """비동기 버전의 `put_writes`입니다.

        이 메서드는 asyncio를 사용하여 별도의 스레드에서 동기 메서드를 실행하는
        `put_writes`에 대한 비동기 래퍼입니다.

        Args:
            config: 쓰기 작업과 연결할 config입니다.
            writes: 저장할 쓰기 작업으로, 각각 (channel, value) 쌍입니다.
            task_id: 쓰기 작업을 생성하는 작업의 식별자입니다.
            task_path: 쓰기 작업을 생성하는 작업의 경로입니다.

        Returns:
            None
        """
        return self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        """thread ID와 연결된 모든 체크포인트 및 쓰기 작업을 삭제합니다.

        Args:
            thread_id: 삭제할 thread ID입니다.

        Returns:
            None
        """
        return self.delete_thread(thread_id)

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"


MemorySaver = InMemorySaver  # 하위 호환성을 위해 유지됨


class PersistentDict(defaultdict):
    """shelve 및 anydbm과 호환되는 API를 가진 영구 딕셔너리입니다.

    딕셔너리는 메모리에 보관되므로 딕셔너리 작업이 일반 딕셔너리만큼 빠르게 실행됩니다.

    디스크에 쓰기는 close 또는 sync까지 지연됩니다(gdbm의 fast 모드와 유사).

    입력 파일 형식은 자동으로 검색됩니다.
    출력 파일 형식은 pickle, json, csv 중에서 선택할 수 있습니다.
    세 가지 직렬화 형식 모두 빠른 C 구현으로 지원됩니다.

    다음에서 수정됨: https://code.activestate.com/recipes/576642-persistent-dict-with-multiple-standard-file-format/

    """

    def __init__(self, *args: Any, filename: str, **kwds: Any) -> None:
        self.flag = "c"  # r=읽기 전용, c=생성, n=새로 만들기
        self.mode = None  # None 또는 0644와 같은 8진수 트리플
        self.format = "pickle"  # 'csv', 'json', 또는 'pickle'
        self.filename = filename
        super().__init__(*args, **kwds)

    def sync(self) -> None:
        "딕셔너리를 디스크에 씁니다"
        if self.flag == "r":
            return
        tempname = self.filename + ".tmp"
        fileobj = open(tempname, "wb" if self.format == "pickle" else "w")
        try:
            self.dump(fileobj)
        except Exception:
            os.remove(tempname)
            raise
        finally:
            fileobj.close()
        shutil.move(tempname, self.filename)  # 원자적 커밋
        if self.mode is not None:
            os.chmod(self.filename, self.mode)

    def close(self) -> None:
        self.sync()
        self.clear()

    def __enter__(self) -> PersistentDict:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def dump(self, fileobj: Any) -> None:
        if self.format == "pickle":
            pickle.dump(dict(self), fileobj, 2)
        else:
            raise NotImplementedError("Unknown format: " + repr(self.format))

    def load(self) -> None:
        # 가장 제한적인 형식부터 가장 제한이 적은 형식까지 시도
        if self.flag == "n":
            return
        with open(self.filename, "rb" if self.format == "pickle" else "r") as fileobj:
            for loader in (pickle.load,):
                fileobj.seek(0)
                try:
                    return self.update(loader(fileobj))
                except EOFError:
                    return
                except Exception:
                    logger.error(f"Failed to load file: {fileobj.name}")
                    raise
            raise ValueError("File not in a supported format")
