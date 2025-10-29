from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, TypeVar, cast

import aiosqlite
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
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from langgraph.checkpoint.sqlite.utils import search_where

T = TypeVar("T", bound=Callable)


class AsyncSqliteSaver(BaseCheckpointSaver[str]):
    """체크포인트를 SQLite 데이터베이스에 저장하는 비동기 체크포인트 세이버입니다.

    이 클래스는 SQLite 데이터베이스를 사용하여 체크포인트를 저장하고 검색하는 비동기 인터페이스를
    제공합니다. 비동기 환경에서 사용하도록 설계되었으며 동기 대안에 비해 I/O 바운드 작업에 대한
    더 나은 성능을 제공합니다.

    Attributes:
        conn (aiosqlite.Connection): 비동기 SQLite 데이터베이스 연결입니다.
        serde (SerializerProtocol): 체크포인트 인코딩/디코딩에 사용되는 직렬화 도구입니다.

    Tip:
        [aiosqlite](https://pypi.org/project/aiosqlite/) 패키지가 필요합니다.
        `pip install aiosqlite`로 설치하세요.

    Warning:
        이 클래스는 비동기 체크포인팅을 지원하지만 SQLite의 쓰기 성능 제한으로 인해
        프로덕션 워크로드에는 권장되지 않습니다.
        프로덕션 사용에는 PostgreSQL과 같은 보다 강력한 데이터베이스를 고려하세요.

    Tip:
        코드 실행 후 **데이터베이스 연결을 닫아야** 합니다. 그렇지 않으면 그래프가
        실행 후 "멈춘" 것처럼 보일 수 있습니다(프로그램은 연결이 닫힐 때까지 종료되지 않기 때문입니다).

        가장 쉬운 방법은 예제에 표시된 대로 `async with` 문을 사용하는 것입니다.

        ```python
        async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
            # 여기에 코드 작성
            graph = builder.compile(checkpointer=saver)
            config = {"configurable": {"thread_id": "thread-1"}}
            async for event in graph.astream_events(..., config, version="v1"):
                print(event)
        ```

    Examples:
        StateGraph 내에서 사용:

        ```pycon
        >>> import asyncio
        >>>
        >>> from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        >>> from langgraph.graph import StateGraph
        >>>
        >>> async def main():
        >>>     builder = StateGraph(int)
        >>>     builder.add_node("add_one", lambda x: x + 1)
        >>>     builder.set_entry_point("add_one")
        >>>     builder.set_finish_point("add_one")
        >>>     async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as memory:
        >>>         graph = builder.compile(checkpointer=memory)
        >>>         coro = graph.ainvoke(1, {"configurable": {"thread_id": "thread-1"}})
        >>>         print(await asyncio.gather(coro))
        >>>
        >>> asyncio.run(main())
        Output: [2]
        ```
        원시 사용:

        ```pycon
        >>> import asyncio
        >>> import aiosqlite
        >>> from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        >>>
        >>> async def main():
        >>>     async with aiosqlite.connect("checkpoints.db") as conn:
        ...         saver = AsyncSqliteSaver(conn)
        ...         config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
        ...         checkpoint = {"ts": "2023-05-03T10:00:00Z", "data": {"key": "value"}, "id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"}
        ...         saved_config = await saver.aput(config, checkpoint, {}, {})
        ...         print(saved_config)
        >>> asyncio.run(main())
        {'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '0c62ca34-ac19-445d-bbb0-5b4984975b2a'}}
        ```
    """

    lock: asyncio.Lock
    is_setup: bool

    def __init__(
        self,
        conn: aiosqlite.Connection,
        *,
        serde: SerializerProtocol | None = None,
    ):
        super().__init__(serde=serde)
        self.jsonplus_serde = JsonPlusSerializer()
        self.conn = conn
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
        self.is_setup = False

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls, conn_string: str
    ) -> AsyncIterator[AsyncSqliteSaver]:
        """연결 문자열에서 새 AsyncSqliteSaver 인스턴스를 생성합니다.

        Args:
            conn_string: SQLite 연결 문자열입니다.

        Yields:
            AsyncSqliteSaver: 새 AsyncSqliteSaver 인스턴스입니다.
        """
        async with aiosqlite.connect(conn_string) as conn:
            yield cls(conn)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """데이터베이스에서 체크포인트 튜플을 가져옵니다.

        이 메서드는 제공된 config를 기반으로 SQLite 데이터베이스에서 체크포인트 튜플을 가져옵니다.
        config에 `checkpoint_id` 키가 포함되어 있으면 일치하는 스레드 ID와 체크포인트 ID를 가진
        체크포인트가 검색됩니다. 그렇지 않으면 주어진 스레드 ID에 대한 최신 체크포인트가 검색됩니다.

        Args:
            config: 체크포인트를 검색하는 데 사용할 config입니다.

        Returns:
            검색된 체크포인트 튜플이거나, 일치하는 체크포인트를 찾지 못한 경우 None입니다.
        """
        try:
            # 메인 스레드에 있는지 확인, 백그라운드 스레드만 차단 가능
            # 오버헤드를 피하기 위해 다른 메서드에서는 확인하지 않음
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "AsyncSqliteSaver에 대한 동기 호출은 다른 스레드에서만 허용됩니다. "
                    "메인 스레드에서는 비동기 인터페이스를 사용하세요. "
                    "예를 들어 `await checkpointer.aget_tuple(...)` 또는 `await "
                    "graph.ainvoke(...)`를 사용하세요."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.aget_tuple(config), self.loop
        ).result()

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """데이터베이스에서 체크포인트 목록을 비동기적으로 조회합니다.

        이 메서드는 제공된 config를 기반으로 SQLite 데이터베이스에서 체크포인트 튜플 목록을 검색합니다.
        체크포인트는 체크포인트 ID를 기준으로 내림차순(최신 우선)으로 정렬됩니다.

        Args:
            config: 체크포인트를 필터링하기 위한 기본 구성입니다.
            filter: 메타데이터에 대한 추가 필터링 기준입니다.
            before: 제공된 경우 지정된 체크포인트 ID 이전의 체크포인트만 반환됩니다.
            limit: 반환할 최대 체크포인트 수입니다.

        Yields:
            일치하는 체크포인트 튜플의 이터레이터입니다.
        """
        try:
            # 메인 스레드에 있는지 확인, 백그라운드 스레드만 차단 가능
            # 오버헤드를 피하기 위해 다른 메서드에서는 확인하지 않음
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "AsyncSqliteSaver에 대한 동기 호출은 다른 스레드에서만 허용됩니다. "
                    "메인 스레드에서는 비동기 인터페이스를 사용하세요. "
                    "예를 들어 `checkpointer.alist(...)` 또는 `await "
                    "graph.ainvoke(...)`를 사용하세요."
                )
        except RuntimeError:
            pass
        aiter_ = self.alist(config, filter=filter, before=before, limit=limit)
        while True:
            try:
                yield asyncio.run_coroutine_threadsafe(
                    anext(aiter_),  # type: ignore[arg-type]  # noqa: F821
                    self.loop,
                ).result()
            except StopAsyncIteration:
                break

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """체크포인트를 데이터베이스에 저장합니다.

        이 메서드는 체크포인트를 SQLite 데이터베이스에 저장합니다. 체크포인트는 제공된
        config 및 해당 부모 config(있는 경우)와 연결됩니다.

        Args:
            config: 체크포인트와 연결할 config입니다.
            checkpoint: 저장할 체크포인트입니다.
            metadata: 체크포인트와 함께 저장할 추가 메타데이터입니다.
            new_versions: 이 쓰기 시점의 새 채널 버전입니다.

        Returns:
            RunnableConfig: 체크포인트 저장 후 업데이트된 구성입니다.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput(config, checkpoint, metadata, new_versions), self.loop
        ).result()

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id, task_path), self.loop
        ).result()

    def delete_thread(self, thread_id: str) -> None:
        """스레드 ID와 연결된 모든 체크포인트 및 쓰기를 삭제합니다.

        Args:
            thread_id: 삭제할 스레드 ID입니다.

        Returns:
            None
        """
        try:
            # 메인 스레드에 있는지 확인, 백그라운드 스레드만 차단 가능
            # 오버헤드를 피하기 위해 다른 메서드에서는 확인하지 않음
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "AsyncSqliteSaver에 대한 동기 호출은 다른 스레드에서만 허용됩니다. "
                    "메인 스레드에서는 비동기 인터페이스를 사용하세요. "
                    "예를 들어 `checkpointer.alist(...)` 또는 `await "
                    "graph.ainvoke(...)`를 사용하세요."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.adelete_thread(thread_id), self.loop
        ).result()

    async def setup(self) -> None:
        """체크포인트 데이터베이스를 비동기적으로 설정합니다.

        이 메서드는 SQLite 데이터베이스에 필요한 테이블이 없는 경우 생성합니다.
        필요할 때 자동으로 호출되며 사용자가 직접 호출해서는 안 됩니다.
        """
        async with self.lock:
            if self.is_setup:
                return
            if not self.conn.is_alive():
                await self.conn
            async with self.conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    type TEXT,
                    checkpoint BLOB,
                    metadata BLOB,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                );
                CREATE TABLE IF NOT EXISTS writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    type TEXT,
                    value BLOB,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                );
                """
            ):
                await self.conn.commit()

            self.is_setup = True

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """데이터베이스에서 체크포인트 튜플을 비동기적으로 가져옵니다.

        이 메서드는 제공된 config를 기반으로 SQLite 데이터베이스에서 체크포인트 튜플을 가져옵니다.
        config에 `checkpoint_id` 키가 포함되어 있으면 일치하는 스레드 ID와 체크포인트 ID를 가진
        체크포인트가 검색됩니다. 그렇지 않으면 주어진 스레드 ID에 대한 최신 체크포인트가 검색됩니다.

        Args:
            config: 체크포인트를 검색하는 데 사용할 config입니다.

        Returns:
            검색된 체크포인트 튜플이거나, 일치하는 체크포인트를 찾지 못한 경우 None입니다.
        """
        await self.setup()
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        async with self.lock, self.conn.cursor() as cur:
            # thread_id에 대한 최신 체크포인트 찾기
            if checkpoint_id := get_checkpoint_id(config):
                await cur.execute(
                    "SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
                    (
                        str(config["configurable"]["thread_id"]),
                        checkpoint_ns,
                        checkpoint_id,
                    ),
                )
            else:
                await cur.execute(
                    "SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? ORDER BY checkpoint_id DESC LIMIT 1",
                    (str(config["configurable"]["thread_id"]), checkpoint_ns),
                )
            # 체크포인트가 발견되면 반환
            if value := await cur.fetchone():
                (
                    thread_id,
                    checkpoint_id,
                    parent_checkpoint_id,
                    type,
                    checkpoint,
                    metadata,
                ) = value
                if not get_checkpoint_id(config):
                    config = {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        }
                    }
                # 대기 중인 쓰기 찾기
                await cur.execute(
                    "SELECT task_id, channel, type, value FROM writes WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? ORDER BY task_id, idx",
                    (
                        str(config["configurable"]["thread_id"]),
                        checkpoint_ns,
                        str(config["configurable"]["checkpoint_id"]),
                    ),
                )
                # 체크포인트와 메타데이터 역직렬화
                return CheckpointTuple(
                    config,
                    self.serde.loads_typed((type, checkpoint)),
                    cast(
                        CheckpointMetadata,
                        (json.loads(metadata) if metadata is not None else {}),
                    ),
                    (
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
                    [
                        (task_id, channel, self.serde.loads_typed((type, value)))
                        async for task_id, channel, type, value in cur
                    ],
                )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """데이터베이스에서 체크포인트 목록을 비동기적으로 조회합니다.

        이 메서드는 제공된 config를 기반으로 SQLite 데이터베이스에서 체크포인트 튜플 목록을 검색합니다.
        체크포인트는 체크포인트 ID를 기준으로 내림차순(최신 우선)으로 정렬됩니다.

        Args:
            config: 체크포인트를 필터링하기 위한 기본 구성입니다.
            filter: 메타데이터에 대한 추가 필터링 기준입니다.
            before: 제공된 경우 지정된 체크포인트 ID 이전의 체크포인트만 반환됩니다.
            limit: 반환할 최대 체크포인트 수입니다.

        Yields:
            일치하는 체크포인트 튜플의 비동기 이터레이터입니다.
        """
        await self.setup()
        where, params = search_where(config, filter, before)
        query = f"""SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
        FROM checkpoints
        {where}
        ORDER BY checkpoint_id DESC"""
        if limit:
            query += f" LIMIT {limit}"
        async with (
            self.lock,
            self.conn.execute(query, params) as cur,
            self.conn.cursor() as wcur,
        ):
            async for (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                parent_checkpoint_id,
                type,
                checkpoint,
                metadata,
            ) in cur:
                await wcur.execute(
                    "SELECT task_id, channel, type, value FROM writes WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? ORDER BY task_id, idx",
                    (thread_id, checkpoint_ns, checkpoint_id),
                )
                yield CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        }
                    },
                    self.serde.loads_typed((type, checkpoint)),
                    cast(
                        CheckpointMetadata,
                        (json.loads(metadata) if metadata is not None else {}),
                    ),
                    (
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
                    [
                        (task_id, channel, self.serde.loads_typed((type, value)))
                        async for task_id, channel, type, value in wcur
                    ],
                )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """체크포인트를 데이터베이스에 비동기적으로 저장합니다.

        이 메서드는 체크포인트를 SQLite 데이터베이스에 저장합니다. 체크포인트는 제공된
        config 및 해당 부모 config(있는 경우)와 연결됩니다.

        Args:
            config: 체크포인트와 연결할 config입니다.
            checkpoint: 저장할 체크포인트입니다.
            metadata: 체크포인트와 함께 저장할 추가 메타데이터입니다.
            new_versions: 이 쓰기 시점의 새 채널 버전입니다.

        Returns:
            RunnableConfig: 체크포인트 저장 후 업데이트된 구성입니다.
        """
        await self.setup()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = json.dumps(
            get_checkpoint_metadata(config, metadata), ensure_ascii=False
        ).encode("utf-8", "ignore")
        async with (
            self.lock,
            self.conn.execute(
                "INSERT OR REPLACE INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(config["configurable"]["thread_id"]),
                    checkpoint_ns,
                    checkpoint["id"],
                    config["configurable"].get("checkpoint_id"),
                    type_,
                    serialized_checkpoint,
                    serialized_metadata,
                ),
            ),
        ):
            await self.conn.commit()
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """체크포인트에 연결된 중간 쓰기를 비동기적으로 저장합니다.

        이 메서드는 체크포인트와 연결된 중간 쓰기를 데이터베이스에 저장합니다.

        Args:
            config: 관련 체크포인트의 구성입니다.
            writes: 저장할 쓰기 목록이며, 각각 (channel, value) 쌍입니다.
            task_id: 쓰기를 생성하는 작업의 식별자입니다.
            task_path: 쓰기를 생성하는 작업의 경로입니다.
        """
        query = (
            "INSERT OR REPLACE INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else "INSERT OR IGNORE INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
        await self.setup()
        async with self.lock, self.conn.cursor() as cur:
            await cur.executemany(
                query,
                [
                    (
                        str(config["configurable"]["thread_id"]),
                        str(config["configurable"]["checkpoint_ns"]),
                        str(config["configurable"]["checkpoint_id"]),
                        task_id,
                        WRITES_IDX_MAP.get(channel, idx),
                        channel,
                        *self.serde.dumps_typed(value),
                    )
                    for idx, (channel, value) in enumerate(writes)
                ],
            )
            await self.conn.commit()

    async def adelete_thread(self, thread_id: str) -> None:
        """스레드 ID와 연결된 모든 체크포인트 및 쓰기를 삭제합니다.

        Args:
            thread_id: 삭제할 스레드 ID입니다.

        Returns:
            None
        """
        async with self.lock, self.conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?",
                (str(thread_id),),
            )
            await cur.execute(
                "DELETE FROM writes WHERE thread_id = ?",
                (str(thread_id),),
            )
            await self.conn.commit()

    def get_next_version(self, current: str | None, channel: None) -> str:
        """채널에 대한 다음 버전 ID를 생성합니다.

        이 메서드는 현재 버전을 기반으로 채널에 대한 새 버전 식별자를 생성합니다.

        Args:
            current (Optional[str]): 채널의 현재 버전 식별자입니다.

        Returns:
            str: 단조 증가하도록 보장된 다음 버전 식별자입니다.
        """
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"
