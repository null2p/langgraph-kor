from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_serializable_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from psycopg import AsyncConnection, AsyncCursor, AsyncPipeline, Capabilities
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from langgraph.checkpoint.postgres import _ainternal
from langgraph.checkpoint.postgres.base import BasePostgresSaver
from langgraph.checkpoint.postgres.shallow import AsyncShallowPostgresSaver

Conn = _ainternal.Conn  # 하위 호환성을 위해


class AsyncPostgresSaver(BasePostgresSaver):
    """Postgres 데이터베이스에 체크포인트를 저장하는 비동기 체크포인터입니다."""

    lock: asyncio.Lock

    def __init__(
        self,
        conn: _ainternal.Conn,
        pipe: AsyncPipeline | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        if isinstance(conn, AsyncConnectionPool) and pipe is not None:
            raise ValueError(
                "파이프라인은 AsyncConnectionPool이 아닌 단일 AsyncConnection에서만 사용해야 합니다."
            )

        self.conn = conn
        self.pipe = pipe
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
        self.supports_pipeline = Capabilities().has_pipeline()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        pipeline: bool = False,
        serde: SerializerProtocol | None = None,
    ) -> AsyncIterator[AsyncPostgresSaver]:
        """연결 문자열에서 새 AsyncPostgresSaver 인스턴스를 생성합니다.

        Args:
            conn_string: Postgres 연결 정보 문자열입니다.
            pipeline: AsyncPipeline 사용 여부입니다.

        Returns:
            AsyncPostgresSaver: 새 AsyncPostgresSaver 인스턴스입니다.
        """
        async with await AsyncConnection.connect(
            conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
        ) as conn:
            if pipeline:
                async with conn.pipeline() as pipe:
                    yield cls(conn=conn, pipe=pipe, serde=serde)
            else:
                yield cls(conn=conn, serde=serde)

    async def setup(self) -> None:
        """체크포인트 데이터베이스를 비동기적으로 설정합니다.

        이 메서드는 Postgres 데이터베이스에 필요한 테이블이 존재하지 않으면 생성하고
        데이터베이스 마이그레이션을 실행합니다. 체크포인터를 처음 사용할 때 사용자가
        직접 호출해야 합니다.
        """
        async with self._cursor() as cur:
            await cur.execute(self.MIGRATIONS[0])
            results = await cur.execute(
                "SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1"
            )
            row = await results.fetchone()
            if row is None:
                version = -1
            else:
                version = row["v"]
            for v, migration in zip(
                range(version + 1, len(self.MIGRATIONS)),
                self.MIGRATIONS[version + 1 :],
                strict=False,
            ):
                await cur.execute(migration)
                await cur.execute(f"INSERT INTO checkpoint_migrations (v) VALUES ({v})")
        if self.pipe:
            await self.pipe.sync()

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """데이터베이스에서 체크포인트를 비동기적으로 나열합니다.

        이 메서드는 제공된 config를 기반으로 Postgres 데이터베이스에서 체크포인트 튜플 리스트를
        검색합니다. 체크포인트는 체크포인트 ID 내림차순(최신 항목 먼저)으로 정렬됩니다.

        Args:
            config: 체크포인트 필터링을 위한 기본 구성입니다.
            filter: 메타데이터에 대한 추가 필터링 기준입니다.
            before: 제공되면 지정된 체크포인트 ID 이전의 체크포인트만 반환됩니다.
            limit: 반환할 체크포인트의 최대 개수입니다.

        Yields:
            일치하는 체크포인트 튜플의 비동기 반복자입니다.
        """
        where, args = self._search_where(config, filter, before)
        query = self.SELECT_SQL + where + " ORDER BY checkpoint_id DESC"
        if limit:
            query += f" LIMIT {limit}"
        # .stream()을 사용하도록 변경하는 경우 커서를 닫아야 합니다
        async with self._cursor() as cur:
            await cur.execute(query, args, binary=True)
            values = await cur.fetchall()
            if not values:
                return
            # 필요한 경우 대기 중인 전송을 마이그레이션합니다
            if to_migrate := [
                v
                for v in values
                if v["checkpoint"]["v"] < 4 and v["parent_checkpoint_id"]
            ]:
                await cur.execute(
                    self.SELECT_PENDING_SENDS_SQL,
                    (
                        values[0]["thread_id"],
                        [v["parent_checkpoint_id"] for v in to_migrate],
                    ),
                )
                grouped_by_parent = defaultdict(list)
                for value in to_migrate:
                    grouped_by_parent[value["parent_checkpoint_id"]].append(value)
                async for sends in cur:
                    for value in grouped_by_parent[sends["checkpoint_id"]]:
                        if value["channel_values"] is None:
                            value["channel_values"] = []
                        self._migrate_pending_sends(
                            sends["sends"],
                            value["checkpoint"],
                            value["channel_values"],
                        )
            for value in values:
                yield await self._load_checkpoint_tuple(value)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """데이터베이스에서 체크포인트 튜플을 비동기적으로 가져옵니다.

        이 메서드는 제공된 config를 기반으로 Postgres 데이터베이스에서 체크포인트 튜플을 검색합니다.
        config에 `checkpoint_id` 키가 포함되어 있으면 일치하는 스레드 ID와 "checkpoint_id"를 가진
        체크포인트가 검색됩니다. 그렇지 않으면 주어진 스레드 ID의 최신 체크포인트가 검색됩니다.

        Args:
            config: 체크포인트 검색에 사용할 config입니다.

        Returns:
            검색된 체크포인트 튜플, 또는 일치하는 체크포인트를 찾지 못한 경우 None입니다.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id:
            args: tuple[Any, ...] = (thread_id, checkpoint_ns, checkpoint_id)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s"
        else:
            args = (thread_id, checkpoint_ns)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s ORDER BY checkpoint_id DESC LIMIT 1"

        async with self._cursor() as cur:
            await cur.execute(
                self.SELECT_SQL + where,
                args,
                binary=True,
            )
            value = await cur.fetchone()
            if value is None:
                return None

            # 필요한 경우 대기 중인 전송을 마이그레이션합니다
            if value["checkpoint"]["v"] < 4 and value["parent_checkpoint_id"]:
                await cur.execute(
                    self.SELECT_PENDING_SENDS_SQL,
                    (thread_id, [value["parent_checkpoint_id"]]),
                )
                if sends := await cur.fetchone():
                    if value["channel_values"] is None:
                        value["channel_values"] = []
                    self._migrate_pending_sends(
                        sends["sends"],
                        value["checkpoint"],
                        value["channel_values"],
                    )

            return await self._load_checkpoint_tuple(value)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """데이터베이스에 체크포인트를 비동기적으로 저장합니다.

        이 메서드는 Postgres 데이터베이스에 체크포인트를 저장합니다. 체크포인트는
        제공된 config 및 해당 부모 config(있는 경우)와 연결됩니다.

        Args:
            config: 체크포인트와 연결할 config입니다.
            checkpoint: 저장할 체크포인트입니다.
            metadata: 체크포인트와 함께 저장할 추가 메타데이터입니다.
            new_versions: 이 쓰기 시점의 새 채널 버전입니다.

        Returns:
            RunnableConfig: 체크포인트 저장 후 업데이트된 구성입니다.
        """
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns")
        checkpoint_id = configurable.pop("checkpoint_id", None)

        copy = checkpoint.copy()
        copy["channel_values"] = copy["channel_values"].copy()
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        # 체크포인트 테이블에 기본값 인라인
        # 나머지는 blobs 테이블에 저장됩니다
        blob_values = {}
        for k, v in checkpoint["channel_values"].items():
            if v is None or isinstance(v, (str, int, float, bool)):
                pass
            else:
                blob_values[k] = copy["channel_values"].pop(k)

        async with self._cursor(pipeline=True) as cur:
            if blob_versions := {
                k: v for k, v in new_versions.items() if k in blob_values
            }:
                await cur.executemany(
                    self.UPSERT_CHECKPOINT_BLOBS_SQL,
                    await asyncio.to_thread(
                        self._dump_blobs,
                        thread_id,
                        checkpoint_ns,
                        blob_values,
                        blob_versions,
                    ),
                )
            await cur.execute(
                self.UPSERT_CHECKPOINTS_SQL,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    checkpoint_id,
                    Jsonb(copy),
                    Jsonb(get_serializable_checkpoint_metadata(config, metadata)),
                ),
            )
        return next_config

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
            writes: 저장할 쓰기 리스트로, 각각 (channel, value) 쌍입니다.
            task_id: 쓰기를 생성하는 작업의 식별자입니다.
        """
        query = (
            self.UPSERT_CHECKPOINT_WRITES_SQL
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else self.INSERT_CHECKPOINT_WRITES_SQL
        )
        params = await asyncio.to_thread(
            self._dump_writes,
            config["configurable"]["thread_id"],
            config["configurable"]["checkpoint_ns"],
            config["configurable"]["checkpoint_id"],
            task_id,
            task_path,
            writes,
        )
        async with self._cursor(pipeline=True) as cur:
            await cur.executemany(query, params)

    async def adelete_thread(self, thread_id: str) -> None:
        """스레드 ID와 연결된 모든 체크포인트 및 쓰기를 삭제합니다.

        Args:
            thread_id: 삭제할 스레드 ID입니다.

        Returns:
            None
        """
        async with self._cursor(pipeline=True) as cur:
            await cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = %s",
                (str(thread_id),),
            )
            await cur.execute(
                "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
                (str(thread_id),),
            )
            await cur.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = %s",
                (str(thread_id),),
            )

    @asynccontextmanager
    async def _cursor(
        self, *, pipeline: bool = False
    ) -> AsyncIterator[AsyncCursor[DictRow]]:
        """컨텍스트 관리자로 데이터베이스 커서를 생성합니다.

        Args:
            pipeline: 컨텍스트 관리자 내부의 DB 작업에 파이프라인을 사용할지 여부입니다.
                AsyncPostgresSaver 인스턴스가 파이프라인으로 초기화되었는지 여부와 관계없이 적용됩니다.
                파이프라인 모드가 지원되지 않으면 트랜잭션 컨텍스트 관리자를 사용하도록 대체됩니다.
        """
        async with self.lock, _ainternal.get_connection(self.conn) as conn:
            if self.pipe:
                # 파이프라인 모드의 연결은 여러 스레드/코루틴에서 동시에 사용할 수 있지만
                # 한 번에 하나의 커서만 사용할 수 있음
                
                try:
                    async with conn.cursor(binary=True, row_factory=dict_row) as cur:
                        yield cur
                finally:
                    if pipeline:
                        await self.pipe.sync()
            elif pipeline:
                # 파이프라인 모드가 아닌 연결은 한 번에 하나의 스레드/코루틴에서만
                # 사용할 수 있으므로 락을 획득함
                if self.supports_pipeline:
                    async with (
                        conn.pipeline(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
                else:
                    # 파이프라인 모드가 지원되지 않을 때 연결의 트랜잭션 컨텍스트 관리자를 사용합니다
                    async with (
                        conn.transaction(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
            else:
                async with conn.cursor(binary=True, row_factory=dict_row) as cur:
                    yield cur

    async def _load_checkpoint_tuple(self, value: DictRow) -> CheckpointTuple:
        """데이터베이스 행을 CheckpointTuple 객체로 변환합니다.

        Args:
            value: 체크포인트 데이터를 포함하는 데이터베이스의 행입니다.

        Returns:
            CheckpointTuple: 구성, 메타데이터, 부모 체크포인트(있는 경우) 및
            대기 중인 쓰기를 포함하는 체크포인트의 구조화된 표현입니다.
        """
        return CheckpointTuple(
            {
                "configurable": {
                    "thread_id": value["thread_id"],
                    "checkpoint_ns": value["checkpoint_ns"],
                    "checkpoint_id": value["checkpoint_id"],
                }
            },
            {
                **value["checkpoint"],
                "channel_values": {
                    **(value["checkpoint"].get("channel_values") or {}),
                    **self._load_blobs(value["channel_values"]),
                },
            },
            value["metadata"],
            (
                {
                    "configurable": {
                        "thread_id": value["thread_id"],
                        "checkpoint_ns": value["checkpoint_ns"],
                        "checkpoint_id": value["parent_checkpoint_id"],
                    }
                }
                if value["parent_checkpoint_id"]
                else None
            ),
            await asyncio.to_thread(self._load_writes, value["pending_writes"]),
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """데이터베이스에서 체크포인트를 나열합니다.

        이 메서드는 제공된 config를 기반으로 Postgres 데이터베이스에서 체크포인트 튜플 리스트를
        검색합니다. 체크포인트는 체크포인트 ID 내림차순(최신 항목 먼저)으로 정렬됩니다.

        Args:
            config: 체크포인트 필터링을 위한 기본 구성입니다.
            filter: 메타데이터에 대한 추가 필터링 기준입니다.
            before: 제공되면 지정된 체크포인트 ID 이전의 체크포인트만 반환됩니다.
            limit: 반환할 체크포인트의 최대 개수입니다.

        Yields:
            일치하는 체크포인트 튜플의 반복자입니다.
        """
        try:
            # 메인 스레드에 있는지 확인하고, 백그라운드 스레드만 차단할 수 있습니다
            # 오버헤드를 피하기 위해 다른 메서드에서는 확인하지 않습니다
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "AsyncPostgresSaver에 대한 동기 호출은 "
                    "다른 스레드에서만 허용됩니다. 메인 스레드에서는 비동기 인터페이스를 사용하세요. "
                    "예: `checkpointer.alist(...)` 또는 `await "
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

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """데이터베이스에서 체크포인트 튜플을 가져옵니다.

        이 메서드는 제공된 config를 기반으로 Postgres 데이터베이스에서 체크포인트 튜플을 검색합니다.
        config에 `checkpoint_id` 키가 포함되어 있으면 일치하는 스레드 ID와 "checkpoint_id"를 가진
        체크포인트가 검색됩니다. 그렇지 않으면 주어진 스레드 ID의 최신 체크포인트가 검색됩니다.

        Args:
            config: 체크포인트 검색에 사용할 config입니다.

        Returns:
            검색된 체크포인트 튜플, 또는 일치하는 체크포인트를 찾지 못한 경우 None입니다.
        """
        try:
            # 메인 스레드에 있는지 확인하고, 백그라운드 스레드만 차단할 수 있습니다
            # 오버헤드를 피하기 위해 다른 메서드에서는 확인하지 않습니다
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "AsyncPostgresSaver에 대한 동기 호출은 "
                    "다른 스레드에서만 허용됩니다. 메인 스레드에서는 비동기 인터페이스를 사용하세요. "
                    "예: `await checkpointer.aget_tuple(...)` 또는 `await "
                    "graph.ainvoke(...)`를 사용하세요."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.aget_tuple(config), self.loop
        ).result()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """데이터베이스에 체크포인트를 저장합니다.

        이 메서드는 Postgres 데이터베이스에 체크포인트를 저장합니다. 체크포인트는
        제공된 config 및 해당 부모 config(있는 경우)와 연결됩니다.

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
        """체크포인트에 연결된 중간 쓰기를 저장합니다.

        이 메서드는 체크포인트와 연결된 중간 쓰기를 데이터베이스에 저장합니다.

        Args:
            config: 관련 체크포인트의 구성입니다.
            writes: 저장할 쓰기 리스트로, 각각 (channel, value) 쌍입니다.
            task_id: 쓰기를 생성하는 작업의 식별자입니다.
            task_path: 쓰기를 생성하는 작업의 경로입니다.
        """
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
            # 메인 스레드에 있는지 확인하고, 백그라운드 스레드만 차단할 수 있습니다
            # 오버헤드를 피하기 위해 다른 메서드에서는 확인하지 않습니다
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "AsyncPostgresSaver에 대한 동기 호출은 "
                    "다른 스레드에서만 허용됩니다. 메인 스레드에서는 비동기 인터페이스를 사용하세요. "
                    "예: `await checkpointer.aget_tuple(...)` 또는 `await "
                    "graph.ainvoke(...)`를 사용하세요."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.adelete_thread(thread_id), self.loop
        ).result()


__all__ = ["AsyncPostgresSaver", "AsyncShallowPostgresSaver", "Conn"]
