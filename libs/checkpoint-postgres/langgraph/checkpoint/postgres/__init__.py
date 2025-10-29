from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
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
from psycopg import Capabilities, Connection, Cursor, Pipeline
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from langgraph.checkpoint.postgres import _internal
from langgraph.checkpoint.postgres.base import BasePostgresSaver
from langgraph.checkpoint.postgres.shallow import ShallowPostgresSaver

Conn = _internal.Conn  # 하위 호환성을 위해


class PostgresSaver(BasePostgresSaver):
    """Postgres 데이터베이스에 체크포인트를 저장하는 체크포인터입니다."""

    lock: threading.Lock

    def __init__(
        self,
        conn: _internal.Conn,
        pipe: Pipeline | None = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        if isinstance(conn, ConnectionPool) and pipe is not None:
            raise ValueError(
                "Pipeline should be used only with a single Connection, not ConnectionPool."
            )

        self.conn = conn
        self.pipe = pipe
        self.lock = threading.Lock()
        self.supports_pipeline = Capabilities().has_pipeline()

    @classmethod
    @contextmanager
    def from_conn_string(
        cls, conn_string: str, *, pipeline: bool = False
    ) -> Iterator[PostgresSaver]:
        """연결 문자열에서 새 PostgresSaver 인스턴스를 생성합니다.

        Args:
            conn_string: Postgres 연결 정보 문자열입니다.
            pipeline: Pipeline 사용 여부입니다.

        Returns:
            PostgresSaver: 새 PostgresSaver 인스턴스입니다.
        """
        with Connection.connect(
            conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
        ) as conn:
            if pipeline:
                with conn.pipeline() as pipe:
                    yield cls(conn, pipe)
            else:
                yield cls(conn)

    def setup(self) -> None:
        """체크포인트 데이터베이스를 비동기적으로 설정합니다.

        이 메서드는 Postgres 데이터베이스에 필요한 테이블이 존재하지 않으면 생성하고
        데이터베이스 마이그레이션을 실행합니다. 체크포인터를 처음 사용할 때 사용자가
        직접 호출해야 합니다.
        """
        with self._cursor() as cur:
            cur.execute(self.MIGRATIONS[0])
            results = cur.execute(
                "SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1"
            )
            row = results.fetchone()
            if row is None:
                version = -1
            else:
                version = row["v"]
            for v, migration in zip(
                range(version + 1, len(self.MIGRATIONS)),
                self.MIGRATIONS[version + 1 :],
                strict=False,
            ):
                cur.execute(migration)
                cur.execute(f"INSERT INTO checkpoint_migrations (v) VALUES ({v})")
        if self.pipe:
            self.pipe.sync()

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """데이터베이스에서 체크포인트를 나열합니다.

        이 메서드는 제공된 구성을 기반으로 Postgres 데이터베이스에서 체크포인트 튜플 목록을
        검색합니다. 체크포인트는 체크포인트 ID의 내림차순으로 정렬됩니다(최신 항목이 먼저).

        Args:
            config: 체크포인트를 나열하는 데 사용할 구성입니다.
            filter: 메타데이터에 대한 추가 필터링 기준입니다.
            before: 제공된 경우, 지정된 체크포인트 ID 이전의 체크포인트만 반환됩니다.
            limit: 반환할 최대 체크포인트 수입니다.

        Yields:
            체크포인트 튜플의 반복자입니다.

        Examples:
            >>> from langgraph.checkpoint.postgres import PostgresSaver
            >>> DB_URI = "postgres://postgres:postgres@localhost:5432/postgres?sslmode=disable"
            >>> with PostgresSaver.from_conn_string(DB_URI) as memory:
            ... # 그래프를 실행한 다음 체크포인트를 나열합니다
            >>>     config = {"configurable": {"thread_id": "1"}}
            >>>     checkpoints = list(memory.list(config, limit=2))
            >>> print(checkpoints)
            [CheckpointTuple(...), CheckpointTuple(...)]

            >>> config = {"configurable": {"thread_id": "1"}}
            >>> before = {"configurable": {"checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875"}}
            >>> with PostgresSaver.from_conn_string(DB_URI) as memory:
            ... # 그래프를 실행한 다음 체크포인트를 나열합니다
            >>>     checkpoints = list(memory.list(config, before=before))
            >>> print(checkpoints)
            [CheckpointTuple(...), ...]
        """
        where, args = self._search_where(config, filter, before)
        query = self.SELECT_SQL + where + " ORDER BY checkpoint_id DESC"
        if limit:
            query += f" LIMIT {limit}"
        # .stream()을 사용하도록 변경하면 커서를 닫아야 합니다
        with self._cursor() as cur:
            cur.execute(query, args)
            values = cur.fetchall()
            if not values:
                return
            # 필요한 경우 보류 중인 전송 마이그레이션
            if to_migrate := [
                v
                for v in values
                if v["checkpoint"]["v"] < 4 and v["parent_checkpoint_id"]
            ]:
                cur.execute(
                    self.SELECT_PENDING_SENDS_SQL,
                    (
                        values[0]["thread_id"],
                        [v["parent_checkpoint_id"] for v in to_migrate],
                    ),
                )
                grouped_by_parent = defaultdict(list)
                for value in to_migrate:
                    grouped_by_parent[value["parent_checkpoint_id"]].append(value)
                for sends in cur:
                    for value in grouped_by_parent[sends["checkpoint_id"]]:
                        if value["channel_values"] is None:
                            value["channel_values"] = []
                        self._migrate_pending_sends(
                            sends["sends"],
                            value["checkpoint"],
                            value["channel_values"],
                        )
            for value in values:
                yield self._load_checkpoint_tuple(value)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """데이터베이스에서 체크포인트 튜플을 가져옵니다.

        이 메서드는 제공된 구성을 기반으로 Postgres 데이터베이스에서 체크포인트 튜플을
        검색합니다. 구성에 `checkpoint_id` 키가 포함된 경우, 일치하는 스레드 ID와 타임스탬프를
        가진 체크포인트가 검색됩니다. 그렇지 않으면 주어진 스레드 ID에 대한 최신 체크포인트가
        검색됩니다.

        Args:
            config: 체크포인트를 검색하는 데 사용할 구성입니다.

        Returns:
            검색된 체크포인트 튜플, 또는 일치하는 체크포인트가 없으면 None입니다.

        Examples:

            Basic:
            >>> config = {"configurable": {"thread_id": "1"}}
            >>> checkpoint_tuple = memory.get_tuple(config)
            >>> print(checkpoint_tuple)
            CheckpointTuple(...)

            With timestamp:

            >>> config = {
            ...    "configurable": {
            ...        "thread_id": "1",
            ...        "checkpoint_ns": "",
            ...        "checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875",
            ...    }
            ... }
            >>> checkpoint_tuple = memory.get_tuple(config)
            >>> print(checkpoint_tuple)
            CheckpointTuple(...)
        """  # noqa
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id:
            args: tuple[Any, ...] = (thread_id, checkpoint_ns, checkpoint_id)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s"
        else:
            args = (thread_id, checkpoint_ns)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s ORDER BY checkpoint_id DESC LIMIT 1"

        with self._cursor() as cur:
            cur.execute(
                self.SELECT_SQL + where,
                args,
            )
            value = cur.fetchone()
            if value is None:
                return None

            # 필요한 경우 보류 중인 전송 마이그레이션
            if value["checkpoint"]["v"] < 4 and value["parent_checkpoint_id"]:
                cur.execute(
                    self.SELECT_PENDING_SENDS_SQL,
                    (thread_id, [value["parent_checkpoint_id"]]),
                )
                if sends := cur.fetchone():
                    if value["channel_values"] is None:
                        value["channel_values"] = []
                    self._migrate_pending_sends(
                        sends["sends"],
                        value["checkpoint"],
                        value["channel_values"],
                    )

            return self._load_checkpoint_tuple(value)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """데이터베이스에 체크포인트를 저장합니다.

        이 메서드는 Postgres 데이터베이스에 체크포인트를 저장합니다. 체크포인트는
        제공된 구성 및 해당 부모 구성(있는 경우)과 연결됩니다.

        Args:
            config: 체크포인트와 연결할 구성입니다.
            checkpoint: 저장할 체크포인트입니다.
            metadata: 체크포인트와 함께 저장할 추가 메타데이터입니다.
            new_versions: 이 쓰기 시점의 새 채널 버전입니다.

        Returns:
            RunnableConfig: 체크포인트 저장 후 업데이트된 구성입니다.

        Examples:

            >>> from langgraph.checkpoint.postgres import PostgresSaver
            >>> DB_URI = "postgres://postgres:postgres@localhost:5432/postgres?sslmode=disable"
            >>> with PostgresSaver.from_conn_string(DB_URI) as memory:
            >>>     config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
            >>>     checkpoint = {"ts": "2024-05-04T06:32:42.235444+00:00", "id": "1ef4f797-8335-6428-8001-8a1503f9b875", "channel_values": {"key": "value"}}
            >>>     saved_config = memory.put(config, checkpoint, {"source": "input", "step": 1, "writes": {"key": "value"}}, {})
            >>> print(saved_config)
            {'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef4f797-8335-6428-8001-8a1503f9b875'}}
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

        # 체크포인트 테이블에 원시 값을 인라인으로 저장
        # 나머지는 blobs 테이블에 저장
        blob_values = {}
        for k, v in checkpoint["channel_values"].items():
            if v is None or isinstance(v, (str, int, float, bool)):
                pass
            else:
                blob_values[k] = copy["channel_values"].pop(k)

        with self._cursor(pipeline=True) as cur:
            if blob_versions := {
                k: v for k, v in new_versions.items() if k in blob_values
            }:
                cur.executemany(
                    self.UPSERT_CHECKPOINT_BLOBS_SQL,
                    self._dump_blobs(
                        thread_id,
                        checkpoint_ns,
                        blob_values,
                        blob_versions,
                    ),
                )
            cur.execute(
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

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """체크포인트에 연결된 중간 쓰기를 저장합니다.

        이 메서드는 체크포인트와 연결된 중간 쓰기를 Postgres 데이터베이스에 저장합니다.

        Args:
            config: 관련 체크포인트의 구성입니다.
            writes: 저장할 쓰기 목록입니다.
            task_id: 쓰기를 생성하는 작업의 식별자입니다.
        """
        query = (
            self.UPSERT_CHECKPOINT_WRITES_SQL
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else self.INSERT_CHECKPOINT_WRITES_SQL
        )
        with self._cursor(pipeline=True) as cur:
            cur.executemany(
                query,
                self._dump_writes(
                    config["configurable"]["thread_id"],
                    config["configurable"]["checkpoint_ns"],
                    config["configurable"]["checkpoint_id"],
                    task_id,
                    task_path,
                    writes,
                ),
            )

    def delete_thread(self, thread_id: str) -> None:
        """스레드 ID와 연결된 모든 체크포인트 및 쓰기를 삭제합니다.

        Args:
            thread_id: 삭제할 스레드 ID입니다.

        Returns:
            None
        """
        with self._cursor(pipeline=True) as cur:
            cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = %s",
                (str(thread_id),),
            )
            cur.execute(
                "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
                (str(thread_id),),
            )
            cur.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = %s",
                (str(thread_id),),
            )

    @contextmanager
    def _cursor(self, *, pipeline: bool = False) -> Iterator[Cursor[DictRow]]:
        """컨텍스트 관리자로 데이터베이스 커서를 생성합니다.

        Args:
            pipeline: 컨텍스트 관리자 내부의 DB 작업에 파이프라인을 사용할지 여부입니다.
                PostgresSaver 인스턴스가 파이프라인으로 초기화되었는지 여부에 관계없이 적용됩니다.
                파이프라인 모드가 지원되지 않으면 트랜잭션 컨텍스트 관리자를 사용하도록 대체합니다.
        """
        with self.lock, _internal.get_connection(self.conn) as conn:
            if self.pipe:
                # 파이프라인 모드의 연결은 여러 스레드/코루틴에서 동시에 사용할 수 있지만,
                # 한 번에 하나의 커서만 사용할 수 있습니다
                try:
                    with conn.cursor(binary=True, row_factory=dict_row) as cur:
                        yield cur
                finally:
                    if pipeline:
                        self.pipe.sync()
            elif pipeline:
                # 파이프라인 모드가 아닌 연결은 한 번에 하나의 스레드/코루틴만 사용할 수 있으므로
                # 잠금을 획득합니다
                if self.supports_pipeline:
                    with (
                        conn.pipeline(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
                else:
                    # 파이프라인 모드가 지원되지 않을 때 연결의 트랜잭션 컨텍스트 관리자 사용
                    with (
                        conn.transaction(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
            else:
                with conn.cursor(binary=True, row_factory=dict_row) as cur:
                    yield cur

    def _load_checkpoint_tuple(self, value: DictRow) -> CheckpointTuple:
        """
        데이터베이스 행을 CheckpointTuple 객체로 변환합니다.

        Args:
            value: 체크포인트 데이터를 포함하는 데이터베이스 행입니다.

        Returns:
            CheckpointTuple: 구성, 메타데이터, 부모 체크포인트(있는 경우) 및
            보류 중인 쓰기를 포함하는 체크포인트의 구조화된 표현입니다.
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
            self._load_writes(value["pending_writes"]),
        )


__all__ = ["PostgresSaver", "BasePostgresSaver", "ShallowPostgresSaver", "Conn"]
