from __future__ import annotations

import json
import random
import sqlite3
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import closing, contextmanager
from typing import Any, cast

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

_AIO_ERROR_MSG = (
    "SqliteSaver는 비동기 메서드를 지원하지 않습니다. "
    "대신 AsyncSqliteSaver를 사용하세요.\n"
    "from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver\n"
    "참고: AsyncSqliteSaver를 사용하려면 aiosqlite 패키지가 필요합니다.\n"
    "설치 방법:\n`pip install aiosqlite`\n"
    "자세한 내용은 https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver"
    "를 참조하세요."
)


class SqliteSaver(BaseCheckpointSaver[str]):
    """체크포인트를 SQLite 데이터베이스에 저장하는 체크포인트 세이버입니다.

    참고:
        이 클래스는 가벼운 동기 사용 사례(데모 및 소규모 프로젝트)를 위한 것이며
        여러 스레드로 확장되지 않습니다.
        `async` 지원이 포함된 유사한 sqlite 세이버를 원하면
        [AsyncSqliteSaver][langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver]를 사용하세요.

    Args:
        conn (sqlite3.Connection): SQLite 데이터베이스 연결입니다.
        serde (Optional[SerializerProtocol]): 체크포인트를 직렬화 및 역직렬화하는 데 사용할 직렬화 도구입니다. 기본값은 JsonPlusSerializerCompat입니다.

    Examples:

        >>> import sqlite3
        >>> from langgraph.checkpoint.sqlite import SqliteSaver
        >>> from langgraph.graph import StateGraph
        >>>
        >>> builder = StateGraph(int)
        >>> builder.add_node("add_one", lambda x: x + 1)
        >>> builder.set_entry_point("add_one")
        >>> builder.set_finish_point("add_one")
        >>> # 새 SqliteSaver 인스턴스 생성
        >>> # 참고: check_same_thread=False는 구현이 락을 사용하므로 괜찮습니다
        >>> # 스레드 안전성을 보장합니다.
        >>> conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
        >>> memory = SqliteSaver(conn)
        >>> graph = builder.compile(checkpointer=memory)
        >>> config = {"configurable": {"thread_id": "1"}}
        >>> graph.get_state(config)
        >>> result = graph.invoke(3, config)
        >>> graph.get_state(config)
        StateSnapshot(values=4, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '0c62ca34-ac19-445d-bbb0-5b4984975b2a'}}, parent_config=None)
    """  # noqa

    conn: sqlite3.Connection
    is_setup: bool

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self.jsonplus_serde = JsonPlusSerializer()
        self.conn = conn
        self.is_setup = False
        self.lock = threading.Lock()

    @classmethod
    @contextmanager
    def from_conn_string(cls, conn_string: str) -> Iterator[SqliteSaver]:
        """연결 문자열에서 새 SqliteSaver 인스턴스를 생성합니다.

        Args:
            conn_string: SQLite 연결 문자열입니다.

        Yields:
            SqliteSaver: 새 SqliteSaver 인스턴스입니다.

        Examples:

            메모리 내:

                with SqliteSaver.from_conn_string(":memory:") as memory:
                    ...

            디스크에:

                with SqliteSaver.from_conn_string("checkpoints.sqlite") as memory:
                    ...
        """
        with closing(
            sqlite3.connect(
                conn_string,
                # https://ricardoanderegg.com/posts/python-sqlite-thread-safety/
                check_same_thread=False,
            )
        ) as conn:
            yield cls(conn)

    def setup(self) -> None:
        """체크포인트 데이터베이스를 설정합니다.

        이 메서드는 SQLite 데이터베이스에 필요한 테이블이 없는 경우 생성합니다.
        필요할 때 자동으로 호출되며 사용자가 직접 호출해서는 안 됩니다.
        """
        if self.is_setup:
            return

        self.conn.executescript(
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
        )

        self.is_setup = True

    @contextmanager
    def cursor(self, transaction: bool = True) -> Iterator[sqlite3.Cursor]:
        """SQLite 데이터베이스에 대한 커서를 가져옵니다.

        이 메서드는 SQLite 데이터베이스에 대한 커서를 반환합니다. SqliteSaver에서 내부적으로
        사용되며 사용자가 직접 호출해서는 안 됩니다.

        Args:
            transaction (bool): 커서가 닫힐 때 트랜잭션을 커밋할지 여부입니다. 기본값은 True입니다.

        Yields:
            sqlite3.Cursor: SQLite 데이터베이스에 대한 커서입니다.
        """
        with self.lock:
            self.setup()
            cur = self.conn.cursor()
            try:
                yield cur
            finally:
                if transaction:
                    self.conn.commit()
                cur.close()

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """데이터베이스에서 체크포인트 튜플을 가져옵니다.

        이 메서드는 제공된 config를 기반으로 SQLite 데이터베이스에서 체크포인트 튜플을 가져옵니다.
        config에 `checkpoint_id` 키가 포함되어 있으면 일치하는 스레드 ID와 체크포인트 ID를 가진
        체크포인트가 검색됩니다. 그렇지 않으면 주어진 스레드 ID에 대한 최신 체크포인트가 검색됩니다.

        Args:
            config: 체크포인트를 검색하는 데 사용할 config입니다.

        Returns:
            검색된 체크포인트 튜플이거나, 일치하는 체크포인트를 찾지 못한 경우 None입니다.

        Examples:

            기본:
            >>> config = {"configurable": {"thread_id": "1"}}
            >>> checkpoint_tuple = memory.get_tuple(config)
            >>> print(checkpoint_tuple)
            CheckpointTuple(...)

            체크포인트 ID 포함:

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
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        with self.cursor(transaction=False) as cur:
            # thread_id에 대한 최신 체크포인트 찾기
            if checkpoint_id := get_checkpoint_id(config):
                cur.execute(
                    "SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
                    (
                        str(config["configurable"]["thread_id"]),
                        checkpoint_ns,
                        checkpoint_id,
                    ),
                )
            else:
                cur.execute(
                    "SELECT thread_id, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? ORDER BY checkpoint_id DESC LIMIT 1",
                    (str(config["configurable"]["thread_id"]), checkpoint_ns),
                )
            # 체크포인트가 발견되면 반환
            if value := cur.fetchone():
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
                cur.execute(
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
                        json.loads(metadata) if metadata is not None else {},
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
                        for task_id, channel, type, value in cur
                    ],
                )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """데이터베이스에서 체크포인트 목록을 조회합니다.

        이 메서드는 제공된 config를 기반으로 SQLite 데이터베이스에서 체크포인트 튜플 목록을 검색합니다.
        체크포인트는 체크포인트 ID를 기준으로 내림차순(최신 우선)으로 정렬됩니다.

        Args:
            config: 체크포인트 목록을 조회하는 데 사용할 config입니다.
            filter: 메타데이터에 대한 추가 필터링 기준입니다.
            before: 제공된 경우 지정된 체크포인트 ID 이전의 체크포인트만 반환됩니다.
            limit: 반환할 최대 체크포인트 수입니다.

        Yields:
            체크포인트 튜플의 이터레이터입니다.

        Examples:
            >>> from langgraph.checkpoint.sqlite import SqliteSaver
            >>> with SqliteSaver.from_conn_string(":memory:") as memory:
            ... # 그래프를 실행한 다음 체크포인트 목록 조회
            >>>     config = {"configurable": {"thread_id": "1"}}
            >>>     checkpoints = list(memory.list(config, limit=2))
            >>> print(checkpoints)
            [CheckpointTuple(...), CheckpointTuple(...)]

            >>> config = {"configurable": {"thread_id": "1"}}
            >>> before = {"configurable": {"checkpoint_id": "1ef4f797-8335-6428-8001-8a1503f9b875"}}
            >>> with SqliteSaver.from_conn_string(":memory:") as memory:
            ... # 그래프를 실행한 다음 체크포인트 목록 조회
            >>>     checkpoints = list(memory.list(config, before=before))
            >>> print(checkpoints)
            [CheckpointTuple(...), ...]
        """
        where, param_values = search_where(config, filter, before)
        query = f"""SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata
        FROM checkpoints
        {where}
        ORDER BY checkpoint_id DESC"""
        if limit:
            query += f" LIMIT {limit}"
        with self.cursor(transaction=False) as cur, closing(self.conn.cursor()) as wcur:
            cur.execute(query, param_values)
            for (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                parent_checkpoint_id,
                type,
                checkpoint,
                metadata,
            ) in cur:
                wcur.execute(
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
                        json.loads(metadata) if metadata is not None else {},
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
                        for task_id, channel, type, value in wcur
                    ],
                )

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

        Examples:

            >>> from langgraph.checkpoint.sqlite import SqliteSaver
            >>> with SqliteSaver.from_conn_string(":memory:") as memory:
            >>>     config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
            >>>     checkpoint = {"ts": "2024-05-04T06:32:42.235444+00:00", "id": "1ef4f797-8335-6428-8001-8a1503f9b875", "channel_values": {"key": "value"}}
            >>>     saved_config = memory.put(config, checkpoint, {"source": "input", "step": 1, "writes": {"key": "value"}}, {})
            >>> print(saved_config)
            {'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef4f797-8335-6428-8001-8a1503f9b875'}}
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = json.dumps(
            get_checkpoint_metadata(config, metadata), ensure_ascii=False
        ).encode("utf-8", "ignore")
        with self.cursor() as cur:
            cur.execute(
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
        """체크포인트에 연결된 중간 쓰기를 저장합니다.

        이 메서드는 체크포인트와 연결된 중간 쓰기를 SQLite 데이터베이스에 저장합니다.

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
        with self.cursor() as cur:
            cur.executemany(
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

    def delete_thread(self, thread_id: str) -> None:
        """스레드 ID와 연결된 모든 체크포인트 및 쓰기를 삭제합니다.

        Args:
            thread_id: 삭제할 스레드 ID입니다.

        Returns:
            None
        """
        with self.cursor() as cur:
            cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?",
                (str(thread_id),),
            )
            cur.execute(
                "DELETE FROM writes WHERE thread_id = ?",
                (str(thread_id),),
            )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """데이터베이스에서 체크포인트 튜플을 비동기적으로 가져옵니다.

        참고:
            이 비동기 메서드는 SqliteSaver 클래스에서 지원되지 않습니다.
            대신 get_tuple()을 사용하거나 [AsyncSqliteSaver][langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver]를 사용하세요.
        """
        raise NotImplementedError(_AIO_ERROR_MSG)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """데이터베이스에서 체크포인트 목록을 비동기적으로 조회합니다.

        참고:
            이 비동기 메서드는 SqliteSaver 클래스에서 지원되지 않습니다.
            대신 list()를 사용하거나 [AsyncSqliteSaver][langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver]를 사용하세요.
        """
        raise NotImplementedError(_AIO_ERROR_MSG)
        yield

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """체크포인트를 데이터베이스에 비동기적으로 저장합니다.

        참고:
            이 비동기 메서드는 SqliteSaver 클래스에서 지원되지 않습니다.
            대신 put()을 사용하거나 [AsyncSqliteSaver][langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver]를 사용하세요.
        """
        raise NotImplementedError(_AIO_ERROR_MSG)

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
