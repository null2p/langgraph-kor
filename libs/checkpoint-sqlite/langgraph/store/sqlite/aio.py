from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, cast

import aiosqlite
import orjson
import sqlite_vec  # type: ignore[import-untyped]
from langgraph.store.base import (
    GetOp,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchOp,
    TTLConfig,
)
from langgraph.store.base.batch import AsyncBatchedBaseStore

from langgraph.store.sqlite.base import (
    _PLACEHOLDER,
    BaseSqliteStore,
    SqliteIndexConfig,
    _decode_ns_text,
    _ensure_index_config,
    _group_ops,
    _row_to_item,
    _row_to_search_item,
)

logger = logging.getLogger(__name__)


class AsyncSqliteStore(AsyncBatchedBaseStore, BaseSqliteStore):
    """선택적 벡터 검색 기능이 있는 비동기 SQLite 기반 저장소입니다.

    이 클래스는 벡터 검색 기능을 지원하는 SQLite 데이터베이스를 사용하여
    데이터를 저장하고 검색하는 비동기 인터페이스를 제공합니다.

    Examples:
        기본 설정 및 사용법:
        ```python
        from langgraph.store.sqlite import AsyncSqliteStore

        async with AsyncSqliteStore.from_conn_string(":memory:") as store:
            await store.setup()  # 마이그레이션 실행

            # 데이터 저장 및 검색
            await store.aput(("users", "123"), "prefs", {"theme": "dark"})
            item = await store.aget(("users", "123"), "prefs")
        ```

        LangChain 임베딩을 사용한 벡터 검색:
        ```python
        from langchain_openai import OpenAIEmbeddings
        from langgraph.store.sqlite import AsyncSqliteStore

        async with AsyncSqliteStore.from_conn_string(
            ":memory:",
            index={
                "dims": 1536,
                "embed": OpenAIEmbeddings(),
                "fields": ["text"]  # 임베딩할 필드 지정
            }
        ) as store:
            await store.setup()  # 마이그레이션 한 번 실행

            # 문서 저장
            await store.aput(("docs",), "doc1", {"text": "Python tutorial"})
            await store.aput(("docs",), "doc2", {"text": "TypeScript guide"})
            await store.aput(("docs",), "doc3", {"text": "Other guide"}, index=False)  # 인덱싱 안 함

            # 유사도로 검색
            results = await store.asearch(("docs",), query="programming guides", limit=2)
        ```

    Warning:
        필요한 테이블과 인덱스를 생성하려면 첫 사용 전에 `setup()`을 호출해야 합니다.

    Note:
        이 클래스는 aiosqlite 패키지가 필요합니다. `pip install aiosqlite`로 설치하세요.
    """

    def __init__(
        self,
        conn: aiosqlite.Connection,
        *,
        deserializer: Callable[[bytes | str | orjson.Fragment], dict[str, Any]]
        | None = None,
        index: SqliteIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ):
        """비동기 SQLite 저장소를 초기화합니다.

        Args:
            conn: SQLite 데이터베이스 연결입니다.
            deserializer: 값에 대한 선택적 사용자 정의 역직렬화 함수입니다.
            index: 선택적 벡터 검색 구성입니다.
            ttl: 선택적 time-to-live 구성입니다.
        """
        super().__init__()
        self._deserializer = deserializer
        self.conn = conn
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
        self.is_setup = False
        self.index_config = index
        if self.index_config:
            self.embeddings, self.index_config = _ensure_index_config(self.index_config)
        else:
            self.embeddings = None
        self.ttl_config = ttl
        self._ttl_sweeper_task: asyncio.Task[None] | None = None
        self._ttl_stop_event = asyncio.Event()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        index: SqliteIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> AsyncIterator[AsyncSqliteStore]:
        """연결 문자열에서 새 AsyncSqliteStore 인스턴스를 생성합니다.

        Args:
            conn_string: SQLite 연결 문자열입니다.
            index: 선택적 벡터 검색 구성입니다.
            ttl: 선택적 time-to-live 구성입니다.

        Returns:
            비동기 컨텍스트 관리자로 래핑된 AsyncSqliteStore 인스턴스입니다.
        """
        async with aiosqlite.connect(conn_string, isolation_level=None) as conn:
            yield cls(conn, index=index, ttl=ttl)

    async def setup(self) -> None:
        """저장소 데이터베이스를 설정합니다.

        이 메서드는 SQLite 데이터베이스에 필요한 테이블이 없는 경우 생성하고
        데이터베이스 마이그레이션을 실행합니다. 첫 사용 전에 호출해야 합니다.
        """
        async with self.lock:
            if self.is_setup:
                return

            # 마이그레이션 테이블이 없으면 생성
            await self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS store_migrations (
                    v INTEGER PRIMARY KEY
                )
                """
            )

            # 현재 마이그레이션 버전 확인
            async with self.conn.execute(
                "SELECT v FROM store_migrations ORDER BY v DESC LIMIT 1"
            ) as cur:
                row = await cur.fetchone()
                if row is None:
                    version = -1
                else:
                    version = row[0]

            # 마이그레이션 적용
            for v, sql in enumerate(self.MIGRATIONS[version + 1 :], start=version + 1):
                await self.conn.executescript(sql)
                await self.conn.execute(
                    "INSERT INTO store_migrations (v) VALUES (?)", (v,)
                )

            # 인덱스 config가 제공된 경우 벡터 마이그레이션 적용
            if self.index_config:
                # 벡터 마이그레이션 테이블이 없으면 생성
                await self.conn.enable_load_extension(True)
                await self.conn.load_extension(sqlite_vec.loadable_path())
                await self.conn.enable_load_extension(False)
                await self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vector_migrations (
                        v INTEGER PRIMARY KEY
                    )
                    """
                )

                # 현재 벡터 마이그레이션 버전 확인
                async with self.conn.execute(
                    "SELECT v FROM vector_migrations ORDER BY v DESC LIMIT 1"
                ) as cur:
                    row = await cur.fetchone()
                    if row is None:
                        version = -1
                    else:
                        version = row[0]

                # 벡터 마이그레이션 적용
                for v, sql in enumerate(
                    self.VECTOR_MIGRATIONS[version + 1 :], start=version + 1
                ):
                    await self.conn.executescript(sql)
                    await self.conn.execute(
                        "INSERT INTO vector_migrations (v) VALUES (?)", (v,)
                    )

            self.is_setup = True

    @asynccontextmanager
    async def _cursor(
        self, *, transaction: bool = True
    ) -> AsyncIterator[aiosqlite.Cursor]:
        """SQLite 데이터베이스에 대한 커서를 가져옵니다.

        Args:
            transaction: 데이터베이스 작업에 트랜잭션을 사용할지 여부입니다.

        Yields:
            SQLite 커서 객체입니다.
        """
        if not self.is_setup:
            await self.setup()
        async with self.lock:
            if transaction:
                await self.conn.execute("BEGIN")

            async with self.conn.cursor() as cur:
                try:
                    yield cur
                finally:
                    if transaction:
                        await self.conn.execute("COMMIT")

    async def sweep_ttl(self) -> int:
        """TTL을 기반으로 만료된 저장소 항목을 삭제합니다.

        Returns:
            int: 삭제된 항목의 수입니다.
        """
        async with self._cursor() as cur:
            await cur.execute(
                """
                DELETE FROM store
                WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
                """
            )
            deleted_count = cur.rowcount
            return deleted_count

    async def start_ttl_sweeper(
        self, sweep_interval_minutes: int | None = None
    ) -> asyncio.Task[None]:
        """TTL을 기반으로 만료된 저장소 항목을 주기적으로 삭제합니다.

        Returns:
            대기하거나 취소할 수 있는 Task입니다.
        """
        if not self.ttl_config:
            return asyncio.create_task(asyncio.sleep(0))

        if self._ttl_sweeper_task is not None and not self._ttl_sweeper_task.done():
            return self._ttl_sweeper_task

        self._ttl_stop_event.clear()

        interval = float(
            sweep_interval_minutes or self.ttl_config.get("sweep_interval_minutes") or 5
        )
        logger.info(f"Starting store TTL sweeper with interval {interval} minutes")

        async def _sweep_loop() -> None:
            while not self._ttl_stop_event.is_set():
                try:
                    try:
                        await asyncio.wait_for(
                            self._ttl_stop_event.wait(),
                            timeout=interval * 60,
                        )
                        break
                    except asyncio.TimeoutError:
                        pass

                    expired_items = await self.sweep_ttl()
                    if expired_items > 0:
                        logger.info(f"Store swept {expired_items} expired items")
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.exception("Store TTL sweep iteration failed", exc_info=exc)

        task = asyncio.create_task(_sweep_loop())
        task.set_name("ttl_sweeper")
        self._ttl_sweeper_task = task
        return task

    async def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        """실행 중인 경우 TTL sweeper 작업을 중지합니다.

        Args:
            timeout: 작업이 중지될 때까지 대기할 최대 시간(초)입니다.
                `None`인 경우 무기한 대기합니다.

        Returns:
            bool: 작업이 성공적으로 중지되었거나 실행 중이 아닌 경우 True,
                작업이 중지되기 전에 타임아웃에 도달한 경우 False입니다.
        """
        if self._ttl_sweeper_task is None or self._ttl_sweeper_task.done():
            return True

        logger.info("Stopping TTL sweeper task")
        self._ttl_stop_event.set()

        if timeout is not None:
            try:
                await asyncio.wait_for(self._ttl_sweeper_task, timeout=timeout)
                success = True
            except asyncio.TimeoutError:
                success = False
        else:
            await self._ttl_sweeper_task
            success = True

        if success:
            self._ttl_sweeper_task = None
            logger.info("TTL sweeper task stopped")
        else:
            logger.warning("Timed out waiting for TTL sweeper task to stop")

        return success

    async def __aenter__(self) -> AsyncSqliteStore:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # 컨텍스트를 종료할 때 TTL sweeper 작업이 중지되도록 함
        if hasattr(self, "_ttl_sweeper_task") and self._ttl_sweeper_task is not None:
            # 작업 중지를 신호하도록 이벤트 설정
            self._ttl_stop_event.set()
            # 차단을 피하기 위해 여기서 작업 완료를 기다리지 않음
            # 작업은 자체적으로 정상적으로 정리됨

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """작업 배치를 비동기적으로 실행합니다.

        Args:
            ops: 실행할 작업의 이터러블입니다.

        Returns:
            작업 결과 목록입니다.
        """
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        async with self._cursor(transaction=True) as cur:
            if GetOp in grouped_ops:
                await self._batch_get_ops(
                    cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]), results, cur
                )

            if SearchOp in grouped_ops:
                await self._batch_search_ops(
                    cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp]),
                    results,
                    cur,
                )

            if ListNamespacesOp in grouped_ops:
                await self._batch_list_namespaces_ops(
                    cast(
                        Sequence[tuple[int, ListNamespacesOp]],
                        grouped_ops[ListNamespacesOp],
                    ),
                    results,
                    cur,
                )

            if PutOp in grouped_ops:
                await self._batch_put_ops(
                    cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp]), cur
                )

        return results

    async def _batch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
        cur: aiosqlite.Cursor,
    ) -> None:
        """배치 GET 작업을 처리합니다.

        Args:
            get_ops: GET 작업 시퀀스입니다.
            results: 결과를 저장할 리스트입니다.
            cur: 데이터베이스 커서입니다.
        """
        # 각 네임스페이스의 모든 작업을 함께 실행하기 위해 네임스페이스별로 모든 쿼리 그룹화
        namespace_queries = defaultdict(list)
        for prepared_query in self._get_batch_GET_ops_queries(get_ops):
            namespace_queries[prepared_query.namespace].append(prepared_query)

        # 각 네임스페이스의 작업 처리
        for namespace, queries in namespace_queries.items():
            # TTL 새로 고침 쿼리 먼저 실행
            for query in queries:
                if query.kind == "refresh":
                    try:
                        await cur.execute(query.query, query.params)
                    except Exception as e:
                        raise ValueError(
                            f"Error executing TTL refresh: \n{query.query}\n{query.params}\n{e}"
                        ) from e

            # Then execute GET queries and process results
            for query in queries:
                if query.kind == "get":
                    try:
                        await cur.execute(query.query, query.params)
                    except Exception as e:
                        raise ValueError(
                            f"Error executing GET query: \n{query.query}\n{query.params}\n{e}"
                        ) from e

                    rows = await cur.fetchall()
                    key_to_row = {
                        row[0]: {
                            "key": row[0],
                            "value": row[1],
                            "created_at": row[2],
                            "updated_at": row[3],
                            "expires_at": row[4] if len(row) > 4 else None,
                            "ttl_minutes": row[5] if len(row) > 5 else None,
                        }
                        for row in rows
                    }

                    # Process results for this query
                    for idx, key in query.items:
                        row = key_to_row.get(key)
                        if row:
                            results[idx] = _row_to_item(
                                namespace, row, loader=self._deserializer
                            )
                        else:
                            results[idx] = None

    async def _batch_put_ops(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
        cur: aiosqlite.Cursor,
    ) -> None:
        """배치 PUT 작업을 처리합니다.

        Args:
            put_ops: PUT 작업 시퀀스입니다.
            cur: 데이터베이스 커서입니다.
        """
        queries, embedding_request = self._prepare_batch_PUT_queries(put_ops)
        if embedding_request:
            if self.embeddings is None:
                # 위에서 embedding_request를 반환하려면 임베딩 config가 필요하므로
                # 여기에 도달하면 안 됨
                raise ValueError(
                    "Embedding configuration is required for vector operations "
                    f"(for semantic search). "
                    f"Please provide an Embeddings when initializing the {self.__class__.__name__}."
                )

            query, txt_params = embedding_request
            # 원시 텍스트를 벡터로 대체하도록 params 업데이트
            vectors = await self.embeddings.aembed_documents(
                [param[-1] for param in txt_params]
            )

            # 벡터를 SQLite 친화적 형식으로 변환
            vector_params = []
            for (ns, k, pathname, _), vector in zip(txt_params, vectors, strict=False):
                vector_params.extend(
                    [ns, k, pathname, sqlite_vec.serialize_float32(vector)]
                )

            queries.append((query, vector_params))

        for query, params in queries:
            await cur.execute(query, params)

    async def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        cur: aiosqlite.Cursor,
    ) -> None:
        """배치 SEARCH 작업을 처리합니다.

        Args:
            search_ops: SEARCH 작업 시퀀스입니다.
            results: 결과를 저장할 리스트입니다.
            cur: 데이터베이스 커서입니다.
        """
        prepared_queries, embedding_requests = self._prepare_batch_search_queries(
            search_ops
        )

        # dot_product 함수가 없으면 설정
        if embedding_requests and self.embeddings:
            vectors = await self.embeddings.aembed_documents(
                [query for _, query in embedding_requests]
            )

            for (embed_req_idx, _), embedding in zip(
                embedding_requests, vectors, strict=False
            ):
                # Find the corresponding query in prepared_queries
                # The embed_req_idx is the original index in search_ops, which should map to prepared_queries
                if embed_req_idx < len(prepared_queries):
                    _params_list: list = prepared_queries[embed_req_idx][1]
                    for i, param in enumerate(_params_list):
                        if param is _PLACEHOLDER:
                            _params_list[i] = sqlite_vec.serialize_float32(embedding)
                else:
                    logger.warning(
                        f"Embedding request index {embed_req_idx} out of bounds for prepared_queries."
                    )

        for (original_op_idx, _), (query, params, needs_refresh) in zip(
            search_ops, prepared_queries, strict=False
        ):
            await cur.execute(query, params)
            rows = await cur.fetchall()

            if needs_refresh and rows and self.ttl_config:
                keys_to_refresh = []
                for row_data in rows:
                    # Assuming row_data[0] is prefix (text), row_data[1] is key (text)
                    # These are raw text values directly from the DB.
                    keys_to_refresh.append((row_data[0], row_data[1]))

                if keys_to_refresh:
                    updates_by_prefix = defaultdict(list)
                    for prefix_text, key_text in keys_to_refresh:
                        updates_by_prefix[prefix_text].append(key_text)

                    for prefix_text, key_list in updates_by_prefix.items():
                        placeholders = ",".join(["?"] * len(key_list))
                        update_query = f"""
                            UPDATE store
                            SET expires_at = DATETIME(CURRENT_TIMESTAMP, '+' || ttl_minutes || ' minutes')
                            WHERE prefix = ? AND key IN ({placeholders}) AND ttl_minutes IS NOT NULL
                        """
                        update_params = (prefix_text, *key_list)
                        try:
                            await cur.execute(update_query, update_params)
                        except Exception as e:
                            logger.error(
                                f"Error during TTL refresh update for search: {e}"
                            )

            # Process rows into items
            if "score" in query:  # Vector search query
                items = [
                    _row_to_search_item(
                        _decode_ns_text(row[0]),  # prefix
                        {
                            "key": row[1],  # key
                            "value": row[2],  # value
                            "created_at": row[3],
                            "updated_at": row[4],
                            "expires_at": row[5] if len(row) > 5 else None,
                            "ttl_minutes": row[6] if len(row) > 6 else None,
                            "score": row[7] if len(row) > 7 else None,
                        },
                        loader=self._deserializer,
                    )
                    for row in rows
                ]
            else:  # Regular search query
                items = [
                    _row_to_search_item(
                        _decode_ns_text(row[0]),  # prefix
                        {
                            "key": row[1],  # key
                            "value": row[2],  # value
                            "created_at": row[3],
                            "updated_at": row[4],
                            "expires_at": row[5] if len(row) > 5 else None,
                            "ttl_minutes": row[6] if len(row) > 6 else None,
                        },
                        loader=self._deserializer,
                    )
                    for row in rows
                ]

            results[original_op_idx] = items

    async def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
        cur: aiosqlite.Cursor,
    ) -> None:
        """배치 LIST NAMESPACES 작업을 처리합니다.

        Args:
            list_ops: LIST NAMESPACES 작업 시퀀스입니다.
            results: 결과를 저장할 리스트입니다.
            cur: 데이터베이스 커서입니다.
        """
        queries = self._get_batch_list_namespaces_queries(list_ops)
        for (query, params), (idx, _) in zip(queries, list_ops, strict=False):
            await cur.execute(query, params)

            rows = await cur.fetchall()
            results[idx] = [_decode_ns_text(row[0]) for row in rows]
