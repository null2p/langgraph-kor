from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, cast

import orjson
from langgraph.store.base import (
    GetOp,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchOp,
)
from langgraph.store.base.batch import AsyncBatchedBaseStore
from psycopg import AsyncConnection, AsyncCursor, AsyncPipeline, Capabilities
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from langgraph.checkpoint.postgres import _ainternal
from langgraph.store.postgres.base import (
    PLACEHOLDER,
    BasePostgresStore,
    PoolConfig,
    PostgresIndexConfig,
    Row,
    TTLConfig,
    _decode_ns_bytes,
    _ensure_index_config,
    _group_ops,
    _row_to_item,
    _row_to_search_item,
)

logger = logging.getLogger(__name__)


class AsyncPostgresStore(AsyncBatchedBaseStore, BasePostgresStore[_ainternal.Conn]):
    """pgvector를 사용한 선택적 벡터 검색 기능이 있는 비동기 Postgres 기반 저장소입니다.

    !!! example "예제"
        기본 설정 및 사용:
        ```python
        from langgraph.store.postgres import AsyncPostgresStore

        conn_string = "postgresql://user:pass@localhost:5432/dbname"

        async with AsyncPostgresStore.from_conn_string(conn_string) as store:
            await store.setup()  # 마이그레이션 실행. 한 번만 수행

            # 데이터 저장 및 검색
            await store.aput(("users", "123"), "prefs", {"theme": "dark"})
            item = await store.aget(("users", "123"), "prefs")
        ```

        LangChain 임베딩을 사용한 벡터 검색:
        ```python
        from langchain.embeddings import init_embeddings
        from langgraph.store.postgres import AsyncPostgresStore

        conn_string = "postgresql://user:pass@localhost:5432/dbname"

        async with AsyncPostgresStore.from_conn_string(
            conn_string,
            index={
                "dims": 1536,
                "embed": init_embeddings("openai:text-embedding-3-small"),
                "fields": ["text"]  # 임베드할 필드 지정. 기본값은 전체 직렬화된 값
            }
        ) as store:
            await store.setup()  # 마이그레이션 실행. 한 번만 수행

            # 문서 저장
            await store.aput(("docs",), "doc1", {"text": "Python tutorial"})
            await store.aput(("docs",), "doc2", {"text": "TypeScript guide"})
            await store.aput(("docs",), "doc3", {"text": "Other guide"}, index=False)  # 인덱싱하지 않음

            # 유사도로 검색
            results = await store.asearch(("docs",), query="programming guides", limit=2)
        ```

        더 나은 성능을 위한 연결 풀링 사용:
        ```python
        from langgraph.store.postgres import AsyncPostgresStore, PoolConfig

        conn_string = "postgresql://user:pass@localhost:5432/dbname"

        async with AsyncPostgresStore.from_conn_string(
            conn_string,
            pool_config=PoolConfig(
                min_size=5,
                max_size=20
            )
        ) as store:
            await store.setup()  # 마이그레이션 실행. 한 번만 수행
            # 연결 풀링과 함께 저장소 사용...
        ```

    Warning:
        다음을 확인하세요:
        1. 필요한 테이블과 인덱스를 생성하기 위해 처음 사용하기 전에 `setup()`을 호출
        2. 벡터 검색을 사용하려면 pgvector 확장이 사용 가능해야 함
        3. 비동기 기능을 위해 Python 3.10+ 사용

    Note:
        의미론적 검색은 기본적으로 비활성화되어 있습니다. 저장소를 생성할 때 `index` 구성을
        제공하여 활성화할 수 있습니다. 이 구성이 없으면 `put` 또는 `aput`에 전달된 모든
        `index` 인수는 효과가 없습니다.

    Note:
        TTL 구성을 제공하는 경우, 만료된 항목을 제거하는 백그라운드 작업을 시작하려면
        명시적으로 `start_ttl_sweeper()`를 호출해야 합니다. 저장소 사용이 끝나면
        `stop_ttl_sweeper()`를 호출하여 리소스를 적절히 정리하세요.
    """

    __slots__ = (
        "_deserializer",
        "pipe",
        "lock",
        "supports_pipeline",
        "index_config",
        "embeddings",
        "ttl_config",
        "_ttl_sweeper_task",
        "_ttl_stop_event",
    )
    supports_ttl: bool = True

    def __init__(
        self,
        conn: _ainternal.Conn,
        *,
        pipe: AsyncPipeline | None = None,
        deserializer: Callable[[bytes | orjson.Fragment], dict[str, Any]] | None = None,
        index: PostgresIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> None:
        if isinstance(conn, AsyncConnectionPool) and pipe is not None:
            raise ValueError(
                "파이프라인은 AsyncConnectionPool이 아닌 단일 AsyncConnection에서만 사용해야 합니다."
            )
        super().__init__()
        self._deserializer = deserializer
        self.conn = conn
        self.pipe = pipe
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()
        self.supports_pipeline = Capabilities().has_pipeline()
        self.index_config = index
        if self.index_config:
            self.embeddings, self.index_config = _ensure_index_config(self.index_config)
        else:
            self.embeddings = None

        self.ttl_config = ttl
        self._ttl_sweeper_task: asyncio.Task[None] | None = None
        self._ttl_stop_event = asyncio.Event()

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """작업 배치를 비동기적으로 실행합니다.

        Args:
            ops: 실행할 작업들의 iterable입니다.

        Returns:
            각 작업의 결과 리스트입니다.
        """
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        async with _ainternal.get_connection(self.conn) as conn:
            if self.pipe:
                async with self.pipe:
                    await self._execute_batch(grouped_ops, results, conn)
            else:
                await self._execute_batch(grouped_ops, results, conn)

        return results

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        pipeline: bool = False,
        pool_config: PoolConfig | None = None,
        index: PostgresIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> AsyncIterator[AsyncPostgresStore]:
        """연결 문자열에서 새 AsyncPostgresStore 인스턴스를 생성합니다.

        Args:
            conn_string: Postgres 연결 정보 문자열입니다.
            pipeline: AsyncPipeline을 사용할지 여부 (단일 연결에만 해당)
            pool_config: 연결 풀 구성입니다.
                제공되면 단일 연결 대신 연결 풀을 생성하여 사용합니다.
                이는 `pipeline` 인수를 재정의합니다.
            index: 임베딩 구성입니다.

        Returns:
            AsyncPostgresStore: 새로운 AsyncPostgresStore 인스턴스입니다.
        """
        if pool_config is not None:
            pc = pool_config.copy()
            async with cast(
                AsyncConnectionPool[AsyncConnection[DictRow]],
                AsyncConnectionPool(
                    conn_string,
                    min_size=pc.pop("min_size", 1),
                    max_size=pc.pop("max_size", None),
                    kwargs={
                        "autocommit": True,
                        "prepare_threshold": 0,
                        "row_factory": dict_row,
                        **(pc.pop("kwargs", None) or {}),
                    },
                    **cast(dict, pc),
                ),
            ) as pool:
                yield cls(conn=pool, index=index, ttl=ttl)
        else:
            async with await AsyncConnection.connect(
                conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
            ) as conn:
                if pipeline:
                    async with conn.pipeline() as pipe:
                        yield cls(conn=conn, pipe=pipe, index=index, ttl=ttl)
                else:
                    yield cls(conn=conn, index=index, ttl=ttl)

    async def setup(self) -> None:
        """저장소 데이터베이스를 비동기적으로 설정합니다.

        이 메서드는 Postgres 데이터베이스에 필요한 테이블이 아직 존재하지 않으면 생성하고
        데이터베이스 마이그레이션을 실행합니다. 저장소를 처음 사용할 때 사용자가 직접
        호출해야 합니다.
        """

        async def _get_version(cur: AsyncCursor[DictRow], table: str) -> int:
            await cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    v INTEGER PRIMARY KEY
                )
            """
            )
            await cur.execute(f"SELECT v FROM {table} ORDER BY v DESC LIMIT 1")
            row = cast(dict, await cur.fetchone())
            if row is None:
                version = -1
            else:
                version = row["v"]
            return version

        async with self._cursor() as cur:
            version = await _get_version(cur, table="store_migrations")
            for v, sql in enumerate(self.MIGRATIONS[version + 1 :], start=version + 1):
                await cur.execute(sql)
                await cur.execute("INSERT INTO store_migrations (v) VALUES (%s)", (v,))

            if self.index_config:
                version = await _get_version(cur, table="vector_migrations")
                for v, migration in enumerate(
                    self.VECTOR_MIGRATIONS[version + 1 :], start=version + 1
                ):
                    sql = migration.sql
                    if migration.params:
                        params = {
                            k: v(self) if v is not None and callable(v) else v
                            for k, v in migration.params.items()
                        }
                        sql = sql % params
                    await cur.execute(sql)
                    await cur.execute(
                        "INSERT INTO vector_migrations (v) VALUES (%s)", (v,)
                    )

    async def sweep_ttl(self) -> int:
        """TTL을 기반으로 만료된 저장소 항목을 삭제합니다.

        Returns:
            int: 삭제된 항목의 수입니다.
        """
        async with self._cursor() as cur:
            await cur.execute(
                """
                DELETE FROM store
                WHERE expires_at IS NOT NULL AND expires_at < NOW()
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
        logger.info(f"{interval}분 간격으로 저장소 TTL 스위퍼를 시작합니다")

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
                        logger.info(f"저장소에서 {expired_items}개의 만료된 항목을 정리했습니다")
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.exception("저장소 TTL 정리 반복이 실패했습니다", exc_info=exc)

        task = asyncio.create_task(_sweep_loop())
        task.set_name("ttl_sweeper")
        self._ttl_sweeper_task = task
        return task

    async def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        """실행 중인 경우 TTL 스위퍼 작업을 중지합니다.

        Args:
            timeout: 작업이 중지될 때까지 대기할 최대 시간(초)입니다.
                `None`이면 무기한 대기합니다.

        Returns:
            bool: 작업이 성공적으로 중지되었거나 실행 중이 아니면 True,
                작업이 중지되기 전에 시간 초과에 도달하면 False입니다.
        """
        if self._ttl_sweeper_task is None or self._ttl_sweeper_task.done():
            return True

        logger.info("TTL 스위퍼 작업을 중지합니다")
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
            logger.info("TTL 스위퍼 작업이 중지되었습니다")
        else:
            logger.warning("TTL 스위퍼 작업 중지를 기다리다 시간 초과되었습니다")

        return success

    async def __aenter__(self) -> AsyncPostgresStore:
        """비동기 컨텍스트 관리자 진입 시 저장소를 반환합니다."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """비동기 컨텍스트 관리자 종료 시 TTL 스위퍼를 정리합니다."""
        # 컨텍스트 종료 시 TTL 스위퍼 작업이 중지되도록 보장
        if hasattr(self, "_ttl_sweeper_task") and self._ttl_sweeper_task is not None:
            # 작업을 중지하도록 이벤트 설정
            self._ttl_stop_event.set()
            # 차단을 피하기 위해 여기서 작업 완료를 기다리지 않음
            # 작업은 자체적으로 정상적으로 정리됨

    async def _execute_batch(
        self,
        grouped_ops: dict,
        results: list[Result],
        conn: AsyncConnection[DictRow],
    ) -> None:
        """그룹화된 작업들을 실행합니다.

        Args:
            grouped_ops: 작업 타입별로 그룹화된 작업 딕셔너리입니다.
            results: 결과를 저장할 리스트입니다.
            conn: 사용할 데이터베이스 연결입니다.
        """
        async with self._cursor(pipeline=True) as cur:
            if GetOp in grouped_ops:
                await self._batch_get_ops(
                    cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]),
                    results,
                    cur,
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
                    cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp]),
                    cur,
                )

    async def _batch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
        cur: AsyncCursor[DictRow],
    ) -> None:
        """배치 GET 작업들을 처리합니다.

        Args:
            get_ops: 인덱스와 GetOp 튜플들의 시퀀스입니다.
            results: 결과를 저장할 리스트입니다.
            cur: 사용할 데이터베이스 커서입니다.
        """
        for query, params, namespace, items in self._get_batch_GET_ops_queries(get_ops):
            await cur.execute(query, params)
            rows = cast(list[Row], await cur.fetchall())
            key_to_row = {row["key"]: row for row in rows}
            for idx, key in items:
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
        cur: AsyncCursor[DictRow],
    ) -> None:
        """배치 PUT 작업들을 처리합니다.

        Args:
            put_ops: 인덱스와 PutOp 튜플들의 시퀀스입니다.
            cur: 사용할 데이터베이스 커서입니다.
        """
        queries, embedding_request = self._prepare_batch_PUT_queries(put_ops)
        if embedding_request:
            if self.embeddings is None:
                # 위에서 embedding_request를 반환하려면 임베딩 구성이 필요하므로
                # 여기에 도달해서는 안 됨
                raise ValueError(
                    "벡터 작업(의미론적 검색용)에는 임베딩 구성이 필요합니다. "
                    f"{self.__class__.__name__}을 초기화할 때 EmbeddingConfig를 제공하세요."
                )
            query, txt_params = embedding_request
            vectors = await self.embeddings.aembed_documents(
                [param[-1] for param in txt_params]
            )
            queries.append(
                (
                    query,
                    [
                        p
                        for (ns, k, pathname, _), vector in zip(
                            txt_params, vectors, strict=False
                        )
                        for p in (ns, k, pathname, vector)
                    ],
                )
            )

        for query, params in queries:
            await cur.execute(query, params)

    async def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        cur: AsyncCursor[DictRow],
    ) -> None:
        """배치 SEARCH 작업들을 처리합니다.

        Args:
            search_ops: 인덱스와 SearchOp 튜플들의 시퀀스입니다.
            results: 결과를 저장할 리스트입니다.
            cur: 사용할 데이터베이스 커서입니다.
        """
        queries, embedding_requests = self._prepare_batch_search_queries(search_ops)

        if embedding_requests and self.embeddings:
            vectors = await self.embeddings.aembed_documents(
                [query for _, query in embedding_requests]
            )
            for (idx, _), vector in zip(embedding_requests, vectors, strict=False):
                _paramslist = queries[idx][1]
                for i in range(len(_paramslist)):
                    if _paramslist[i] is PLACEHOLDER:
                        _paramslist[i] = vector

        for (idx, _), (query, params) in zip(search_ops, queries, strict=False):
            await cur.execute(query, params)
            rows = cast(list[Row], await cur.fetchall())
            items = [
                _row_to_search_item(
                    _decode_ns_bytes(row["prefix"]), row, loader=self._deserializer
                )
                for row in rows
            ]
            results[idx] = items

    async def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
        cur: AsyncCursor[DictRow],
    ) -> None:
        """배치 LIST_NAMESPACES 작업들을 처리합니다.

        Args:
            list_ops: 인덱스와 ListNamespacesOp 튜플들의 시퀀스입니다.
            results: 결과를 저장할 리스트입니다.
            cur: 사용할 데이터베이스 커서입니다.
        """
        queries = self._get_batch_list_namespaces_queries(list_ops)
        for (query, params), (idx, _) in zip(queries, list_ops, strict=False):
            await cur.execute(query, params)
            rows = cast(list[dict], await cur.fetchall())
            namespaces = [_decode_ns_bytes(row["truncated_prefix"]) for row in rows]
            results[idx] = namespaces

    @asynccontextmanager
    async def _cursor(
        self, *, pipeline: bool = False
    ) -> AsyncIterator[AsyncCursor[DictRow]]:
        """컨텍스트 관리자로 데이터베이스 커서를 생성합니다.

        Args:
            pipeline: 컨텍스트 관리자 내부의 DB 작업에 파이프라인을 사용할지 여부입니다.
                PostgresStore 인스턴스가 파이프라인으로 초기화되었는지 여부와 관계없이 적용됩니다.
                파이프라인 모드가 지원되지 않으면 트랜잭션 컨텍스트 관리자를 사용하도록 대체됩니다.
        """
        async with _ainternal.get_connection(self.conn) as conn:
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
                        self.lock,
                        conn.pipeline(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
                else:
                    async with (
                        self.lock,
                        conn.transaction(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
            else:
                async with (
                    self.lock,
                    conn.cursor(binary=True, row_factory=dict_row) as cur,
                ):
                    yield cur
