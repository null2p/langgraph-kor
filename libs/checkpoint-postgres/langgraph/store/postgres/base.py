from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NamedTuple,
    TypeVar,
    cast,
)

import orjson
from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)
from psycopg import Capabilities, Connection, Cursor, Pipeline
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from typing_extensions import TypedDict

from langgraph.checkpoint.postgres import _ainternal as _ainternal
from langgraph.checkpoint.postgres import _internal as _pg_internal

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class Migration(NamedTuple):
    """선택적 조건 및 매개변수가 있는 데이터베이스 마이그레이션입니다."""

    sql: str
    params: dict[str, Any] | None = None
    condition: Callable[[BasePostgresStore], bool] | None = None


MIGRATIONS: Sequence[str] = [
    """
CREATE TABLE IF NOT EXISTS store (
    -- 'prefix'는 문서의 'namespace'를 나타냅니다
    prefix text NOT NULL,
    key text NOT NULL,
    value jsonb NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (prefix, key)
);
""",
    """
-- prefix로 더 빠른 조회를 위해
CREATE INDEX CONCURRENTLY IF NOT EXISTS store_prefix_idx ON store USING btree (prefix text_pattern_ops);
""",
    """
-- store 테이블에 expires_at 열 추가
ALTER TABLE store
ADD COLUMN IF NOT EXISTS expires_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS ttl_minutes INT;
""",
    """
-- 효율적인 TTL 스위핑을 위한 인덱스 추가
CREATE INDEX IF NOT EXISTS idx_store_expires_at ON store (expires_at)
WHERE expires_at IS NOT NULL;
""",
]

VECTOR_MIGRATIONS: Sequence[Migration] = [
    Migration(
        """
CREATE EXTENSION IF NOT EXISTS vector;
""",
    ),
    Migration(
        """
CREATE TABLE IF NOT EXISTS store_vectors (
    prefix text NOT NULL,
    key text NOT NULL,
    field_name text NOT NULL,
    embedding %(vector_type)s(%(dims)s),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (prefix, key, field_name),
    FOREIGN KEY (prefix, key) REFERENCES store(prefix, key) ON DELETE CASCADE
);
""",
        params={
            "dims": lambda store: store.index_config["dims"],
            "vector_type": lambda store: (
                cast(PostgresIndexConfig, store.index_config)
                .get("ann_index_config", {})
                .get("vector_type", "vector")
            ),
        },
    ),
    Migration(
        """
CREATE INDEX CONCURRENTLY IF NOT EXISTS store_vectors_embedding_idx ON store_vectors 
    USING %(index_type)s (embedding %(ops)s)%(index_params)s;
""",
        condition=lambda store: bool(
            store.index_config and _get_index_params(store)[0] != "flat"
        ),
        params={
            "index_type": lambda store: _get_index_params(store)[0],
            "ops": lambda store: _get_vector_type_ops(store),
            "index_params": lambda store: (
                " WITH ("
                + ", ".join(f"{k}={v}" for k, v in _get_index_params(store)[1].items())
                + ")"
                if _get_index_params(store)[1]
                else ""
            ),
        },
    ),
]


C = TypeVar("C", bound=_pg_internal.Conn | _ainternal.Conn)


class PoolConfig(TypedDict, total=False):
    """PostgreSQL 연결을 위한 연결 풀 설정입니다.

    연결 수명 주기와 리소스 활용을 제어합니다:
    - 작은 풀(1-5)은 낮은 동시성 워크로드에 적합
    - 큰 풀은 동시 요청을 처리하지만 더 많은 리소스를 소비
    - max_size 설정은 부하 시 리소스 고갈을 방지
    """

    min_size: int
    """풀에서 유지되는 최소 연결 수입니다. 기본값은 1입니다."""

    max_size: int | None
    """풀에서 허용되는 최대 연결 수입니다. None은 무제한을 의미합니다."""

    kwargs: dict
    """풀의 각 연결에 전달되는 추가 연결 인수입니다.

    자동으로 설정되는 기본 kwargs:
    - autocommit: True
    - prepare_threshold: 0
    - row_factory: dict_row
    """


class ANNIndexConfig(TypedDict, total=False):
    """PostgreSQL 저장소의 벡터 인덱스 구성입니다."""

    kind: Literal["hnsw", "ivfflat", "flat"]
    """사용할 인덱스 유형: 'hnsw'는 Hierarchical Navigable Small World, 'ivfflat'은 Inverted File Flat입니다."""
    vector_type: Literal["vector", "halfvec"]
    """사용할 벡터 저장소 유형입니다.
    옵션:
    - 'vector': 일반 벡터(기본값)
    - 'halfvec': 메모리 사용량을 줄이기 위한 반정밀도 벡터
    """


class HNSWConfig(ANNIndexConfig, total=False):
    """HNSW(Hierarchical Navigable Small World) 인덱스를 위한 구성입니다."""

    kind: Literal["hnsw"]  # type: ignore[misc]
    m: int
    """레이어당 최대 연결 수입니다. 기본값은 16입니다."""
    ef_construction: int
    """인덱스 구성을 위한 동적 후보 리스트의 크기입니다. 기본값은 64입니다."""


class IVFFlatConfig(ANNIndexConfig, total=False):
    """IVFFlat 인덱스는 벡터를 리스트로 나누고, 쿼리 벡터에 가장 가까운 리스트의 하위 집합을 검색합니다. HNSW보다 빌드 시간이 빠르고 메모리를 덜 사용하지만, 쿼리 성능(속도-재현율 트레이드오프 측면에서)은 낮습니다.

    좋은 재현율을 달성하기 위한 세 가지 핵심:
    1. 테이블에 일부 데이터가 있은 후 인덱스 생성
    2. 적절한 리스트 수 선택 - 최대 1M 행의 경우 rows / 1000, 1M 행 이상의 경우 sqrt(rows)부터 시작하는 것이 좋음
    3. 쿼리 시 적절한 프로브 수 지정(높을수록 재현율이 좋고, 낮을수록 속도가 좋음) - sqrt(lists)부터 시작하는 것이 좋음
    """

    kind: Literal["ivfflat"]  # type: ignore[misc]
    nlist: int
    """IVF 인덱스를 위한 역 리스트(클러스터) 수입니다.

    인덱스 구조에 사용되는 클러스터 수를 결정합니다.
    값이 높을수록 검색 속도를 향상시킬 수 있지만 인덱스 크기와 빌드 시간이 증가합니다.
    일반적으로 인덱스의 벡터 수의 제곱근으로 설정됩니다.
    """


class PostgresIndexConfig(IndexConfig, total=False):
    """pgvector 전용 옵션을 사용한 PostgreSQL 저장소의 벡터 임베딩 구성입니다.

    EmbeddingConfig를 확장하여 pgvector 인덱스 및 벡터 유형에 대한 추가 구성을 제공합니다.
    """

    ann_index_config: ANNIndexConfig
    """선택한 인덱스 유형(HNSW 또는 IVF Flat)에 대한 특정 구성입니다."""
    distance_type: Literal["l2", "inner_product", "cosine"]
    """벡터 유사도 검색에 사용할 거리 측정 항목:
    - 'l2': 유클리드 거리
    - 'inner_product': 내적
    - 'cosine': 코사인 유사도
    """


class BasePostgresStore(Generic[C]):
    MIGRATIONS = MIGRATIONS
    VECTOR_MIGRATIONS = VECTOR_MIGRATIONS
    conn: C
    _deserializer: Callable[[bytes | orjson.Fragment], dict[str, Any]] | None
    index_config: PostgresIndexConfig | None

    def _get_batch_GET_ops_queries(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
    ) -> list[tuple[str, tuple, tuple[str, ...], list]]:
        """
        네임스페이스당 여러 키를 가져오고 (선택적으로 TTL을 새로 고침하는) 쿼리를 빌드합니다.

        반환되는 각 요소는 다음의 튜플입니다:
        (sql_query_string, sql_params, namespace, items_for_this_namespace)

        여기서 items_for_this_namespace는 (idx, key, refresh_ttl)의 원래 리스트입니다.
        """

        namespace_groups = defaultdict(list)
        refresh_ttls = defaultdict(list)
        for idx, op in get_ops:
            namespace_groups[op.namespace].append((idx, op.key))
            refresh_ttls[op.namespace].append(op.refresh_ttl)

        results = []
        for namespace, items in namespace_groups.items():
            _, keys = zip(*items, strict=False)
            this_refresh_ttls = refresh_ttls[namespace]

            query = """
                WITH passed_in AS (
                    SELECT unnest(%s::text[]) AS key,
                        unnest(%s::bool[])  AS do_refresh
                ),
                updated AS (
                    UPDATE store s
                    SET expires_at = NOW() + (s.ttl_minutes || ' minutes')::interval
                    FROM passed_in p
                    WHERE s.prefix = %s
                    AND s.key    = p.key
                    AND p.do_refresh = TRUE
                    AND s.ttl_minutes IS NOT NULL
                    RETURNING s.key
                )
                SELECT s.key, s.value, s.created_at, s.updated_at
                FROM store s
                JOIN passed_in p ON s.key = p.key
                WHERE s.prefix = %s
            """
            ns_text = _namespace_to_text(namespace)
            params = (
                list(keys),  # -> unnest(%s::text[])
                list(this_refresh_ttls),  # -> unnest(%s::bool[])
                ns_text,  # -> prefix = %s (for UPDATE)
                ns_text,  # -> prefix = %s (for final SELECT)
            )
            results.append((query, params, namespace, items))

        return results

    def _prepare_batch_PUT_queries(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
    ) -> tuple[
        list[tuple[str, Sequence]],
        tuple[str, Sequence[tuple[str, str, str, str]]] | None,
    ]:
        dedupped_ops: dict[tuple[tuple[str, ...], str], PutOp] = {}
        for _, op in put_ops:
            dedupped_ops[(op.namespace, op.key)] = op

        inserts: list[PutOp] = []
        deletes: list[PutOp] = []
        for op in dedupped_ops.values():
            if op.value is None:
                deletes.append(op)
            else:
                inserts.append(op)

        queries: list[tuple[str, Sequence]] = []

        if deletes:
            namespace_groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
            for op in deletes:
                namespace_groups[op.namespace].append(op.key)
            for namespace, keys in namespace_groups.items():
                placeholders = ",".join(["%s"] * len(keys))
                query = (
                    f"DELETE FROM store WHERE prefix = %s AND key IN ({placeholders})"
                )
                params = (_namespace_to_text(namespace), *keys)
                queries.append((query, params))
        embedding_request: tuple[str, Sequence[tuple[str, str, str, str]]] | None = None
        if inserts:
            values = []
            insertion_params = []
            vector_values = []
            embedding_request_params = []
            # TTL 만료 처리

            # 먼저 메인 저장소 삽입 처리
            for op in inserts:
                if op.ttl is not None:
                    expires_at_str = f"NOW() + INTERVAL '{op.ttl * 60} seconds'"
                    ttl_minutes = op.ttl
                else:
                    expires_at_str = "NULL"
                    ttl_minutes = None

                values.append(
                    f"(%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, {expires_at_str}, %s)"
                )
                insertion_params.extend(
                    [
                        _namespace_to_text(op.namespace),
                        op.key,
                        Jsonb(cast(dict, op.value)),
                        ttl_minutes,
                    ]
                )

            # 그런 다음 구성된 경우 임베딩 처리
            if self.index_config:
                for op in inserts:
                    if op.index is False:
                        continue
                    value = op.value
                    ns = _namespace_to_text(op.namespace)
                    k = op.key

                    if op.index is None:
                        paths = cast(dict, self.index_config)["__tokenized_fields"]
                    else:
                        paths = [(ix, tokenize_path(ix)) for ix in op.index]

                    for path, tokenized_path in paths:
                        texts = get_text_at_path(value, tokenized_path)
                        for i, text in enumerate(texts):
                            pathname = f"{path}.{i}" if len(texts) > 1 else path
                            vector_values.append(
                                "(%s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                            )
                            embedding_request_params.append((ns, k, pathname, text))

            values_str = ",".join(values)
            query = f"""
                INSERT INTO store (prefix, key, value, created_at, updated_at, expires_at, ttl_minutes)
                VALUES {values_str}
                ON CONFLICT (prefix, key) DO UPDATE
                SET value = EXCLUDED.value,
                    updated_at = CURRENT_TIMESTAMP,
                    expires_at = EXCLUDED.expires_at,
                    ttl_minutes = EXCLUDED.ttl_minutes
            """
            queries.append((query, insertion_params))

            if vector_values:
                values_str = ",".join(vector_values)
                query = f"""
                    INSERT INTO store_vectors (prefix, key, field_name, embedding, created_at, updated_at)
                    VALUES {values_str}
                    ON CONFLICT (prefix, key, field_name) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        updated_at = CURRENT_TIMESTAMP
                """
                embedding_request = (query, embedding_request_params)

        return queries, embedding_request

    def _prepare_batch_search_queries(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
    ) -> tuple[
        list[tuple[str, list[None | str | list[float]]]],  # queries, params
        list[tuple[int, str]],  # idx, query_text pairs to embed
    ]:
        """
        SearchOp당 SQL 쿼리(선택적 TTL 새로 고침 포함) 및 임베딩 요청을 빌드합니다.
        반환값:
        - queries: (SQL, param_list)의 리스트
        - embedding_requests: (search_ops의_원래_인덱스, text_query)의 리스트
        """

        queries = []
        embedding_requests = []
        for idx, (_, op) in enumerate(search_ops):
            filter_params = []
            filter_clauses = []
            if op.filter:
                for key, value in op.filter.items():
                    if isinstance(value, dict):
                        for op_name, val in value.items():
                            condition, params_ = self._get_filter_condition(
                                key, op_name, val
                            )
                            filter_clauses.append(condition)
                            filter_params.extend(params_)
                    else:
                        filter_clauses.append("value->%s = %s::jsonb")
                        filter_params.extend([key, orjson.dumps(value).decode("utf-8")])

            ns_condition = "TRUE"
            ns_param: Sequence[str] | None = None
            if op.namespace_prefix:
                ns_condition = "store.prefix LIKE %s"
                ns_param = (f"{_namespace_to_text(op.namespace_prefix)}%",)
            else:
                ns_param = ()

            extra_filters = (
                " AND " + " AND ".join(filter_clauses) if filter_clauses else ""
            )

            if op.query and self.index_config:
                # 나중에 텍스트를 임베딩할 것이므로 요청을 기록합니다.
                embedding_requests.append((idx, op.query))

                score_operator, post_operator = get_distance_operator(self)
                post_operator = post_operator.replace("scored", "uniq")
                vector_type = self.index_config.get("ann_index_config", {}).get(
                    "vector_type", "vector"
                )

                # 해밍 비트 벡터 또는 "일반" 벡터용
                if (
                    vector_type == "bit"
                    and cast(dict, self.index_config).get("distance_type") == "hamming"
                ):
                    score_operator = score_operator % (
                        "%s",
                        cast(dict, self.index_config)["dims"],
                    )
                else:
                    score_operator = score_operator % ("%s", vector_type)

                vectors_per_doc_estimate = cast(dict, self.index_config)[
                    "__estimated_num_vectors"
                ]
                expanded_limit = (op.limit * vectors_per_doc_estimate * 2) + 1

                # "sub_scored"는 메인 벡터 검색을 수행합니다
                # 그런 다음 DISTINCT ON을 사용하여 저장소에 중복이 있을 수 있는 경우 삭제합니다
                # 마지막으로 limit 및 offset을 적용합니다
                vector_search_cte = f"""
                        SELECT store.prefix, store.key, store.value, store.created_at, store.updated_at,
                            {score_operator} AS neg_score
                        FROM store
                        JOIN store_vectors sv ON store.prefix = sv.prefix AND store.key = sv.key
                        WHERE {ns_condition} {extra_filters}
                        ORDER BY {score_operator} ASC
                        LIMIT %s
                    """

                search_results_sql = f"""
                        WITH scored AS (
                            {vector_search_cte}
                        )
                        SELECT uniq.prefix, uniq.key, uniq.value, uniq.created_at, uniq.updated_at,
                            {post_operator} AS score
                        FROM (
                            SELECT DISTINCT ON (scored.prefix, scored.key)
                                scored.prefix, scored.key, scored.value, scored.created_at, scored.updated_at, scored.neg_score
                            FROM scored
                            ORDER BY scored.prefix, scored.key, scored.neg_score ASC
                        ) uniq
                        ORDER BY score DESC
                        LIMIT %s
                        OFFSET %s
                    """

                search_results_params = [
                    PLACEHOLDER,
                    *ns_param,
                    *filter_params,
                    PLACEHOLDER,
                    expanded_limit,
                    op.limit,
                    op.offset,
                ]

            else:
                base_query = f"""
                        SELECT store.prefix, store.key, store.value, store.created_at, store.updated_at, NULL AS score
                        FROM store
                        WHERE {ns_condition} {extra_filters}
                        ORDER BY store.updated_at DESC
                        LIMIT %s
                        OFFSET %s
                    """
                search_results_sql = base_query
                search_results_params = [
                    *ns_param,
                    *filter_params,
                    op.limit,
                    op.offset,
                ]

            if op.refresh_ttl:
                # 전체 기본 쿼리를 CTE로 래핑한 다음 "update_at"을 수행합니다
                final_sql = f"""
                        WITH search_results AS (
                            {search_results_sql}
                        ),
                        updated AS (
                            UPDATE store s
                            SET expires_at = NOW() + (s.ttl_minutes || ' minutes')::interval
                            FROM search_results sr
                            WHERE s.prefix = sr.prefix
                            AND s.key = sr.key
                            AND s.ttl_minutes IS NOT NULL
                        )
                        SELECT sr.prefix, sr.key, sr.value, sr.created_at, sr.updated_at, sr.score
                        FROM search_results sr
                    """
                final_params = search_results_params[:]  # 복사
            else:
                final_sql = search_results_sql
                final_params = search_results_params
            queries.append((final_sql, final_params))

        return queries, embedding_requests

    def _get_batch_list_namespaces_queries(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
    ) -> list[tuple[str, Sequence]]:
        queries: list[tuple[str, Sequence]] = []
        for _, op in list_ops:
            query = r"""
                SELECT DISTINCT ON (truncated_prefix) truncated_prefix, prefix
                FROM (
                    SELECT
                        prefix,
                        CASE
                            WHEN %s::integer IS NOT NULL THEN
                                (SELECT STRING_AGG(part, '.' ORDER BY idx)
                                 FROM (
                                     SELECT part, ROW_NUMBER() OVER () AS idx
                                     FROM UNNEST(REGEXP_SPLIT_TO_ARRAY(prefix, '\.')) AS part
                                     LIMIT %s::integer
                                 ) subquery
                                )
                            ELSE prefix
                        END AS truncated_prefix
                    FROM store
            """
            params: list[Any] = [op.max_depth, op.max_depth]

            conditions = []
            if op.match_conditions:
                for condition in op.match_conditions:
                    if condition.match_type == "prefix":
                        conditions.append("prefix LIKE %s")
                        params.append(
                            f"{_namespace_to_text(condition.path, handle_wildcards=True)}%"
                        )
                    elif condition.match_type == "suffix":
                        conditions.append("prefix LIKE %s")
                        params.append(
                            f"%{_namespace_to_text(condition.path, handle_wildcards=True)}"
                        )
                    else:
                        logger.warning(
                            f"Unknown match_type in list_namespaces: {condition.match_type}"
                        )

            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += ") AS subquery "

            query += " ORDER BY truncated_prefix LIMIT %s OFFSET %s"
            params.extend([op.limit, op.offset])
            queries.append((query, tuple(params)))

        return queries

    def _get_filter_condition(self, key: str, op: str, value: Any) -> tuple[str, list]:
        """필터 조건을 생성하는 헬퍼 함수입니다."""
        if op == "$eq":
            return "value->%s = %s::jsonb", [key, json.dumps(value)]
        elif op == "$gt":
            return "value->>%s > %s", [key, str(value)]
        elif op == "$gte":
            return "value->>%s >= %s", [key, str(value)]
        elif op == "$lt":
            return "value->>%s < %s", [key, str(value)]
        elif op == "$lte":
            return "value->>%s <= %s", [key, str(value)]
        elif op == "$ne":
            return "value->%s != %s::jsonb", [key, json.dumps(value)]
        else:
            raise ValueError(f"Unsupported operator: {op}")


class PostgresStore(BaseStore, BasePostgresStore[_pg_internal.Conn]):
    """pgvector를 사용한 선택적 벡터 검색 기능이 있는 Postgres 기반 저장소입니다.

    !!! example "예제"
        기본 설정 및 사용:
        ```python
        from langgraph.store.postgres import PostgresStore
        from psycopg import Connection

        conn_string = "postgresql://user:pass@localhost:5432/dbname"

        # 직접 연결 사용
        with Connection.connect(conn_string) as conn:
            store = PostgresStore(conn)
            store.setup() # 마이그레이션 실행. 한 번만 수행

            # 데이터 저장 및 검색
            store.put(("users", "123"), "prefs", {"theme": "dark"})
            item = store.get(("users", "123"), "prefs")
        ```

        또는 편리한 from_conn_string 헬퍼 사용:
        ```python
        from langgraph.store.postgres import PostgresStore

        conn_string = "postgresql://user:pass@localhost:5432/dbname"

        with PostgresStore.from_conn_string(conn_string) as store:
            store.setup()

            # 데이터 저장 및 검색
            store.put(("users", "123"), "prefs", {"theme": "dark"})
            item = store.get(("users", "123"), "prefs")
        ```

        LangChain 임베딩을 사용한 벡터 검색:
        ```python
        from langchain.embeddings import init_embeddings
        from langgraph.store.postgres import PostgresStore

        conn_string = "postgresql://user:pass@localhost:5432/dbname"

        with PostgresStore.from_conn_string(
            conn_string,
            index={
                "dims": 1536,
                "embed": init_embeddings("openai:text-embedding-3-small"),
                "fields": ["text"]  # 임베드할 필드 지정. 기본값은 전체 직렬화된 값
            }
        ) as store:
            store.setup() # 마이그레이션 실행을 위해 한 번 수행

            # 문서 저장
            store.put(("docs",), "doc1", {"text": "Python tutorial"})
            store.put(("docs",), "doc2", {"text": "TypeScript guide"})
            store.put(("docs",), "doc2", {"text": "Other guide"}, index=False) # 인덱싱하지 않음

            # 유사도로 검색
            results = store.search(("docs",), query="programming guides", limit=2)
        ```

    Note:
        의미론적 검색은 기본적으로 비활성화되어 있습니다. 저장소를 생성할 때 `index` 구성을
        제공하여 활성화할 수 있습니다. 이 구성이 없으면 `put` 또는 `aput`에 전달된 모든
        `index` 인수는 효과가 없습니다.

    Warning:
        처음 사용하기 전에 필요한 테이블과 인덱스를 생성하기 위해 `setup()`을 호출하세요.
        벡터 검색을 사용하려면 pgvector 확장이 사용 가능해야 합니다.

    Note:
        TTL 구성을 제공하는 경우, 만료된 항목을 제거하는 백그라운드 스레드를 시작하려면
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
        "_ttl_sweeper_thread",
        "_ttl_stop_event",
    )
    supports_ttl: bool = True

    def __init__(
        self,
        conn: _pg_internal.Conn,
        *,
        pipe: Pipeline | None = None,
        deserializer: Callable[[bytes | orjson.Fragment], dict[str, Any]] | None = None,
        index: PostgresIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> None:
        super().__init__()
        self._deserializer = deserializer
        self.conn = conn
        self.pipe = pipe
        self.supports_pipeline = Capabilities().has_pipeline()
        self.lock = threading.Lock()
        self.index_config = index
        if self.index_config:
            self.embeddings, self.index_config = _ensure_index_config(self.index_config)
        else:
            self.embeddings = None
        self.ttl_config = ttl
        self._ttl_sweeper_thread: threading.Thread | None = None
        self._ttl_stop_event = threading.Event()

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
        *,
        pipeline: bool = False,
        pool_config: PoolConfig | None = None,
        index: PostgresIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> Iterator[PostgresStore]:
        """연결 문자열에서 새 PostgresStore 인스턴스를 생성합니다.

        Args:
            conn_string: Postgres 연결 정보 문자열입니다.
            pipeline: Pipeline을 사용할지 여부입니다.
            pool_config: 연결 풀 구성입니다.
                제공되면 단일 연결 대신 연결 풀을 생성하여 사용합니다.
                이는 `pipeline` 인수를 재정의합니다.
            index: 저장소를 위한 인덱스 구성입니다.
            ttl: 저장소를 위한 TTL 구성입니다.

        Returns:
            PostgresStore: 새로운 PostgresStore 인스턴스입니다.
        """
        if pool_config is not None:
            pc = pool_config.copy()
            with cast(
                ConnectionPool[Connection[DictRow]],
                ConnectionPool(
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
            with Connection.connect(
                conn_string, autocommit=True, prepare_threshold=0, row_factory=dict_row
            ) as conn:
                if pipeline:
                    with conn.pipeline() as pipe:
                        yield cls(conn, pipe=pipe, index=index, ttl=ttl)
                else:
                    yield cls(conn, index=index, ttl=ttl)

    def sweep_ttl(self) -> int:
        """TTL을 기반으로 만료된 저장소 항목을 삭제합니다.

        Returns:
            int: 삭제된 항목의 수입니다.
        """
        with self._cursor() as cur:
            cur.execute(
                """
                DELETE FROM store
                WHERE expires_at IS NOT NULL AND expires_at < NOW()
                """
            )
            deleted_count = cur.rowcount
            return deleted_count

    def start_ttl_sweeper(
        self, sweep_interval_minutes: int | None = None
    ) -> concurrent.futures.Future[None]:
        """TTL을 기반으로 만료된 저장소 항목을 주기적으로 삭제합니다.

        Returns:
            대기하거나 취소할 수 있는 Future입니다.
        """
        if not self.ttl_config:
            future: concurrent.futures.Future[None] = concurrent.futures.Future()
            future.set_result(None)
            return future

        if self._ttl_sweeper_thread and self._ttl_sweeper_thread.is_alive():
            logger.info("TTL sweeper thread is already running")
            # 기존 스레드를 취소하는 데 사용할 수 있는 future를 반환합니다
            future = concurrent.futures.Future()
            future.add_done_callback(
                lambda f: self._ttl_stop_event.set() if f.cancelled() else None
            )
            return future

        self._ttl_stop_event.clear()

        interval = float(
            sweep_interval_minutes or self.ttl_config.get("sweep_interval_minutes") or 5
        )
        logger.info(f"Starting store TTL sweeper with interval {interval} minutes")

        future = concurrent.futures.Future()

        def _sweep_loop() -> None:
            try:
                while not self._ttl_stop_event.is_set():
                    if self._ttl_stop_event.wait(interval * 60):
                        break

                    try:
                        expired_items = self.sweep_ttl()
                        if expired_items > 0:
                            logger.info(f"Store swept {expired_items} expired items")
                    except Exception as exc:
                        logger.exception(
                            "Store TTL sweep iteration failed", exc_info=exc
                        )
                future.set_result(None)
            except Exception as exc:
                future.set_exception(exc)

        thread = threading.Thread(target=_sweep_loop, daemon=True, name="ttl-sweeper")
        self._ttl_sweeper_thread = thread
        thread.start()

        future.add_done_callback(
            lambda f: self._ttl_stop_event.set() if f.cancelled() else None
        )
        return future

    def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        """실행 중인 경우 TTL 스위퍼 스레드를 중지합니다.

        Args:
            timeout: 스레드가 중지될 때까지 대기할 최대 시간(초)입니다.
                `None`이면 무기한 대기합니다.

        Returns:
            bool: 스레드가 성공적으로 중지되었거나 실행 중이 아니면 True,
                스레드가 중지되기 전에 시간 초과에 도달하면 False입니다.
        """
        if not self._ttl_sweeper_thread or not self._ttl_sweeper_thread.is_alive():
            return True

        logger.info("Stopping TTL sweeper thread")
        self._ttl_stop_event.set()

        self._ttl_sweeper_thread.join(timeout)
        success = not self._ttl_sweeper_thread.is_alive()

        if success:
            self._ttl_sweeper_thread = None
            logger.info("TTL sweeper thread stopped")
        else:
            logger.warning("Timed out waiting for TTL sweeper thread to stop")

        return success

    def __del__(self) -> None:
        """객체가 가비지 수집될 때 TTL 스위퍼 스레드가 중지되도록 보장합니다."""
        if hasattr(self, "_ttl_stop_event") and hasattr(self, "_ttl_sweeper_thread"):
            self.stop_ttl_sweeper(timeout=0.1)

    @contextmanager
    def _cursor(self, *, pipeline: bool = False) -> Iterator[Cursor[DictRow]]:
        """컨텍스트 관리자로 데이터베이스 커서를 생성합니다.

        Args:
            pipeline: 컨텍스트 관리자 내부의 DB 작업에 파이프라인을 사용할지 여부입니다.
                PostgresStore 인스턴스가 파이프라인으로 초기화되었는지 여부와 관계없이 적용됩니다.
                파이프라인 모드가 지원되지 않으면 트랜잭션 컨텍스트 관리자를 사용하도록 대체됩니다.
        """
        with _pg_internal.get_connection(self.conn) as conn:
            if self.pipe:
                # 파이프라인 모드의 연결은 여러 스레드/코루틴에서 동시에 사용할 수 있지만
                # 한 번에 하나의 커서만 사용할 수 있음
                
                try:
                    with conn.cursor(binary=True, row_factory=dict_row) as cur:
                        yield cur
                finally:
                    if pipeline:
                        self.pipe.sync()
            elif pipeline:
                # 파이프라인 모드가 아닌 연결은 한 번에 하나의 스레드/코루틴에서만
                # 사용할 수 있으므로 락을 획득함
                if self.supports_pipeline:
                    with (
                        self.lock,
                        conn.pipeline(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
                else:
                    with (
                        self.lock,
                        conn.transaction(),
                        conn.cursor(binary=True, row_factory=dict_row) as cur,
                    ):
                        yield cur
            else:
                with conn.cursor(binary=True, row_factory=dict_row) as cur:
                    yield cur

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        with self._cursor(pipeline=True) as cur:
            if GetOp in grouped_ops:
                self._batch_get_ops(
                    cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]), results, cur
                )

            if SearchOp in grouped_ops:
                self._batch_search_ops(
                    cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp]),
                    results,
                    cur,
                )

            if ListNamespacesOp in grouped_ops:
                self._batch_list_namespaces_ops(
                    cast(
                        Sequence[tuple[int, ListNamespacesOp]],
                        grouped_ops[ListNamespacesOp],
                    ),
                    results,
                    cur,
                )
            if PutOp in grouped_ops:
                self._batch_put_ops(
                    cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp]), cur
                )

        return results

    def _batch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
        cur: Cursor[DictRow],
    ) -> None:
        for query, params, namespace, items in self._get_batch_GET_ops_queries(get_ops):
            cur.execute(query, params)
            rows = cast(list[Row], cur.fetchall())
            key_to_row = {row["key"]: row for row in rows}
            for idx, key in items:
                row = key_to_row.get(key)
                if row:
                    results[idx] = _row_to_item(
                        namespace, row, loader=self._deserializer
                    )
                else:
                    results[idx] = None

    def _batch_put_ops(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
        cur: Cursor[DictRow],
    ) -> None:
        queries, embedding_request = self._prepare_batch_PUT_queries(put_ops)
        if embedding_request:
            if self.embeddings is None:
                # 위에서 embedding_request를 반환하려면 임베딩 구성이 필요하므로
                # 여기에 도달해서는 안 됨
                raise ValueError(
                    "Embedding configuration is required for vector operations "
                    f"(for semantic search). "
                    f"Please provide an Embeddings when initializing the {self.__class__.__name__}."
                )
            query, txt_params = embedding_request
            # 원시 텍스트를 벡터로 교체하도록 매개변수를 업데이트합니다
            vectors = self.embeddings.embed_documents(
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
            cur.execute(query, params)

    def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        cur: Cursor[DictRow],
    ) -> None:
        queries, embedding_requests = self._prepare_batch_search_queries(search_ops)

        if embedding_requests and self.embeddings:
            embeddings = self.embeddings.embed_documents(
                [query for _, query in embedding_requests]
            )
            for (idx, _), embedding in zip(
                embedding_requests, embeddings, strict=False
            ):
                _paramslist = queries[idx][1]
                for i in range(len(_paramslist)):
                    if _paramslist[i] is PLACEHOLDER:
                        _paramslist[i] = embedding

        for (idx, _), (query, params) in zip(search_ops, queries, strict=False):
            cur.execute(query, params)
            rows = cast(list[Row], cur.fetchall())
            results[idx] = [
                _row_to_search_item(
                    _decode_ns_bytes(row["prefix"]), row, loader=self._deserializer
                )
                for row in rows
            ]

    def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
        cur: Cursor[DictRow],
    ) -> None:
        for (query, params), (idx, _) in zip(
            self._get_batch_list_namespaces_queries(list_ops), list_ops, strict=False
        ):
            cur.execute(query, params)
            results[idx] = [_decode_ns_bytes(row["truncated_prefix"]) for row in cur]

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.get_running_loop().run_in_executor(None, self.batch, ops)

    def setup(self) -> None:
        """저장소 데이터베이스를 설정합니다.

        이 메서드는 Postgres 데이터베이스에 필요한 테이블이 아직 존재하지 않으면 생성하고
        데이터베이스 마이그레이션을 실행합니다. 저장소를 처음 사용할 때 사용자가 직접
        호출해야 합니다.
        """

        def _get_version(cur: Cursor[dict[str, Any]], table: str) -> int:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    v INTEGER PRIMARY KEY
                )
            """
            )
            cur.execute(f"SELECT v FROM {table} ORDER BY v DESC LIMIT 1")
            row = cast(dict, cur.fetchone())
            if row is None:
                version = -1
            else:
                version = row["v"]
            return version

        with self._cursor() as cur:
            version = _get_version(cur, table="store_migrations")
            for v, sql in enumerate(self.MIGRATIONS[version + 1 :], start=version + 1):
                try:
                    cur.execute(sql)
                    cur.execute("INSERT INTO store_migrations (v) VALUES (%s)", (v,))
                except Exception as e:
                    logger.error(
                        f"Failed to apply migration {v}.\nSql={sql}\nError={e}"
                    )
                    raise

            if self.index_config:
                version = _get_version(cur, table="vector_migrations")
                for v, migration in enumerate(
                    self.VECTOR_MIGRATIONS[version + 1 :], start=version + 1
                ):
                    if migration.condition and not migration.condition(self):
                        continue
                    sql = migration.sql
                    if migration.params:
                        params = {
                            k: v(self) if v is not None and callable(v) else v
                            for k, v in migration.params.items()
                        }
                        sql = sql % params
                    cur.execute(sql)
                    cur.execute("INSERT INTO vector_migrations (v) VALUES (%s)", (v,))


class Row(TypedDict):
    key: str
    value: Any
    prefix: str
    created_at: datetime
    updated_at: datetime


# 비공개 유틸리티

_DEFAULT_ANN_CONFIG = ANNIndexConfig(
    vector_type="vector",
)


def _get_vector_type_ops(store: BasePostgresStore) -> str:
    """구성을 기반으로 벡터 유형 연산자 클래스를 가져옵니다."""
    if not store.index_config:
        return "vector_cosine_ops"

    config = store.index_config
    index_config = config.get("ann_index_config", _DEFAULT_ANN_CONFIG).copy()
    vector_type = cast(str, index_config.get("vector_type", "vector"))
    if vector_type not in ("vector", "halfvec"):
        raise ValueError(
            f"Vector type must be 'vector' or 'halfvec', got {vector_type}"
        )

    distance_type = config.get("distance_type", "cosine")

    # 일반 벡터용
    type_prefix = {"vector": "vector", "halfvec": "halfvec"}[vector_type]

    if distance_type not in ("l2", "inner_product", "cosine"):
        raise ValueError(
            f"Vector type {vector_type} only supports 'l2', 'inner_product', or 'cosine' distance, got {distance_type}"
        )

    distance_suffix = {
        "l2": "l2_ops",
        "inner_product": "ip_ops",
        "cosine": "cosine_ops",
    }[distance_type]

    return f"{type_prefix}_{distance_suffix}"


def _get_index_params(store: Any) -> tuple[str, dict[str, Any]]:
    """구성을 기반으로 인덱스 유형 및 구성을 가져옵니다."""
    if not store.index_config:
        return "hnsw", {}

    config = cast(PostgresIndexConfig, store.index_config)
    index_config = config.get("ann_index_config", _DEFAULT_ANN_CONFIG).copy()
    kind = index_config.pop("kind", "hnsw")
    index_config.pop("vector_type", None)
    return kind, index_config


def _namespace_to_text(
    namespace: tuple[str, ...], handle_wildcards: bool = False
) -> str:
    """네임스페이스 튜플을 텍스트 문자열로 변환합니다."""
    if handle_wildcards:
        namespace = tuple("%" if val == "*" else val for val in namespace)
    return ".".join(namespace)


def _row_to_item(
    namespace: tuple[str, ...],
    row: Row,
    *,
    loader: Callable[[bytes | orjson.Fragment], dict[str, Any]] | None = None,
) -> Item:
    """데이터베이스의 행을 Item으로 변환합니다.

    Args:
        namespace: Item 네임스페이스
        row: 데이터베이스 행
        loader: dict가 아닌 값을 위한 선택적 값 로더
    """
    val = row["value"]
    if not isinstance(val, dict):
        val = (loader or _json_loads)(val)

    kwargs = {
        "key": row["key"],
        "namespace": namespace,
        "value": val,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }

    return Item(**kwargs)


def _row_to_search_item(
    namespace: tuple[str, ...],
    row: Row,
    *,
    loader: Callable[[bytes | orjson.Fragment], dict[str, Any]] | None = None,
) -> SearchItem:
    """데이터베이스의 행을 Item으로 변환합니다."""
    loader = loader or _json_loads
    val = row["value"]
    score = row.get("score")
    if score is not None:
        try:
            score = float(score)  # type: ignore[arg-type]
        except ValueError:
            logger.warning("Invalid score: %s", score)
            score = None
    return SearchItem(
        value=val if isinstance(val, dict) else loader(val),
        key=row["key"],
        namespace=namespace,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        score=score,
    )


def _group_ops(ops: Iterable[Op]) -> tuple[dict[type, list[tuple[int, Op]]], int]:
    grouped_ops: dict[type, list[tuple[int, Op]]] = defaultdict(list)
    tot = 0
    for idx, op in enumerate(ops):
        grouped_ops[type(op)].append((idx, op))
        tot += 1
    return grouped_ops, tot


def _json_loads(content: bytes | orjson.Fragment) -> Any:
    if isinstance(content, orjson.Fragment):
        if hasattr(content, "buf"):
            content = content.buf
        else:
            if isinstance(content.contents, bytes):
                content = content.contents
            else:
                content = content.contents.encode()
    return orjson.loads(cast(bytes, content))


def _decode_ns_bytes(namespace: str | bytes | list) -> tuple[str, ...]:
    if isinstance(namespace, list):
        return tuple(namespace)
    if isinstance(namespace, bytes):
        namespace = namespace.decode()[1:]
    return tuple(namespace.split("."))


def get_distance_operator(store: Any) -> tuple[str, str]:
    """구성을 기반으로 거리 연산자 및 점수 표현식을 가져옵니다."""
    # 참고: 현재 벡터 및 비벡터 필터를 혼합하는 PGVector의 지원에 대한 제한으로 인해
    # ANN 인덱스를 사용하지 않습니다
    # 인덱스를 사용하려면 PGVector가 다음을 기대합니다:
    #  - 표현식이 아닌 연산자로 ORDER BY (부정도 차단함)
    #  - 오름차순 정렬
    #  - 모든 WHERE 절은 부분 인덱스 위에 있어야 함.
    # 이 중 하나라도 위반하면 순차 스캔을 사용합니다
    # 자세한 내용은 https://github.com/pgvector/pgvector/issues/216 및
    # pgvector 문서를 참조하세요.
    if not store.index_config:
        raise ValueError(
            "Embedding configuration is required for vector operations "
            f"(for semantic search). "
            f"Please provide an Embeddings when initializing the {store.__class__.__name__}."
        )

    config = cast(PostgresIndexConfig, store.index_config)
    distance_type = config.get("distance_type", "cosine")

    # 연산자와 점수 표현식을 반환합니다
    # 연산자는 CTE에서 사용되며 오름차순 ORDER
    # 정렬 절과 호환됩니다.
    # 점수 표현식은 최종 쿼리에서 사용되며
    # 내림차순 ORDER 정렬 절 및 유사도 점수가 무엇이어야 하는지에 대한
    # 사용자의 기대와 호환됩니다.
    if distance_type == "l2":
        # 최종: "-(sv.embedding <-> %s::%s)"
        # 정렬 순서가 동일하도록 "l2 유사도"를 반환합니다
        return "sv.embedding <-> %s::%s", "-scored.neg_score"
    elif distance_type == "inner_product":
        # 최종: "-(sv.embedding <#> %s::%s)"
        return "sv.embedding <#> %s::%s", "-(scored.neg_score)"
    else:  # 코사인 유사도
        # 최종:  "1 - (sv.embedding <=> %s::%s)"
        return "sv.embedding <=> %s::%s", "1 - scored.neg_score"


def _ensure_index_config(
    index_config: PostgresIndexConfig,
) -> tuple[Embeddings | None, PostgresIndexConfig]:
    index_config = index_config.copy()
    tokenized: list[tuple[str, Literal["$"] | list[str]]] = []
    tot = 0
    fields = index_config.get("fields") or ["$"]
    if isinstance(fields, str):
        fields = [fields]
    if not isinstance(fields, list):
        raise ValueError(f"Text fields must be a list or a string. Got {fields}")
    for p in fields:
        if p == "$":
            tokenized.append((p, "$"))
            tot += 1
        else:
            toks = tokenize_path(p)
            tokenized.append((p, toks))
            tot += len(toks)
    index_config["__tokenized_fields"] = tokenized
    index_config["__estimated_num_vectors"] = tot
    embeddings = ensure_embeddings(
        index_config.get("embed"),
    )
    return embeddings, index_config


PLACEHOLDER = object()
