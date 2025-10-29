from __future__ import annotations

import concurrent.futures
import datetime
import logging
import re
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal, NamedTuple, cast

import orjson
import sqlite_vec  # type: ignore[import-untyped]
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

_AIO_ERROR_MSG = (
    "SqliteStore는 비동기 메서드를 지원하지 않습니다. "
    "대신 AsyncSqliteStore를 사용하세요.\n"
    "from langgraph.store.sqlite.aio import AsyncSqliteStore\n"
)

logger = logging.getLogger(__name__)

MIGRATIONS = [
    """
CREATE TABLE IF NOT EXISTS store (
    -- 'prefix'는 문서의 'namespace'를 나타냄
    prefix text NOT NULL,
    key text NOT NULL,
    value text NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (prefix, key)
);
""",
    """
-- prefix로 더 빠른 조회를 위함
CREATE INDEX IF NOT EXISTS store_prefix_idx ON store (prefix);
""",
    """
-- store 테이블에 expires_at 컬럼 추가
ALTER TABLE store
ADD COLUMN expires_at TIMESTAMP;
""",
    """
-- store 테이블에 ttl_minutes 컬럼 추가
ALTER TABLE store
ADD COLUMN ttl_minutes REAL;
""",
    """
-- 효율적인 TTL 정리를 위한 인덱스 추가
CREATE INDEX IF NOT EXISTS idx_store_expires_at ON store (expires_at)
WHERE expires_at IS NOT NULL;
""",
]

VECTOR_MIGRATIONS = [
    """
CREATE TABLE IF NOT EXISTS store_vectors (
    prefix text NOT NULL,
    key text NOT NULL,
    field_name text NOT NULL,
    embedding BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (prefix, key, field_name),
    FOREIGN KEY (prefix, key) REFERENCES store(prefix, key) ON DELETE CASCADE
);
""",
]


class SqliteIndexConfig(IndexConfig):
    """SQLite 저장소의 벡터 임베딩에 대한 구성입니다."""

    pass


def _namespace_to_text(
    namespace: tuple[str, ...], handle_wildcards: bool = False
) -> str:
    """네임스페이스 튜플을 텍스트 문자열로 변환합니다."""
    if handle_wildcards:
        namespace = tuple("%" if val == "*" else val for val in namespace)
    return ".".join(namespace)


def _decode_ns_text(namespace: str) -> tuple[str, ...]:
    """네임스페이스 문자열을 튜플로 변환합니다."""
    return tuple(namespace.split("."))


def _validate_filter_key(key: str) -> None:
    """필터 키가 SQL 쿼리에서 안전하게 사용될 수 있는지 검증합니다.

    Args:
        key: 검증할 필터 키입니다.

    Raises:
        ValueError: SQL 인젝션을 가능하게 하는 잘못된 문자가 포함된 경우
    """
    # 알파벳 숫자, 밑줄, 점 및 하이픈 허용
    # SQL 인젝션을 방지하면서 일반적인 JSON 속성 이름을 포함
    if not re.match(r"^[a-zA-Z0-9_.-]+$", key):
        raise ValueError(
            f"Invalid filter key: '{key}'. Filter keys must contain only alphanumeric characters, underscores, dots, and hyphens."
        )


def _json_loads(content: bytes | str | orjson.Fragment) -> Any:
    if isinstance(content, orjson.Fragment):
        if hasattr(content, "buf"):
            content = content.buf
        else:
            if isinstance(content.contents, bytes):
                content = content.contents
            else:
                content = content.contents.encode()
        return orjson.loads(cast(bytes, content))
    elif isinstance(content, bytes):
        return orjson.loads(content)
    else:
        return orjson.loads(content)


def _row_to_item(
    namespace: tuple[str, ...],
    row: dict[str, Any],
    *,
    loader: Callable[[bytes | str | orjson.Fragment], dict[str, Any]] | None = None,
) -> Item:
    """데이터베이스의 행을 Item으로 변환합니다."""
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
    row: dict[str, Any],
    *,
    loader: Callable[[bytes | str | orjson.Fragment], dict[str, Any]] | None = None,
) -> SearchItem:
    """데이터베이스의 행을 SearchItem으로 변환합니다."""
    loader = loader or _json_loads
    val = row["value"]
    score = row.get("score")
    if score is not None:
        try:
            score = float(score)
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


class PreparedGetQuery(NamedTuple):
    query: str  # 실행할 메인 쿼리
    params: tuple  # 메인 쿼리의 매개변수
    namespace: tuple[str, ...]  # 네임스페이스 정보
    items: list  # 이 쿼리가 대상으로 하는 항목 목록
    kind: Literal["get", "refresh"]


class BaseSqliteStore:
    """SQLite 저장소의 공유 베이스 클래스입니다."""

    MIGRATIONS = MIGRATIONS
    VECTOR_MIGRATIONS = VECTOR_MIGRATIONS
    supports_ttl = True
    index_config: SqliteIndexConfig | None = None
    ttl_config: TTLConfig | None = None

    def _get_batch_GET_ops_queries(
        self, get_ops: Sequence[tuple[int, GetOp]]
    ) -> list[PreparedGetQuery]:
        """
        네임스페이스당 여러 키를 가져오고 (선택적으로 TTL을 새로 고치는) 쿼리를 빌드합니다.

        다음을 포함할 수 있는 PreparedGetQuery 객체 목록을 반환합니다:
        - TTL 새로 고침 작업을 위한 kind='refresh' 쿼리
        - 데이터 검색 작업을 위한 kind='get' 쿼리
        """
        namespace_groups = defaultdict(list)
        refresh_ttls = defaultdict(list)
        for idx, op in get_ops:
            namespace_groups[op.namespace].append((idx, op.key))
            refresh_ttls[op.namespace].append(getattr(op, "refresh_ttl", False))

        results = []
        for namespace, items in namespace_groups.items():
            _, keys = zip(*items, strict=False)
            this_refresh_ttls = refresh_ttls[namespace]
            refresh_ttl_any = any(this_refresh_ttls)

            # 데이터를 가져오기 위한 메인 쿼리는 항상 추가
            select_query = f"""
                SELECT key, value, created_at, updated_at, expires_at, ttl_minutes
                FROM store
                WHERE prefix = ? AND key IN ({",".join(["?"] * len(keys))})
            """
            select_params = (_namespace_to_text(namespace), *keys)
            results.append(
                PreparedGetQuery(select_query, select_params, namespace, items, "get")
            )

            # 필요한 경우 TTL 새로 고침 쿼리 추가
            if (
                refresh_ttl_any
                and self.ttl_config
                and self.ttl_config.get("refresh_on_read", False)
            ):
                placeholders = ",".join(["?"] * len(keys))
                update_query = f"""
                    UPDATE store
                    SET expires_at = DATETIME(CURRENT_TIMESTAMP, '+' || ttl_minutes || ' minutes')
                    WHERE prefix = ? 
                    AND key IN ({placeholders})
                    AND ttl_minutes IS NOT NULL
                """
                update_params = (_namespace_to_text(namespace), *keys)
                results.append(
                    PreparedGetQuery(
                        update_query, update_params, namespace, items, "refresh"
                    )
                )

        return results

    def _prepare_batch_PUT_queries(
        self, put_ops: Sequence[tuple[int, PutOp]]
    ) -> tuple[
        list[tuple[str, Sequence]],
        tuple[str, Sequence[tuple[str, str, str, str]]] | None,
    ]:
        # 마지막 쓰기가 우선
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
                placeholders = ",".join(["?" for _ in keys])
                query = (
                    f"DELETE FROM store WHERE prefix = ? AND key IN ({placeholders})"
                )
                params = (_namespace_to_text(namespace), *keys)
                queries.append((query, params))

        embedding_request: tuple[str, Sequence[tuple[str, str, str, str]]] | None = None
        if inserts:
            values = []
            insertion_params = []
            vector_values = []
            embedding_request_params = []
            now = datetime.datetime.now(datetime.timezone.utc)

            # 먼저 메인 저장소 삽입 처리
            for op in inserts:
                if op.ttl is None:
                    expires_at = None
                else:
                    expires_at = now + datetime.timedelta(minutes=op.ttl)
                values.append("(?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)")
                insertion_params.extend(
                    [
                        _namespace_to_text(op.namespace),
                        op.key,
                        orjson.dumps(cast(dict, op.value)),
                        expires_at,
                        op.ttl,
                    ]
                )

            # 그 다음 구성된 경우 임베딩 처리
            if self.index_config:
                for op in inserts:
                    if op.index is False:
                        continue
                    value = op.value
                    ns = _namespace_to_text(op.namespace)
                    k = op.key

                    if op.index is None:
                        paths = self.index_config["__tokenized_fields"]
                    else:
                        paths = [(ix, tokenize_path(ix)) for ix in op.index]

                    for path, tokenized_path in paths:
                        texts = get_text_at_path(value, tokenized_path)
                        for i, text in enumerate(texts):
                            pathname = f"{path}.{i}" if len(texts) > 1 else path
                            vector_values.append(
                                "(?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                            )
                            embedding_request_params.append((ns, k, pathname, text))

            values_str = ",".join(values)
            query = f"""
                INSERT OR REPLACE INTO store (prefix, key, value, created_at, updated_at, expires_at, ttl_minutes)
                VALUES {values_str}
            """
            queries.append((query, insertion_params))

            if vector_values:
                values_str = ",".join(vector_values)
                query = f"""
                    INSERT OR REPLACE INTO store_vectors (prefix, key, field_name, embedding, created_at, updated_at)
                    VALUES {values_str}
                """
                embedding_request = (query, embedding_request_params)

        return queries, embedding_request

    def _prepare_batch_search_queries(
        self, search_ops: Sequence[tuple[int, SearchOp]]
    ) -> tuple[
        list[
            tuple[str, list[None | str | list[float]], bool]
        ],  # queries, params, needs_refresh
        list[tuple[int, str]],  # idx, query_text pairs to embed
    ]:
        """
        SearchOp별 SQL 쿼리(선택적 TTL 새로 고침 플래그 포함)와 임베딩 요청을 빌드합니다.
        Returns:
        - queries: (SQL, param_list, needs_ttl_refresh_flag)의 목록
        - embedding_requests: (original_index_in_search_ops, text_query) 쌍의 목록
        """
        queries = []
        embedding_requests = []

        for idx, (_, op) in enumerate(search_ops):
            # 먼저 필터 조건 빌드
            filter_params = []
            filter_conditions = []
            if op.filter:
                for key, value in op.filter.items():
                    _validate_filter_key(key)

                    if isinstance(value, dict):
                        for op_name, val in value.items():
                            condition, filter_params_ = self._get_filter_condition(
                                key, op_name, val
                            )
                            filter_conditions.append(condition)
                            filter_params.extend(filter_params_)
                    else:
                        # SQLite json_extract는 따옴표 없는 문자열 값을 반환
                        if isinstance(value, str):
                            filter_conditions.append(
                                "json_extract(value, '$."
                                + key
                                + "') = '"
                                + value.replace("'", "''")
                                + "'"
                            )
                        elif value is None:
                            filter_conditions.append(
                                "json_extract(value, '$." + key + "') IS NULL"
                            )
                        elif isinstance(value, bool):
                            # SQLite JSON은 boolean을 정수로 저장
                            filter_conditions.append(
                                "json_extract(value, '$."
                                + key
                                + "') = "
                                + ("1" if value else "0")
                            )
                        elif isinstance(value, (int, float)):
                            filter_conditions.append(
                                "json_extract(value, '$." + key + "') = " + str(value)
                            )
                        else:
                            # 복잡한 객체(list, dict, ...) – JSON 텍스트 비교
                            filter_conditions.append(
                                "json_extract(value, '$." + key + "') = ?"
                            )
                            # orjson.dumps는 bytes 반환 → SQLite가 TEXT로 인식하도록 str로 디코드
                            filter_params.append(orjson.dumps(value).decode())

            # 벡터 검색 분기
            if op.query and self.index_config:
                embedding_requests.append((idx, op.query))

                # 거리 타입에 따라 유사도 함수와 점수 표현식 선택
                distance_type = self.index_config.get("distance_type", "cosine")

                if distance_type == "cosine":
                    score_expr = "1.0 - vec_distance_cosine(sv.embedding, ?)"
                elif distance_type == "l2":
                    score_expr = "vec_distance_L2(sv.embedding, ?)"
                elif distance_type == "inner_product":
                    # 내적의 경우, 높은 값이 더 좋도록 하기 위해 결과를 부정합니다
                    # 내적 유사도는 벡터가 더 유사할수록 높기 때문입니다
                    score_expr = "-1 * vec_distance_L1(sv.embedding, ?)"
                else:
                    # 기본값은 코사인 유사도
                    score_expr = "1.0 - vec_distance_cosine(sv.embedding, ?)"

                filter_str = (
                    ""
                    if not filter_conditions
                    else " AND " + " AND ".join(filter_conditions)
                )
                if op.namespace_prefix:
                    prefix_filter_str = f"WHERE s.prefix LIKE ? {filter_str} "
                    ns_args: Sequence = (f"{_namespace_to_text(op.namespace_prefix)}%",)
                else:
                    ns_args = ()
                    if filter_str:
                        prefix_filter_str = f"WHERE {filter_str[5:]} "
                    else:
                        prefix_filter_str = ""

                # CTE를 사용하여 점수를 계산하고, 고유한 결과를 위해 SQLite 호환 방식 사용
                base_query = f"""
                    WITH scored AS (
                        SELECT s.prefix, s.key, s.value, s.created_at, s.updated_at, s.expires_at, s.ttl_minutes,
                            {score_expr} AS score
                        FROM store s
                        JOIN store_vectors sv ON s.prefix = sv.prefix AND s.key = sv.key
                        {prefix_filter_str}
                            ORDER BY score DESC 
                        LIMIT ?
                    ),
                    ranked AS (
                        SELECT prefix, key, value, created_at, updated_at, expires_at, ttl_minutes, score,
                                ROW_NUMBER() OVER (PARTITION BY prefix, key ORDER BY score DESC) as rn
                        FROM scored
                    )
                    SELECT prefix, key, value, created_at, updated_at, expires_at, ttl_minutes, score
                    FROM ranked
                    WHERE rn = 1
                        ORDER BY score DESC
                    LIMIT ?
                    OFFSET ?
                    """
                params = [
                    _PLACEHOLDER,  # Vector placeholder
                    *ns_args,
                    *filter_params,
                    op.limit * 2,  # Expanded limit for better results
                    op.limit,
                    op.offset,
                ]
            # 일반 검색 분기 (벡터 검색 없음)
            else:
                base_query = """
                    SELECT prefix, key, value, created_at, updated_at, expires_at, ttl_minutes, NULL as score
                    FROM store
                    WHERE prefix LIKE ?
                """
                params = [f"{_namespace_to_text(op.namespace_prefix)}%"]

                if filter_conditions:
                    params.extend(filter_params)
                    base_query += " AND " + " AND ".join(filter_conditions)

                base_query += " ORDER BY updated_at DESC"
                base_query += " LIMIT ? OFFSET ?"
                params.extend([op.limit, op.offset])

                # 쿼리 디버깅
                logger.debug(f"Search query: {base_query}")
                logger.debug(f"Search params: {params}")

            # TTL 새로 고침이 필요한지 확인
            needs_ttl_refresh = bool(
                op.refresh_ttl
                and self.ttl_config
                and self.ttl_config.get("refresh_on_read", False)
            )

            # base_query가 이제 final_sql이고, 새로 고침 플래그를 전달
            final_sql = base_query
            final_params = params

            queries.append((final_sql, final_params, needs_ttl_refresh))

        return queries, embedding_requests

    def _get_batch_list_namespaces_queries(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
    ) -> list[tuple[str, Sequence]]:
        queries: list[tuple[str, Sequence]] = []

        for _, op in list_ops:
            where_clauses: list[str] = []
            params: list[Any] = []

            if op.match_conditions:
                for cond in op.match_conditions:
                    if cond.match_type == "prefix":
                        where_clauses.append("prefix LIKE ?")
                        params.append(
                            f"{_namespace_to_text(cond.path, handle_wildcards=True)}%"
                        )
                    elif cond.match_type == "suffix":
                        where_clauses.append("prefix LIKE ?")
                        params.append(
                            f"%{_namespace_to_text(cond.path, handle_wildcards=True)}"
                        )
                    else:
                        logger.warning(
                            "list_namespaces에서 알 수 없는 match_type: %s", cond.match_type
                        )

            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            if op.max_depth is not None:
                query = f"""
                    WITH RECURSIVE split(original, truncated, remainder, depth) AS (
                        SELECT
                            prefix          AS original,
                            ''              AS truncated,
                            prefix          AS remainder,
                            0               AS depth
                        FROM (SELECT DISTINCT prefix FROM store {where_sql})

                        UNION ALL

                        SELECT
                            original,
                            CASE
                                WHEN depth = 0
                                    THEN substr(remainder,
                                                1,
                                                CASE
                                                    WHEN instr(remainder, '.') > 0
                                                        THEN instr(remainder, '.') - 1
                                                    ELSE length(remainder)
                                                END)
                                ELSE
                                    truncated || '.' ||
                                    substr(remainder,
                                        1,
                                        CASE
                                            WHEN instr(remainder, '.') > 0
                                                THEN instr(remainder, '.') - 1
                                            ELSE length(remainder)
                                        END)
                            END                              AS truncated,
                            CASE
                                WHEN instr(remainder, '.') > 0
                                    THEN substr(remainder, instr(remainder, '.') + 1)
                                ELSE ''
                            END                              AS remainder,
                            depth + 1                       AS depth
                        FROM split
                        WHERE remainder <> ''
                            AND depth < ?
                    )
                    SELECT DISTINCT truncated AS prefix
                    FROM split
                    WHERE depth = ? OR remainder = ''
                    ORDER BY prefix
                    LIMIT ? OFFSET ?
                """
                params.extend([op.max_depth, op.max_depth, op.limit, op.offset])

            else:
                query = f"""
                    SELECT DISTINCT prefix
                    FROM store
                    {where_sql}
                    ORDER BY prefix
                    LIMIT ? OFFSET ?
                """
                params.extend([op.limit, op.offset])

            queries.append((query, tuple(params)))

        return queries

    def _get_filter_condition(self, key: str, op: str, value: Any) -> tuple[str, list]:
        """Helper to generate filter conditions."""
        _validate_filter_key(key)

        # SQLite JSON 추출 비교를 위해 값을 올바르게 포맷해야 함
        if op == "$eq":
            if isinstance(value, str):
                # 따옴표 없는 json_extract 결과에 대한 올바른 따옴표 처리를 통한 직접 문자열 비교
                return (
                    f"json_extract(value, '$.{key}') = '"
                    + value.replace("'", "''")
                    + "'",
                    [],
                )
            elif value is None:
                return f"json_extract(value, '$.{key}') IS NULL", []
            elif isinstance(value, bool):
                # SQLite JSON은 boolean을 정수로 저장
                return f"json_extract(value, '$.{key}') = {1 if value else 0}", []
            elif isinstance(value, (int, float)):
                return f"json_extract(value, '$.{key}') = {value}", []
            else:
                return f"json_extract(value, '$.{key}') = ?", [orjson.dumps(value)]
        elif op == "$gt":
            # 숫자 값의 경우 SQLite는 문자열이 아닌 숫자로 비교해야 함
            if isinstance(value, (int, float)):
                return f"CAST(json_extract(value, '$.{key}') AS REAL) > {value}", []
            elif isinstance(value, str):
                return (
                    f"json_extract(value, '$.{key}') > '"
                    + value.replace("'", "''")
                    + "'",
                    [],
                )
            else:
                return f"json_extract(value, '$.{key}') > ?", [orjson.dumps(value)]
        elif op == "$gte":
            if isinstance(value, (int, float)):
                return f"CAST(json_extract(value, '$.{key}') AS REAL) >= {value}", []
            elif isinstance(value, str):
                return (
                    f"json_extract(value, '$.{key}') >= '"
                    + value.replace("'", "''")
                    + "'",
                    [],
                )
            else:
                return f"json_extract(value, '$.{key}') >= ?", [orjson.dumps(value)]
        elif op == "$lt":
            if isinstance(value, (int, float)):
                return f"CAST(json_extract(value, '$.{key}') AS REAL) < {value}", []
            elif isinstance(value, str):
                return (
                    f"json_extract(value, '$.{key}') < '"
                    + value.replace("'", "''")
                    + "'",
                    [],
                )
            else:
                return f"json_extract(value, '$.{key}') < ?", [orjson.dumps(value)]
        elif op == "$lte":
            if isinstance(value, (int, float)):
                return f"CAST(json_extract(value, '$.{key}') AS REAL) <= {value}", []
            elif isinstance(value, str):
                return (
                    f"json_extract(value, '$.{key}') <= '"
                    + value.replace("'", "''")
                    + "'",
                    [],
                )
            else:
                return f"json_extract(value, '$.{key}') <= ?", [orjson.dumps(value)]
        elif op == "$ne":
            if isinstance(value, str):
                return (
                    f"json_extract(value, '$.{key}') != '"
                    + value.replace("'", "''")
                    + "'",
                    [],
                )
            elif value is None:
                return f"json_extract(value, '$.{key}') IS NOT NULL", []
            elif isinstance(value, bool):
                return f"json_extract(value, '$.{key}') != {1 if value else 0}", []
            elif isinstance(value, (int, float)):
                return f"json_extract(value, '$.{key}') != {value}", []
            else:
                return f"json_extract(value, '$.{key}') != ?", [orjson.dumps(value)]
        else:
            raise ValueError(f"Unsupported operator: {op}")


class SqliteStore(BaseSqliteStore, BaseStore):
    """선택적 벡터 검색 기능을 갖춘 SQLite 기반 저장소입니다.

    Examples:
        기본 설정 및 사용법:
        ```python
        from langgraph.store.sqlite import SqliteStore
        import sqlite3

        conn = sqlite3.connect(":memory:")
        store = SqliteStore(conn)
        store.setup()  # 마이그레이션 실행. 한 번만 수행

        # 데이터 저장 및 검색
        store.put(("users", "123"), "prefs", {"theme": "dark"})
        item = store.get(("users", "123"), "prefs")
        ```

        또는 편리한 from_conn_string 헬퍼 사용:
        ```python
        from langgraph.store.sqlite import SqliteStore

        with SqliteStore.from_conn_string(":memory:") as store:
            store.setup()

            # 데이터 저장 및 검색
            store.put(("users", "123"), "prefs", {"theme": "dark"})
            item = store.get(("users", "123"), "prefs")
        ```

        LangChain 임베딩을 사용한 벡터 검색:
        ```python
        from langchain.embeddings import OpenAIEmbeddings
        from langgraph.store.sqlite import SqliteStore

        with SqliteStore.from_conn_string(
            ":memory:",
            index={
                "dims": 1536,
                "embed": OpenAIEmbeddings(),
                "fields": ["text"]  # 임베딩할 필드 지정
            }
        ) as store:
            store.setup()  # 마이그레이션 실행

            # 문서 저장
            store.put(("docs",), "doc1", {"text": "Python tutorial"})
            store.put(("docs",), "doc2", {"text": "TypeScript guide"})
            store.put(("docs",), "doc3", {"text": "Other guide"}, index=False)  # 인덱싱하지 않음

            # 유사도로 검색
            results = store.search(("docs",), query="programming guides", limit=2)
        ```

    Note:
        의미론적 검색은 기본적으로 비활성화되어 있습니다. 저장소를 만들 때 `index` 구성을
        제공하여 활성화할 수 있습니다. 이 구성이 없으면 `put` 또는 `aput`에 전달된 모든
        `index` 인수는 효과가 없습니다.

    Warning:
        필요한 테이블과 인덱스를 생성하기 위해 첫 사용 전에 반드시 `setup()`을 호출하세요.
    """

    MIGRATIONS = MIGRATIONS
    VECTOR_MIGRATIONS = VECTOR_MIGRATIONS
    supports_ttl = True

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        deserializer: Callable[[bytes | str | orjson.Fragment], dict[str, Any]]
        | None = None,
        index: SqliteIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ):
        super().__init__()
        self._deserializer = deserializer
        self.conn = conn
        self.lock = threading.Lock()
        self.is_setup = False
        self.index_config = index
        if self.index_config:
            self.embeddings, self.index_config = _ensure_index_config(self.index_config)
        else:
            self.embeddings = None
        self.ttl_config = ttl
        self._ttl_sweeper_thread: threading.Thread | None = None
        self._ttl_stop_event = threading.Event()

    def _get_batch_GET_ops_queries(
        self, get_ops: Sequence[tuple[int, GetOp]]
    ) -> list[PreparedGetQuery]:
        """
        네임스페이스당 여러 키를 가져오고 (선택적으로 TTL을 새로 고치는) 쿼리를 빌드합니다.

        다음을 포함할 수 있는 PreparedGetQuery 객체 목록을 반환합니다:
        - TTL 새로 고침 작업을 위한 kind='refresh' 쿼리
        - 데이터 검색 작업을 위한 kind='get' 쿼리
        """
        namespace_groups = defaultdict(list)
        refresh_ttls = defaultdict(list)
        for idx, op in get_ops:
            namespace_groups[op.namespace].append((idx, op.key))
            refresh_ttls[op.namespace].append(getattr(op, "refresh_ttl", False))

        results = []
        for namespace, items in namespace_groups.items():
            _, keys = zip(*items, strict=False)
            this_refresh_ttls = refresh_ttls[namespace]
            refresh_ttl_any = any(this_refresh_ttls)

            # 데이터를 가져오기 위한 메인 쿼리는 항상 추가
            select_query = f"""
                SELECT key, value, created_at, updated_at, expires_at, ttl_minutes
                FROM store
                WHERE prefix = ? AND key IN ({",".join(["?"] * len(keys))})
            """
            select_params = (_namespace_to_text(namespace), *keys)
            results.append(
                PreparedGetQuery(select_query, select_params, namespace, items, "get")
            )

            # 필요한 경우 TTL 새로 고침 쿼리 추가
            if (
                refresh_ttl_any
                and self.ttl_config
                and self.ttl_config.get("refresh_on_read", False)
            ):
                placeholders = ",".join(["?"] * len(keys))
                update_query = f"""
                    UPDATE store
                    SET expires_at = DATETIME(CURRENT_TIMESTAMP, '+' || ttl_minutes || ' minutes')
                    WHERE prefix = ? 
                    AND key IN ({placeholders})
                    AND ttl_minutes IS NOT NULL
                """
                update_params = (_namespace_to_text(namespace), *keys)
                results.append(
                    PreparedGetQuery(
                        update_query, update_params, namespace, items, "refresh"
                    )
                )

        return results

    def _get_filter_condition(self, key: str, op: str, value: Any) -> tuple[str, list]:
        """Helper to generate filter conditions."""
        _validate_filter_key(key)

        # SQLite JSON 추출 비교를 위해 값을 올바르게 포맷해야 함
        if op == "$eq":
            if isinstance(value, str):
                # 따옴표 없는 json_extract 결과에 대한 올바른 따옴표 처리를 통한 직접 문자열 비교
                return (
                    f"json_extract(value, '$.{key}') = '"
                    + value.replace("'", "''")
                    + "'",
                    [],
                )
            elif value is None:
                return f"json_extract(value, '$.{key}') IS NULL", []
            elif isinstance(value, bool):
                # SQLite JSON은 boolean을 정수로 저장
                return f"json_extract(value, '$.{key}') = {1 if value else 0}", []
            elif isinstance(value, (int, float)):
                return f"json_extract(value, '$.{key}') = {value}", []
            else:
                return f"json_extract(value, '$.{key}') = ?", [orjson.dumps(value)]
        elif op == "$gt":
            # 숫자 값의 경우 SQLite는 문자열이 아닌 숫자로 비교해야 함
            if isinstance(value, (int, float)):
                return f"CAST(json_extract(value, '$.{key}') AS REAL) > {value}", []
            elif isinstance(value, str):
                return (
                    f"json_extract(value, '$.{key}') > '"
                    + value.replace("'", "''")
                    + "'",
                    [],
                )
            else:
                return f"json_extract(value, '$.{key}') > ?", [orjson.dumps(value)]
        elif op == "$gte":
            if isinstance(value, (int, float)):
                return f"CAST(json_extract(value, '$.{key}') AS REAL) >= {value}", []
            elif isinstance(value, str):
                return (
                    f"json_extract(value, '$.{key}') >= '"
                    + value.replace("'", "''")
                    + "'",
                    [],
                )
            else:
                return f"json_extract(value, '$.{key}') >= ?", [orjson.dumps(value)]
        elif op == "$lt":
            if isinstance(value, (int, float)):
                return f"CAST(json_extract(value, '$.{key}') AS REAL) < {value}", []
            elif isinstance(value, str):
                return (
                    f"json_extract(value, '$.{key}') < '"
                    + value.replace("'", "''")
                    + "'",
                    [],
                )
            else:
                return f"json_extract(value, '$.{key}') < ?", [orjson.dumps(value)]
        elif op == "$lte":
            if isinstance(value, (int, float)):
                return f"CAST(json_extract(value, '$.{key}') AS REAL) <= {value}", []
            elif isinstance(value, str):
                return (
                    f"json_extract(value, '$.{key}') <= '"
                    + value.replace("'", "''")
                    + "'",
                    [],
                )
            else:
                return f"json_extract(value, '$.{key}') <= ?", [orjson.dumps(value)]
        elif op == "$ne":
            if isinstance(value, str):
                return (
                    f"json_extract(value, '$.{key}') != '"
                    + value.replace("'", "''")
                    + "'",
                    [],
                )
            elif value is None:
                return f"json_extract(value, '$.{key}') IS NOT NULL", []
            elif isinstance(value, bool):
                return f"json_extract(value, '$.{key}') != {1 if value else 0}", []
            elif isinstance(value, (int, float)):
                return f"json_extract(value, '$.{key}') != {value}", []
            else:
                return f"json_extract(value, '$.{key}') != ?", [orjson.dumps(value)]
        else:
            raise ValueError(f"Unsupported operator: {op}")

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
        *,
        index: SqliteIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> Iterator[SqliteStore]:
        """연결 문자열로부터 새로운 SqliteStore 인스턴스를 생성합니다.

        Args:
            conn_string (str): SQLite 연결 문자열입니다.
            index (Optional[SqliteIndexConfig]): 저장소의 인덱스 구성입니다.
            ttl (Optional[TTLConfig]): 저장소의 TTL(time-to-live) 구성입니다.

        Returns:
            SqliteStore: 새로운 SqliteStore 인스턴스입니다.
        """
        conn = sqlite3.connect(
            conn_string,
            check_same_thread=False,
            isolation_level=None,  # autocommit mode
        )
        try:
            yield cls(conn, index=index, ttl=ttl)
        finally:
            conn.close()

    @contextmanager
    def _cursor(self, *, transaction: bool = True) -> Iterator[sqlite3.Cursor]:
        """컨텍스트 매니저로 데이터베이스 커서를 생성합니다.

        Args:
            transaction (bool): DB 작업에 트랜잭션을 사용할지 여부
        """
        if not self.is_setup:
            self.setup()
        with self.lock:
            if transaction:
                self.conn.execute("BEGIN")

            cur = self.conn.cursor()
            try:
                yield cur
            finally:
                if transaction:
                    self.conn.execute("COMMIT")
                cur.close()

    def setup(self) -> None:
        """저장소 데이터베이스를 설정합니다.

        이 메서드는 SQLite 데이터베이스에 필요한 테이블이 없으면 생성하고
        데이터베이스 마이그레이션을 실행합니다. 첫 사용 전에 호출해야 합니다.
        """

        with self.lock:
            if self.is_setup:
                return
            # 마이그레이션 테이블이 없으면 생성
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS store_migrations (
                    v INTEGER PRIMARY KEY
                )
                """
            )

            # 현재 마이그레이션 버전 확인
            cur = self.conn.execute(
                "SELECT v FROM store_migrations ORDER BY v DESC LIMIT 1"
            )
            row = cur.fetchone()
            if row is None:
                version = -1
            else:
                version = row[0]

            # 마이그레이션 적용
            for v, sql in enumerate(self.MIGRATIONS[version + 1 :], start=version + 1):
                self.conn.executescript(sql)
                self.conn.execute("INSERT INTO store_migrations (v) VALUES (?)", (v,))

            # 인덱스 구성이 제공된 경우 벡터 마이그레이션 적용
            if self.index_config:
                # 벡터 마이그레이션 테이블이 없으면 생성
                self.conn.enable_load_extension(True)
                sqlite_vec.load(self.conn)
                self.conn.enable_load_extension(False)
                self.conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS vector_migrations (
                        v INTEGER PRIMARY KEY
                    )
                    """
                )

                # 현재 벡터 마이그레이션 버전 확인
                cur = self.conn.execute(
                    "SELECT v FROM vector_migrations ORDER BY v DESC LIMIT 1"
                )
                row = cur.fetchone()
                if row is None:
                    version = -1
                else:
                    version = row[0]

                # 벡터 마이그레이션 적용
                for v, sql in enumerate(
                    self.VECTOR_MIGRATIONS[version + 1 :], start=version + 1
                ):
                    self.conn.executescript(sql)
                    self.conn.execute(
                        "INSERT INTO vector_migrations (v) VALUES (?)", (v,)
                    )

            self.is_setup = True

    def sweep_ttl(self) -> int:
        """TTL을 기반으로 만료된 저장소 항목을 삭제합니다.

        Returns:
            int: 삭제된 항목의 개수입니다.
        """
        with self._cursor() as cur:
            cur.execute(
                """
                DELETE FROM store
                WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
                """
            )
            deleted_count = cur.rowcount
            return deleted_count

    def start_ttl_sweeper(
        self, sweep_interval_minutes: int | None = None
    ) -> concurrent.futures.Future[None]:
        """TTL을 기반으로 주기적으로 만료된 저장소 항목을 삭제합니다.

        Returns:
            대기하거나 취소할 수 있는 Future입니다.
        """
        if not self.ttl_config:
            future: concurrent.futures.Future[None] = concurrent.futures.Future()
            future.set_result(None)
            return future

        if self._ttl_sweeper_thread and self._ttl_sweeper_thread.is_alive():
            logger.info("TTL 스위퍼 스레드가 이미 실행 중입니다")
            # 기존 스레드를 취소하는 데 사용할 수 있는 future 반환
            future = concurrent.futures.Future()
            future.add_done_callback(
                lambda f: self._ttl_stop_event.set() if f.cancelled() else None
            )
            return future

        self._ttl_stop_event.clear()

        interval = float(
            sweep_interval_minutes or self.ttl_config.get("sweep_interval_minutes") or 5
        )
        logger.info(f"{interval}분 간격으로 저장소 TTL 스위퍼를 시작합니다")

        future = concurrent.futures.Future()

        def _sweep_loop() -> None:
            try:
                while not self._ttl_stop_event.is_set():
                    if self._ttl_stop_event.wait(interval * 60):
                        break

                    try:
                        expired_items = self.sweep_ttl()
                        if expired_items > 0:
                            logger.info(f"저장소에서 {expired_items}개의 만료된 항목을 정리했습니다")
                    except Exception as exc:
                        logger.exception(
                            "저장소 TTL 정리 반복 실패", exc_info=exc
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
            timeout: 스레드가 중지될 때까지 기다릴 최대 시간(초)입니다.
                `None`이면 무한정 기다립니다.

        Returns:
            bool: 스레드가 성공적으로 중지되었거나 실행 중이 아니었으면 True,
                스레드가 중지되기 전에 시간 초과에 도달하면 False입니다.
        """
        if not self._ttl_sweeper_thread or not self._ttl_sweeper_thread.is_alive():
            return True

        logger.info("TTL 스위퍼 스레드를 중지하는 중입니다")
        self._ttl_stop_event.set()

        self._ttl_sweeper_thread.join(timeout)
        success = not self._ttl_sweeper_thread.is_alive()

        if success:
            self._ttl_sweeper_thread = None
            logger.info("TTL 스위퍼 스레드가 중지되었습니다")
        else:
            logger.warning("TTL 스위퍼 스레드가 중지될 때까지 기다리는 중 시간 초과되었습니다")

        return success

    def __del__(self) -> None:
        """객체가 가비지 수집될 때 TTL 스위퍼 스레드가 중지되도록 보장합니다."""
        if hasattr(self, "_ttl_stop_event") and hasattr(self, "_ttl_sweeper_thread"):
            self.stop_ttl_sweeper(timeout=0.1)

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """작업 배치를 실행합니다.

        Args:
            ops (Iterable[Op]): 실행할 작업 목록

        Returns:
            list[Result]: 작업의 결과
        """
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        with self._cursor(transaction=True) as cur:
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
        cur: sqlite3.Cursor,
    ) -> None:
        # 각 네임스페이스의 모든 작업을 함께 실행하기 위해 네임스페이스별로 모든 쿼리 그룹화
        namespace_queries = defaultdict(list)
        for prepared_query in self._get_batch_GET_ops_queries(get_ops):
            namespace_queries[prepared_query.namespace].append(prepared_query)

        # 각 네임스페이스의 작업 처리
        for namespace, queries in namespace_queries.items():
            # TTL 새로 고침 쿼리를 먼저 실행
            for query in queries:
                if query.kind == "refresh":
                    try:
                        cur.execute(query.query, query.params)
                    except Exception as e:
                        raise ValueError(
                            f"Error executing TTL refresh: \n{query.query}\n{query.params}\n{e}"
                        ) from e

            # 그 다음 GET 쿼리를 실행하고 결과 처리
            for query in queries:
                if query.kind == "get":
                    try:
                        cur.execute(query.query, query.params)
                    except Exception as e:
                        raise ValueError(
                            f"Error executing GET query: \n{query.query}\n{query.params}\n{e}"
                        ) from e

                    rows = cur.fetchall()
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

                    # 이 쿼리의 결과 처리
                    for idx, key in query.items:
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
        cur: sqlite3.Cursor,
    ) -> None:
        queries, embedding_request = self._prepare_batch_PUT_queries(put_ops)
        if embedding_request:
            if self.embeddings is None:
                # 위에서 embedding_request를 반환하려면 임베딩 구성이 필요하므로
                # 여기에 도달하면 안 됨
                raise ValueError(
                    "Embedding configuration is required for vector operations "
                    f"(for semantic search). "
                    f"{self.__class__.__name__}를 초기화할 때 Embeddings를 제공하세요."
                )
            query, txt_params = embedding_request
            # 원시 텍스트를 벡터로 바꾸기 위해 매개변수 업데이트
            vectors = self.embeddings.embed_documents(
                [param[-1] for param in txt_params]
            )

            # 벡터를 SQLite 호환 형식으로 변환
            vector_params = []
            for (ns, k, pathname, _), vector in zip(txt_params, vectors, strict=False):
                vector_params.extend(
                    [ns, k, pathname, sqlite_vec.serialize_float32(vector)]
                )

            queries.append((query, vector_params))

        for query, params in queries:
            cur.execute(query, params)

    def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        cur: sqlite3.Cursor,
    ) -> None:
        prepared_queries, embedding_requests = self._prepare_batch_search_queries(
            search_ops
        )

        # 유사도 함수가 없으면 설정
        if embedding_requests and self.embeddings:
            # 검색 쿼리에 대한 임베딩 생성
            embeddings = self.embeddings.embed_documents(
                [query for _, query in embedding_requests]
            )

            # 플레이스홀더를 실제 임베딩으로 교체
            for (embed_req_idx, _), embedding in zip(
                embedding_requests, embeddings, strict=False
            ):
                if embed_req_idx < len(prepared_queries):
                    _params_list: list = prepared_queries[embed_req_idx][1]
                    for i, param in enumerate(_params_list):
                        if param is _PLACEHOLDER:
                            _params_list[i] = sqlite_vec.serialize_float32(embedding)
                else:
                    logger.warning(
                        f"임베딩 요청 인덱스 {embed_req_idx}가 prepared_queries의 범위를 벗어났습니다."
                    )

        for (original_op_idx, _), (query, params, needs_refresh) in zip(
            search_ops, prepared_queries, strict=False
        ):
            cur.execute(query, params)
            rows = cur.fetchall()

            if needs_refresh and rows and self.ttl_config:
                keys_to_refresh = []
                for row_data in rows:
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
                            cur.execute(update_query, update_params)
                        except Exception as e:
                            logger.error(
                                f"검색에 대한 TTL 새로 고침 업데이트 중 오류 발생: {e}"
                            )

            if "score" in query:  # 벡터 검색 쿼리
                items = [
                    _row_to_search_item(
                        _decode_ns_text(row[0]),
                        {
                            "key": row[1],
                            "value": row[2],
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
            else:  # 일반 검색 쿼리
                items = [
                    _row_to_search_item(
                        _decode_ns_text(row[0]),
                        {
                            "key": row[1],
                            "value": row[2],
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

    def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
        cur: sqlite3.Cursor,
    ) -> None:
        queries = self._get_batch_list_namespaces_queries(list_ops)
        for (query, params), (idx, _) in zip(queries, list_ops, strict=False):
            cur.execute(query, params)
            results[idx] = [_decode_ns_text(row[0]) for row in cur.fetchall()]

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """비동기 배치 작업 - SqliteStore에서는 지원되지 않습니다.

        비동기 작업의 경우 AsyncSqliteStore를 사용하세요.
        """
        raise NotImplementedError(_AIO_ERROR_MSG)


# 헬퍼 함수


def _ensure_index_config(
    index_config: SqliteIndexConfig,
) -> tuple[Any, SqliteIndexConfig]:
    """인덱스 구성을 처리하고 검증합니다."""
    index_config = index_config.copy()
    tokenized: list[tuple[str, Literal["$"] | list[str]]] = []
    tot = 0
    text_fields = index_config.get("text_fields") or ["$"]
    if isinstance(text_fields, str):
        text_fields = [text_fields]
    if not isinstance(text_fields, list):
        raise ValueError(f"Text fields must be a list or a string. Got {text_fields}")
    for p in text_fields:
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


_PLACEHOLDER = object()
