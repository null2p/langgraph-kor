# type: ignore
from __future__ import annotations

import re
import time
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings
from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchOp,
)
from psycopg import Connection

from langgraph.store.postgres import PostgresStore
from tests.conftest import (
    DEFAULT_URI,
    VECTOR_TYPES,
    CharacterEmbeddings,
)

TTL_SECONDS = 6
TTL_MINUTES = TTL_SECONDS / 60


@pytest.fixture(scope="function", params=["default", "pipe", "pool"])
def store(request) -> PostgresStore:
    database = f"test_{uuid4().hex[:16]}"
    uri_parts = DEFAULT_URI.split("/")
    uri_base = "/".join(uri_parts[:-1])
    query_params = ""
    if "?" in uri_parts[-1]:
        _, query_params = uri_parts[-1].split("?", 1)
        query_params = "?" + query_params

    conn_string = f"{uri_base}/{database}{query_params}"
    admin_conn_string = DEFAULT_URI
    ttl_config = {
        "default_ttl": TTL_MINUTES,
        "refresh_on_read": True,
        "sweep_interval_minutes": TTL_MINUTES / 2,
    }
    with Connection.connect(admin_conn_string, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {database}")
    try:
        with PostgresStore.from_conn_string(conn_string, ttl=ttl_config) as store:
            store.MIGRATIONS = [
                (
                    mig.replace("ttl_minutes INT;", "ttl_minutes FLOAT;")
                    if isinstance(mig, str)
                    else mig
                )
                for mig in store.MIGRATIONS
            ]
            store.setup()

        if request.param == "pipe":
            with PostgresStore.from_conn_string(
                conn_string,
                pipeline=True,
                ttl=ttl_config,
            ) as store:
                store.start_ttl_sweeper()
                yield store

                store.stop_ttl_sweeper()
        elif request.param == "pool":
            with PostgresStore.from_conn_string(
                conn_string,
                pool_config={"min_size": 1, "max_size": 10},
                ttl=ttl_config,
            ) as store:
                store.start_ttl_sweeper()
                yield store

                store.stop_ttl_sweeper()
        else:  # default
            with PostgresStore.from_conn_string(conn_string, ttl=ttl_config) as store:
                store.start_ttl_sweeper()
                yield store

                store.stop_ttl_sweeper()
    finally:
        with Connection.connect(admin_conn_string, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE {database}")


def test_batch_order(store: PostgresStore) -> None:
    # 테스트 데이터 설정
    store.put(("test", "foo"), "key1", {"data": "value1"})
    store.put(("test", "bar"), "key2", {"data": "value2"})

    ops = [
        GetOp(namespace=("test", "foo"), key="key1"),
        PutOp(namespace=("test", "bar"), key="key2", value={"data": "value2"}),
        SearchOp(
            namespace_prefix=("test",), filter={"data": "value1"}, limit=10, offset=0
        ),
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=10, offset=0),
        GetOp(namespace=("test",), key="key3"),
    ]

    results = store.batch(ops)
    assert len(results) == 5
    assert isinstance(results[0], Item)
    assert isinstance(results[0].value, dict)
    assert results[0].value == {"data": "value1"}
    assert results[0].key == "key1"
    assert results[1] is None  # Put 작업은 None을 반환합니다
    assert isinstance(results[2], list)
    assert len(results[2]) == 1
    assert isinstance(results[3], list)
    assert len(results[3]) > 0  # 최소한 우리 테스트 네임스페이스들을 포함해야 함
    assert results[4] is None  # 존재하지 않는 키 returns None

    # 재정렬된 작업 테스트
    ops_reordered = [
        SearchOp(namespace_prefix=("test",), filter=None, limit=5, offset=0),
        GetOp(namespace=("test", "bar"), key="key2"),
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=5, offset=0),
        PutOp(namespace=("test",), key="key3", value={"data": "value3"}),
        GetOp(namespace=("test", "foo"), key="key1"),
    ]

    results_reordered = store.batch(ops_reordered)
    assert len(results_reordered) == 5
    assert isinstance(results_reordered[0], list)
    assert len(results_reordered[0]) >= 2  # 최소한 우리의 두 테스트 아이템을 찾아야 함
    assert isinstance(results_reordered[1], Item)
    assert results_reordered[1].value == {"data": "value2"}
    assert results_reordered[1].key == "key2"
    assert isinstance(results_reordered[2], list)
    assert len(results_reordered[2]) > 0
    assert results_reordered[3] is None  # Put 작업은 None을 반환합니다
    assert isinstance(results_reordered[4], Item)
    assert results_reordered[4].value == {"data": "value1"}
    assert results_reordered[4].key == "key1"


def test_batch_get_ops(store: PostgresStore) -> None:
    # 테스트 데이터 설정
    store.put(("test",), "key1", {"data": "value1"})
    store.put(("test",), "key2", {"data": "value2"})

    ops = [
        GetOp(namespace=("test",), key="key1"),
        GetOp(namespace=("test",), key="key2"),
        GetOp(namespace=("test",), key="key3"),  # 존재하지 않는 키
    ]

    results = store.batch(ops)

    assert len(results) == 3
    assert results[0] is not None
    assert results[1] is not None
    assert results[2] is None
    assert results[0].key == "key1"
    assert results[1].key == "key2"


def test_batch_put_ops(store: PostgresStore) -> None:
    ops = [
        PutOp(namespace=("test",), key="key1", value={"data": "value1"}),
        PutOp(namespace=("test",), key="key2", value={"data": "value2"}),
        PutOp(namespace=("test",), key="key3", value=None),  # 삭제 작업
    ]

    results = store.batch(ops)
    assert len(results) == 3
    assert all(result is None for result in results)

    # put이 제대로 작동했는지 확인
    item1 = store.get(("test",), "key1")
    item2 = store.get(("test",), "key2")
    item3 = store.get(("test",), "key3")

    assert item1 and item1.value == {"data": "value1"}
    assert item2 and item2.value == {"data": "value2"}
    assert item3 is None


def test_batch_search_ops(store: PostgresStore) -> None:
    # 테스트 데이터 설정
    test_data = [
        (("test", "foo"), "key1", {"data": "value1", "tag": "a"}),
        (("test", "bar"), "key2", {"data": "value2", "tag": "a"}),
        (("test", "baz"), "key3", {"data": "value3", "tag": "b"}),
    ]
    for namespace, key, value in test_data:
        store.put(namespace, key, value)

    ops = [
        SearchOp(namespace_prefix=("test",), filter={"tag": "a"}, limit=10, offset=0),
        SearchOp(namespace_prefix=("test",), filter=None, limit=2, offset=0),
        SearchOp(namespace_prefix=("test", "foo"), filter=None, limit=10, offset=0),
    ]

    results = store.batch(ops)
    assert len(results) == 3

    # 첫 번째 검색은 태그가 있는 항목을 찾아야 함 "a"
    assert len(results[0]) == 2
    assert all(item.value["tag"] == "a" for item in results[0])

    # 두 번째 검색은 첫 2개 항목을 반환해야 함
    assert len(results[1]) == 2

    # 세 번째 검색은 항목만 찾아야 함 in test/foo namespace
    assert len(results[2]) == 1
    assert results[2][0].namespace == ("test", "foo")


def test_batch_list_namespaces_ops(store: PostgresStore) -> None:
    # 테스트 데이터 설정 with various namespaces
    test_data = [
        (("test", "documents", "public"), "doc1", {"content": "public doc"}),
        (("test", "documents", "private"), "doc2", {"content": "private doc"}),
        (("test", "images", "public"), "img1", {"content": "public image"}),
        (("prod", "documents", "public"), "doc3", {"content": "prod doc"}),
    ]
    for namespace, key, value in test_data:
        store.put(namespace, key, value)

    ops = [
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=10, offset=0),
        ListNamespacesOp(match_conditions=None, max_depth=2, limit=10, offset=0),
        ListNamespacesOp(
            match_conditions=[MatchCondition("suffix", "public")],
            max_depth=None,
            limit=10,
            offset=0,
        ),
    ]

    results = store.batch(ops)
    assert len(results) == 3

    # 첫 번째 작업은 모든 네임스페이스를 나열해야 함
    assert len(results[0]) == len(test_data)

    # 두 번째 작업은 깊이 2까지의 네임스페이스만 반환해야 함
    assert all(len(ns) <= 2 for ns in results[1])

    # 세 번째 작업은 다음으로 끝나는 네임스페이스만 반환해야 함 "public"
    assert all(ns[-1] == "public" for ns in results[2])


def test_basic_store_ops(store) -> None:
    namespace = ("test", "documents")
    item_id = "doc1"
    item_value = {"title": "Test Document", "content": "Hello, World!"}

    store.put(namespace, item_id, item_value)
    item = store.get(namespace, item_id)

    assert item
    assert item.namespace == namespace
    assert item.key == item_id
    assert item.value == item_value

    # 업데이트 테스트
    updated_value = {"title": "Updated Document", "content": "Hello, Updated!"}
    store.put(namespace, item_id, updated_value)
    updated_item = store.get(namespace, item_id)

    assert updated_item.value == updated_value
    assert updated_item.updated_at > item.updated_at

    # 존재하지 않는 네임스페이스에서 가져오기 테스트
    different_namespace = ("test", "other_documents")
    item_in_different_namespace = store.get(different_namespace, item_id)
    assert item_in_different_namespace is None

    # 삭제 테스트
    store.delete(namespace, item_id)
    deleted_item = store.get(namespace, item_id)
    assert deleted_item is None


def test_list_namespaces(store) -> None:
    # 테스트 데이터 생성 with various namespaces
    test_namespaces = [
        ("test", "documents", "public"),
        ("test", "documents", "private"),
        ("test", "images", "public"),
        ("test", "images", "private"),
        ("prod", "documents", "public"),
        ("prod", "documents", "private"),
    ]

    # 테스트 데이터 삽입
    for namespace in test_namespaces:
        store.put(namespace, "dummy", {"content": "dummy"})

    # 다양한 필터로 목록 조회 테스트
    all_namespaces = store.list_namespaces()
    assert len(all_namespaces) == len(test_namespaces)

    # 접두사 필터링 테스트
    test_prefix_namespaces = store.list_namespaces(prefix=["test"])
    assert len(test_prefix_namespaces) == 4
    assert all(ns[0] == "test" for ns in test_prefix_namespaces)

    # 접미사 필터링 테스트
    public_namespaces = store.list_namespaces(suffix=["public"])
    assert len(public_namespaces) == 3
    assert all(ns[-1] == "public" for ns in public_namespaces)

    # 최대 깊이 테스트
    depth_2_namespaces = store.list_namespaces(max_depth=2)
    assert all(len(ns) <= 2 for ns in depth_2_namespaces)

    # 페이지네이션 테스트
    paginated_namespaces = store.list_namespaces(limit=3)
    assert len(paginated_namespaces) == 3

    # 정리
    for namespace in test_namespaces:
        store.delete(namespace, "dummy")


def test_search(store) -> None:
    # 테스트 데이터 생성
    test_data = [
        (
            ("test", "docs"),
            "doc1",
            {"title": "First Doc", "author": "Alice", "tags": ["important"]},
        ),
        (
            ("test", "docs"),
            "doc2",
            {"title": "Second Doc", "author": "Bob", "tags": ["draft"]},
        ),
        (
            ("test", "images"),
            "img1",
            {"title": "Image 1", "author": "Alice", "tags": ["final"]},
        ),
    ]

    for namespace, key, value in test_data:
        store.put(namespace, key, value)

    # 기본 검색 테스트
    all_items = store.search(["test"])
    assert len(all_items) == 3

    # 네임스페이스 필터링 테스트
    docs_items = store.search(["test", "docs"])
    assert len(docs_items) == 2
    assert all(item.namespace == ("test", "docs") for item in docs_items)

    # 값 필터링 테스트
    alice_items = store.search(["test"], filter={"author": "Alice"})
    assert len(alice_items) == 2
    assert all(item.value["author"] == "Alice" for item in alice_items)

    # 페이지네이션 테스트
    paginated_items = store.search(["test"], limit=2)
    assert len(paginated_items) == 2

    offset_items = store.search(["test"], offset=2)
    assert len(offset_items) == 1

    # 정리
    for namespace, key, _ in test_data:
        store.delete(namespace, key)


@contextmanager
def _create_vector_store(
    vector_type: str,
    distance_type: str,
    fake_embeddings: Embeddings,
    text_fields: list[str] | None = None,
    enable_ttl: bool = True,
) -> PostgresStore:
    """벡터 검색이 활성화된 저장소를 생성합니다."""
    database = f"test_{uuid4().hex[:16]}"
    uri_parts = DEFAULT_URI.split("/")
    uri_base = "/".join(uri_parts[:-1])
    query_params = ""
    if "?" in uri_parts[-1]:
        db_name, query_params = uri_parts[-1].split("?", 1)
        query_params = "?" + query_params

    conn_string = f"{uri_base}/{database}{query_params}"
    admin_conn_string = DEFAULT_URI

    index_config = {
        "dims": fake_embeddings.dims,
        "embed": fake_embeddings,
        "ann_index_config": {
            "vector_type": vector_type,
        },
        "distance_type": distance_type,
        "fields": text_fields,
    }

    with Connection.connect(admin_conn_string, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {database}")
    try:
        with PostgresStore.from_conn_string(
            conn_string,
            index=index_config,
            ttl={"default_ttl": 2, "refresh_on_read": True} if enable_ttl else None,
        ) as store:
            store.setup()
            with store._cursor() as cur:
                # 마이그레이션 인덱스 삭제
                cur.execute("DROP TABLE IF EXISTS store_migrations")
            store.setup()  # 마이그레이션이 멱등성이 아니면 실패함
            yield store
    finally:
        with Connection.connect(admin_conn_string, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE {database}")


_vector_params = [
    (vector_type, distance_type, True)
    for vector_type in VECTOR_TYPES
    for distance_type in (
        ["hamming"] if vector_type == "bit" else ["l2", "inner_product", "cosine"]
    )
]
_vector_params += [(*_vector_params[-1][:2], False)]


@pytest.fixture(
    scope="function",
    params=_vector_params,
    ids=lambda p: f"{p[0]}_{p[1]}",
)
def vector_store(
    request,
    fake_embeddings: Embeddings,
) -> PostgresStore:
    """벡터 검색이 활성화된 저장소를 생성합니다."""
    vector_type, distance_type, enable_ttl = request.param
    with _create_vector_store(
        vector_type, distance_type, fake_embeddings, enable_ttl=enable_ttl
    ) as store:
        yield store


def test_vector_store_initialization(
    vector_store: PostgresStore, fake_embeddings: CharacterEmbeddings
) -> None:
    """임베딩 설정으로 저장소 초기화를 테스트합니다."""
    # 저장소는 임베딩 설정으로 초기화되어야 함
    assert vector_store.index_config is not None
    assert vector_store.index_config["dims"] == fake_embeddings.dims
    assert vector_store.index_config["embed"] == fake_embeddings


def test_vector_insert_with_auto_embedding(vector_store: PostgresStore) -> None:
    """자동으로 임베딩되는 항목 삽입을 테스트합니다."""
    docs = [
        ("doc1", {"text": "short text"}),
        ("doc2", {"text": "longer text document"}),
        ("doc3", {"text": "longest text document here"}),
        ("doc4", {"description": "text in description field"}),
        ("doc5", {"content": "text in content field"}),
        ("doc6", {"body": "text in body field"}),
    ]

    for key, value in docs:
        vector_store.put(("test",), key, value)

    results = vector_store.search(("test",), query="long text")
    assert len(results) > 0

    doc_order = [r.key for r in results]
    assert "doc2" in doc_order
    assert "doc3" in doc_order


def test_vector_update_with_embedding(vector_store: PostgresStore) -> None:
    """항목 업데이트 시 임베딩이 올바르게 업데이트되는지 테스트합니다."""
    vector_store.put(("test",), "doc1", {"text": "zany zebra Xerxes"})
    vector_store.put(("test",), "doc2", {"text": "something about dogs"})
    vector_store.put(("test",), "doc3", {"text": "text about birds"})

    results_initial = vector_store.search(("test",), query="Zany Xerxes")
    assert len(results_initial) > 0
    assert results_initial[0].key == "doc1"
    initial_score = results_initial[0].score

    vector_store.put(("test",), "doc1", {"text": "new text about dogs"})

    results_after = vector_store.search(("test",), query="Zany Xerxes")
    after_score = next((r.score for r in results_after if r.key == "doc1"), 0.0)
    assert after_score < initial_score

    results_new = vector_store.search(("test",), query="new text about dogs")
    for r in results_new:
        if r.key == "doc1":
            assert r.score > after_score

    # 이것은 인덱싱하지 않음
    vector_store.put(("test",), "doc4", {"text": "new text about dogs"}, index=False)
    results_new = vector_store.search(("test",), query="new text about dogs", limit=3)
    assert not any(r.key == "doc4" for r in results_new)


@pytest.mark.parametrize("refresh_ttl", [True, False])
def test_vector_search_with_filters(
    vector_store: PostgresStore, refresh_ttl: bool
) -> None:
    """벡터 검색과 필터를 결합하여 테스트합니다."""
    # 테스트 문서 삽입
    docs = [
        ("doc1", {"text": "red apple", "color": "red", "score": 4.5}),
        ("doc2", {"text": "red car", "color": "red", "score": 3.0}),
        ("doc3", {"text": "green apple", "color": "green", "score": 4.0}),
        ("doc4", {"text": "blue car", "color": "blue", "score": 3.5}),
    ]

    for key, value in docs:
        vector_store.put(("test",), key, value)

    results = vector_store.search(
        ("test",), query="apple", filter={"color": "red"}, refresh_ttl=refresh_ttl
    )
    assert len(results) == 2
    assert results[0].key == "doc1"

    results = vector_store.search(
        ("test",), query="car", filter={"color": "red"}, refresh_ttl=refresh_ttl
    )
    assert len(results) == 2
    assert results[0].key == "doc2"

    results = vector_store.search(
        ("test",),
        query="bbbbluuu",
        filter={"score": {"$gt": 3.2}},
        refresh_ttl=refresh_ttl,
    )
    assert len(results) == 3
    assert results[0].key == "doc4"

    # 여러 필터
    results = vector_store.search(
        ("test",), query="apple", filter={"score": {"$gte": 4.0}, "color": "green"}
    )
    assert len(results) == 1
    assert results[0].key == "doc3"


def test_vector_search_pagination(vector_store: PostgresStore) -> None:
    """벡터 검색의 페이지네이션을 테스트합니다."""
    # 여러 유사한 문서 삽입
    for i in range(5):
        vector_store.put(("test",), f"doc{i}", {"text": f"test document number {i}"})

    # 다양한 페이지 크기로 테스트
    results_page1 = vector_store.search(("test",), query="test", limit=2)
    results_page2 = vector_store.search(("test",), query="test", limit=2, offset=2)

    assert len(results_page1) == 2
    assert len(results_page2) == 2
    assert results_page1[0].key != results_page2[0].key

    # 모든 결과 가져오기
    all_results = vector_store.search(("test",), query="test", limit=10)
    assert len(all_results) == 5


def test_vector_search_edge_cases(vector_store: PostgresStore) -> None:
    """벡터 검색의 엣지 케이스를 테스트합니다."""
    vector_store.put(("test",), "doc1", {"text": "test document"})

    results = vector_store.search(("test",), query="")
    assert len(results) == 1

    results = vector_store.search(("test",), query=None)
    assert len(results) == 1

    long_query = "test " * 100
    results = vector_store.search(("test",), query=long_query)
    assert len(results) == 1

    special_query = "test!@#$%^&*()"
    results = vector_store.search(("test",), query=special_query)
    assert len(results) == 1


@pytest.mark.parametrize(
    "vector_type,distance_type",
    [
        ("vector", "cosine"),
        ("vector", "inner_product"),
        ("halfvec", "cosine"),
        ("halfvec", "inner_product"),
    ],
)
def test_embed_with_path_sync(
    request: Any,
    fake_embeddings: CharacterEmbeddings,
    vector_type: str,
    distance_type: str,
) -> None:
    """Postgres 저장소에서 특정 텍스트 필드를 사용한 벡터 검색을 테스트합니다."""
    with _create_vector_store(
        vector_type,
        distance_type,
        fake_embeddings,
        text_fields=["key0", "key1", "key3"],
    ) as store:
        # 이것은 2개의 벡터로 표현됩니다
        doc1 = {
            # key0 생략 - 오류가 발생하지 않는지 확인
            "key1": "xxx",
            "key2": "yyy",
            "key3": "zzz",
        }
        # 이것은 3개의 벡터로 표현됩니다
        doc2 = {
            "key0": "uuu",
            "key1": "vvv",
            "key2": "www",
            "key3": "xxx",
        }
        store.put(("test",), "doc1", doc1)
        store.put(("test",), "doc2", doc2)

        # doc2.key3 and doc1.key1 both would have the highest score
        results = store.search(("test",), query="xxx")
        assert len(results) == 2
        assert results[0].key != results[1].key
        ascore = results[0].score
        bscore = results[1].score
        assert ascore == pytest.approx(bscore, abs=1e-3)

        # ~doc2만 일치
        results = store.search(("test",), query="uuu")
        assert len(results) == 2
        assert results[0].key != results[1].key
        assert results[0].key == "doc2"
        assert results[0].score > results[1].score
        assert ascore == pytest.approx(results[0].score, abs=1e-3)

        # ~doc1만 일치
        results = store.search(("test",), query="zzz")
        assert len(results) == 2
        assert results[0].key != results[1].key
        assert results[0].key == "doc1"
        assert results[0].score > results[1].score
        assert ascore == pytest.approx(results[0].score, abs=1e-3)

        # 인덱싱되지 않음 - 둘 다 낮은 결과를 가짐. Not zero (because we're projecting)
        # but less than the above.
        results = store.search(("test",), query="www")
        assert len(results) == 2
        assert results[0].key != results[1].key
        assert results[0].score < ascore
        assert results[1].score < ascore


@pytest.mark.parametrize(
    "vector_type,distance_type",
    [
        ("vector", "cosine"),
        ("vector", "inner_product"),
        ("halfvec", "cosine"),
        ("halfvec", "inner_product"),
    ],
)
def test_embed_with_path_operation_config(
    request: Any,
    fake_embeddings: CharacterEmbeddings,
    vector_type: str,
    distance_type: str,
) -> None:
    """벡터 검색을 위한 작업 수준의 필드 설정을 테스트합니다."""

    with _create_vector_store(
        vector_type,
        distance_type,
        fake_embeddings,
        text_fields=["key17"],  # 테스트 데이터와 일치하지 않는 기본 필드
    ) as store:
        doc3 = {
            "key0": "aaa",
            "key1": "bbb",
            "key2": "ccc",
            "key3": "ddd",
        }
        doc4 = {
            "key0": "eee",
            "key1": "bbb",  # doc3.key1과 동일
            "key2": "fff",
            "key3": "ggg",
        }

        store.put(("test",), "doc3", doc3, index=["key0", "key1"])
        store.put(("test",), "doc4", doc4, index=["key1", "key3"])

        results = store.search(("test",), query="aaa")
        assert len(results) == 2
        assert results[0].key == "doc3"
        assert len(set(r.key for r in results)) == 2
        assert results[0].score > results[1].score

        results = store.search(("test",), query="ggg")
        assert len(results) == 2
        assert results[0].key == "doc4"
        assert results[0].score > results[1].score

        results = store.search(("test",), query="bbb")
        assert len(results) == 2
        assert results[0].key != results[1].key
        assert results[0].score == pytest.approx(results[1].score, abs=1e-3)

        results = store.search(("test",), query="ccc")
        assert len(results) == 2
        assert all(
            r.score < 0.9 for r in results
        )  # 인덱싱되지 않은 필드는 낮은 점수를 가져야 함

        # index=False 동작 테스트
        doc5 = {
            "key0": "hhh",
            "key1": "iii",
        }
        store.put(("test",), "doc5", doc5, index=False)
        results = store.search(("test",))
        assert len(results) == 3
        assert all(r.score is None for r in results), f"{results}"
        assert any(r.key == "doc5" for r in results)

        results = store.search(("test",), query="hhh")
        # TODO: 현재 결과가 충분하지 않은 경우 추가 결과를 채우지 않습니다
        # 벡터 검색 중에 반환됩니다.
        # assert len(results) == 3
        # doc5_result = next(r for r in results if r.key == "doc5")
        # assert doc5_result.score is None


def _cosine_similarity(X: list[float], Y: list[list[float]]) -> list[float]:
    """
    벡터 X와 행렬 Y 간의 코사인 유사도를 계산합니다.
    효율성을 위해 numpy를 지연 임포트합니다.
    """

    similarities = []
    for y in Y:
        dot_product = sum(a * b for a, b in zip(X, y, strict=False))
        norm1 = sum(a * a for a in X) ** 0.5
        norm2 = sum(a * a for a in y) ** 0.5
        similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        similarities.append(similarity)

    return similarities


def _inner_product(X: list[float], Y: list[list[float]]) -> list[float]:
    """
    벡터 X와 행렬 Y 간의 내적을 계산합니다.
    효율성을 위해 numpy를 지연 임포트합니다.
    """

    similarities = []
    for y in Y:
        similarity = sum(a * b for a, b in zip(X, y, strict=False))
        similarities.append(similarity)

    return similarities


def _neg_l2_distance(X: list[float], Y: list[list[float]]) -> list[float]:
    """
    벡터 X와 행렬 Y 간의 L2 거리를 계산합니다.
    효율성을 위해 numpy를 지연 임포트합니다.
    """

    similarities = []
    for y in Y:
        similarity = sum((a - b) ** 2 for a, b in zip(X, y, strict=False)) ** 0.5
        similarities.append(-similarity)

    return similarities


@pytest.mark.parametrize(
    "vector_type,distance_type",
    [
        ("vector", "cosine"),
        ("vector", "inner_product"),
        ("halfvec", "l2"),
    ],
)
@pytest.mark.parametrize("query", ["aaa", "bbb", "ccc", "abcd", "poisson"])
def test_scores(
    fake_embeddings: CharacterEmbeddings,
    vector_type: str,
    distance_type: str,
    query: str,
) -> None:
    """벡터 검색을 위한 작업 수준의 필드 설정을 테스트합니다."""
    with _create_vector_store(
        vector_type,
        distance_type,
        fake_embeddings,
        text_fields=["key0"],
    ) as store:
        doc = {
            "key0": "aaa",
        }
        store.put(("test",), "doc", doc, index=["key0", "key1"])

        results = store.search((), query=query)
        vec0 = fake_embeddings.embed_query(doc["key0"])
        vec1 = fake_embeddings.embed_query(query)
        if distance_type == "cosine":
            similarities = _cosine_similarity(vec1, [vec0])
        elif distance_type == "inner_product":
            similarities = _inner_product(vec1, [vec0])
        elif distance_type == "l2":
            similarities = _neg_l2_distance(vec1, [vec0])

        assert len(results) == 1
        assert results[0].score == pytest.approx(similarities[0], abs=1e-3)


def test_nonnull_migrations() -> None:
    _leading_comment_remover = re.compile(r"^/\*.*?\*/")
    for migration in PostgresStore.MIGRATIONS:
        statement = _leading_comment_remover.sub("", migration).split()[0]
        assert statement.strip()


def test_store_ttl(store):
    # TTL이 1분 = 60초라고 가정
    ns = ("foo",)
    store.put(
        ns,
        key="item1",
        value={"foo": "bar"},
        ttl=TTL_MINUTES,  # type: ignore
    )
    time.sleep(TTL_SECONDS - 2)
    res = store.get(ns, key="item1", refresh_ttl=True)
    assert res is not None
    time.sleep(TTL_SECONDS - 2)
    results = store.search(ns, query="foo", refresh_ttl=True)
    assert len(results) == 1
    time.sleep(TTL_SECONDS - 2)
    res = store.get(ns, key="item1", refresh_ttl=False)
    assert res is not None
    time.sleep(TTL_SECONDS - 1)
    # 이제 경과 시간이 (TTL_SECONDS-2)*2 > TTL_SECONDS + TTL_SECONDS/2
    res = store.search(ns, query="bar", refresh_ttl=False)
    assert len(res) == 0


@pytest.mark.parametrize(
    "vector_type,distance_type",
    [
        ("vector", "cosine"),
        ("vector", "inner_product"),
        ("halfvec", "cosine"),
        ("halfvec", "inner_product"),
    ],
)
def test_non_ascii(
    request: Any,
    fake_embeddings: CharacterEmbeddings,
    vector_type: str,
    distance_type: str,
) -> None:
    """ASCII가 아닌 문자 지원을 테스트합니다"""
    with _create_vector_store(vector_type, distance_type, fake_embeddings) as store:
        store.put(("user_123", "memories"), "1", {"text": "这是中文"})  # 중국어
        store.put(
            ("user_123", "memories"), "2", {"text": "これは日本語です"}
        )  # 일본어
        store.put(("user_123", "memories"), "3", {"text": "이건 한국어야"})  # 한국어
        store.put(("user_123", "memories"), "4", {"text": "Это русский"})  # 러시아어
        store.put(("user_123", "memories"), "5", {"text": "यह रूसी है"})  # 힌디어

        result1 = store.search(("user_123", "memories"), query="这是中文")
        result2 = store.search(("user_123", "memories"), query="これは日本語です")
        result3 = store.search(("user_123", "memories"), query="이건 한국어야")
        result4 = store.search(("user_123", "memories"), query="Это русский")
        result5 = store.search(("user_123", "memories"), query="यह रूसी है")

        assert result1[0].key == "1"
        assert result2[0].key == "2"
        assert result3[0].key == "3"
        assert result4[0].key == "4"
        assert result5[0].key == "5"
