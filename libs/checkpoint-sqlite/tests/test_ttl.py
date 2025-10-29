"""SQLite 저장소의 TTL(Time-To-Live) 기능을 테스트합니다."""

import asyncio
import os
import tempfile
import time
from collections.abc import Generator

import pytest
from langgraph.store.base import TTLConfig

from langgraph.store.sqlite import SqliteStore
from langgraph.store.sqlite.aio import AsyncSqliteStore


@pytest.fixture
def temp_db_file() -> Generator[str, None, None]:
    """테스트용 임시 데이터베이스 파일을 생성합니다."""
    fd, path = tempfile.mkstemp()
    os.close(fd)
    yield path
    os.unlink(path)


def test_ttl_basic(temp_db_file: str) -> None:
    """동기 API를 사용한 기본 TTL 기능을 테스트합니다."""
    ttl_seconds = 1
    ttl_minutes = ttl_seconds / 60

    with SqliteStore.from_conn_string(
        temp_db_file, ttl={"default_ttl": ttl_minutes}
    ) as store:
        store.setup()

        store.put(("test",), "item1", {"value": "test"})

        item = store.get(("test",), "item1")
        assert item is not None
        assert item.value["value"] == "test"

        time.sleep(ttl_seconds + 1.0)

        store.sweep_ttl()

        item = store.get(("test",), "item1")
        assert item is None


@pytest.mark.flaky(retries=3)
def test_ttl_refresh(temp_db_file: str) -> None:
    """읽기 시 TTL 갱신을 테스트합니다."""
    ttl_seconds = 1
    ttl_minutes = ttl_seconds / 60

    with SqliteStore.from_conn_string(
        temp_db_file, ttl={"default_ttl": ttl_minutes, "refresh_on_read": True}
    ) as store:
        store.setup()

        # TTL을 가진 항목 저장
        store.put(("test",), "item1", {"value": "test"})

        # 만료 시간에 거의 다가갈 때까지 대기
        time.sleep(ttl_seconds - 0.5)
        swept = store.sweep_ttl()
        assert swept == 0

        # 항목을 가져오고 TTL 갱신
        item = store.get(("test",), "item1", refresh_ttl=True)
        assert item is not None

        time.sleep(ttl_seconds - 0.5)
        swept = store.sweep_ttl()
        assert swept == 0

        # 항목을 가져오기, 여전히 존재해야 함
        item = store.get(("test",), "item1")
        assert item is not None
        assert item.value["value"] == "test"

        # 다시 대기하지만 이번에는 갱신하지 않음
        time.sleep(ttl_seconds + 0.75)

        swept = store.sweep_ttl()
        assert swept == 1

        # 이제 항목이 사라져야 함
        item = store.get(("test",), "item1")
        assert item is None


def test_ttl_sweeper(temp_db_file: str) -> None:
    """TTL 스위퍼 스레드를 테스트합니다."""
    ttl_seconds = 2
    ttl_minutes = ttl_seconds / 60

    ttl_config: TTLConfig = {
        "default_ttl": ttl_minutes,
        "sweep_interval_minutes": ttl_minutes / 2,
    }
    with SqliteStore.from_conn_string(
        temp_db_file,
        ttl=ttl_config,
    ) as store:
        store.setup()

        # TTL 스위퍼 시작
        store.start_ttl_sweeper()

        # TTL을 가진 항목 저장
        store.put(("test",), "item1", {"value": "test"})

        # 초기에는 항목이 존재해야 함
        item = store.get(("test",), "item1")
        assert item is not None

        # TTL이 만료되고 스위퍼가 실행될 때까지 대기
        time.sleep(ttl_seconds + (ttl_seconds / 2) + 0.5)

        # 이제 항목이 사라져야 함 (자동으로 스윕됨)
        item = store.get(("test",), "item1")
        assert item is None

        # 스위퍼 중지
        store.stop_ttl_sweeper()


@pytest.mark.flaky(retries=3)
def test_ttl_custom_value(temp_db_file: str) -> None:
    """항목별 사용자 정의 TTL 값을 테스트합니다."""
    with SqliteStore.from_conn_string(temp_db_file) as store:
        store.setup()

        # 서로 다른 TTL을 가진 항목들 저장
        store.put(("test",), "item1", {"value": "short"}, ttl=1 / 60)  # 1초
        store.put(("test",), "item2", {"value": "long"}, ttl=3 / 60)  # 3초

        # 짧은 TTL을 가진 항목
        time.sleep(2)  # 짧은 TTL이 만료될 때까지 대기
        store.sweep_ttl()

        # 짧은 TTL 항목은 사라지고, 긴 TTL 항목은 남아 있어야 함
        item1 = store.get(("test",), "item1")
        item2 = store.get(("test",), "item2")
        assert item1 is None
        assert item2 is not None

        # 두 번째 항목의 TTL이 만료될 때까지 대기
        time.sleep(4)
        store.sweep_ttl()

        # 이제 둘 다 사라져야 함
        item2 = store.get(("test",), "item2")
        assert item2 is None


@pytest.mark.flaky(retries=3)
def test_ttl_override_default(temp_db_file: str) -> None:
    """항목 수준에서 기본 TTL을 재정의하는 것을 테스트합니다."""
    with SqliteStore.from_conn_string(
        temp_db_file,
        ttl={"default_ttl": 5 / 60},  # 기본값 5초
    ) as store:
        store.setup()

        # 기본값보다 짧은 TTL을 가진 항목 저장
        store.put(("test",), "item1", {"value": "override"}, ttl=1 / 60)  # 1초

        # 기본 TTL을 가진 항목 저장
        store.put(("test",), "item2", {"value": "default"})  # 기본값 5초 사용

        # TTL이 없는 항목 저장
        store.put(("test",), "item3", {"value": "permanent"}, ttl=None)

        # 재정의된 TTL이 만료될 때까지 대기
        time.sleep(2)
        store.sweep_ttl()

        # 결과 확인
        item1 = store.get(("test",), "item1")
        item2 = store.get(("test",), "item2")
        item3 = store.get(("test",), "item3")

        assert item1 is None  # 만료되어야 함
        assert item2 is not None  # 기본 TTL, 여전히 존재해야 함
        assert item3 is not None  # TTL 없음, 여전히 존재해야 함

        # 기본 TTL이 만료될 때까지 대기
        time.sleep(4)
        store.sweep_ttl()

        # 결과를 다시 확인
        item2 = store.get(("test",), "item2")
        item3 = store.get(("test",), "item3")

        assert item2 is None  # 기본 TTL 항목은 사라져야 함
        assert item3 is not None  # TTL 없는 항목은 여전히 존재해야 함


@pytest.mark.flaky(retries=3)
def test_search_with_ttl(temp_db_file: str) -> None:
    """검색 작업과 함께 TTL을 테스트합니다."""
    ttl_seconds = 1
    ttl_minutes = ttl_seconds / 60

    with SqliteStore.from_conn_string(
        temp_db_file, ttl={"default_ttl": ttl_minutes}
    ) as store:
        store.setup()

        # 항목들 저장
        store.put(("test",), "item1", {"value": "apple"})
        store.put(("test",), "item2", {"value": "banana"})

        # 만료 전 검색
        results = store.search(("test",), filter={"value": "apple"})
        assert len(results) == 1
        assert results[0].key == "item1"

        # TTL이 만료될 때까지 대기
        time.sleep(ttl_seconds + 1)
        store.sweep_ttl()

        # 만료 후 검색
        results = store.search(("test",), filter={"value": "apple"})
        assert len(results) == 0


@pytest.mark.asyncio
async def test_async_ttl_basic(temp_db_file: str) -> None:
    """비동기 API를 사용한 기본 TTL 기능을 테스트합니다."""
    ttl_seconds = 1
    ttl_minutes = ttl_seconds / 60

    async with AsyncSqliteStore.from_conn_string(
        temp_db_file, ttl={"default_ttl": ttl_minutes}
    ) as store:
        await store.setup()

        # TTL을 가진 항목 저장
        await store.aput(("test",), "item1", {"value": "test"})

        # 만료 전에 항목 가져오기
        item = await store.aget(("test",), "item1")
        assert item is not None
        assert item.value["value"] == "test"

        # TTL이 만료될 때까지 대기
        await asyncio.sleep(ttl_seconds + 1.0)

        # 스위퍼 스레드 없이 수동 스윕 필요
        await store.sweep_ttl()

        # 이제 항목이 사라져야 함
        item = await store.aget(("test",), "item1")
        assert item is None


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3)
async def test_async_ttl_refresh(temp_db_file: str) -> None:
    """비동기 API를 사용하여 읽기 시 TTL 갱신을 테스트합니다."""
    ttl_seconds = 1
    ttl_minutes = ttl_seconds / 60

    async with AsyncSqliteStore.from_conn_string(
        temp_db_file, ttl={"default_ttl": ttl_minutes, "refresh_on_read": True}
    ) as store:
        await store.setup()

        # TTL을 가진 항목 저장
        await store.aput(("test",), "item1", {"value": "test"})

        # 만료 시간에 거의 다가갈 때까지 대기
        await asyncio.sleep(ttl_seconds - 0.5)

        # 항목을 가져오고 TTL 갱신
        item = await store.aget(("test",), "item1", refresh_ttl=True)
        assert item is not None

        # 다시 대기 - 갱신하지 않았다면 지금쯤 만료되었을 것임
        await asyncio.sleep(ttl_seconds - 0.5)

        # 항목을 가져오기, 여전히 존재해야 함
        item = await store.aget(("test",), "item1")
        assert item is not None
        assert item.value["value"] == "test"

        # 다시 대기하지만 이번에는 갱신하지 않음
        await asyncio.sleep(ttl_seconds + 1.0)

        # 수동 스윕
        await store.sweep_ttl()

        # 이제 항목이 사라져야 함
        item = await store.aget(("test",), "item1")
        assert item is None


@pytest.mark.asyncio
async def test_async_ttl_sweeper(temp_db_file: str) -> None:
    """비동기 API를 사용하여 TTL 스위퍼 스레드를 테스트합니다."""
    ttl_seconds = 2
    ttl_minutes = ttl_seconds / 60

    ttl_config: TTLConfig = {
        "default_ttl": ttl_minutes,
        "sweep_interval_minutes": ttl_minutes / 2,
    }

    async with AsyncSqliteStore.from_conn_string(
        temp_db_file,
        ttl=ttl_config,
    ) as store:
        await store.setup()

        # TTL 스위퍼 시작
        await store.start_ttl_sweeper()

        # TTL을 가진 항목 저장
        await store.aput(("test",), "item1", {"value": "test"})

        # 초기에는 항목이 존재해야 함
        item = await store.aget(("test",), "item1")
        assert item is not None

        # TTL이 만료되고 스위퍼가 실행될 때까지 대기
        await asyncio.sleep(ttl_seconds + (ttl_seconds / 2) + 0.5)

        # 이제 항목이 사라져야 함 (자동으로 스윕됨)
        item = await store.aget(("test",), "item1")
        assert item is None

        # 스위퍼 중지
        await store.stop_ttl_sweeper()


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3)
async def test_async_search_with_ttl(temp_db_file: str) -> None:
    """비동기 API를 사용하여 검색 작업과 함께 TTL을 테스트합니다."""
    ttl_seconds = 1
    ttl_minutes = ttl_seconds / 60

    async with AsyncSqliteStore.from_conn_string(
        temp_db_file, ttl={"default_ttl": ttl_minutes}
    ) as store:
        await store.setup()

        # 항목들 저장
        await store.aput(("test",), "item1", {"value": "apple"})
        await store.aput(("test",), "item2", {"value": "banana"})

        # 만료 전 검색
        results = await store.asearch(("test",), filter={"value": "apple"})
        assert len(results) == 1
        assert results[0].key == "item1"

        # TTL이 만료될 때까지 대기
        await asyncio.sleep(ttl_seconds + 1)
        await store.sweep_ttl()

        # 만료 후 검색
        results = await store.asearch(("test",), filter={"value": "apple"})
        assert len(results) == 0


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3)
async def test_async_asearch_refresh_ttl(temp_db_file: str) -> None:
    """비동기 API를 사용하여 asearch에서 TTL 갱신을 테스트합니다."""
    ttl_seconds = 4.0  # 타이밍에 덜 민감하도록 TTL 증가
    ttl_minutes = ttl_seconds / 60.0

    async with AsyncSqliteStore.from_conn_string(
        temp_db_file, ttl={"default_ttl": ttl_minutes, "refresh_on_read": True}
    ) as store:
        await store.setup()

        namespace = ("docs", "user1")
        # t=0: 항목 저장, t=4.0초에 만료
        await store.aput(namespace, "item1", {"text": "content1", "id": 1})
        await store.aput(namespace, "item2", {"text": "content2", "id": 2})

        # t=3.0초: (sleep ttl_seconds * 0.75 = 3초 이후)
        await asyncio.sleep(ttl_seconds * 0.75)

        # item1에 대해 refresh_ttl=True로 asearch 수행.
        # item1의 TTL이 갱신되어야 함. 새로운 만료 시간: t=3.0초 + 4.0초 = t=7.0초.
        # item2의 TTL은 영향받지 않음. t=4.0초에 만료됨.
        searched_items = await store.asearch(
            namespace, filter={"id": 1}, refresh_ttl=True
        )
        assert len(searched_items) == 1
        assert searched_items[0].key == "item1"

        # t=5.0초: (sleep ttl_seconds * 0.5 = 2초 더 대기. 총 경과 시간: 3초 + 2초 = 5초)
        await asyncio.sleep(ttl_seconds * 0.5)
        # 이 시점에서:
        # - item1 (asearch로 갱신됨)은 t=7.0초에 만료될 예정. 아직 살아있어야 함.
        # - item2 (원래 TTL)는 t=4.0초에 만료됨. 스윕 후 사라져야 함.

        await store.sweep_ttl()

        # item1 확인 (asearch 갱신으로 인해 존재해야 함)
        item1_check1 = await store.aget(namespace, "item1", refresh_ttl=False)
        assert item1_check1 is not None, (
            "Item1은 asearch 갱신 후와 첫 번째 스윕 이후에도 존재해야 합니다"
        )
        assert item1_check1.value["text"] == "content1"

        # item2 확인 (사라져야 함)
        item2_check1 = await store.aget(namespace, "item2", refresh_ttl=False)
        assert item2_check1 is None, (
            "Item2는 원래 TTL이 만료된 후 사라져야 합니다"
        )

        # t=7.5초: (sleep ttl_seconds * 0.625 = 2.5초 더 대기. 총 경과 시간: 5초 + 2.5초 = 7.5초)
        await asyncio.sleep(ttl_seconds * 0.625)
        # 이 시점에서:
        # - item1 (asearch로 갱신됨, t=7.0초에 만료)은 스윕 후 사라져야 함.

        await store.sweep_ttl()

        # item1 다시 확인 (이제 사라져야 함)
        item1_final_check = await store.aget(namespace, "item1", refresh_ttl=False)
        assert item1_final_check is None, (
            "Item1은 갱신된 TTL이 만료된 후 사라져야 합니다"
        )
