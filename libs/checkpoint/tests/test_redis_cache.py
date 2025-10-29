"""Redis 캐시 구현을 위한 단위 테스트입니다."""

import time

import pytest
import redis

from langgraph.cache.base import FullKey
from langgraph.cache.redis import RedisCache


class TestRedisCache:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """테스트용 Redis 클라이언트와 캐시를 설정합니다."""
        self.client = redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=False
        )
        try:
            self.client.ping()
        except redis.ConnectionError:
            pytest.skip("Redis server not available")

        self.cache: RedisCache = RedisCache(self.client, prefix="test:cache:")

        # 각 테스트 전에 정리
        self.client.flushdb()

    def teardown_method(self) -> None:
        """각 테스트 후에 정리합니다."""
        try:
            self.client.flushdb()
        except Exception:
            pass

    def test_basic_set_and_get(self) -> None:
        """기본적인 set과 get 작업을 테스트합니다."""
        keys: list[FullKey] = [(("graph", "node"), "key1")]
        values = {keys[0]: ({"result": 42}, None)}

        # 값 설정
        self.cache.set(values)

        # 값 가져오기
        result = self.cache.get(keys)
        assert len(result) == 1
        assert result[keys[0]] == {"result": 42}

    def test_batch_operations(self) -> None:
        """배치 set과 get 작업을 테스트합니다."""
        keys: list[FullKey] = [
            (("graph", "node1"), "key1"),
            (("graph", "node2"), "key2"),
            (("other", "node"), "key3"),
        ]
        values = {
            keys[0]: ({"result": 1}, None),
            keys[1]: ({"result": 2}, 60),  # TTL 포함
            keys[2]: ({"result": 3}, None),
        }

        # 값들 설정
        self.cache.set(values)

        # 모든 값 가져오기
        result = self.cache.get(keys)
        assert len(result) == 3
        assert result[keys[0]] == {"result": 1}
        assert result[keys[1]] == {"result": 2}
        assert result[keys[2]] == {"result": 3}

    def test_ttl_behavior(self) -> None:
        """TTL(time-to-live) 기능을 테스트합니다."""
        key: FullKey = (("graph", "node"), "ttl_key")
        values = {key: ({"data": "expires_soon"}, 1)}  # 1초 TTL

        # TTL과 함께 설정
        self.cache.set(values)

        # 즉시 사용 가능해야 함
        result = self.cache.get([key])
        assert len(result) == 1
        assert result[key] == {"data": "expires_soon"}

        # 만료 대기
        time.sleep(1.1)

        # 만료되어야 함
        result = self.cache.get([key])
        assert len(result) == 0

    def test_namespace_isolation(self) -> None:
        """서로 다른 네임스페이스가 격리되는지 테스트합니다."""
        key1: FullKey = (("graph1", "node"), "same_key")
        key2: FullKey = (("graph2", "node"), "same_key")

        values = {key1: ({"graph": 1}, None), key2: ({"graph": 2}, None)}

        self.cache.set(values)

        result = self.cache.get([key1, key2])
        assert result[key1] == {"graph": 1}
        assert result[key2] == {"graph": 2}

    def test_clear_all(self) -> None:
        """모든 캐시된 값을 지우는 것을 테스트합니다."""
        keys: list[FullKey] = [
            (("graph", "node1"), "key1"),
            (("graph", "node2"), "key2"),
        ]
        values = {keys[0]: ({"result": 1}, None), keys[1]: ({"result": 2}, None)}

        self.cache.set(values)

        # 데이터 존재 확인
        result = self.cache.get(keys)
        assert len(result) == 2

        # 모두 지우기
        self.cache.clear()

        # 데이터가 사라졌는지 확인
        result = self.cache.get(keys)
        assert len(result) == 0

    def test_clear_by_namespace(self) -> None:
        """네임스페이스별로 캐시된 값을 지우는 것을 테스트합니다."""
        keys: list[FullKey] = [
            (("graph1", "node"), "key1"),
            (("graph2", "node"), "key2"),
            (("graph1", "other"), "key3"),
        ]
        values = {
            keys[0]: ({"result": 1}, None),
            keys[1]: ({"result": 2}, None),
            keys[2]: ({"result": 3}, None),
        }

        self.cache.set(values)

        # graph1 네임스페이스만 지우기
        self.cache.clear([("graph1", "node"), ("graph1", "other")])

        # graph1은 지워지고 graph2는 유지되어야 함
        result = self.cache.get(keys)
        assert len(result) == 1
        assert result[keys[1]] == {"result": 2}

    def test_empty_operations(self) -> None:
        """빈 키/값에 대한 동작을 테스트합니다."""
        # 빈 get
        result = self.cache.get([])
        assert result == {}

        # 빈 set
        self.cache.set({})  # 오류를 발생시키지 않아야 함

    def test_nonexistent_keys(self) -> None:
        """존재하지 않는 키를 가져오는 것을 테스트합니다."""
        keys: list[FullKey] = [(("graph", "node"), "nonexistent")]
        result = self.cache.get(keys)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_async_operations(self) -> None:
        """동기 Redis 클라이언트로 비동기 set과 get 작업을 테스트합니다."""
        # 동기 Redis 클라이언트와 캐시 생성 (메인 통합 테스트와 유사)
        client = redis.Redis(host="localhost", port=6379, db=1, decode_responses=False)
        try:
            client.ping()
        except Exception:
            pytest.skip("Redis not available")

        cache: RedisCache = RedisCache(client, prefix="test:async:")

        keys: list[FullKey] = [(("graph", "node"), "async_key")]
        values = {keys[0]: ({"async": True}, None)}

        # 비동기 set (동기로 위임)
        await cache.aset(values)

        # 비동기 get (동기로 위임)
        result = await cache.aget(keys)
        assert len(result) == 1
        assert result[keys[0]] == {"async": True}

        # 정리
        client.flushdb()

    @pytest.mark.asyncio
    async def test_async_clear(self) -> None:
        """동기 Redis 클라이언트로 비동기 clear 작업을 테스트합니다."""
        # 동기 Redis 클라이언트와 캐시 생성 (메인 통합 테스트와 유사)
        client = redis.Redis(host="localhost", port=6379, db=1, decode_responses=False)
        try:
            client.ping()
        except Exception:
            pytest.skip("Redis not available")

        cache: RedisCache = RedisCache(client, prefix="test:async:")

        keys: list[FullKey] = [(("graph", "node"), "key")]
        values = {keys[0]: ({"data": "test"}, None)}

        await cache.aset(values)

        # 데이터 존재 확인
        result = await cache.aget(keys)
        assert len(result) == 1

        # 모두 지우기 (동기로 위임)
        await cache.aclear()

        # 데이터가 사라졌는지 확인
        result = await cache.aget(keys)
        assert len(result) == 0

        # 정리
        client.flushdb()

    def test_redis_unavailable_get(self) -> None:
        """get 작업 중 Redis를 사용할 수 없을 때의 동작을 테스트합니다."""
        # 존재하지 않는 Redis 서버로 캐시 생성
        bad_client = redis.Redis(
            host="nonexistent", port=9999, socket_connect_timeout=0.1
        )
        cache: RedisCache = RedisCache(bad_client, prefix="test:cache:")

        keys: list[FullKey] = [(("graph", "node"), "key")]
        result = cache.get(keys)

        # Redis를 사용할 수 없을 때 빈 dict를 반환해야 함
        assert result == {}

    def test_redis_unavailable_set(self) -> None:
        """set 작업 중 Redis를 사용할 수 없을 때의 동작을 테스트합니다."""
        # 존재하지 않는 Redis 서버로 캐시 생성
        bad_client = redis.Redis(
            host="nonexistent", port=9999, socket_connect_timeout=0.1
        )
        cache: RedisCache = RedisCache(bad_client, prefix="test:cache:")

        keys: list[FullKey] = [(("graph", "node"), "key")]
        values = {keys[0]: ({"data": "test"}, None)}

        # Redis를 사용할 수 없을 때 예외를 발생시키지 않아야 함
        cache.set(values)  # 조용히 실패해야 함

    @pytest.mark.asyncio
    async def test_redis_unavailable_async(self) -> None:
        """Redis를 사용할 수 없을 때의 비동기 동작을 테스트합니다."""
        # 존재하지 않는 Redis 서버로 동기 캐시 생성 (메인 통합 테스트와 유사)
        bad_client = redis.Redis(
            host="nonexistent", port=9999, socket_connect_timeout=0.1
        )
        cache: RedisCache = RedisCache(bad_client, prefix="test:cache:")

        keys: list[FullKey] = [(("graph", "node"), "key")]
        values = {keys[0]: ({"data": "test"}, None)}

        # get에 대해 빈 dict를 반환해야 함 (동기로 위임)
        result = await cache.aget(keys)
        assert result == {}

        # set에 대해 예외를 발생시키지 않아야 함 (동기로 위임)
        await cache.aset(values)  # 조용히 실패해야 함

    def test_corrupted_data_handling(self) -> None:
        """Redis에서 손상된 데이터를 처리하는 것을 테스트합니다."""
        # 먼저 유효한 데이터 설정
        keys: list[FullKey] = [(("graph", "node"), "valid_key")]
        values = {keys[0]: ({"data": "valid"}, None)}
        self.cache.set(values)

        # 수동으로 손상된 데이터 삽입
        corrupted_key = self.cache._make_key(("graph", "node"), "corrupted_key")
        self.client.set(corrupted_key, b"invalid:data:format:too:many:colons")

        # 손상된 항목을 건너뛰고 유효한 항목만 반환해야 함
        all_keys: list[FullKey] = [keys[0], (("graph", "node"), "corrupted_key")]
        result = self.cache.get(all_keys)

        assert len(result) == 1
        assert result[keys[0]] == {"data": "valid"}

    def test_key_parsing_edge_cases(self) -> None:
        """엣지 케이스에서 키 파싱을 테스트합니다."""
        # 빈 네임스페이스 테스트
        key1: FullKey = ((), "empty_ns")
        values = {key1: ({"data": "empty_ns"}, None)}
        self.cache.set(values)
        result = self.cache.get([key1])
        assert result[key1] == {"data": "empty_ns"}

        # 특수 문자가 있는 네임스페이스 테스트
        key2: FullKey = (
            ("graph:with:colons", "node-with-dashes"),
            "key_with_underscores",
        )
        values = {key2: ({"data": "special_chars"}, None)}
        self.cache.set(values)
        result = self.cache.get([key2])
        assert result[key2] == {"data": "special_chars"}

    def test_large_data_serialization(self) -> None:
        """큰 데이터 객체를 처리하는 것을 테스트합니다."""
        # 큰 데이터 구조 생성
        large_data = {"large_list": list(range(1000)), "nested": {"data": "x" * 1000}}
        key: FullKey = (("graph", "node"), "large_key")
        values = {key: (large_data, None)}

        self.cache.set(values)
        result = self.cache.get([key])

        assert len(result) == 1
        assert result[key] == large_data
