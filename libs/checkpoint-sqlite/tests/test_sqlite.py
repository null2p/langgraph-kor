from typing import Any, cast

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.utils import _metadata_predicate, search_where


class TestSqliteSaver:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        # 테스트 설정을 위한 객체
        self.config_1: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-1",
                # 하위 호환성 테스트를 위함
                "checkpoint_id": "1",
                "checkpoint_ns": "",
            }
        }
        self.config_2: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_id": "2",
                "checkpoint_ns": "",
            }
        }
        self.config_3: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_id": "2-inner",
                "checkpoint_ns": "inner",
            }
        }

        self.chkpnt_1: Checkpoint = empty_checkpoint()
        self.chkpnt_2: Checkpoint = create_checkpoint(self.chkpnt_1, {}, 1)
        self.chkpnt_3: Checkpoint = empty_checkpoint()

        self.metadata_1: CheckpointMetadata = {
            "source": "input",
            "step": 2,
            "writes": {},
            "score": 1,
        }
        self.metadata_2: CheckpointMetadata = {
            "source": "loop",
            "step": 1,
            "writes": {"foo": "bar"},
            "score": None,
        }
        self.metadata_3: CheckpointMetadata = {}

    def test_combined_metadata(self) -> None:
        with SqliteSaver.from_conn_string(":memory:") as saver:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-2",
                    "checkpoint_ns": "",
                    "__super_private_key": "super_private_value",
                },
                "metadata": {"run_id": "my_run_id"},
            }
            saver.put(config, self.chkpnt_2, self.metadata_2, {})
            checkpoint = saver.get_tuple(config)
            assert checkpoint is not None and checkpoint.metadata == {
                **self.metadata_2,
                "run_id": "my_run_id",
            }

    def test_search(self) -> None:
        with SqliteSaver.from_conn_string(":memory:") as saver:
            # 테스트 설정
            # 체크포인트 저장
            saver.put(self.config_1, self.chkpnt_1, self.metadata_1, {})
            saver.put(self.config_2, self.chkpnt_2, self.metadata_2, {})
            saver.put(self.config_3, self.chkpnt_3, self.metadata_3, {})

            # 메서드 호출 / 어설션
            query_1 = {"source": "input"}  # 1개의 키로 검색
            query_2 = {
                "step": 1,
                "writes": {"foo": "bar"},
            }  # 여러 키로 검색
            query_3: dict[str, Any] = {}  # 키 없이 검색, 모든 체크포인트 반환
            query_4 = {"source": "update", "step": 1}  # 일치하는 항목 없음

            search_results_1 = list(saver.list(None, filter=query_1))
            assert len(search_results_1) == 1
            assert search_results_1[0].metadata == self.metadata_1

            search_results_2 = list(saver.list(None, filter=query_2))
            assert len(search_results_2) == 1
            assert search_results_2[0].metadata == self.metadata_2

            search_results_3 = list(saver.list(None, filter=query_3))
            assert len(search_results_3) == 3

            search_results_4 = list(saver.list(None, filter=query_4))
            assert len(search_results_4) == 0

            # config로 검색 (기본적으로 모든 네임스페이스의 체크포인트 검색)
            search_results_5 = list(
                saver.list({"configurable": {"thread_id": "thread-2"}})
            )
            assert len(search_results_5) == 2
            assert {
                search_results_5[0].config["configurable"]["checkpoint_ns"],
                search_results_5[1].config["configurable"]["checkpoint_ns"],
            } == {"", "inner"}

            # before 매개변수로 검색
            search_results_6 = list(saver.list(None, before=search_results_5[1].config))
            assert len(search_results_6) == 1
            assert search_results_6[0].config["configurable"]["thread_id"] == "thread-1"

            # limit 매개변수로 검색
            search_results_7 = list(
                saver.list({"configurable": {"thread_id": "thread-2"}}, limit=1)
            )
            assert len(search_results_7) == 1
            assert search_results_7[0].config["configurable"]["thread_id"] == "thread-2"

    def test_search_where(self) -> None:
        # 메서드 호출 / 어설션
        expected_predicate_1 = "WHERE json_extract(CAST(metadata AS TEXT), '$.source') = ? AND json_extract(CAST(metadata AS TEXT), '$.step') = ? AND json_extract(CAST(metadata AS TEXT), '$.writes') = ? AND json_extract(CAST(metadata AS TEXT), '$.score') = ? AND checkpoint_id < ?"
        expected_param_values_1 = ["input", 2, "{}", 1, "1"]
        assert search_where(
            None, cast(dict[str, Any], self.metadata_1), self.config_1
        ) == (
            expected_predicate_1,
            expected_param_values_1,
        )

    def test_metadata_predicate(self) -> None:
        # 메서드 호출 / 어설션
        expected_predicate_1 = [
            "json_extract(CAST(metadata AS TEXT), '$.source') = ?",
            "json_extract(CAST(metadata AS TEXT), '$.step') = ?",
            "json_extract(CAST(metadata AS TEXT), '$.writes') = ?",
            "json_extract(CAST(metadata AS TEXT), '$.score') = ?",
        ]
        expected_predicate_2 = [
            "json_extract(CAST(metadata AS TEXT), '$.source') = ?",
            "json_extract(CAST(metadata AS TEXT), '$.step') = ?",
            "json_extract(CAST(metadata AS TEXT), '$.writes') = ?",
            "json_extract(CAST(metadata AS TEXT), '$.score') IS ?",
        ]
        expected_predicate_3: list[str] = []

        expected_param_values_1 = ["input", 2, "{}", 1]
        expected_param_values_2 = ["loop", 1, '{"foo":"bar"}', None]
        expected_param_values_3: list[Any] = []

        assert _metadata_predicate(cast(dict[str, Any], self.metadata_1)) == (
            expected_predicate_1,
            expected_param_values_1,
        )
        assert _metadata_predicate(cast(dict[str, Any], self.metadata_2)) == (
            expected_predicate_2,
            expected_param_values_2,
        )
        assert _metadata_predicate(cast(dict[str, Any], self.metadata_3)) == (
            expected_predicate_3,
            expected_param_values_3,
        )

    async def test_informative_async_errors(self) -> None:
        with SqliteSaver.from_conn_string(":memory:") as saver:
            # 메서드 호출 / 어설션
            with pytest.raises(NotImplementedError, match="AsyncSqliteSaver"):
                await saver.aget(self.config_1)
            with pytest.raises(NotImplementedError, match="AsyncSqliteSaver"):
                await saver.aget_tuple(self.config_1)
            with pytest.raises(NotImplementedError, match="AsyncSqliteSaver"):
                async for _ in saver.alist(self.config_1):
                    pass
