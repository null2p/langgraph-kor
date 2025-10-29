from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class TestAsyncSqliteSaver:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        # 테스트 설정을 위한 객체
        self.config_1: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-1",
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

    async def test_combined_metadata(self) -> None:
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": "thread-2",
                    "checkpoint_ns": "",
                    "__super_private_key": "super_private_value",
                },
                "metadata": {"run_id": "my_run_id"},
            }
            await saver.aput(config, self.chkpnt_2, self.metadata_2, {})
            checkpoint = await saver.aget_tuple(config)
            assert checkpoint is not None and checkpoint.metadata == {
                **self.metadata_2,
                "run_id": "my_run_id",
            }

    async def test_asearch(self) -> None:
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            await saver.aput(self.config_1, self.chkpnt_1, self.metadata_1, {})
            await saver.aput(self.config_2, self.chkpnt_2, self.metadata_2, {})
            await saver.aput(self.config_3, self.chkpnt_3, self.metadata_3, {})

            # 메서드 호출 / 어설션
            query_1 = {"source": "input"}  # 1개의 키로 검색
            query_2 = {
                "step": 1,
                "writes": {"foo": "bar"},
            }  # 여러 키로 검색
            query_3: dict[str, Any] = {}  # 키 없이 검색, 모든 체크포인트 반환
            query_4 = {"source": "update", "step": 1}  # 일치하는 항목 없음

            search_results_1 = [c async for c in saver.alist(None, filter=query_1)]
            assert len(search_results_1) == 1
            assert search_results_1[0].metadata == self.metadata_1

            search_results_2 = [c async for c in saver.alist(None, filter=query_2)]
            assert len(search_results_2) == 1
            assert search_results_2[0].metadata == self.metadata_2

            search_results_3 = [c async for c in saver.alist(None, filter=query_3)]
            assert len(search_results_3) == 3

            search_results_4 = [c async for c in saver.alist(None, filter=query_4)]
            assert len(search_results_4) == 0

            # config로 검색 (기본적으로 모든 네임스페이스의 체크포인트 검색)
            search_results_5 = [
                c
                async for c in saver.alist({"configurable": {"thread_id": "thread-2"}})
            ]
            assert len(search_results_5) == 2
            assert {
                search_results_5[0].config["configurable"]["checkpoint_ns"],
                search_results_5[1].config["configurable"]["checkpoint_ns"],
            } == {"", "inner"}

            # TODO: before 및 limit 매개변수 테스트
