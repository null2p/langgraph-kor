import operator
from collections.abc import Sequence
from functools import partial
from random import choice
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.graph.state import StateGraph


def wide_dict(n: int) -> StateGraph:
    class State(TypedDict):
        messages: Annotated[list, operator.add]
        trigger_events: Annotated[list, operator.add]
        """그래프에 의해 변환되는 외부 이벤트입니다."""
        primary_issue_medium: Annotated[str, lambda x, y: y or x]
        autoresponse: Annotated[dict | None, lambda _, y: y]  # 항상 덮어쓰기
        issue: Annotated[dict | None, lambda x, y: y if y else x]
        relevant_rules: list[dict] | None
        """현재 대화와 관련된 규칙집에서 가져온 SOP입니다."""
        memory_docs: list[dict] | None
        """현재 대화와 관련된 메모리 서비스에서 가져온 메모리 문서입니다."""
        categorizations: Annotated[list[dict], operator.add]
        """AI가 자동 생성한 이슈 분류입니다."""
        responses: Annotated[list[dict], operator.add]
        """AI가 추천한 초안 응답입니다."""

        user_info: Annotated[dict | None, lambda x, y: y if y is not None else x]
        """현재 사용자 상태입니다 (이메일 기준)."""
        crm_info: Annotated[dict | None, lambda x, y: y if y is not None else x]
        """현재 사용자가 속한 조직의 CRM 정보입니다."""
        email_thread_id: Annotated[str | None, lambda x, y: y if y is not None else x]
        """현재 이메일 스레드 ID입니다."""
        slack_participants: Annotated[dict, operator.or_]
        """현재 슬랙 참여자의 증가하는 목록입니다."""
        bot_id: str | None
        """슬랙 채널의 봇 사용자 ID입니다."""
        notified_assignees: Annotated[dict, operator.or_]

    list_fields = {
        "messages",
        "trigger_events",
        "categorizations",
        "responses",
        "memory_docs",
        "relevant_rules",
    }
    dict_fields = {
        "user_info",
        "crm_info",
        "slack_participants",
        "notified_assignees",
        "autoresponse",
        "issue",
    }

    def read_write(read: str, write: Sequence[str], input: State) -> dict:
        val = input.get(read)
        val = {val: val} if isinstance(val, str) else val
        val_single = val[-1] if isinstance(val, list) else val
        val_list = val if isinstance(val, list) else [val]
        return {
            k: val_list
            if k in list_fields
            else val_single
            if k in dict_fields
            else "".join(choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n))
            for k in write
        }

    builder = StateGraph(State)
    builder.add_edge(START, "one")
    builder.add_node(
        "one",
        partial(read_write, "messages", ["trigger_events", "primary_issue_medium"]),
    )
    builder.add_edge("one", "two")
    builder.add_node(
        "two",
        partial(read_write, "trigger_events", ["autoresponse", "issue"]),
    )
    builder.add_edge("two", "three")
    builder.add_edge("two", "four")
    builder.add_node(
        "three",
        partial(read_write, "autoresponse", ["relevant_rules"]),
    )
    builder.add_node(
        "four",
        partial(
            read_write,
            "trigger_events",
            ["categorizations", "responses", "memory_docs"],
        ),
    )
    builder.add_node(
        "five",
        partial(
            read_write,
            "categorizations",
            [
                "user_info",
                "crm_info",
                "email_thread_id",
                "slack_participants",
                "bot_id",
                "notified_assignees",
            ],
        ),
    )
    builder.add_edge(["three", "four"], "five")
    builder.add_edge("five", "six")
    builder.add_node(
        "six",
        partial(read_write, "responses", ["messages"]),
    )
    builder.add_conditional_edges(
        "six", lambda state: END if len(state["messages"]) > n else "one"
    )

    return builder


if __name__ == "__main__":
    import asyncio

    import uvloop
    from langgraph.checkpoint.memory import InMemorySaver

    graph = wide_dict(1000).compile(checkpointer=InMemorySaver())
    input = {
        "messages": [
            {
                str(i) * 10: {
                    str(j) * 10: ["hi?" * 10, True, 1, 6327816386138, None] * 5
                    for j in range(50)
                }
                for i in range(50)
            }
        ]
    }
    config = {"configurable": {"thread_id": "1"}, "recursion_limit": 20000000000}

    async def run():
        async for c in graph.astream(input, config=config):
            print(c.keys())

    uvloop.install()
    asyncio.run(run())
