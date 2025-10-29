import operator
from collections.abc import Sequence
from functools import partial
from random import choice
from typing import Annotated

from pydantic import BaseModel, Field, field_validator

from langgraph.constants import END, START
from langgraph.graph.state import StateGraph


def pydantic_state(n: int) -> StateGraph:
    class State(BaseModel):
        messages: Annotated[list, operator.add] = Field(default_factory=list)

        @field_validator("messages", mode="after")
        @classmethod
        def validate_messages(cls, v):
            if not isinstance(v, list):
                raise TypeError("messages must be a list")
            for msg in v:
                if not isinstance(msg, dict):
                    raise TypeError("messages must be a list of dicts")
                if not all(isinstance(k, str) for k in msg.keys()):
                    raise TypeError("messages must be a list of dicts with str keys")
            return v

        trigger_events: Annotated[list, operator.add] = Field(default_factory=list)
        """그래프에 의해 변환되는 외부 이벤트입니다."""

        @field_validator("trigger_events", mode="after")
        @classmethod
        def validate_trigger_events(cls, v):
            if not isinstance(v, list):
                raise TypeError("trigger_events must be a list")
            for event in v:
                if not isinstance(event, dict):
                    raise TypeError("trigger_events must be a list of dicts")
                if not all(isinstance(k, str) for k in event.keys()):
                    raise TypeError(
                        "trigger_events must be a list of dicts with str keys"
                    )
            return v

        primary_issue_medium: Annotated[str, lambda x, y: y or x] = Field(
            default="email"
        )
        """현재 대화의 주요 이슈 매체입니다."""

        @field_validator("primary_issue_medium", mode="after")
        @classmethod
        def validate_primary_issue_medium(cls, v):
            if not isinstance(v, str):
                raise TypeError("primary_issue_medium must be a string")
            return v

        autoresponse: Annotated[dict | None, lambda _, y: y] = Field(
            default=None
        )  # 항상 덮어쓰기

        @field_validator("autoresponse", mode="after")
        @classmethod
        def validate_autoresponse(cls, v):
            if v is not None and not isinstance(v, dict):
                raise TypeError("autoresponse must be a dict or None")
            return v

        issue: Annotated[dict | None, lambda x, y: y if y else x] = Field(default=None)

        @field_validator("issue", mode="after")
        @classmethod
        def validate_issue(cls, v):
            if v is not None and not isinstance(v, dict):
                raise TypeError("issue must be a dict or None")
            return v

        relevant_rules: list[dict] | None = Field(default=None)
        """현재 대화와 관련된 규칙집에서 가져온 SOP입니다."""

        @field_validator("relevant_rules", mode="after")
        @classmethod
        def validate_relevant_rules(cls, v):
            if v is None:
                return v
            if not isinstance(v, list):
                raise TypeError("relevant_rules must be a list or None")
            for rule in v:
                if not isinstance(rule, dict):
                    raise TypeError("relevant_rules must be a list of dicts")
                if not all(isinstance(k, str) for k in rule.keys()):
                    raise TypeError(
                        "relevant_rules must be a list of dicts with str keys"
                    )
            return v

        memory_docs: list[dict] | None = Field(default=None)
        """현재 대화와 관련된 메모리 서비스에서 가져온 메모리 문서입니다."""

        @field_validator("memory_docs", mode="after")
        @classmethod
        def validate_memory_docs(cls, v):
            if v is None:
                return v
            if not isinstance(v, list):
                raise TypeError("memory_docs must be a list or None")
            for doc in v:
                if not isinstance(doc, dict):
                    raise TypeError("memory_docs must be a list of dicts")
                if not all(isinstance(k, str) for k in doc.keys()):
                    raise TypeError("memory_docs must be a list of dicts with str keys")
            return v

        categorizations: Annotated[list[dict], operator.add] = Field(
            default_factory=list
        )
        """AI가 자동 생성한 이슈 분류입니다."""

        @field_validator("categorizations", mode="after")
        @classmethod
        def validate_categorizations(cls, v):
            if not isinstance(v, list):
                raise TypeError("categorizations must be a list")
            for categorization in v:
                if not isinstance(categorization, dict):
                    raise TypeError("categorizations must be a list of dicts")
                if not all(isinstance(k, str) for k in categorization.keys()):
                    raise TypeError(
                        "categorizations must be a list of dicts with str keys"
                    )
            return v

        responses: Annotated[list[dict], operator.add] = Field(default_factory=list)
        """AI가 추천한 초안 응답입니다."""

        @field_validator("responses", mode="after")
        @classmethod
        def validate_responses(cls, v):
            if not isinstance(v, list):
                raise TypeError("responses must be a list")
            for response in v:
                if not isinstance(response, dict):
                    raise TypeError("responses must be a list of dicts")
                if not all(isinstance(k, str) for k in response.keys()):
                    raise TypeError("responses must be a list of dicts with str keys")
            return v

        user_info: Annotated[dict | None, lambda x, y: y if y is not None else x] = (
            Field(default=None)
        )
        """현재 사용자 상태입니다 (이메일 기준)."""

        @field_validator("user_info", mode="after")
        @classmethod
        def validate_user_info(cls, v):
            if v is not None and not isinstance(v, dict):
                raise TypeError("user_info must be a dict or None")
            return v

        crm_info: Annotated[dict | None, lambda x, y: y if y is not None else x] = (
            Field(default=None)
        )
        """현재 사용자가 속한 조직의 CRM 정보입니다."""

        @field_validator("crm_info", mode="after")
        @classmethod
        def validate_crm_info(cls, v):
            if v is not None and not isinstance(v, dict):
                raise TypeError("crm_info must be a dict or None")
            return v

        email_thread_id: Annotated[
            str | None, lambda x, y: y if y is not None else x
        ] = Field(default=None)
        """현재 이메일 스레드 ID입니다."""

        @field_validator("email_thread_id", mode="after")
        @classmethod
        def validate_email_thread_id(cls, v):
            if v is not None and not isinstance(v, str):
                raise TypeError("email_thread_id must be a string or None")
            return v

        slack_participants: Annotated[dict, operator.or_] = Field(default_factory=dict)
        """현재 슬랙 참여자의 증가하는 목록입니다."""

        @field_validator("slack_participants", mode="after")
        @classmethod
        def validate_slack_participants(cls, v):
            if not isinstance(v, dict):
                raise TypeError("slack_participants must be a dict")
            for participant in v:
                if not isinstance(participant, str):
                    raise TypeError("slack_participants must be a dict with str keys")
            return v

        bot_id: str | None = Field(default=None)
        """슬랙 채널의 봇 사용자 ID입니다."""

        @field_validator("bot_id", mode="after")
        @classmethod
        def validate_bot_id(cls, v):
            if v is not None and not isinstance(v, str):
                raise TypeError("bot_id must be a string or None")
            return v

        notified_assignees: Annotated[dict, operator.or_] = Field(default_factory=dict)

        @field_validator("notified_assignees", mode="after")
        def validate_notified_assignees(cls, v):
            if not isinstance(v, dict):
                raise TypeError("notified_assignees must be a dict")
            for assignee in v:
                if not isinstance(assignee, str):
                    raise TypeError("notified_assignees must be a dict with str keys")
            return v

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
        val = getattr(input, read)
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
        "six", lambda state: END if len(state.messages) > n else "one"
    )

    return builder


if __name__ == "__main__":
    import asyncio

    import uvloop
    from langgraph.checkpoint.memory import InMemorySaver

    graph = pydantic_state(1000).compile(checkpointer=InMemorySaver())
    input = {
        "messages": [
            {
                str(i) * 10: {
                    str(j) * 10: ["hi?" * 10, True, 1, 6327816386138, None] * 5
                    for j in range(5)
                }
                for i in range(5)
            }
        ]
    }
    config = {"configurable": {"thread_id": "1"}, "recursion_limit": 20000000000}

    async def run():
        async for c in graph.astream(input, config=config):
            print(c.keys())

    uvloop.install()
    asyncio.run(run())
