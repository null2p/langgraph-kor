from __future__ import annotations

import uuid
import warnings
from collections.abc import Callable, Sequence
from functools import partial
from typing import (
    Annotated,
    Any,
    Literal,
    cast,
)

from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    BaseMessageChunk,
    MessageLikeRepresentation,
    RemoveMessage,
    convert_to_messages,
    message_chunk_to_message,
)
from typing_extensions import TypedDict, deprecated

from langgraph._internal._constants import CONF, CONFIG_KEY_SEND, NS_SEP
from langgraph.graph.state import StateGraph
from langgraph.warnings import LangGraphDeprecatedSinceV10

__all__ = (
    "add_messages",
    "MessagesState",
    "MessageGraph",
)

Messages = list[MessageLikeRepresentation] | MessageLikeRepresentation

REMOVE_ALL_MESSAGES = "__remove_all__"


def _add_messages_wrapper(func: Callable) -> Callable[[Messages, Messages], Messages]:
    def _add_messages(
        left: Messages | None = None, right: Messages | None = None, **kwargs: Any
    ) -> Messages | Callable[[Messages, Messages], Messages]:
        if left is not None and right is not None:
            return func(left, right, **kwargs)
        elif left is not None or right is not None:
            msg = (
                f"Must specify non-null arguments for both 'left' and 'right'. Only "
                f"received: '{'left' if left else 'right'}'."
            )
            raise ValueError(msg)
        else:
            return partial(func, **kwargs)

    _add_messages.__doc__ = func.__doc__
    return cast(Callable[[Messages, Messages], Messages], _add_messages)


@_add_messages_wrapper
def add_messages(
    left: Messages,
    right: Messages,
    *,
    format: Literal["langchain-openai"] | None = None,
) -> Messages:
    """두 메시지 목록을 병합하여 ID로 기존 메시지를 업데이트합니다.

    기본적으로 새 메시지가 기존 메시지와 동일한 ID를 갖지 않는 한,
    상태가 "추가 전용"이 되도록 합니다.

    인자:
        left: `Messages`의 기본 목록입니다.
        right: 기본 목록에 병합할 `Messages` 목록 (또는 단일 `Message`)입니다.
        format: 메시지를 반환할 형식입니다. `None`이면 `Messages`가
            그대로 반환됩니다. `langchain-openai`이면 `Messages`가
            OpenAI 메시지 형식과 일치하도록 내용이 포맷된 `BaseMessage` 객체로 반환되며,
            이는 내용이 문자열, `'text'` 블록 또는 `'image_url'` 블록일 수 있고
            도구 응답이 자체 `ToolMessage` 객체로 반환된다는 의미입니다.

            !!! important "요구 사항"

                이 기능을 사용하려면 `langchain-core>=0.3.11`이 설치되어 있어야 합니다.

    반환:
        `right`의 메시지가 `left`에 병합된 새 메시지 목록입니다.
        `right`의 메시지가 `left`의 메시지와 동일한 ID를 가지면,
            `right`의 메시지가 `left`의 메시지를 대체합니다.

    예제:
        ```python title="Basic usage"
        from langchain_core.messages import AIMessage, HumanMessage

        msgs1 = [HumanMessage(content="Hello", id="1")]
        msgs2 = [AIMessage(content="Hi there!", id="2")]
        add_messages(msgs1, msgs2)
        # [HumanMessage(content='Hello', id='1'), AIMessage(content='Hi there!', id='2')]
        ```

        ```python title="Overwrite existing message"
        msgs1 = [HumanMessage(content="Hello", id="1")]
        msgs2 = [HumanMessage(content="Hello again", id="1")]
        add_messages(msgs1, msgs2)
        # [HumanMessage(content='Hello again', id='1')]
        ```

        ```python title="Use in a StateGraph"
        from typing import Annotated
        from typing_extensions import TypedDict
        from langgraph.graph import StateGraph


        class State(TypedDict):
            messages: Annotated[list, add_messages]


        builder = StateGraph(State)
        builder.add_node("chatbot", lambda state: {"messages": [("assistant", "Hello")]})
        builder.set_entry_point("chatbot")
        builder.set_finish_point("chatbot")
        graph = builder.compile()
        graph.invoke({})
        # {'messages': [AIMessage(content='Hello', id=...)]}
        ```

        ```python title="Use OpenAI message format"
        from typing import Annotated
        from typing_extensions import TypedDict
        from langgraph.graph import StateGraph, add_messages


        class State(TypedDict):
            messages: Annotated[list, add_messages(format="langchain-openai")]


        def chatbot_node(state: State) -> list:
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Here's an image:",
                                "cache_control": {"type": "ephemeral"},
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": "1234",
                                },
                            },
                        ],
                    },
                ]
            }


        builder = StateGraph(State)
        builder.add_node("chatbot", chatbot_node)
        builder.set_entry_point("chatbot")
        builder.set_finish_point("chatbot")
        graph = builder.compile()
        graph.invoke({"messages": []})
        # {
        #     'messages': [
        #         HumanMessage(
        #             content=[
        #                 {"type": "text", "text": "Here's an image:"},
        #                 {
        #                     "type": "image_url",
        #                     "image_url": {"url": "data:image/jpeg;base64,1234"},
        #                 },
        #             ],
        #         ),
        #     ]
        # }
        ```

    """
    remove_all_idx = None
    # 리스트로 변환
    if not isinstance(left, list):
        left = [left]  # type: ignore[assignment]
    if not isinstance(right, list):
        right = [right]  # type: ignore[assignment]
    # 메시지로 변환
    left = [
        message_chunk_to_message(cast(BaseMessageChunk, m))
        for m in convert_to_messages(left)
    ]
    right = [
        message_chunk_to_message(cast(BaseMessageChunk, m))
        for m in convert_to_messages(right)
    ]
    # 누락된 id 할당
    for m in left:
        if m.id is None:
            m.id = str(uuid.uuid4())
    for idx, m in enumerate(right):
        if m.id is None:
            m.id = str(uuid.uuid4())
        if isinstance(m, RemoveMessage) and m.id == REMOVE_ALL_MESSAGES:
            remove_all_idx = idx

    if remove_all_idx is not None:
        return right[remove_all_idx + 1 :]

    # 병합
    merged = left.copy()
    merged_by_id = {m.id: i for i, m in enumerate(merged)}
    ids_to_remove = set()
    for m in right:
        if (existing_idx := merged_by_id.get(m.id)) is not None:
            if isinstance(m, RemoveMessage):
                ids_to_remove.add(m.id)
            else:
                ids_to_remove.discard(m.id)
                merged[existing_idx] = m
        else:
            if isinstance(m, RemoveMessage):
                raise ValueError(
                    f"Attempting to delete a message with an ID that doesn't exist ('{m.id}')"
                )

            merged_by_id[m.id] = len(merged)
            merged.append(m)
    merged = [m for m in merged if m.id not in ids_to_remove]

    if format == "langchain-openai":
        merged = _format_messages(merged)
    elif format:
        msg = f"Unrecognized {format=}. Expected one of 'langchain-openai', None."
        raise ValueError(msg)
    else:
        pass

    return merged


@deprecated(
    "MessageGraph is deprecated in langgraph 1.0.0, to be removed in 2.0.0. Please use StateGraph with a `messages` key instead.",
    category=None,
)
class MessageGraph(StateGraph):
    """모든 노드가 메시지 목록을 입력으로 받고 하나 이상의 메시지를 출력으로 반환하는 StateGraph입니다.

    MessageGraph는 전체 상태가 단일 추가 전용* 메시지 목록인 StateGraph의 서브클래스입니다.
    MessageGraph의 각 노드는 메시지 목록을 입력으로 받고 0개 이상의
    메시지를 출력으로 반환합니다. `add_messages` 함수는 각 노드의 출력 메시지를
    그래프 상태의 기존 메시지 목록에 병합하는 데 사용됩니다.

    예제:
        ```pycon
        >>> from langgraph.graph.message import MessageGraph
        ...
        >>> builder = MessageGraph()
        >>> builder.add_node("chatbot", lambda state: [("assistant", "Hello!")])
        >>> builder.set_entry_point("chatbot")
        >>> builder.set_finish_point("chatbot")
        >>> builder.compile().invoke([("user", "Hi there.")])
        [HumanMessage(content="Hi there.", id='...'), AIMessage(content="Hello!", id='...')]
        ```

        ```pycon
        >>> from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
        >>> from langgraph.graph.message import MessageGraph
        ...
        >>> builder = MessageGraph()
        >>> builder.add_node(
        ...     "chatbot",
        ...     lambda state: [
        ...         AIMessage(
        ...             content="Hello!",
        ...             tool_calls=[{"name": "search", "id": "123", "args": {"query": "X"}}],
        ...         )
        ...     ],
        ... )
        >>> builder.add_node(
        ...     "search", lambda state: [ToolMessage(content="Searching...", tool_call_id="123")]
        ... )
        >>> builder.set_entry_point("chatbot")
        >>> builder.add_edge("chatbot", "search")
        >>> builder.set_finish_point("search")
        >>> builder.compile().invoke([HumanMessage(content="Hi there. Can you search for X?")])
        {'messages': [HumanMessage(content="Hi there. Can you search for X?", id='b8b7d8f4-7f4d-4f4d-9c1d-f8b8d8f4d9c1'),
                     AIMessage(content="Hello!", id='f4d9c1d8-8d8f-4d9c-b8b7-d8f4f4d9c1d8'),
                     ToolMessage(content="Searching...", id='d8f4f4d9-c1d8-4f4d-b8b7-d8f4f4d9c1d8', tool_call_id="123")]}
        ```
    """

    def __init__(self) -> None:
        warnings.warn(
            "MessageGraph is deprecated in LangGraph v1.0.0, to be removed in v2.0.0. Please use StateGraph with a `messages` key instead.",
            category=LangGraphDeprecatedSinceV10,
            stacklevel=2,
        )
        super().__init__(Annotated[list[AnyMessage], add_messages])  # type: ignore[arg-type]


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def _format_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    try:
        from langchain_core.messages import convert_to_openai_messages
    except ImportError:
        msg = (
            "Must have langchain-core>=0.3.11 installed to use automatic message "
            "formatting (format='langchain-openai'). Please update your langchain-core "
            "version or remove the 'format' flag. Returning un-formatted "
            "messages."
        )
        warnings.warn(msg)
        return list(messages)
    else:
        return convert_to_messages(convert_to_openai_messages(messages))


def push_message(
    message: MessageLikeRepresentation | BaseMessageChunk,
    *,
    state_key: str | None = "messages",
) -> AnyMessage:
    """`messages` / `messages-tuple` 스트림 모드에 메시지를 수동으로 작성합니다.

    `state_key`가 `None`이 아닌 경우 `state_key`에 지정된 채널에 자동으로 작성됩니다.
    """

    from langchain_core.callbacks.base import (
        BaseCallbackHandler,
        BaseCallbackManager,
    )

    from langgraph.config import get_config
    from langgraph.pregel._messages import StreamMessagesHandler

    config = get_config()
    message = next(x for x in convert_to_messages([message]))

    if message.id is None:
        raise ValueError("Message ID is required")

    if isinstance(config["callbacks"], BaseCallbackManager):
        manager = config["callbacks"]
        handlers = manager.handlers
    elif isinstance(config["callbacks"], list) and all(
        isinstance(x, BaseCallbackHandler) for x in config["callbacks"]
    ):
        handlers = config["callbacks"]

    if stream_handler := next(
        (x for x in handlers if isinstance(x, StreamMessagesHandler)), None
    ):
        metadata = config["metadata"]
        message_meta = (
            tuple(cast(str, metadata["langgraph_checkpoint_ns"]).split(NS_SEP)),
            metadata,
        )
        stream_handler._emit(message_meta, message, dedupe=False)

    if state_key:
        config[CONF][CONFIG_KEY_SEND]([(state_key, message)])

    return message
