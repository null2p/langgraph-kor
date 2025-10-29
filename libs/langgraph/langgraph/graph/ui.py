from __future__ import annotations

from typing import Any, Literal, cast
from uuid import uuid4

from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict

from langgraph.config import get_config, get_stream_writer
from langgraph.constants import CONF

__all__ = (
    "UIMessage",
    "RemoveUIMessage",
    "AnyUIMessage",
    "push_ui_message",
    "delete_ui_message",
    "ui_message_reducer",
)


class UIMessage(TypedDict):
    """LangGraph의 UI 업데이트를 위한 메시지 타입입니다.

    이 TypedDict는 UI 상태를 업데이트하기 위해 전송할 수 있는 UI 메시지를 나타냅니다.
    렌더링할 UI 컴포넌트와 그 속성에 대한 정보를 포함합니다.

    속성:
        type: 이것이 UI 메시지임을 나타내는 리터럴 타입입니다.
        id: UI 메시지의 고유 식별자입니다.
        name: 렌더링할 UI 컴포넌트의 이름입니다.
        props: UI 컴포넌트에 전달할 속성입니다.
        metadata: UI 메시지에 대한 추가 메타데이터입니다.
    """

    type: Literal["ui"]
    id: str
    name: str
    props: dict[str, Any]
    metadata: dict[str, Any]


class RemoveUIMessage(TypedDict):
    """LangGraph에서 UI 컴포넌트를 제거하기 위한 메시지 타입입니다.

    이 TypedDict는 현재 상태에서 UI 컴포넌트를 제거하기 위해
    전송할 수 있는 메시지를 나타냅니다.

    속성:
        type: 이것이 remove-ui 메시지임을 나타내는 리터럴 타입입니다.
        id: 제거할 UI 메시지의 고유 식별자입니다.
    """

    type: Literal["remove-ui"]
    id: str


AnyUIMessage = UIMessage | RemoveUIMessage


def push_ui_message(
    name: str,
    props: dict[str, Any],
    *,
    id: str | None = None,
    metadata: dict[str, Any] | None = None,
    message: AnyMessage | None = None,
    state_key: str | None = "ui",
    merge: bool = False,
) -> UIMessage:
    """UI 상태를 업데이트하기 위해 새 UI 메시지를 푸시합니다.

    이 함수는 UI에 렌더링될 UI 메시지를 생성하고 전송합니다.
    또한 새 UI 메시지로 그래프 상태를 업데이트합니다.

    인자:
        name: 렌더링할 UI 컴포넌트의 이름입니다.
        props: UI 컴포넌트에 전달할 속성입니다.
        id: UI 메시지의 선택적 고유 식별자입니다.
            제공되지 않으면 무작위 UUID가 생성됩니다.
        metadata: UI 메시지에 대한 선택적 추가 메타데이터입니다.
        message: UI 메시지와 연결할 선택적 메시지 객체입니다.
        state_key: UI 메시지가 저장되는 그래프 상태의 키입니다.
            기본값은 "ui"입니다.
        merge: 기존 UI 메시지와 props를 병합할지(True) 아니면
            교체할지(False) 여부입니다. 기본값은 False입니다.

    반환:
        생성된 UI 메시지입니다.

    예제:
        ```python
        push_ui_message(
            name="component-name",
            props={"content": "Hello world"},
        )
        ```

    """
    from langgraph._internal._constants import CONFIG_KEY_SEND

    writer = get_stream_writer()
    config = get_config()

    message_id = None
    if message:
        if isinstance(message, dict) and "id" in message:
            message_id = message.get("id")
        elif hasattr(message, "id"):
            message_id = message.id

    evt: UIMessage = {
        "type": "ui",
        "id": id or str(uuid4()),
        "name": name,
        "props": props,
        "metadata": {
            "merge": merge,
            "run_id": config.get("run_id", None),
            "tags": config.get("tags", None),
            "name": config.get("run_name", None),
            **(metadata or {}),
            **({"message_id": message_id} if message_id else {}),
        },
    }

    writer(evt)
    if state_key:
        config[CONF][CONFIG_KEY_SEND]([(state_key, evt)])

    return evt


def delete_ui_message(id: str, *, state_key: str = "ui") -> RemoveUIMessage:
    """UI 상태에서 ID로 UI 메시지를 삭제합니다.

    이 함수는 현재 상태에서 UI 컴포넌트를 제거하는 메시지를 생성하고 전송합니다.
    또한 UI 메시지를 제거하도록 그래프 상태를 업데이트합니다.

    인자:
        id: 제거할 UI 컴포넌트의 고유 식별자입니다.
        state_key: UI 메시지가 저장되는 그래프 상태의 키입니다. 기본값은 "ui"입니다.

    반환:
        제거 UI 메시지입니다.

    예제:
        ```python
        delete_ui_message("message-123")
        ```

    """
    from langgraph._internal._constants import CONFIG_KEY_SEND

    writer = get_stream_writer()
    config = get_config()

    evt: RemoveUIMessage = {"type": "remove-ui", "id": id}

    writer(evt)
    config[CONF][CONFIG_KEY_SEND]([(state_key, evt)])

    return evt


def ui_message_reducer(
    left: list[AnyUIMessage] | AnyUIMessage,
    right: list[AnyUIMessage] | AnyUIMessage,
) -> list[AnyUIMessage]:
    """UI 메시지 제거를 지원하면서 두 UI 메시지 목록을 병합합니다.

    이 함수는 두 UI 메시지 목록을 결합하며, 일반 UI 메시지와
    `remove-ui` 메시지를 모두 처리합니다. `remove-ui` 메시지가 발견되면
    현재 상태에서 일치하는 ID를 가진 모든 UI 메시지를 제거합니다.

    인자:
        left: 첫 번째 UI 메시지 목록 또는 단일 UI 메시지입니다.
        right: 두 번째 UI 메시지 목록 또는 단일 UI 메시지입니다.

    반환:
        제거가 적용된 결합된 UI 메시지 목록입니다.

    예제:
        ```python
        messages = ui_message_reducer(
            [{"type": "ui", "id": "1", "name": "Chat", "props": {}}],
            {"type": "remove-ui", "id": "1"},
        )
        ```

    """
    if not isinstance(left, list):
        left = [left]

    if not isinstance(right, list):
        right = [right]

    # 메시지 병합
    merged = left.copy()
    merged_by_id = {m.get("id"): i for i, m in enumerate(merged)}
    ids_to_remove = set()

    for msg in right:
        msg_id = msg.get("id")

        if (existing_idx := merged_by_id.get(msg_id)) is not None:
            if msg.get("type") == "remove-ui":
                ids_to_remove.add(msg_id)
            else:
                ids_to_remove.discard(msg_id)

                if cast(UIMessage, msg).get("metadata", {}).get("merge", False):
                    prev_msg = merged[existing_idx]
                    msg = msg.copy()
                    msg["props"] = {**prev_msg["props"], **msg["props"]}

                merged[existing_idx] = msg
        else:
            if msg.get("type") == "remove-ui":
                raise ValueError(
                    f"Attempting to delete an UI message with an ID that doesn't exist ('{msg_id}')"
                )

            merged_by_id[msg_id] = len(merged)
            merged.append(msg)

    merged = [m for m in merged if m.get("id") not in ids_to_remove]
    return merged
