from typing import Literal

from langgraph.warnings import LangGraphDeprecatedSinceV10
from typing_extensions import TypedDict, deprecated


@deprecated(
    "HumanInterruptConfig가 `langchain.agents.interrupt`로 이동되었습니다. import를 `from langchain.agents.interrupt import HumanInterruptConfig`로 업데이트하세요.",
    category=LangGraphDeprecatedSinceV10,
)
class HumanInterruptConfig(TypedDict):
    """human interrupt에 허용되는 작업을 정의하는 설정입니다.

    그래프가 인간 입력을 위해 일시 중지될 때 사용 가능한 상호 작용 옵션을 제어합니다.

    Attributes:
        allow_ignore: 인간이 현재 단계를 무시/건너뛸 수 있는지 여부
        allow_respond: 인간이 텍스트 응답/피드백을 제공할 수 있는지 여부
        allow_edit: 인간이 제공된 콘텐츠/상태를 편집할 수 있는지 여부
        allow_accept: 인간이 현재 상태를 수락/승인할 수 있는지 여부
    """

    allow_ignore: bool
    allow_respond: bool
    allow_edit: bool
    allow_accept: bool


@deprecated(
    "ActionRequest가 `langchain.agents.interrupt`로 이동되었습니다. import를 `from langchain.agents.interrupt import ActionRequest`로 업데이트하세요.",
    category=LangGraphDeprecatedSinceV10,
)
class ActionRequest(TypedDict):
    """그래프 실행 내에서 인간 작업에 대한 요청을 나타냅니다.

    작업에 필요한 작업 유형과 관련 인수를 포함합니다.

    Attributes:
        action: 요청되는 작업의 유형 또는 이름 (예: "Approve XYZ action")
        args: 작업에 필요한 인수의 키-값 쌍
    """

    action: str
    args: dict


@deprecated(
    "HumanInterrupt가 `langchain.agents.interrupt`로 이동되었습니다. import를 `from langchain.agents.interrupt import HumanInterrupt`로 업데이트하세요.",
    category=LangGraphDeprecatedSinceV10,
)
class HumanInterrupt(TypedDict):
    """인간 개입이 필요한 그래프에 의해 트리거된 interrupt를 나타냅니다.

    실행이 인간 입력을 위해 일시 중지될 때 `interrupt` 함수에 전달됩니다.

    Attributes:
        action_request: 인간으로부터 요청되는 특정 작업
        config: 허용되는 작업을 정의하는 설정
        description: 필요한 입력에 대한 선택적 상세 설명

    Example:
        ```python
        # 상태에서 도구 호출을 추출하고 interrupt 요청을 생성합니다
        request = HumanInterrupt(
            action_request=ActionRequest(
                action="run_command",  # 요청되는 작업
                args={"command": "ls", "args": ["-l"]}  # 작업에 대한 인수
            ),
            config=HumanInterruptConfig(
                allow_ignore=True,    # 이 단계 건너뛰기 허용
                allow_respond=True,   # 텍스트 피드백 허용
                allow_edit=False,     # 편집 불허
                allow_accept=True     # 직접 수락 허용
            ),
            description="Please review the command before execution"
        )
        # interrupt 요청을 전송하고 응답을 가져옵니다
        response = interrupt([request])[0]
        ```
    """

    action_request: ActionRequest
    config: HumanInterruptConfig
    description: str | None


class HumanResponse(TypedDict):
    """그래프 실행이 재개될 때 반환되는, interrupt에 대한 인간의 응답입니다.

    Attributes:
        type: 응답 유형:
            - "accept": 변경 없이 현재 상태를 승인합니다
            - "ignore": 현재 단계를 건너뛰기/무시합니다
            - "response": 텍스트 피드백 또는 지시사항을 제공합니다
            - "edit": 현재 상태/콘텐츠를 수정합니다
        args: 응답 페이로드:
            - None: ignore/accept 작업의 경우
            - str: 텍스트 응답의 경우
            - ActionRequest: 업데이트된 콘텐츠를 포함한 edit 작업의 경우
    """

    type: Literal["accept", "ignore", "response", "edit"]
    args: None | str | ActionRequest
