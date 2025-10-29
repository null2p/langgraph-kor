"""이 모듈은 langchain 그래프에서 도구 호출을 검증하는 데 사용할 수 있는 ValidationNode 클래스를 제공합니다.
모델 출력의 tool_calls에 pydantic 스키마를 적용하고, 검증된 콘텐츠와 함께 ToolMessage를 반환합니다.
스키마가 유효하지 않은 경우 오류 메시지와 함께 ToolMessage를 반환합니다. ValidationNode는
"messages" 키가 있는 StateGraph에서 사용할 수 있습니다. 여러 도구 호출이 요청되면 병렬로 실행됩니다.
"""

from collections.abc import Callable, Sequence
from typing import (
    Any,
    cast,
)

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import (
    RunnableConfig,
)
from langchain_core.runnables.config import get_executor_for_config
from langchain_core.tools import BaseTool, create_schema_from_function
from langchain_core.utils.pydantic import is_basemodel_subclass
from langgraph._internal._runnable import RunnableCallable
from langgraph.warnings import LangGraphDeprecatedSinceV10
from pydantic import BaseModel, ValidationError
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import ValidationError as ValidationErrorV1
from typing_extensions import deprecated


def _default_format_error(
    error: BaseException,
    call: ToolCall,
    schema: type[BaseModel] | type[BaseModelV1],
) -> str:
    """기본 오류 포맷 함수입니다."""
    return f"{repr(error)}\n\n모든 검증 오류를 수정한 후 응답하세요."


@deprecated(
    "ValidationNode는 더 이상 사용되지 않습니다. 커스텀 도구 오류 처리와 함께 `langchain.agents`의 `create_agent`를 사용하세요.",
    category=LangGraphDeprecatedSinceV10,
)
class ValidationNode(RunnableCallable):
    """마지막 AIMessage의 모든 도구 요청을 검증하는 노드입니다.

    "messages" 키가 있는 StateGraph에서 사용할 수 있습니다.

    !!! note

        이 노드는 실제로 도구를 **실행하지** 않으며, 도구 호출만 검증합니다.
        이는 원본 메시지와 도구 ID를 잃지 않고 복잡한 스키마를 준수하는
        구조화된 출력을 생성해야 하는 추출 및 기타 사용 사례에 유용합니다
        (다중 턴 대화에서 사용).

    Returns:
        (Union[Dict[str, List[ToolMessage]], Sequence[ToolMessage]]): 검증된 콘텐츠 또는 오류 메시지가 포함된 ToolMessage 목록입니다.

    Example:
        ```python title="모델이 유효한 응답을 생성하도록 재촉구하는 예제 사용법:"
        from typing import Literal, Annotated
        from typing_extensions import TypedDict

        from langchain_anthropic import ChatAnthropic
        from pydantic import BaseModel, field_validator

        from langgraph.graph import END, START, StateGraph
        from langgraph.prebuilt import ValidationNode
        from langgraph.graph.message import add_messages

        class SelectNumber(BaseModel):
            a: int

            @field_validator("a")
            def a_must_be_meaningful(cls, v):
                if v != 37:
                    raise ValueError("Only 37 is allowed")
                return v

        builder = StateGraph(Annotated[list, add_messages])
        llm = ChatAnthropic(model="claude-3-5-haiku-latest").bind_tools([SelectNumber])
        builder.add_node("model", llm)
        builder.add_node("validation", ValidationNode([SelectNumber]))
        builder.add_edge(START, "model")

        def should_validate(state: list) -> Literal["validation", "__end__"]:
            if state[-1].tool_calls:
                return "validation"
            return END

        builder.add_conditional_edges("model", should_validate)

        def should_reprompt(state: list) -> Literal["model", "__end__"]:
            for msg in state[::-1]:
                # 도구 호출 중 어느 것도 오류가 아님
                if msg.type == "ai":
                    return END
                if msg.additional_kwargs.get("is_error"):
                    return "model"
            return END

        builder.add_conditional_edges("validation", should_reprompt)

        graph = builder.compile()
        res = graph.invoke(("user", "Select a number, any number"))
        # 재시도 로직을 표시합니다
        for msg in res:
            msg.pretty_print()
        ```
    """

    def __init__(
        self,
        schemas: Sequence[BaseTool | type[BaseModel] | Callable],
        *,
        format_error: Callable[[BaseException, ToolCall, type[BaseModel]], str]
        | None = None,
        name: str = "validation",
        tags: list[str] | None = None,
    ) -> None:
        """ValidationNode를 초기화합니다.

        Args:
            schemas: 도구 호출을 검증할 스키마 목록입니다. 다음 중 하나일 수 있습니다:
                - pydantic BaseModel 클래스
                - BaseTool 인스턴스 (args_schema가 사용됨)
                - 함수 (함수 시그니처로부터 스키마가 생성됨)
            format_error: 예외, ToolCall, 스키마를 받아 포맷된 오류 문자열을 반환하는 함수입니다.
                기본적으로 예외 repr과 검증 오류 수정 후 응답하라는 메시지를 반환합니다.
            name: 노드의 이름입니다.
            tags: 노드에 추가할 태그 목록입니다.
        """
        super().__init__(self._func, None, name=name, tags=tags, trace=False)
        self._format_error = format_error or _default_format_error
        self.schemas_by_name: dict[str, type[BaseModel]] = {}
        for schema in schemas:
            if isinstance(schema, BaseTool):
                if schema.args_schema is None:
                    raise ValueError(
                        f"Tool {schema.name} does not have an args_schema defined."
                    )
                elif not isinstance(
                    schema.args_schema, type
                ) or not is_basemodel_subclass(schema.args_schema):
                    raise ValueError(
                        "Validation node only works with tools that have a pydantic BaseModel args_schema. "
                        f"Got {schema.name} with args_schema: {schema.args_schema}."
                    )
                self.schemas_by_name[schema.name] = schema.args_schema
            elif isinstance(schema, type) and issubclass(
                schema, (BaseModel, BaseModelV1)
            ):
                self.schemas_by_name[schema.__name__] = cast(type[BaseModel], schema)
            elif callable(schema):
                base_model = create_schema_from_function("Validation", schema)
                self.schemas_by_name[schema.__name__] = base_model
            else:
                raise ValueError(
                    f"Unsupported input to ValidationNode. Expected BaseModel, tool or function. Got: {type(schema)}."
                )

    def _get_message(
        self, input: list[AnyMessage] | dict[str, Any]
    ) -> tuple[str, AIMessage]:
        """입력에서 마지막 AIMessage를 추출합니다."""
        if isinstance(input, list):
            output_type = "list"
            messages: list = input
        elif messages := input.get("messages", []):
            output_type = "dict"
        else:
            raise ValueError("No message found in input")
        message: AnyMessage = messages[-1]
        if not isinstance(message, AIMessage):
            raise ValueError("Last message is not an AIMessage")
        return output_type, message

    def _func(
        self, input: list[AnyMessage] | dict[str, Any], config: RunnableConfig
    ) -> Any:
        """도구 호출을 동기적으로 검증하고 실행합니다."""
        output_type, message = self._get_message(input)

        def run_one(call: ToolCall) -> ToolMessage:
            schema = self.schemas_by_name[call["name"]]
            try:
                if issubclass(schema, BaseModel):
                    output = schema.model_validate(call["args"])
                    content = output.model_dump_json()
                elif issubclass(schema, BaseModelV1):
                    output = schema.validate(call["args"])
                    content = output.json()
                else:
                    raise ValueError(
                        f"Unsupported schema type: {type(schema)}. Expected BaseModel or BaseModelV1."
                    )
                return ToolMessage(
                    content=content,
                    name=call["name"],
                    tool_call_id=cast(str, call["id"]),
                )
            except (ValidationError, ValidationErrorV1) as e:
                return ToolMessage(
                    content=self._format_error(e, call, schema),
                    name=call["name"],
                    tool_call_id=cast(str, call["id"]),
                    additional_kwargs={"is_error": True},
                )

        with get_executor_for_config(config) as executor:
            outputs = [*executor.map(run_one, message.tool_calls)]
            if output_type == "list":
                return outputs
            else:
                return {"messages": outputs}
