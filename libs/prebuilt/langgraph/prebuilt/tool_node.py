"""LangGraph 워크플로우를 위한 도구 실행 노드입니다.

이 모듈은 LangGraph에서 도구를 실행하기 위한 사전 구축된 기능을 제공합니다.

도구는 모델이 외부 시스템, API, 데이터베이스와 상호 작용하거나
계산을 수행하기 위해 호출할 수 있는 함수입니다.

이 모듈은 여러 주요 디자인 패턴을 구현합니다:
- 효율성을 위한 여러 도구 호출의 병렬 실행
- 사용자 정의 가능한 오류 메시지를 가진 강력한 오류 처리
- 그래프 상태에 접근해야 하는 도구를 위한 상태 주입
- 영구 저장소가 필요한 도구를 위한 저장소 주입
- 고급 제어 흐름을 위한 명령 기반 상태 업데이트

주요 구성 요소:
    ToolNode: LangGraph 워크플로우에서 도구를 실행하는 메인 클래스
    InjectedState: 도구에 그래프 상태를 주입하기 위한 어노테이션
    InjectedStore: 도구에 영구 저장소를 주입하기 위한 어노테이션
    tools_condition: 도구 호출을 기반으로 조건부 라우팅을 위한 유틸리티 함수

일반적인 사용법:
    ```python
    from langchain_core.tools import tool
    from langgraph.prebuilt import ToolNode

    @tool
    def my_tool(x: int) -> str:
        return f"Result: {x}"

    tool_node = ToolNode([my_tool])
    ```
"""

import asyncio
import inspect
import json
import types
from collections.abc import Callable, Sequence
from copy import copy, deepcopy
from dataclasses import replace
from typing import (
    Annotated,
    Any,
    Literal,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    RemoveMessage,
    ToolCall,
    ToolMessage,
    convert_to_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import (
    get_config_list,
    get_executor_for_config,
)
from langchain_core.tools import BaseTool, InjectedToolArg
from langchain_core.tools import tool as create_tool
from langchain_core.tools.base import (
    TOOL_MESSAGE_BLOCK_TYPES,
    get_all_basemodel_annotations,
)
from langgraph._internal._runnable import RunnableCallable
from langgraph.errors import GraphBubbleUp
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.store.base import BaseStore
from langgraph.types import Command, Send
from pydantic import BaseModel

INVALID_TOOL_NAME_ERROR_TEMPLATE = (
    "Error: {requested_tool} is not a valid tool, try one of [{available_tools}]."
)
TOOL_CALL_ERROR_TEMPLATE = "Error: {error}\n Please fix your mistakes."


def msg_content_output(output: Any) -> str | list[dict]:
    """도구 출력을 유효한 메시지 콘텐츠 형식으로 변환합니다.

    LangChain ToolMessage는 문자열 콘텐츠 또는 콘텐츠 블록 목록을 허용합니다.
    이 함수는 가능한 경우 구조화된 데이터를 보존하려고 시도하고,
    JSON 직렬화 또는 문자열 변환으로 폴백하여 도구 출력이 메시지 소비를 위해
    적절하게 포맷되도록 보장합니다.

    Args:
        output: 도구 실행의 원시 출력입니다. 모든 타입이 가능합니다.

    Returns:
        출력이 이미 구조화된 콘텐츠를 위한 올바른 형식인 경우
        출력의 문자열 표현 또는 콘텐츠 블록 목록입니다.

    Note:
        이 함수는 모든 가능한 메시지 콘텐츠 형식을 지원하는 대신
        JSON 직렬화를 기본값으로 사용하여 이전 버전과의 호환성을 우선시합니다.
    """
    if isinstance(output, str):
        return output
    elif isinstance(output, list) and all(
        [
            isinstance(x, dict) and x.get("type") in TOOL_MESSAGE_BLOCK_TYPES
            for x in output
        ]
    ):
        return output
    # 기술적으로 문자열 목록도 유효한 메시지 콘텐츠이지만,
    # 현재 모든 채팅 모델이 이를 지원하는지 충분히 테스트되지 않았습니다.
    # 그리고 이전 버전과의 호환성을 위해 기존 ToolNode 사용을 중단하지 않도록
    # 확인하고 싶습니다.
    else:
        try:
            return json.dumps(output, ensure_ascii=False)
        except Exception:
            return str(output)


def _handle_tool_error(
    e: Exception,
    *,
    flag: bool | str | Callable[..., str] | tuple[type[Exception], ...],
) -> str:
    """예외 처리 설정에 따라 오류 메시지 콘텐츠를 생성합니다.

    이 함수는 ToolNode의 handle_tool_errors 매개변수를 통해 구성된
    다양한 오류 처리 전략을 지원하여 오류 메시지 생성 로직을 중앙 집중화합니다.

    Args:
        e: 도구 실행 중 발생한 예외입니다.
        flag: 오류 처리 방법에 대한 설정입니다. 다음 중 하나일 수 있습니다:
            - bool: `True`인 경우 기본 오류 템플릿을 사용합니다
            - str: 이 문자열을 오류 메시지로 사용합니다
            - Callable: 예외와 함께 이 함수를 호출하여 오류 메시지를 가져옵니다
            - tuple: 이 컨텍스트에서는 사용되지 않음 (호출자가 처리)

    Returns:
        ToolMessage에 포함할 오류 메시지를 포함하는 문자열입니다.

    Raises:
        ValueError: flag가 지원되는 타입 중 하나가 아닌 경우.

    Note:
        튜플 케이스는 이 함수가 직접 처리하지 않고
        호출자가 예외 타입 검사를 통해 처리합니다.
    """
    if isinstance(flag, (bool, tuple)):
        content = TOOL_CALL_ERROR_TEMPLATE.format(error=repr(e))
    elif isinstance(flag, str):
        content = flag
    elif callable(flag):
        content = flag(e)
    else:
        raise ValueError(
            f"Got unexpected type of `handle_tool_error`. Expected bool, str "
            f"or callable. Received: {flag}"
        )
    return content


def _infer_handled_types(handler: Callable[..., str]) -> tuple[type[Exception], ...]:
    """커스텀 오류 핸들러 함수가 처리하는 예외 타입을 추론합니다.

    이 함수는 커스텀 오류 핸들러의 타입 어노테이션을 분석하여
    처리하도록 설계된 예외 타입을 결정합니다. 이를 통해 특정 예외만
    핸들러에 의해 포착되고 처리되는 타입 안전 오류 처리가 가능합니다.

    Args:
        handler: 예외를 받아 오류 메시지 문자열을 반환하는 callable입니다.
                첫 번째 매개변수(self/cls가 있는 경우 그 다음)는 처리할
                예외 타입으로 타입 어노테이션되어야 합니다.

    Returns:
        핸들러가 처리할 수 있는 예외 타입의 튜플입니다. 이전 버전과의 호환성을 위해
        특정 타입 정보가 없으면 (Exception,)을 반환합니다.

    Raises:
        ValueError: 핸들러의 어노테이션에 Exception이 아닌 타입이 포함되어 있거나
                   Union 타입에 Exception이 아닌 타입이 포함된 경우.

    Note:
        이 함수는 여러 예외 타입을 다르게 처리해야 하는 핸들러를 위해
        단일 예외 타입과 Union 타입을 모두 지원합니다.
    """
    sig = inspect.signature(handler)
    params = list(sig.parameters.values())
    if params:
        # 메서드인 경우 첫 번째 인수는 일반적으로 'self' 또는 'cls'입니다
        if params[0].name in ["self", "cls"] and len(params) == 2:
            first_param = params[1]
        else:
            first_param = params[0]

        type_hints = get_type_hints(handler)
        if first_param.name in type_hints:
            origin = get_origin(first_param.annotation)
            # typing.Union과 types.UnionType (Python 3.10+ X | Y 구문) 모두 처리합니다
            if origin is Union or origin is types.UnionType:
                args = get_args(first_param.annotation)
                if all(issubclass(arg, Exception) for arg in args):
                    return tuple(args)
                else:
                    raise ValueError(
                        "All types in the error handler error annotation must be "
                        "Exception types. For example, "
                        "`def custom_handler(e: Union[ValueError, TypeError])`. "
                        f"Got '{first_param.annotation}' instead."
                    )

            exception_type = type_hints[first_param.name]
            if Exception in exception_type.__mro__:
                return (exception_type,)
            else:
                raise ValueError(
                    f"Arbitrary types are not supported in the error handler "
                    f"signature. Please annotate the error with either a "
                    f"specific Exception type or a union of Exception types. "
                    "For example, `def custom_handler(e: ValueError)` or "
                    "`def custom_handler(e: Union[ValueError, TypeError])`. "
                    f"Got '{exception_type}' instead."
                )

    # 타입 정보가 없으면 이전 버전과의 호환성을 위해
    # (Exception,)을 반환합니다.
    return (Exception,)


class ToolNode(RunnableCallable):
    """마지막 AIMessage에서 호출된 도구를 실행하는 노드입니다.

    "messages" 상태 키(또는 ToolNode의 'messages_key'를 통해 전달된 커스텀 키)가 있는 StateGraph에서 사용할 수 있습니다.
    여러 도구 호출이 요청되면 병렬로 실행됩니다. 출력은 각 도구 호출마다
    하나씩 ToolMessage 목록이 됩니다.

    도구 호출은 `ToolCall` dict 목록으로 직접 전달할 수도 있습니다.

    Example:
        간단한 도구와 함께 기본 사용법:

        ```python
        from langgraph.prebuilt import ToolNode
        from langchain_core.tools import tool

        @tool
        def calculator(a: int, b: int) -> int:
            \"\"\"두 숫자를 더합니다.\"\"\"
            return a + b

        tool_node = ToolNode([calculator])
        ```

        커스텀 오류 처리:

        ```python
        def handle_math_errors(e: ZeroDivisionError) -> str:
            return "0으로 나눌 수 없습니다!"

        tool_node = ToolNode([calculator], handle_tool_errors=handle_math_errors)
        ```

        직접 도구 호출 실행:

        ```python
        tool_calls = [{"name": "calculator", "args": {"a": 5, "b": 3}, "id": "1", "type": "tool_call"}]
        result = tool_node.invoke(tool_calls)
        ```

    Note:
        ToolNode는 다음 세 가지 형식 중 하나의 입력을 예상합니다:
        1. 메시지 목록을 포함하는 messages 키가 있는 딕셔너리
        2. 메시지 목록을 직접
        3. 도구 호출 딕셔너리 목록

        메시지 형식을 사용할 때 마지막 메시지는 tool_calls가 채워진
        AIMessage여야 합니다. 노드는 자동으로 이러한 도구 호출을
        동시에 추출하고 처리합니다.

        상태 주입 또는 저장소 액세스와 관련된 고급 사용 사례의 경우, 도구에
        InjectedState 또는 InjectedStore를 어노테이션하여 그래프 컨텍스트를
        자동으로 받을 수 있습니다.
    """

    name: str = "ToolNode"

    def __init__(
        self,
        tools: Sequence[BaseTool | Callable],
        *,
        name: str = "tools",
        tags: list[str] | None = None,
        handle_tool_errors: bool
        | str
        | Callable[..., str]
        | tuple[type[Exception], ...] = True,
        messages_key: str = "messages",
    ) -> None:
        """제공된 도구와 설정으로 ToolNode를 초기화합니다.

        Args:
            tools: 이 노드가 호출할 수 있는 도구의 시퀀스입니다. 도구는
                BaseTool 인스턴스 또는 도구로 변환될 일반 함수일 수 있습니다.
            name: 그래프에서 이 노드의 이름 식별자입니다. 디버깅 및
                시각화에 사용됩니다.
            tags: 필터링 및 구성을 위해 노드와 연결할 선택적 메타데이터 태그입니다.
            handle_tool_errors: 도구 실행 중 오류 처리를 위한 설정입니다.
                기본값은 True입니다. 여러 전략을 지원합니다:

                - True: 모든 오류를 포착하고 예외 세부 정보가 포함된 기본
                    오류 템플릿과 함께 ToolMessage를 반환합니다.
                - str: 모든 오류를 포착하고 이 커스텀 오류 메시지 문자열과
                    함께 ToolMessage를 반환합니다.
                - tuple[type[Exception], ...]: 지정된 타입의 예외만 포착하고
                    이에 대한 기본 오류 메시지를 반환합니다.
                - Callable[..., str]: callable의 시그니처와 일치하는 예외를 포착하고
                    예외와 함께 호출한 문자열 결과를 반환합니다.
                - False: 오류 처리를 완전히 비활성화하여 예외가 전파되도록 합니다.
            messages_key: 메시지 목록을 포함하는 상태 딕셔너리의 키입니다.
                이 동일한 키가 출력 ToolMessage에도 사용됩니다.
        """
        super().__init__(self._func, self._afunc, name=name, tags=tags, trace=False)
        self.tools_by_name: dict[str, BaseTool] = {}
        self.tool_to_state_args: dict[str, dict[str, str | None]] = {}
        self.tool_to_store_arg: dict[str, str | None] = {}
        self.handle_tool_errors = handle_tool_errors
        self.messages_key = messages_key
        for tool_ in tools:
            if not isinstance(tool_, BaseTool):
                tool_ = create_tool(tool_)
            self.tools_by_name[tool_.name] = tool_
            self.tool_to_state_args[tool_.name] = _get_state_args(tool_)
            self.tool_to_store_arg[tool_.name] = _get_store_arg(tool_)

    def _func(
        self,
        input: list[AnyMessage] | dict[str, Any] | BaseModel,
        config: RunnableConfig,
        *,
        store: BaseStore | None,
    ) -> Any:
        tool_calls, input_type = self._parse_input(input, store)
        config_list = get_config_list(config, len(tool_calls))
        input_types = [input_type] * len(tool_calls)
        with get_executor_for_config(config) as executor:
            outputs = [
                *executor.map(self._run_one, tool_calls, input_types, config_list)
            ]

        return self._combine_tool_outputs(outputs, input_type)

    async def _afunc(
        self,
        input: list[AnyMessage] | dict[str, Any] | BaseModel,
        config: RunnableConfig,
        *,
        store: BaseStore | None,
    ) -> Any:
        tool_calls, input_type = self._parse_input(input, store)
        outputs = await asyncio.gather(
            *(self._arun_one(call, input_type, config) for call in tool_calls)
        )

        return self._combine_tool_outputs(outputs, input_type)

    def _combine_tool_outputs(
        self,
        outputs: list[ToolMessage],
        input_type: Literal["list", "dict", "tool_calls"],
    ) -> list[Command | list[ToolMessage] | dict[str, list[ToolMessage]]]:
        # preserve existing behavior for non-command tool outputs for backwards
        # compatibility
        if not any(isinstance(output, Command) for output in outputs):
            # TypedDict, pydantic, dataclass, etc. should all be able to load from dict
            return outputs if input_type == "list" else {self.messages_key: outputs}

        # LangGraph will automatically handle list of Command and non-command node
        # updates
        combined_outputs: list[
            Command | list[ToolMessage] | dict[str, list[ToolMessage]]
        ] = []

        # combine all parent commands with goto into a single parent command
        parent_command: Command | None = None
        for output in outputs:
            if isinstance(output, Command):
                if (
                    output.graph is Command.PARENT
                    and isinstance(output.goto, list)
                    and all(isinstance(send, Send) for send in output.goto)
                ):
                    if parent_command:
                        parent_command = replace(
                            parent_command,
                            goto=cast(list[Send], parent_command.goto) + output.goto,
                        )
                    else:
                        parent_command = Command(graph=Command.PARENT, goto=output.goto)
                else:
                    combined_outputs.append(output)
            else:
                combined_outputs.append(
                    [output] if input_type == "list" else {self.messages_key: [output]}
                )

        if parent_command:
            combined_outputs.append(parent_command)
        return combined_outputs

    def _run_one(
        self,
        call: ToolCall,
        input_type: Literal["list", "dict", "tool_calls"],
        config: RunnableConfig,
    ) -> ToolMessage:
        """Run a single tool call synchronously."""
        if invalid_tool_message := self._validate_tool_call(call):
            return invalid_tool_message
        try:
            call_args = {**call, **{"type": "tool_call"}}
            response = self.tools_by_name[call["name"]].invoke(call_args, config)

        # GraphInterrupt is a special exception that will always be raised.
        # It can be triggered in the following scenarios,
        # Where GraphInterrupt(GraphBubbleUp) is raised from an `interrupt` invocation most commonly:
        # (1) a GraphInterrupt is raised inside a tool
        # (2) a GraphInterrupt is raised inside a graph node for a graph called as a tool
        # (3) a GraphInterrupt is raised when a subgraph is interrupted inside a graph called as a tool
        # (2 and 3 can happen in a "supervisor w/ tools" multi-agent architecture)
        except GraphBubbleUp as e:
            raise e
        except Exception as e:
            if isinstance(self.handle_tool_errors, tuple):
                handled_types: tuple = self.handle_tool_errors
            elif callable(self.handle_tool_errors):
                handled_types = _infer_handled_types(self.handle_tool_errors)
            else:
                # default behavior is catching all exceptions
                handled_types = (Exception,)

            # Unhandled
            if not self.handle_tool_errors or not isinstance(e, handled_types):
                raise e
            # Handled
            else:
                content = _handle_tool_error(e, flag=self.handle_tool_errors)
            return ToolMessage(
                content=content,
                name=call["name"],
                tool_call_id=call["id"],
                status="error",
            )

        if isinstance(response, Command):
            return self._validate_tool_command(response, call, input_type)
        elif isinstance(response, ToolMessage):
            response.content = cast(str | list, msg_content_output(response.content))
            return response
        else:
            raise TypeError(
                f"Tool {call['name']} returned unexpected type: {type(response)}"
            )

    async def _arun_one(
        self,
        call: ToolCall,
        input_type: Literal["list", "dict", "tool_calls"],
        config: RunnableConfig,
    ) -> ToolMessage:
        """Run a single tool call asynchronously."""
        if invalid_tool_message := self._validate_tool_call(call):
            return invalid_tool_message

        try:
            call_args = {**call, **{"type": "tool_call"}}
            response = await self.tools_by_name[call["name"]].ainvoke(call_args, config)

        # GraphInterrupt is a special exception that will always be raised.
        # It can be triggered in the following scenarios,
        # Where GraphInterrupt(GraphBubbleUp) is raised from an `interrupt` invocation most commonly:
        # (1) a GraphInterrupt is raised inside a tool
        # (2) a GraphInterrupt is raised inside a graph node for a graph called as a tool
        # (3) a GraphInterrupt is raised when a subgraph is interrupted inside a graph called as a tool
        # (2 and 3 can happen in a "supervisor w/ tools" multi-agent architecture)
        except GraphBubbleUp as e:
            raise e
        except Exception as e:
            if isinstance(self.handle_tool_errors, tuple):
                handled_types: tuple = self.handle_tool_errors
            elif callable(self.handle_tool_errors):
                handled_types = _infer_handled_types(self.handle_tool_errors)
            else:
                # default behavior is catching all exceptions
                handled_types = (Exception,)

            # Unhandled
            if not self.handle_tool_errors or not isinstance(e, handled_types):
                raise e
            # Handled
            else:
                content = _handle_tool_error(e, flag=self.handle_tool_errors)

            return ToolMessage(
                content=content,
                name=call["name"],
                tool_call_id=call["id"],
                status="error",
            )

        if isinstance(response, Command):
            return self._validate_tool_command(response, call, input_type)
        elif isinstance(response, ToolMessage):
            response.content = cast(str | list, msg_content_output(response.content))
            return response
        else:
            raise TypeError(
                f"Tool {call['name']} returned unexpected type: {type(response)}"
            )

    def _parse_input(
        self,
        input: list[AnyMessage] | dict[str, Any] | BaseModel,
        store: BaseStore | None,
    ) -> tuple[list[ToolCall], Literal["list", "dict", "tool_calls"]]:
        input_type: Literal["list", "dict", "tool_calls"]
        if isinstance(input, list):
            if isinstance(input[-1], dict) and input[-1].get("type") == "tool_call":
                input_type = "tool_calls"
                tool_calls = cast(list[ToolCall], input)
                return tool_calls, input_type
            else:
                input_type = "list"
                messages = input
        elif isinstance(input, dict) and (messages := input.get(self.messages_key, [])):
            input_type = "dict"
        elif messages := getattr(input, self.messages_key, []):
            # Assume dataclass-like state that can coerce from dict
            input_type = "dict"
        else:
            raise ValueError("No message found in input")

        try:
            latest_ai_message = next(
                m for m in reversed(messages) if isinstance(m, AIMessage)
            )
        except StopIteration:
            raise ValueError("No AIMessage found in input")

        tool_calls = [
            self.inject_tool_args(call, input, store)
            for call in latest_ai_message.tool_calls
        ]
        return tool_calls, input_type

    def _validate_tool_call(self, call: ToolCall) -> ToolMessage | None:
        if (requested_tool := call["name"]) not in self.tools_by_name:
            content = INVALID_TOOL_NAME_ERROR_TEMPLATE.format(
                requested_tool=requested_tool,
                available_tools=", ".join(self.tools_by_name.keys()),
            )
            return ToolMessage(
                content, name=requested_tool, tool_call_id=call["id"], status="error"
            )
        else:
            return None

    def _inject_state(
        self,
        tool_call: ToolCall,
        input: list[AnyMessage] | dict[str, Any] | BaseModel,
    ) -> ToolCall:
        state_args = self.tool_to_state_args[tool_call["name"]]
        if state_args and isinstance(input, list):
            required_fields = list(state_args.values())
            if (
                len(required_fields) == 1
                and required_fields[0] == self.messages_key
                or required_fields[0] is None
            ):
                input = {self.messages_key: input}
            else:
                err_msg = (
                    f"Invalid input to ToolNode. Tool {tool_call['name']} requires "
                    f"graph state dict as input."
                )
                if any(state_field for state_field in state_args.values()):
                    required_fields_str = ", ".join(f for f in required_fields if f)
                    err_msg += f" State should contain fields {required_fields_str}."
                raise ValueError(err_msg)

        if isinstance(input, dict):
            tool_state_args = {
                tool_arg: input[state_field] if state_field else input
                for tool_arg, state_field in state_args.items()
            }
        else:
            tool_state_args = {
                tool_arg: getattr(input, state_field) if state_field else input
                for tool_arg, state_field in state_args.items()
            }

        tool_call["args"] = {
            **tool_call["args"],
            **tool_state_args,
        }
        return tool_call

    def _inject_store(self, tool_call: ToolCall, store: BaseStore | None) -> ToolCall:
        store_arg = self.tool_to_store_arg[tool_call["name"]]
        if not store_arg:
            return tool_call

        if store is None:
            raise ValueError(
                "Cannot inject store into tools with InjectedStore annotations - "
                "please compile your graph with a store."
            )

        tool_call["args"] = {
            **tool_call["args"],
            store_arg: store,
        }
        return tool_call

    def inject_tool_args(
        self,
        tool_call: ToolCall,
        input: list[AnyMessage] | dict[str, Any] | BaseModel,
        store: BaseStore | None,
    ) -> ToolCall:
        """Inject graph state and store into tool call arguments.

        This method enables tools to access graph context that should not be controlled
        by the model. Tools can declare dependencies on graph state or persistent storage
        using InjectedState and InjectedStore annotations. This method automatically
        identifies these dependencies and injects the appropriate values.

        The injection process preserves the original tool call structure while adding
        the necessary context arguments. This allows tools to be both model-callable
        and context-aware without exposing internal state management to the model.

        Args:
            tool_call: The tool call dictionary to augment with injected arguments.
                Must contain 'name', 'args', 'id', and 'type' fields.
            input: The current graph state to inject into tools requiring state access.
                Can be a message list, state dictionary, or BaseModel instance.
            store: The persistent store instance to inject into tools requiring storage.
                Will be None if no store is configured for the graph.

        Returns:
            A new ToolCall dictionary with the same structure as the input but with
            additional arguments injected based on the tool's annotation requirements.

        Raises:
            ValueError: If a tool requires store injection but no store is provided,
                       or if state injection requirements cannot be satisfied.

        Note:
            This method is automatically called during tool execution but can also
            be used manually when working with the Send API or custom routing logic.
            The injection is performed on a copy of the tool call to avoid mutating
            the original.
        """
        if tool_call["name"] not in self.tools_by_name:
            return tool_call

        tool_call_copy: ToolCall = copy(tool_call)
        tool_call_with_state = self._inject_state(tool_call_copy, input)
        tool_call_with_store = self._inject_store(tool_call_with_state, store)
        return tool_call_with_store

    def _validate_tool_command(
        self,
        command: Command,
        call: ToolCall,
        input_type: Literal["list", "dict", "tool_calls"],
    ) -> Command:
        if isinstance(command.update, dict):
            # input type is dict when ToolNode is invoked with a dict input (e.g. {"messages": [AIMessage(..., tool_calls=[...])]})
            if input_type not in ("dict", "tool_calls"):
                raise ValueError(
                    f"Tools can provide a dict in Command.update only when using dict with '{self.messages_key}' key as ToolNode input, "
                    f"got: {command.update} for tool '{call['name']}'"
                )

            updated_command = deepcopy(command)
            state_update = cast(dict[str, Any], updated_command.update) or {}
            messages_update = state_update.get(self.messages_key, [])
        elif isinstance(command.update, list):
            # input type is list when ToolNode is invoked with a list input (e.g. [AIMessage(..., tool_calls=[...])])
            if input_type != "list":
                raise ValueError(
                    f"Tools can provide a list of messages in Command.update only when using list of messages as ToolNode input, "
                    f"got: {command.update} for tool '{call['name']}'"
                )

            updated_command = deepcopy(command)
            messages_update = updated_command.update
        else:
            return command

        # convert to message objects if updates are in a dict format
        messages_update = convert_to_messages(messages_update)

        # no validation needed if all messages are being removed
        if messages_update == [RemoveMessage(id=REMOVE_ALL_MESSAGES)]:
            return updated_command

        has_matching_tool_message = False
        for message in messages_update:
            if not isinstance(message, ToolMessage):
                continue

            if message.tool_call_id == call["id"]:
                message.name = call["name"]
                has_matching_tool_message = True

        # validate that we always have a ToolMessage matching the tool call in
        # Command.update if command is sent to the CURRENT graph
        if updated_command.graph is None and not has_matching_tool_message:
            example_update = (
                '`Command(update={"messages": [ToolMessage("Success", tool_call_id=tool_call_id), ...]}, ...)`'
                if input_type == "dict"
                else '`Command(update=[ToolMessage("Success", tool_call_id=tool_call_id), ...], ...)`'
            )
            raise ValueError(
                f"Expected to have a matching ToolMessage in Command.update for tool '{call['name']}', got: {messages_update}. "
                "Every tool call (LLM requesting to call a tool) in the message history MUST have a corresponding ToolMessage. "
                f"You can fix it by modifying the tool to return {example_update}."
            )
        return updated_command


def tools_condition(
    state: list[AnyMessage] | dict[str, Any] | BaseModel,
    messages_key: str = "messages",
) -> Literal["tools", "__end__"]:
    """Conditional routing function for tool-calling workflows.

    This utility function implements the standard conditional logic for ReAct-style
    agents: if the last AI message contains tool calls, route to the tool execution
    node; otherwise, end the workflow. This pattern is fundamental to most tool-calling
    agent architectures.

    The function handles multiple state formats commonly used in LangGraph applications,
    making it flexible for different graph designs while maintaining consistent behavior.

    Args:
        state: The current graph state to examine for tool calls. Supported formats:
            - Dictionary containing a messages key (for StateGraph)
            - BaseModel instance with a messages attribute
        messages_key: The key or attribute name containing the message list in the state.
            This allows customization for graphs using different state schemas.
            Defaults to "messages".

    Returns:
        Either "tools" if tool calls are present in the last AI message, or "__end__"
        to terminate the workflow. These are the standard routing destinations for
        tool-calling conditional edges.

    Raises:
        ValueError: If no messages can be found in the provided state format.

    Example:
        Basic usage in a ReAct agent:

        ```python
        from langgraph.graph import StateGraph
        from langgraph.prebuilt import ToolNode, tools_condition
        from typing_extensions import TypedDict

        class State(TypedDict):
            messages: list

        graph = StateGraph(State)
        graph.add_node("llm", call_model)
        graph.add_node("tools", ToolNode([my_tool]))
        graph.add_conditional_edges(
            "llm",
            tools_condition,  # Routes to "tools" or "__end__"
            {"tools": "tools", "__end__": "__end__"}
        )
        ```

        Custom messages key:

        ```python
        def custom_condition(state):
            return tools_condition(state, messages_key="chat_history")
        ```

    Note:
        This function is designed to work seamlessly with ToolNode and standard
        LangGraph patterns. It expects the last message to be an AIMessage when
        tool calls are present, which is the standard output format for tool-calling
        language models.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"


class InjectedState(InjectedToolArg):
    """Annotation for injecting graph state into tool arguments.

    This annotation enables tools to access graph state without exposing state
    management details to the language model. Tools annotated with InjectedState
    receive state data automatically during execution while remaining invisible
    to the model's tool-calling interface.

    Example:
        ```python
        from typing import List
        from typing_extensions import Annotated, TypedDict

        from langchain_core.messages import BaseMessage, AIMessage
        from langchain_core.tools import tool

        from langgraph.prebuilt import InjectedState, ToolNode


        class AgentState(TypedDict):
            messages: List[BaseMessage]
            foo: str

        @tool
        def state_tool(x: int, state: Annotated[dict, InjectedState]) -> str:
            '''Do something with state.'''
            if len(state["messages"]) > 2:
                return state["foo"] + str(x)
            else:
                return "not enough messages"

        @tool
        def foo_tool(x: int, foo: Annotated[str, InjectedState("foo")]) -> str:
            '''Do something else with state.'''
            return foo + str(x + 1)

        node = ToolNode([state_tool, foo_tool])

        tool_call1 = {"name": "state_tool", "args": {"x": 1}, "id": "1", "type": "tool_call"}
        tool_call2 = {"name": "foo_tool", "args": {"x": 1}, "id": "2", "type": "tool_call"}
        state = {
            "messages": [AIMessage("", tool_calls=[tool_call1, tool_call2])],
            "foo": "bar",
        }
        node.invoke(state)
        ```

        ```pycon
        [
            ToolMessage(content='not enough messages', name='state_tool', tool_call_id='1'),
            ToolMessage(content='bar2', name='foo_tool', tool_call_id='2')
        ]
        ```

    Note:
        - InjectedState arguments are automatically excluded from tool schemas
          presented to language models
        - ToolNode handles the injection process during execution
        - Tools can mix regular arguments (controlled by the model) with injected
          arguments (controlled by the system)
        - State injection occurs after the model generates tool calls but before
          tool execution
    """  # noqa: E501

    def __init__(self, field: str | None = None) -> None:
        """Initialize InjectedState annotation.

        Args:
            field: Optional key to extract from the state dictionary. If `None`, the entire
                state is injected. If specified, only that field's value is injected.
                This allows tools to request specific state components rather than
                processing the full state structure.
        """
        self.field = field


class InjectedStore(InjectedToolArg):
    """Annotation for injecting persistent store into tool arguments.

    This annotation enables tools to access LangGraph's persistent storage system
    without exposing storage details to the language model. Tools annotated with
    InjectedStore receive the store instance automatically during execution while
    remaining invisible to the model's tool-calling interface.

    The store provides persistent, cross-session data storage that tools can use
    for maintaining context, user preferences, or any other data that needs to
    persist beyond individual workflow executions.

    !!! Warning
        `InjectedStore` annotation requires `langchain-core >= 0.3.8`

    Example:
        ```python
        from typing_extensions import Annotated
        from langchain_core.tools import tool
        from langgraph.store.memory import InMemoryStore
        from langgraph.prebuilt import InjectedStore, ToolNode

        @tool
        def save_preference(
            key: str,
            value: str,
            store: Annotated[Any, InjectedStore()]
        ) -> str:
            \"\"\"Save user preference to persistent storage.\"\"\"
            store.put(("preferences",), key, value)
            return f"Saved {key} = {value}"

        @tool
        def get_preference(
            key: str,
            store: Annotated[Any, InjectedStore()]
        ) -> str:
            \"\"\"Retrieve user preference from persistent storage.\"\"\"
            result = store.get(("preferences",), key)
            return result.value if result else "Not found"
        ```

        Usage with ToolNode and graph compilation:

        ```python
        from langgraph.graph import StateGraph
        from langgraph.store.memory import InMemoryStore

        store = InMemoryStore()
        tool_node = ToolNode([save_preference, get_preference])

        graph = StateGraph(State)
        graph.add_node("tools", tool_node)
        compiled_graph = graph.compile(store=store)  # Store is injected automatically
        ```

        Cross-session persistence:

        ```python
        # First session
        result1 = graph.invoke({"messages": [HumanMessage("Save my favorite color as blue")]})

        # Later session - data persists
        result2 = graph.invoke({"messages": [HumanMessage("What's my favorite color?")]})
        ```

    Note:
        - InjectedStore arguments are automatically excluded from tool schemas
          presented to language models
        - The store instance is automatically injected by ToolNode during execution
        - Tools can access namespaced storage using the store's get/put methods
        - Store injection requires the graph to be compiled with a store instance
        - Multiple tools can share the same store instance for data consistency
    """  # noqa: E501


def _is_injection(
    type_arg: Any, injection_type: type[InjectedState] | type[InjectedStore]
) -> bool:
    """Check if a type argument represents an injection annotation.

    This utility function determines whether a type annotation indicates that
    an argument should be injected with state or store data. It handles both
    direct annotations and nested annotations within Union or Annotated types.

    Args:
        type_arg: The type argument to check for injection annotations.
        injection_type: The injection type to look for (InjectedState or InjectedStore).

    Returns:
        True if the type argument contains the specified injection annotation.
    """
    if isinstance(type_arg, injection_type) or (
        isinstance(type_arg, type) and issubclass(type_arg, injection_type)
    ):
        return True
    origin_ = get_origin(type_arg)
    if origin_ is Union or origin_ is Annotated:
        return any(_is_injection(ta, injection_type) for ta in get_args(type_arg))
    return False


def _get_state_args(tool: BaseTool) -> dict[str, str | None]:
    """Extract state injection mappings from tool annotations.

    This function analyzes a tool's input schema to identify arguments that should
    be injected with graph state. It processes InjectedState annotations to build
    a mapping of tool argument names to state field names.

    Args:
        tool: The tool to analyze for state injection requirements.

    Returns:
        A dictionary mapping tool argument names to state field names. If a field
        name is None, the entire state should be injected for that argument.
    """
    full_schema = tool.get_input_schema()
    tool_args_to_state_fields: dict = {}

    for name, type_ in get_all_basemodel_annotations(full_schema).items():
        injections = [
            type_arg
            for type_arg in get_args(type_)
            if _is_injection(type_arg, InjectedState)
        ]
        if len(injections) > 1:
            raise ValueError(
                "A tool argument should not be annotated with InjectedState more than "
                f"once. Received arg {name} with annotations {injections}."
            )
        elif len(injections) == 1:
            injection = injections[0]
            if isinstance(injection, InjectedState) and injection.field:
                tool_args_to_state_fields[name] = injection.field
            else:
                tool_args_to_state_fields[name] = None
        else:
            pass
    return tool_args_to_state_fields


def _get_store_arg(tool: BaseTool) -> str | None:
    """Extract store injection argument from tool annotations.

    This function analyzes a tool's input schema to identify the argument that
    should be injected with the graph store. Only one store argument is supported
    per tool.

    Args:
        tool: The tool to analyze for store injection requirements.

    Returns:
        The name of the argument that should receive the store injection, or None
        if no store injection is required.

    Raises:
        ValueError: If a tool argument has multiple InjectedStore annotations.
    """
    full_schema = tool.get_input_schema()
    for name, type_ in get_all_basemodel_annotations(full_schema).items():
        injections = [
            type_arg
            for type_arg in get_args(type_)
            if _is_injection(type_arg, InjectedStore)
        ]
        if len(injections) > 1:
            raise ValueError(
                "A tool argument should not be annotated with InjectedStore more than "
                f"once. Received arg {name} with annotations {injections}."
            )
        elif len(injections) == 1:
            return name
        else:
            pass

    return None
