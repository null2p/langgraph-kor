import inspect
import warnings
from collections.abc import Awaitable, Callable, Sequence
from typing import (
    Annotated,
    Any,
    Literal,
    TypeVar,
    cast,
    get_type_hints,
)

from langchain_core.language_models import (
    BaseChatModel,
    LanguageModelInput,
    LanguageModelLike,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import (
    Runnable,
    RunnableBinding,
    RunnableConfig,
    RunnableSequence,
)
from langchain_core.tools import BaseTool
from langgraph._internal._runnable import RunnableCallable, RunnableLike
from langgraph._internal._typing import MISSING
from langgraph.errors import ErrorCode, create_error_message
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.managed import RemainingSteps
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Send
from langgraph.typing import ContextT
from langgraph.warnings import LangGraphDeprecatedSinceV10
from pydantic import BaseModel
from typing_extensions import NotRequired, TypedDict, deprecated

from langgraph.prebuilt.tool_node import ToolNode

StructuredResponse = dict | BaseModel
StructuredResponseSchema = dict | type[BaseModel]


@deprecated(
    "AgentState가 `langchain.agents`로 이동되었습니다. import를 `from langchain.agents import AgentState`로 업데이트하세요.",
    category=LangGraphDeprecatedSinceV10,
)
class AgentState(TypedDict):
    """에이전트의 상태입니다."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    remaining_steps: NotRequired[RemainingSteps]


@deprecated(
    "AgentStatePydantic가 `langchain.agents`로 이동되었습니다. import를 `from langchain.agents import AgentStatePydantic`로 업데이트하세요.",
    category=LangGraphDeprecatedSinceV10,
)
class AgentStatePydantic(BaseModel):
    """에이전트의 상태입니다."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    remaining_steps: RemainingSteps = 25


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=LangGraphDeprecatedSinceV10,
        message="AgentState has been moved to langchain.agents.*",
    )

    @deprecated(
        "AgentStateWithStructuredResponse가 `langchain.agents`로 이동되었습니다. import를 `from langchain.agents import AgentStateWithStructuredResponse`로 업데이트하세요.",
        category=LangGraphDeprecatedSinceV10,
    )
    class AgentStateWithStructuredResponse(AgentState):
        """구조화된 응답을 포함하는 에이전트의 상태입니다."""

        structured_response: StructuredResponse


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=LangGraphDeprecatedSinceV10,
        message="AgentStatePydantic has been moved to langchain.agents.*",
    )

    @deprecated(
        "AgentStateWithStructuredResponsePydantic가 `langchain.agents`로 이동되었습니다. import를 `from langchain.agents import AgentStateWithStructuredResponsePydantic`로 업데이트하세요.",
        category=LangGraphDeprecatedSinceV10,
    )
    class AgentStateWithStructuredResponsePydantic(AgentStatePydantic):
        """구조화된 응답을 포함하는 에이전트의 상태입니다."""

        structured_response: StructuredResponse


StateSchema = TypeVar("StateSchema", bound=AgentState | AgentStatePydantic)
StateSchemaType = type[StateSchema]

PROMPT_RUNNABLE_NAME = "Prompt"

Prompt = (
    SystemMessage
    | str
    | Callable[[StateSchema], LanguageModelInput]
    | Runnable[StateSchema, LanguageModelInput]
)


def _get_state_value(state: StateSchema, key: str, default: Any = None) -> Any:
    return (
        state.get(key, default)
        if isinstance(state, dict)
        else getattr(state, key, default)
    )


def _get_prompt_runnable(prompt: Prompt | None) -> Runnable:
    prompt_runnable: Runnable
    if prompt is None:
        prompt_runnable = RunnableCallable(
            lambda state: _get_state_value(state, "messages"), name=PROMPT_RUNNABLE_NAME
        )
    elif isinstance(prompt, str):
        _system_message: BaseMessage = SystemMessage(content=prompt)
        prompt_runnable = RunnableCallable(
            lambda state: [_system_message] + _get_state_value(state, "messages"),
            name=PROMPT_RUNNABLE_NAME,
        )
    elif isinstance(prompt, SystemMessage):
        prompt_runnable = RunnableCallable(
            lambda state: [prompt] + _get_state_value(state, "messages"),
            name=PROMPT_RUNNABLE_NAME,
        )
    elif inspect.iscoroutinefunction(prompt):
        prompt_runnable = RunnableCallable(
            None,
            prompt,
            name=PROMPT_RUNNABLE_NAME,
        )
    elif callable(prompt):
        prompt_runnable = RunnableCallable(
            prompt,
            name=PROMPT_RUNNABLE_NAME,
        )
    elif isinstance(prompt, Runnable):
        prompt_runnable = prompt
    else:
        raise ValueError(f"Got unexpected type for `prompt`: {type(prompt)}")

    return prompt_runnable


def _should_bind_tools(
    model: LanguageModelLike, tools: Sequence[BaseTool], num_builtin: int = 0
) -> bool:
    if isinstance(model, RunnableSequence):
        model = next(
            (
                step
                for step in model.steps
                if isinstance(step, (RunnableBinding, BaseChatModel))
            ),
            model,
        )

    if not isinstance(model, RunnableBinding):
        return True

    if "tools" not in model.kwargs:
        return True

    bound_tools = model.kwargs["tools"]
    if len(tools) != len(bound_tools) - num_builtin:
        raise ValueError(
            "Number of tools in the model.bind_tools() and tools passed to create_react_agent must match"
            f" Got {len(tools)} tools, expected {len(bound_tools) - num_builtin}"
        )

    tool_names = set(tool.name for tool in tools)
    bound_tool_names = set()
    for bound_tool in bound_tools:
        # OpenAI 스타일 도구
        if bound_tool.get("type") == "function":
            bound_tool_name = bound_tool["function"]["name"]
        # Anthropic 스타일 도구
        elif bound_tool.get("name"):
            bound_tool_name = bound_tool["name"]
        else:
            # 알 수 없는 도구 타입이므로 무시합니다
            continue

        bound_tool_names.add(bound_tool_name)

    if missing_tools := tool_names - bound_tool_names:
        raise ValueError(f"Missing tools '{missing_tools}' in the model.bind_tools()")

    return False


def _get_model(model: LanguageModelLike) -> BaseChatModel:
    """RunnableBinding에서 기본 모델을 가져오거나 모델 자체를 반환합니다."""
    if isinstance(model, RunnableSequence):
        model = next(
            (
                step
                for step in model.steps
                if isinstance(step, (RunnableBinding, BaseChatModel))
            ),
            model,
        )

    if isinstance(model, RunnableBinding):
        model = model.bound

    if not isinstance(model, BaseChatModel):
        raise TypeError(
            f"Expected `model` to be a ChatModel or RunnableBinding (e.g. model.bind_tools(...)), got {type(model)}"
        )

    return model


def _validate_chat_history(
    messages: Sequence[BaseMessage],
) -> None:
    """AIMessage의 모든 도구 호출에 해당하는 ToolMessage가 있는지 검증합니다."""
    all_tool_calls = [
        tool_call
        for message in messages
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
    ]
    tool_call_ids_with_results = {
        message.tool_call_id for message in messages if isinstance(message, ToolMessage)
    }
    tool_calls_without_results = [
        tool_call
        for tool_call in all_tool_calls
        if tool_call["id"] not in tool_call_ids_with_results
    ]
    if not tool_calls_without_results:
        return

    error_message = create_error_message(
        message="해당하는 ToolMessage가 없는 tool_calls를 가진 AIMessage를 발견했습니다. "
        f"해당 도구 호출의 처음 몇 개는 다음과 같습니다: {tool_calls_without_results[:3]}.\n\n"
        "메시지 기록의 모든 도구 호출(LLM이 도구를 호출하도록 요청)은 해당하는 ToolMessage "
        "(LLM에 반환할 도구 실행 결과)를 가져야 합니다 - 이는 대부분의 LLM 공급자에서 요구됩니다.",
        error_code=ErrorCode.INVALID_CHAT_HISTORY,
    )
    raise ValueError(error_message)


@deprecated(
    "create_react_agent가 `langchain.agents`로 이동되었습니다. import를 `from langchain.agents import create_agent`로 업데이트하세요.",
    category=LangGraphDeprecatedSinceV10,
)
def create_react_agent(
    model: str
    | LanguageModelLike
    | Callable[[StateSchema, Runtime[ContextT]], BaseChatModel]
    | Callable[[StateSchema, Runtime[ContextT]], Awaitable[BaseChatModel]]
    | Callable[
        [StateSchema, Runtime[ContextT]], Runnable[LanguageModelInput, BaseMessage]
    ]
    | Callable[
        [StateSchema, Runtime[ContextT]],
        Awaitable[Runnable[LanguageModelInput, BaseMessage]],
    ],
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | ToolNode,
    *,
    prompt: Prompt | None = None,
    response_format: StructuredResponseSchema
    | tuple[str, StructuredResponseSchema]
    | None = None,
    pre_model_hook: RunnableLike | None = None,
    post_model_hook: RunnableLike | None = None,
    state_schema: StateSchemaType | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    version: Literal["v1", "v2"] = "v2",
    name: str | None = None,
    **deprecated_kwargs: Any,
) -> CompiledStateGraph:
    """중지 조건이 충족될 때까지 루프에서 도구를 호출하는 에이전트 그래프를 생성합니다.

    `create_react_agent` 사용에 대한 자세한 내용은 [Agents](https://langchain-ai.github.io/langgraph/agents/overview/) 문서를 참조하세요.

    Args:
        model: 에이전트를 위한 언어 모델입니다. 정적 및 동적
            모델 선택을 지원합니다.

            - **정적 모델**: 채팅 모델 인스턴스 (예: `ChatOpenAI()`) 또는
              문자열 식별자 (예: `"openai:gpt-4"`)
            - **동적 모델**: 런타임 컨텍스트에 따라 다른 모델을 반환하는
              `(state, runtime) -> BaseChatModel` 시그니처를 가진 callable입니다.
              모델에 `.bind_tools()` 또는 기타 설정을 통해 도구가 바인딩된 경우,
              반환 타입은 Runnable[LanguageModelInput, BaseMessage]이어야 합니다.
              코루틴도 지원되어 비동기 모델 선택이 가능합니다.

            동적 함수는 그래프 상태와 런타임을 받아, 컨텍스트에 따른
            모델 선택을 가능하게 합니다. `BaseChatModel` 인스턴스를 반환해야 합니다.
            도구 호출의 경우 `.bind_tools()`를 사용하여 도구를 바인딩합니다.
            바인딩된 도구는 `tools` 매개변수의 하위 집합이어야 합니다.

            동적 모델 예제:
            ```python
            from dataclasses import dataclass

            @dataclass
            class ModelContext:
                model_name: str = "gpt-3.5-turbo"

            # 모델을 전역적으로 인스턴스화
            gpt4_model = ChatOpenAI(model="gpt-4")
            gpt35_model = ChatOpenAI(model="gpt-3.5-turbo")

            def select_model(state: AgentState, runtime: Runtime[ModelContext]) -> ChatOpenAI:
                model_name = runtime.context.model_name
                model = gpt4_model if model_name == "gpt-4" else gpt35_model
                return model.bind_tools(tools)
            ```

            !!! note "동적 모델 요구사항"

                반환된 모델이 `.bind_tools()`를 통해 적절한 도구가 바인딩되어 있고
                필요한 기능을 지원하는지 확인하세요. 바인딩된 도구는
                `tools` 매개변수에 지정된 도구의 하위 집합이어야 합니다.

        tools: 도구 목록 또는 `ToolNode` 인스턴스입니다.
            빈 목록이 제공되면, 에이전트는 도구 호출 없이 단일 LLM 노드로 구성됩니다.
        prompt: LLM을 위한 선택적 프롬프트입니다. 여러 형태로 제공될 수 있습니다:

            - str: SystemMessage로 변환되어 state["messages"]의 메시지 목록 시작 부분에 추가됩니다.
            - SystemMessage: state["messages"]의 메시지 목록 시작 부분에 추가됩니다.
            - Callable: 이 함수는 전체 그래프 상태를 받아 출력이 언어 모델로 전달됩니다.
            - Runnable: 이 runnable은 전체 그래프 상태를 받아 출력이 언어 모델로 전달됩니다.

        response_format: 최종 에이전트 출력을 위한 선택적 스키마입니다.

            제공되면, 출력이 주어진 스키마에 맞게 포맷되어 'structured_response' 상태 키에 반환됩니다.
            제공되지 않으면, `structured_response`는 출력 상태에 존재하지 않습니다.
            다음과 같이 전달될 수 있습니다:

                - OpenAI function/tool 스키마,
                - JSON 스키마,
                - TypedDict 클래스,
                - 또는 Pydantic 클래스.
                - 튜플 (prompt, schema), 여기서 schema는 위의 것 중 하나입니다.
                    프롬프트는 구조화된 응답을 생성하는 데 사용되는 모델과 함께 사용됩니다.

            !!! Important
                `response_format`은 모델이 `.with_structured_output`을 지원해야 합니다

            !!! Note
                그래프는 에이전트 루프가 완료된 후 구조화된 응답을 생성하기 위해 LLM에 별도의 호출을 수행합니다.
                이것이 구조화된 응답을 얻는 유일한 전략은 아닙니다. [이 가이드](https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/)에서 더 많은 옵션을 확인하세요.

        pre_model_hook: `agent` 노드(즉, LLM을 호출하는 노드) 앞에 추가할 선택적 노드입니다.
            긴 메시지 기록 관리(예: 메시지 트리밍, 요약 등)에 유용합니다.
            Pre-model hook은 현재 그래프 상태를 받아 다음 형태의 상태 업데이트를 반환하는 callable 또는 runnable이어야 합니다
                ```python
                # `messages` 또는 `llm_input_messages` 중 적어도 하나는 반드시 제공되어야 합니다
                {
                    # 제공되면, 상태의 `messages`를 업데이트합니다
                    "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), ...],
                    # 제공되면, LLM의 입력으로 사용되며,
                    # 상태의 `messages`를 업데이트하지 않습니다
                    "llm_input_messages": [...],
                    # 전파되어야 하는 기타 상태 키
                    ...
                }
                ```

            !!! Important
                `messages` 또는 `llm_input_messages` 중 적어도 하나는 반드시 제공되어야 하며 `agent` 노드의 입력으로 사용됩니다.
                나머지 키는 그래프 상태에 추가됩니다.

            !!! Warning
                pre-model hook에서 `messages`를 반환하는 경우, 다음과 같이 `messages` 키를 덮어써야 합니다:

                ```python
                {
                    "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *new_messages]
                    ...
                }
                ```
        post_model_hook: `agent` 노드(즉, LLM을 호출하는 노드) 뒤에 추가할 선택적 노드입니다.
            human-in-the-loop, 가드레일, 검증 또는 기타 후처리를 구현하는 데 유용합니다.
            Post-model hook은 현재 그래프 상태를 받아 상태 업데이트를 반환하는 callable 또는 runnable이어야 합니다.

            !!! Note
                `version="v2"`에서만 사용 가능합니다.
        state_schema: 그래프 상태를 정의하는 선택적 상태 스키마입니다.
            `messages`와 `remaining_steps` 키를 가져야 합니다.
            기본값은 이 두 키를 정의하는 `AgentState`입니다.
            !!! Note
                `remaining_steps`는 react 에이전트가 수행할 수 있는 단계 수를 제한하는 데 사용됩니다.
                대략 `recursion_limit` - `total_steps_taken`으로 계산됩니다.
                `remaining_steps`가 2보다 작고 응답에 도구 호출이 있는 경우,
                react 에이전트는 "Sorry, need more steps to process this request."
                내용의 최종 AI Message를 반환합니다.
                이 경우 `GraphRecusionError`는 발생하지 않습니다.

        context_schema: 런타임 컨텍스트를 위한 선택적 스키마입니다.
        checkpointer: 선택적 체크포인트 저장 객체입니다. 단일 스레드(예: 단일 대화)에 대해
            그래프의 상태를 유지하는 데(예: 채팅 메모리로) 사용됩니다.
        store: 선택적 저장소 객체입니다. 여러 스레드(예: 여러 대화/사용자)에 걸쳐
            데이터를 유지하는 데 사용됩니다.
        interrupt_before: 중단할 노드 이름의 선택적 목록입니다.
            다음 중 하나여야 합니다: "agent", "tools".
            작업을 수행하기 전에 사용자 확인 또는 기타 중단을 추가하려는 경우 유용합니다.
        interrupt_after: 다음에 중단할 노드 이름의 선택적 목록입니다.
            다음 중 하나여야 합니다: "agent", "tools".
            직접 반환하거나 출력에 대한 추가 처리를 실행하려는 경우 유용합니다.
        debug: 디버그 모드를 활성화할지 여부를 나타내는 플래그입니다.
        version: 생성할 그래프의 버전을 결정합니다.
            다음 중 하나일 수 있습니다:

            - `"v1"`: 도구 노드가 단일 메시지를 처리합니다. 메시지의 모든 도구
                호출은 도구 노드 내에서 병렬로 실행됩니다.
            - `"v2"`: 도구 노드가 도구 호출을 처리합니다.
                도구 호출은 [Send](https://langchain-ai.github.io/langgraph/concepts/low_level/#send)
                API를 사용하여 도구 노드의 여러 인스턴스에 분산됩니다.
        name: CompiledStateGraph의 선택적 이름입니다.
            이 이름은 ReAct 에이전트 그래프를 서브그래프 노드로 다른 그래프에 추가할 때 자동으로 사용됩니다 -
            다중 에이전트 시스템 구축에 특히 유용합니다.

    !!! warning "`config_schema` 더 이상 사용되지 않음"
        `config_schema` 매개변수는 v0.6.0에서 더 이상 사용되지 않으며 v2.0.0에서 지원이 제거될 예정입니다.
        런 범위 컨텍스트의 스키마를 지정하려면 대신 `context_schema`를 사용하세요.


    Returns:
        채팅 상호 작용에 사용할 수 있는 컴파일된 LangChain runnable입니다.

    "agent" 노드는 메시지 목록(프롬프트 적용 후)으로 언어 모델을 호출합니다.
    결과 AIMessage에 `tool_calls`가 포함되어 있으면, 그래프는 ["tools"][langgraph.prebuilt.tool_node.ToolNode]를 호출합니다.
    "tools" 노드는 도구를 실행하고(`tool_call`당 1개의 도구) 응답을 `ToolMessage` 객체로
    메시지 목록에 추가합니다. 그런 다음 agent 노드가 언어 모델을 다시 호출합니다.
    응답에 더 이상 `tool_calls`가 없을 때까지 프로세스가 반복됩니다.
    그런 다음 에이전트는 "messages" 키를 포함하는 딕셔너리로 전체 메시지 목록을 반환합니다.

    ``` mermaid
        sequenceDiagram
            participant U as User
            participant A as LLM
            participant T as Tools
            U->>A: Initial input
            Note over A: Prompt + LLM
            loop while tool_calls present
                A->>T: Execute tools
                T-->>A: ToolMessage for each tool_calls
            end
            A->>U: Return final state
    ```

    Example:
        ```python
        from langgraph.prebuilt import create_react_agent

        def check_weather(location: str) -> str:
            '''지정된 위치의 날씨 예보를 반환합니다.'''
            return f"It's always sunny in {location}"

        graph = create_react_agent(
            "anthropic:claude-3-7-sonnet-latest",
            tools=[check_weather],
            prompt="You are a helpful assistant",
        )
        inputs = {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
        for chunk in graph.stream(inputs, stream_mode="updates"):
            print(chunk)
        ```
    """
    if (
        config_schema := deprecated_kwargs.pop("config_schema", MISSING)
    ) is not MISSING:
        warnings.warn(
            "`config_schema` is deprecated and will be removed. Please use `context_schema` instead.",
            category=LangGraphDeprecatedSinceV10,
        )

        if context_schema is None:
            context_schema = config_schema

    if len(deprecated_kwargs) > 0:
        raise TypeError(
            f"create_react_agent() got unexpected keyword arguments: {deprecated_kwargs}"
        )

    if version not in ("v1", "v2"):
        raise ValueError(
            f"Invalid version {version}. Supported versions are 'v1' and 'v2'."
        )

    if state_schema is not None:
        required_keys = {"messages", "remaining_steps"}
        if response_format is not None:
            required_keys.add("structured_response")

        schema_keys = set(get_type_hints(state_schema))
        if missing_keys := required_keys - set(schema_keys):
            raise ValueError(f"Missing required key(s) {missing_keys} in state_schema")

    if state_schema is None:
        state_schema = (
            AgentStateWithStructuredResponse
            if response_format is not None
            else AgentState
        )

    llm_builtin_tools: list[dict] = []
    if isinstance(tools, ToolNode):
        tool_classes = list(tools.tools_by_name.values())
        tool_node = tools
    else:
        llm_builtin_tools = [t for t in tools if isinstance(t, dict)]
        tool_node = ToolNode([t for t in tools if not isinstance(t, dict)])
        tool_classes = list(tool_node.tools_by_name.values())

    is_dynamic_model = not isinstance(model, (str, Runnable)) and callable(model)
    is_async_dynamic_model = is_dynamic_model and inspect.iscoroutinefunction(model)

    tool_calling_enabled = len(tool_classes) > 0

    if not is_dynamic_model:
        if isinstance(model, str):
            try:
                from langchain.chat_models import (  # type: ignore[import-not-found]
                    init_chat_model,
                )
            except ImportError:
                raise ImportError(
                    "Please install langchain (`pip install langchain`) to "
                    "use '<provider>:<model>' string syntax for `model` parameter."
                )

            model = cast(BaseChatModel, init_chat_model(model))

        if (
            _should_bind_tools(model, tool_classes, num_builtin=len(llm_builtin_tools))  # type: ignore[arg-type]
            and len(tool_classes + llm_builtin_tools) > 0
        ):
            model = cast(BaseChatModel, model).bind_tools(
                tool_classes + llm_builtin_tools  # type: ignore[operator]
            )

        static_model: Runnable | None = _get_prompt_runnable(prompt) | model  # type: ignore[operator]
    else:
        # 동적 모델의 경우 런타임에 runnable을 생성합니다
        static_model = None

    # 실행 후 return_directly로 구성된 도구가 있는 경우,
    # 그래프는 이러한 도구가 호출되었는지 확인해야 합니다
    should_return_direct = {t.name for t in tool_classes if t.return_direct}

    def _resolve_model(
        state: StateSchema, runtime: Runtime[ContextT]
    ) -> LanguageModelLike:
        """정적 및 동적 모델을 모두 처리하여 사용할 모델을 확인합니다."""
        if is_dynamic_model:
            return _get_prompt_runnable(prompt) | model(state, runtime)  # type: ignore[operator]
        else:
            return static_model

    async def _aresolve_model(
        state: StateSchema, runtime: Runtime[ContextT]
    ) -> LanguageModelLike:
        """정적 및 동적 모델을 모두 처리하여 사용할 모델을 비동기적으로 확인합니다."""
        if is_async_dynamic_model:
            resolved_model = await model(state, runtime)  # type: ignore[misc,operator]
            return _get_prompt_runnable(prompt) | resolved_model
        elif is_dynamic_model:
            return _get_prompt_runnable(prompt) | model(state, runtime)  # type: ignore[operator]
        else:
            return static_model

    def _are_more_steps_needed(state: StateSchema, response: BaseMessage) -> bool:
        has_tool_calls = isinstance(response, AIMessage) and response.tool_calls
        all_tools_return_direct = (
            all(call["name"] in should_return_direct for call in response.tool_calls)
            if isinstance(response, AIMessage)
            else False
        )
        remaining_steps = _get_state_value(state, "remaining_steps", None)
        if remaining_steps is not None:
            if remaining_steps < 1 and all_tools_return_direct:
                return True
            elif remaining_steps < 2 and has_tool_calls:
                return True

        return False

    def _get_model_input_state(state: StateSchema) -> StateSchema:
        if pre_model_hook is not None:
            messages = (
                _get_state_value(state, "llm_input_messages")
            ) or _get_state_value(state, "messages")
            error_msg = f"Expected input to call_model to have 'llm_input_messages' or 'messages' key, but got {state}"
        else:
            messages = _get_state_value(state, "messages")
            error_msg = (
                f"Expected input to call_model to have 'messages' key, but got {state}"
            )

        if messages is None:
            raise ValueError(error_msg)

        _validate_chat_history(messages)
        # 프롬프트가 예상하는 대로 `messages` 키 아래에 메시지를 전달합니다
        if isinstance(state_schema, type) and issubclass(state_schema, BaseModel):
            state.messages = messages  # type: ignore
        else:
            state["messages"] = messages  # type: ignore

        return state

    # 모델을 호출하는 함수를 정의합니다
    def call_model(
        state: StateSchema, runtime: Runtime[ContextT], config: RunnableConfig
    ) -> StateSchema:
        if is_async_dynamic_model:
            msg = (
                "비동기 모델 callable이 제공되었지만 에이전트가 동기적으로 호출되었습니다. "
                "agent.ainvoke() 또는 agent.astream()을 사용하거나, "
                "동기 모델 callable을 제공하세요."
            )
            raise RuntimeError(msg)

        model_input = _get_model_input_state(state)

        if is_dynamic_model:
            # 런타임에 동적 모델을 확인하고 프롬프트를 적용합니다
            dynamic_model = _resolve_model(state, runtime)
            response = cast(AIMessage, dynamic_model.invoke(model_input, config))  # type: ignore[arg-type]
        else:
            response = cast(AIMessage, static_model.invoke(model_input, config))  # type: ignore[union-attr]

        # AIMessage에 에이전트 이름을 추가합니다
        response.name = name

        if _are_more_steps_needed(state, response):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        # 리스트를 반환합니다. 이는 기존 리스트에 추가될 것이기 때문입니다
        return {"messages": [response]}

    async def acall_model(
        state: StateSchema, runtime: Runtime[ContextT], config: RunnableConfig
    ) -> StateSchema:
        model_input = _get_model_input_state(state)

        if is_dynamic_model:
            # 런타임에 동적 모델을 확인하고 프롬프트를 적용합니다
            # (동기 및 비동기 모두 지원)
            dynamic_model = await _aresolve_model(state, runtime)
            response = cast(AIMessage, await dynamic_model.ainvoke(model_input, config))  # type: ignore[arg-type]
        else:
            response = cast(AIMessage, await static_model.ainvoke(model_input, config))  # type: ignore[union-attr]

        # AIMessage에 에이전트 이름을 추가합니다
        response.name = name
        if _are_more_steps_needed(state, response):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        # 리스트를 반환합니다. 이는 기존 리스트에 추가될 것이기 때문입니다
        return {"messages": [response]}

    input_schema: StateSchemaType
    if pre_model_hook is not None:
        # state_schema를 상속하고 'llm_input_messages'를 추가하는 스키마를 동적으로 생성합니다
        if isinstance(state_schema, type) and issubclass(state_schema, BaseModel):
            # Pydantic 스키마의 경우
            from pydantic import create_model

            input_schema = create_model(
                "CallModelInputSchema",
                llm_input_messages=(list[AnyMessage], ...),
                __base__=state_schema,
            )
        else:
            # TypedDict 스키마의 경우
            class CallModelInputSchema(state_schema):  # type: ignore
                llm_input_messages: list[AnyMessage]

            input_schema = CallModelInputSchema
    else:
        input_schema = state_schema

    def generate_structured_response(
        state: StateSchema, runtime: Runtime[ContextT], config: RunnableConfig
    ) -> StateSchema:
        if is_async_dynamic_model:
            msg = (
                "비동기 모델 callable이 제공되었지만 에이전트가 동기적으로 호출되었습니다. "
                "agent.ainvoke() 또는 agent.astream()을 사용하거나, 동기 모델 callable을 제공하세요."
            )
            raise RuntimeError(msg)

        messages = _get_state_value(state, "messages")
        structured_response_schema = response_format
        if isinstance(response_format, tuple):
            system_prompt, structured_response_schema = response_format
            messages = [SystemMessage(content=system_prompt)] + list(messages)

        resolved_model = _resolve_model(state, runtime)
        model_with_structured_output = _get_model(
            resolved_model
        ).with_structured_output(
            cast(StructuredResponseSchema, structured_response_schema)
        )
        response = model_with_structured_output.invoke(messages, config)
        return {"structured_response": response}

    async def agenerate_structured_response(
        state: StateSchema, runtime: Runtime[ContextT], config: RunnableConfig
    ) -> StateSchema:
        messages = _get_state_value(state, "messages")
        structured_response_schema = response_format
        if isinstance(response_format, tuple):
            system_prompt, structured_response_schema = response_format
            messages = [SystemMessage(content=system_prompt)] + list(messages)

        resolved_model = await _aresolve_model(state, runtime)
        model_with_structured_output = _get_model(
            resolved_model
        ).with_structured_output(
            cast(StructuredResponseSchema, structured_response_schema)
        )
        response = await model_with_structured_output.ainvoke(messages, config)
        return {"structured_response": response}

    if not tool_calling_enabled:
        # 새 그래프를 정의합니다
        workflow = StateGraph(state_schema=state_schema, context_schema=context_schema)
        workflow.add_node(
            "agent",
            RunnableCallable(call_model, acall_model),
            input_schema=input_schema,
        )
        if pre_model_hook is not None:
            workflow.add_node("pre_model_hook", pre_model_hook)  # type: ignore[arg-type]
            workflow.add_edge("pre_model_hook", "agent")
            entrypoint = "pre_model_hook"
        else:
            entrypoint = "agent"

        workflow.set_entry_point(entrypoint)

        if post_model_hook is not None:
            workflow.add_node("post_model_hook", post_model_hook)  # type: ignore[arg-type]
            workflow.add_edge("agent", "post_model_hook")

        if response_format is not None:
            workflow.add_node(
                "generate_structured_response",
                RunnableCallable(
                    generate_structured_response,
                    agenerate_structured_response,
                ),
            )
            if post_model_hook is not None:
                workflow.add_edge("post_model_hook", "generate_structured_response")
            else:
                workflow.add_edge("agent", "generate_structured_response")

        return workflow.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            debug=debug,
            name=name,
        )

    # 계속할지 여부를 결정하는 함수를 정의합니다
    def should_continue(state: StateSchema) -> str | list[Send]:
        messages = _get_state_value(state, "messages")
        last_message = messages[-1]
        # 함수 호출이 없으면 완료합니다
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            if post_model_hook is not None:
                return "post_model_hook"
            elif response_format is not None:
                return "generate_structured_response"
            else:
                return END
        # 그렇지 않고 함수 호출이 있으면 계속합니다
        else:
            if version == "v1":
                return "tools"
            elif version == "v2":
                if post_model_hook is not None:
                    return "post_model_hook"
                tool_calls = [
                    tool_node.inject_tool_args(call, state, store)  # type: ignore[arg-type]
                    for call in last_message.tool_calls
                ]
                return [Send("tools", [tool_call]) for tool_call in tool_calls]

    # 새 그래프를 정의합니다
    workflow = StateGraph(
        state_schema=state_schema or AgentState, context_schema=context_schema
    )

    # 순환할 두 개의 노드를 정의합니다
    workflow.add_node(
        "agent",
        RunnableCallable(call_model, acall_model),
        input_schema=input_schema,
    )
    workflow.add_node("tools", tool_node)

    # "agent"(LLM 호출 노드) 전에 매번 호출될
    # pre-model hook 노드를 선택적으로 추가합니다
    if pre_model_hook is not None:
        workflow.add_node("pre_model_hook", pre_model_hook)  # type: ignore[arg-type]
        workflow.add_edge("pre_model_hook", "agent")
        entrypoint = "pre_model_hook"
    else:
        entrypoint = "agent"

    # 엔트리포인트를 `agent`로 설정합니다
    # 이는 이 노드가 첫 번째로 호출되는 노드임을 의미합니다
    workflow.set_entry_point(entrypoint)

    agent_paths = []
    post_model_hook_paths = [entrypoint, "tools"]

    # post_model_hook이 제공되면 post model hook 노드를 추가합니다
    if post_model_hook is not None:
        workflow.add_node("post_model_hook", post_model_hook)  # type: ignore[arg-type]
        agent_paths.append("post_model_hook")
        workflow.add_edge("agent", "post_model_hook")
    else:
        agent_paths.append("tools")

    # response_format이 제공되면 구조화된 출력 노드를 추가합니다
    if response_format is not None:
        workflow.add_node(
            "generate_structured_response",
            RunnableCallable(
                generate_structured_response,
                agenerate_structured_response,
            ),
        )
        if post_model_hook is not None:
            post_model_hook_paths.append("generate_structured_response")
        else:
            agent_paths.append("generate_structured_response")
    else:
        if post_model_hook is not None:
            post_model_hook_paths.append(END)
        else:
            agent_paths.append(END)

    if post_model_hook is not None:

        def post_model_hook_router(state: StateSchema) -> str | list[Send]:
            """post_model_hook 이후 다음 노드로 라우팅합니다.

            다음 중 하나로 라우팅합니다:
            * "tools": 해당하는 메시지 없이 대기 중인 도구 호출이 있는 경우.
            * "generate_structured_response": 대기 중인 도구 호출이 없고 response_format이 지정된 경우.
            * END: 대기 중인 도구 호출이 없고 response_format이 지정되지 않은 경우.
            """

            messages = _get_state_value(state, "messages")
            tool_messages = [
                m.tool_call_id for m in messages if isinstance(m, ToolMessage)
            ]
            last_ai_message = next(
                m for m in reversed(messages) if isinstance(m, AIMessage)
            )
            pending_tool_calls = [
                c for c in last_ai_message.tool_calls if c["id"] not in tool_messages
            ]

            if pending_tool_calls:
                pending_tool_calls = [
                    tool_node.inject_tool_args(call, state, store)  # type: ignore[arg-type]
                    for call in pending_tool_calls
                ]
                return [Send("tools", [tool_call]) for tool_call in pending_tool_calls]
            elif isinstance(messages[-1], ToolMessage):
                return entrypoint
            elif response_format is not None:
                return "generate_structured_response"
            else:
                return END

        workflow.add_conditional_edges(
            "post_model_hook",
            post_model_hook_router,
            path_map=post_model_hook_paths,
        )

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        path_map=agent_paths,
    )

    def route_tool_responses(state: StateSchema) -> str:
        for m in reversed(_get_state_value(state, "messages")):
            if not isinstance(m, ToolMessage):
                break
            if m.name in should_return_direct:
                return END

        # `return_direct`를 가진 도구가 다른 `Send`에서 실행된
        # 병렬 도구 호출의 경우를 처리합니다
        if isinstance(m, AIMessage) and m.tool_calls:
            if any(call["name"] in should_return_direct for call in m.tool_calls):
                return END

        return entrypoint

    if should_return_direct:
        workflow.add_conditional_edges(
            "tools", route_tool_responses, path_map=[entrypoint, END]
        )
    else:
        workflow.add_edge("tools", entrypoint)

    # 마지막으로 컴파일합니다!
    # 이것은 LangChain Runnable로 컴파일되며,
    # 다른 runnable과 동일하게 사용할 수 있습니다
    return workflow.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        name=name,
    )


# 이전 버전과의 호환성을 유지합니다
create_tool_calling_executor = create_react_agent

__all__ = [
    "create_react_agent",
    "create_tool_calling_executor",
    "AgentState",
    "AgentStatePydantic",
    "AgentStateWithStructuredResponse",
    "AgentStateWithStructuredResponsePydantic",
]
