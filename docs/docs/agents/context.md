# Context

**Context engineering**은 AI 애플리케이션이 작업을 수행할 수 있도록 올바른 정보와 도구를 올바른 형식으로 제공하는 동적 시스템을 구축하는 관행입니다. Context는 두 가지 주요 차원으로 특징지을 수 있습니다:

1. **가변성(mutability)**에 따라:
    - **Static context**: 실행 중 변경되지 않는 불변 데이터 (예: 사용자 메타데이터, 데이터베이스 연결, 도구)
    - **Dynamic context**: 애플리케이션이 실행되면서 변화하는 가변 데이터 (예: 대화 히스토리, 중간 결과, 도구 호출 관찰 결과)
2. **생명 주기(lifetime)**에 따라:
    - **Runtime context**: 단일 실행 또는 호출에 범위가 지정된 데이터
    - **Cross-conversation context**: 여러 대화 또는 세션에 걸쳐 유지되는 데이터

!!! tip "Runtime context vs LLM context"

    Runtime context는 로컬 컨텍스트를 의미합니다: 코드가 실행되기 위해 필요한 데이터 및 의존성입니다. 다음을 의미하지 **않습니다**:

    * LLM context, 즉 LLM의 프롬프트에 전달되는 데이터.
    * "context window", 즉 LLM에 전달할 수 있는 최대 토큰 수.

    Runtime context는 LLM context를 최적화하는 데 사용할 수 있습니다. 예를 들어, runtime context의 사용자 메타데이터를 사용하여 사용자 선호도를 가져와서 context window에 제공할 수 있습니다.

LangGraph는 가변성과 생명 주기 차원을 결합하여 context를 관리하는 세 가지 방법을 제공합니다:

:::python

| Context 타입                                                                                | 설명                                            | 가변성  | 생명 주기         | 접근 방법                           |
| ------------------------------------------------------------------------------------------- | ------------------------------------------------------ | ---------- | ------------------ | --------------------------------------- |
| [**Static runtime context**](#static-runtime-context)                                       | 시작 시 전달되는 사용자 메타데이터, 도구, db 연결 | Static     | Single run         | `context` argument to `invoke`/`stream` |
| [**Dynamic runtime context (state)**](#dynamic-runtime-context-state)                       | 단일 실행 중 변화하는 가변 데이터          | Dynamic    | Single run         | LangGraph state object                  |
| [**Dynamic cross-conversation context (store)**](#dynamic-cross-conversation-context-store) | 대화 간에 공유되는 영구 데이터            | Dynamic    | Cross-conversation | LangGraph store                         |

## Static runtime context

**Static runtime context**는 실행 시작 시 `invoke`/`stream`의 `context` 인수를 통해 애플리케이션에 전달되는 사용자 메타데이터, 도구, 데이터베이스 연결과 같은 불변 데이터를 나타냅니다. 이 데이터는 실행 중 변경되지 않습니다.

!!! version-added "Added in version 0.6.0: `context` replaces `config['configurable']`"

    Runtime context는 이제 `invoke`/`stream`의 `context` 인수로 전달되며,
    이는 `config['configurable']`에 애플리케이션 구성을 전달하던 이전 패턴을 대체합니다.

```python
@dataclass
class ContextSchema:
    user_name: str

graph.invoke( # (1)!
    {"messages": [{"role": "user", "content": "hi!"}]}, # (2)!
    # highlight-next-line
    context={"user_name": "John Smith"} # (3)!
)
```

1. 이것은 에이전트 또는 그래프의 호출입니다. `invoke` 메서드는 제공된 입력으로 기본 그래프를 실행합니다.
2. 이 예제는 메시지를 입력으로 사용하며, 이는 일반적이지만 애플리케이션은 다른 입력 구조를 사용할 수 있습니다.
3. 여기에서 runtime 데이터를 전달합니다. `context` 매개변수를 사용하면 에이전트가 실행 중에 사용할 수 있는 추가 의존성을 제공할 수 있습니다.

=== "Agent prompt"

    ```python
    from langchain_core.messages import AnyMessage
    from langgraph.runtime import get_runtime
    from langgraph.prebuilt.chat_agent_executor import AgentState
    from langgraph.prebuilt import create_react_agent

    # highlight-next-line
    def prompt(state: AgentState) -> list[AnyMessage]:
        runtime = get_runtime(ContextSchema)
        system_msg = f"You are a helpful assistant. Address the user as {runtime.context.user_name}."
        return [{"role": "system", "content": system_msg}] + state["messages"]

    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_weather],
        prompt=prompt,
        context_schema=ContextSchema
    )

    agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
        # highlight-next-line
        context={"user_name": "John Smith"}
    )
    ```

    * 자세한 내용은 [Agents](../agents/agents.md)를 참조하세요.

=== "Workflow node"

    ```python
    from langgraph.runtime import Runtime

    # highlight-next-line
    def node(state: State, runtime: Runtime[ContextSchema]):
        user_name = runtime.context.user_name
        ...
    ```

    * 자세한 내용은 [Graph API](https://langchain-ai.github.io/langgraph/how-tos/graph-api/#add-runtime-configuration)를 참조하세요.

=== "In a tool"

    ```python
    from langgraph.runtime import get_runtime

    @tool
    # highlight-next-line
    def get_user_email() -> str:
        """Retrieve user information based on user ID."""
        # simulate fetching user info from a database
        runtime = get_runtime(ContextSchema)
        email = get_user_email_from_db(runtime.context.user_name)
        return email
    ```

    자세한 내용은 [tool calling guide](../how-tos/tool-calling.md#configuration)를 참조하세요.

!!! tip

    `Runtime` 객체는 static context 및 활성 store, stream writer와 같은 다른 유틸리티에 액세스하는 데 사용할 수 있습니다.
    자세한 내용은 [Runtime][langgraph.runtime.Runtime] 문서를 참조하세요.

:::

:::js

| Context 타입                                                                                | 설명                                   | 가변성  | 생명 주기         |
| ------------------------------------------------------------------------------------------- | --------------------------------------------- | ---------- | ------------------ |
| [**Config**](#config-static-context)                                                        | 실행 시작 시 전달되는 데이터             | Static     | Single run         |
| [**Dynamic runtime context (state)**](#dynamic-runtime-context-state)                       | 단일 실행 중 변화하는 가변 데이터 | Dynamic    | Single run         |
| [**Dynamic cross-conversation context (store)**](#dynamic-cross-conversation-context-store) | 대화 간에 공유되는 영구 데이터   | Dynamic    | Cross-conversation |

## Config (static context)

Config는 사용자 메타데이터 또는 API 키와 같은 불변 데이터를 위한 것입니다. 실행 중에 변경되지 않는 값이 있을 때 사용합니다.

**"configurable"**이라는 키를 사용하여 구성을 지정합니다. 이 키는 이 목적으로 예약되어 있습니다.

```typescript
await graph.invoke(
  // (1)!
  { messages: [{ role: "user", content: "hi!" }] }, // (2)!
  // highlight-next-line
  { configurable: { user_id: "user_123" } } // (3)!
);
```

:::

## Dynamic runtime context (state)

**Dynamic runtime context**는 단일 실행 중 변화할 수 있는 가변 데이터를 나타내며 LangGraph state 객체를 통해 관리됩니다. 여기에는 대화 히스토리, 중간 결과, 도구 또는 LLM 출력에서 파생된 값이 포함됩니다. LangGraph에서 state 객체는 실행 중 [단기 메모리](../concepts/memory.md) 역할을 합니다.

=== "In an agent"

    예제는 **prompt**에 state를 통합하는 방법을 보여줍니다.

    State는 에이전트의 **도구**에서도 액세스할 수 있으며, 필요에 따라 state를 읽거나 업데이트할 수 있습니다. 자세한 내용은 [tool calling guide](../how-tos/tool-calling.md#short-term-memory)를 참조하세요.

    :::python
    ```python
    from langchain_core.messages import AnyMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.prebuilt import create_react_agent
    from langgraph.prebuilt.chat_agent_executor import AgentState

    # highlight-next-line
    class CustomState(AgentState): # (1)!
        user_name: str

    def prompt(
        # highlight-next-line
        state: CustomState
    ) -> list[AnyMessage]:
        user_name = state["user_name"]
        system_msg = f"You are a helpful assistant. User's name is {user_name}"
        return [{"role": "system", "content": system_msg}] + state["messages"]

    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[...],
        # highlight-next-line
        state_schema=CustomState, # (2)!
        prompt=prompt
    )

    agent.invoke({
        "messages": "hi!",
        "user_name": "John Smith"
    })
    ```

    1. `AgentState` 또는 `MessagesState`를 확장하는 커스텀 state 스키마를 정의합니다.
    2. 커스텀 state 스키마를 에이전트에 전달합니다. 이를 통해 에이전트가 실행 중에 state에 액세스하고 수정할 수 있습니다.
    :::

    :::js
    ```typescript
    import type { BaseMessage } from "@langchain/core/messages";
    import { createReactAgent } from "@langchain/langgraph/prebuilt";
    import { MessagesZodState } from "@langchain/langgraph";
    import { z } from "zod";

    // highlight-next-line
    const CustomState = z.object({ // (1)!
      messages: MessagesZodState.shape.messages,
      userName: z.string(),
    });

    const prompt = (
      // highlight-next-line
      state: z.infer<typeof CustomState>
    ): BaseMessage[] => {
      const userName = state.userName;
      const systemMsg = `You are a helpful assistant. User's name is ${userName}`;
      return [{ role: "system", content: systemMsg }, ...state.messages];
    };

    const agent = createReactAgent({
      llm: model,
      tools: [...],
      // highlight-next-line
      stateSchema: CustomState, // (2)!
      stateModifier: prompt,
    });

    await agent.invoke({
      messages: [{ role: "user", content: "hi!" }],
      userName: "John Smith",
    });
    ```

    1. `MessagesZodState`를 확장하거나 새 스키마를 생성하는 커스텀 state 스키마를 정의합니다.
    2. 커스텀 state 스키마를 에이전트에 전달합니다. 이를 통해 에이전트가 실행 중에 state에 액세스하고 수정할 수 있습니다.
    :::

=== "In a workflow"

    :::python
    ```python
    from typing_extensions import TypedDict
    from langchain_core.messages import AnyMessage
    from langgraph.graph import StateGraph

    # highlight-next-line
    class CustomState(TypedDict): # (1)!
        messages: list[AnyMessage]
        extra_field: int

    # highlight-next-line
    def node(state: CustomState): # (2)!
        messages = state["messages"]
        ...
        return { # (3)!
            # highlight-next-line
            "extra_field": state["extra_field"] + 1
        }

    builder = StateGraph(State)
    builder.add_node(node)
    builder.set_entry_point("node")
    graph = builder.compile()
    ```

    1. 커스텀 state를 정의합니다
    2. 모든 노드나 도구에서 state에 액세스합니다
    3. Graph API는 state와 최대한 쉽게 작동하도록 설계되었습니다. 노드의 반환 값은 state에 대한 요청된 업데이트를 나타냅니다.
    :::

    :::js
    ```typescript
    import type { BaseMessage } from "@langchain/core/messages";
    import { StateGraph, MessagesZodState, START } from "@langchain/langgraph";
    import { z } from "zod";

    // highlight-next-line
    const CustomState = z.object({ // (1)!
      messages: MessagesZodState.shape.messages,
      extraField: z.number(),
    });

    const builder = new StateGraph(CustomState)
      .addNode("node", async (state) => { // (2)!
        const messages = state.messages;
        // ...
        return { // (3)!
          // highlight-next-line
          extraField: state.extraField + 1,
        };
      })
      .addEdge(START, "node");

    const graph = builder.compile();
    ```

    1. 커스텀 state를 정의합니다
    2. 모든 노드나 도구에서 state에 액세스합니다
    3. Graph API는 state와 최대한 쉽게 작동하도록 설계되었습니다. 노드의 반환 값은 state에 대한 요청된 업데이트를 나타냅니다.
    :::

!!! tip "Turning on memory"

    메모리를 활성화하는 방법에 대한 자세한 내용은 [memory guide](../how-tos/memory/add-memory.md)를 참조하세요. 이는 여러 호출에 걸쳐 에이전트의 state를 유지할 수 있는 강력한 기능입니다. 그렇지 않으면 state는 단일 실행으로만 범위가 지정됩니다.

## Dynamic cross-conversation context (store)

**Dynamic cross-conversation context**는 여러 대화 또는 세션에 걸쳐 지속되는 영구적이고 가변적인 데이터를 나타내며 LangGraph store를 통해 관리됩니다. 여기에는 사용자 프로필, 선호도, 과거 상호작용이 포함됩니다. LangGraph store는 여러 실행에 걸친 [장기 메모리](../concepts/memory.md#long-term-memory) 역할을 합니다. 이는 영구적인 사실(예: 사용자 프로필, 선호도, 이전 상호작용)을 읽거나 업데이트하는 데 사용할 수 있습니다.

자세한 내용은 [Memory guide](../how-tos/memory/add-memory.md)를 참조하세요.
