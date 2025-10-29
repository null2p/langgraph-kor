---
search:
  boost: 2
tags:
  - agent
hide:
  - tags
---

# Running agents

에이전트는 `.invoke()` / `await .ainvoke()`를 사용한 전체 응답이나 `.stream()` / `.astream()`을 사용한 **점진적** [스트리밍](../how-tos/streaming.md) 출력 모두에 대해 동기 및 비동기 실행을 지원합니다. 이 섹션에서는 입력을 제공하고, 출력을 해석하고, 스트리밍을 활성화하고, 실행 제한을 제어하는 방법을 설명합니다.

## Basic usage

에이전트는 두 가지 주요 모드로 실행할 수 있습니다:

:::python

- **동기(Synchronous)** `.invoke()` 또는 `.stream()` 사용
- **비동기(Asynchronous)** `await .ainvoke()` 또는 `.astream()`과 함께 `async for` 사용
  :::

:::js

- **동기(Synchronous)** `.invoke()` 또는 `.stream()` 사용
- **비동기(Asynchronous)** `await .invoke()` 또는 `.stream()`과 함께 `for await` 사용
  :::

:::python
=== "Sync invocation"

    ```python
    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(...)

    # highlight-next-line
    response = agent.invoke({"messages": [{"role": "user", "content": "what is the weather in sf"}]})
    ```

=== "Async invocation"

    ```python
    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(...)
    # highlight-next-line
    response = await agent.ainvoke({"messages": [{"role": "user", "content": "what is the weather in sf"}]})
    ```

:::

:::js

```typescript
import { createReactAgent } from "@langchain/langgraph/prebuilt";

const agent = createReactAgent(...);
// highlight-next-line
const response = await agent.invoke({
    "messages": [
        { "role": "user", "content": "what is the weather in sf" }
    ]
});
```

:::

## Inputs and outputs

에이전트는 `messages` 리스트를 입력으로 예상하는 언어 모델을 사용합니다. 따라서 에이전트의 입력과 출력은 에이전트 [state](../concepts/low_level.md#working-with-messages-in-graph-state)의 `messages` 키 아래에 `messages` 리스트로 저장됩니다.

## Input format

에이전트 입력은 `messages` 키를 가진 딕셔너리여야 합니다. 지원되는 형식은 다음과 같습니다:

:::python
| 형식 | 예시 |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------|
| String | `{"messages": "Hello"}` — [HumanMessage](https://python.langchain.com/docs/concepts/messages/#humanmessage)로 해석됨 |
| Message dictionary | `{"messages": {"role": "user", "content": "Hello"}}` |
| List of messages | `{"messages": [{"role": "user", "content": "Hello"}]}` |
| With custom state | `{"messages": [{"role": "user", "content": "Hello"}], "user_name": "Alice"}` — 커스텀 `state_schema`를 사용하는 경우 |
:::

:::js
| 형식 | 예시 |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------|
| String | `{"messages": "Hello"}` — [HumanMessage](https://js.langchain.com/docs/concepts/messages/#humanmessage)로 해석됨 |
| Message dictionary | `{"messages": {"role": "user", "content": "Hello"}}` |
| List of messages | `{"messages": [{"role": "user", "content": "Hello"}]}` |
| With custom state | `{"messages": [{"role": "user", "content": "Hello"}], "user_name": "Alice"}` — 커스텀 state 정의를 사용하는 경우 |
:::

:::python
메시지는 자동으로 LangChain의 내부 메시지 형식으로 변환됩니다. [LangChain messages](https://python.langchain.com/docs/concepts/messages/#langchain-messages)에 대한 자세한 내용은 LangChain 문서를 참조하세요.
:::

:::js
메시지는 자동으로 LangChain의 내부 메시지 형식으로 변환됩니다. [LangChain messages](https://js.langchain.com/docs/concepts/messages/#langchain-messages)에 대한 자세한 내용은 LangChain 문서를 참조하세요.
:::

!!! tip "Using custom agent state"

    :::python
    에이전트의 state 스키마에 정의된 추가 필드를 입력 딕셔너리에 직접 제공할 수 있습니다. 이를 통해 런타임 데이터나 이전 도구 출력을 기반으로 동적 동작이 가능합니다.
    자세한 내용은 [context guide](./context.md)를 참조하세요.
    :::

    :::js
    에이전트의 state에 정의된 추가 필드를 state 정의에 직접 제공할 수 있습니다. 이를 통해 런타임 데이터나 이전 도구 출력을 기반으로 동적 동작이 가능합니다.
    자세한 내용은 [context guide](./context.md)를 참조하세요.
    :::

!!! note

    :::python
    `messages`에 대한 문자열 입력은 [HumanMessage](https://python.langchain.com/docs/concepts/messages/#humanmessage)로 변환됩니다. 이는 문자열로 전달될 때 [SystemMessage](https://python.langchain.com/docs/concepts/messages/#systemmessage)로 해석되는 `create_react_agent`의 `prompt` 매개변수와 다릅니다.
    :::

    :::js
    `messages`에 대한 문자열 입력은 [HumanMessage](https://js.langchain.com/docs/concepts/messages/#humanmessage)로 변환됩니다. 이는 문자열로 전달될 때 [SystemMessage](https://js.langchain.com/docs/concepts/messages/#systemmessage)로 해석되는 `createReactAgent`의 `prompt` 매개변수와 다릅니다.
    :::

## Output format

:::python
에이전트 출력은 다음을 포함하는 딕셔너리입니다:

- `messages`: 실행 중 교환된 모든 메시지 리스트 (사용자 입력, assistant 응답, 도구 호출).
- 선택적으로, [structured output](./agents.md#6-configure-structured-output)이 구성된 경우 `structured_response`.
- 커스텀 `state_schema`를 사용하는 경우, 정의된 필드에 해당하는 추가 키도 출력에 존재할 수 있습니다. 이들은 도구 실행이나 프롬프트 로직에서 업데이트된 state 값을 보유할 수 있습니다.
:::

:::js
에이전트 출력은 다음을 포함하는 딕셔너리입니다:

- `messages`: 실행 중 교환된 모든 메시지 리스트 (사용자 입력, assistant 응답, 도구 호출).
- 선택적으로, [structured output](./agents.md#6-configure-structured-output)이 구성된 경우 `structuredResponse`.
- 커스텀 state 정의를 사용하는 경우, 정의된 필드에 해당하는 추가 키도 출력에 존재할 수 있습니다. 이들은 도구 실행이나 프롬프트 로직에서 업데이트된 state 값을 보유할 수 있습니다.
:::

커스텀 state 스키마 작업 및 context 액세스에 대한 자세한 내용은 [context guide](./context.md)를 참조하세요.

## Streaming output

에이전트는 보다 반응성 높은 애플리케이션을 위해 스트리밍 응답을 지원합니다. 여기에는 다음이 포함됩니다:

- 각 단계 후의 **진행 상황 업데이트**
- 생성되는 **LLM 토큰**
- 실행 중의 **커스텀 도구 메시지**

스트리밍은 동기 및 비동기 모드 모두에서 사용할 수 있습니다:

:::python
=== "Sync streaming"

    ```python
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
        stream_mode="updates"
    ):
        print(chunk)
    ```

=== "Async streaming"

    ```python
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
        stream_mode="updates"
    ):
        print(chunk)
    ```

:::

:::js

```typescript
for await (const chunk of agent.stream(
  { messages: [{ role: "user", content: "what is the weather in sf" }] },
  { streamMode: "updates" }
)) {
  console.log(chunk);
}
```

:::

!!! tip

    자세한 내용은 [streaming guide](../how-tos/streaming.md)를 참조하세요.

## Max iterations

:::python
에이전트 실행을 제어하고 무한 루프를 방지하려면 재귀 제한을 설정합니다. 이는 `GraphRecursionError`가 발생하기 전에 에이전트가 수행할 수 있는 최대 단계 수를 정의합니다. 런타임에 또는 `.with_config()`를 통해 에이전트를 정의할 때 `recursion_limit`을 구성할 수 있습니다:
:::

:::js
에이전트 실행을 제어하고 무한 루프를 방지하려면 재귀 제한을 설정합니다. 이는 `GraphRecursionError`가 발생하기 전에 에이전트가 수행할 수 있는 최대 단계 수를 정의합니다. 런타임에 또는 `.withConfig()`를 통해 에이전트를 정의할 때 `recursionLimit`을 구성할 수 있습니다:
:::

:::python
=== "Runtime"

    ```python
    from langgraph.errors import GraphRecursionError
    from langgraph.prebuilt import create_react_agent

    max_iterations = 3
    # highlight-next-line
    recursion_limit = 2 * max_iterations + 1
    agent = create_react_agent(
        model="anthropic:claude-3-5-haiku-latest",
        tools=[get_weather]
    )

    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": "what's the weather in sf"}]},
            # highlight-next-line
            {"recursion_limit": recursion_limit},
        )
    except GraphRecursionError:
        print("Agent stopped due to max iterations.")
    ```

=== "`.with_config()`"

    ```python
    from langgraph.errors import GraphRecursionError
    from langgraph.prebuilt import create_react_agent

    max_iterations = 3
    # highlight-next-line
    recursion_limit = 2 * max_iterations + 1
    agent = create_react_agent(
        model="anthropic:claude-3-5-haiku-latest",
        tools=[get_weather]
    )
    # highlight-next-line
    agent_with_recursion_limit = agent.with_config(recursion_limit=recursion_limit)

    try:
        response = agent_with_recursion_limit.invoke(
            {"messages": [{"role": "user", "content": "what's the weather in sf"}]},
        )
    except GraphRecursionError:
        print("Agent stopped due to max iterations.")
    ```

:::

:::js
=== "Runtime"

    ```typescript
    import { GraphRecursionError } from "@langchain/langgraph";
    import { ChatAnthropic } from "@langchain/langgraph/prebuilt";
    import { createReactAgent } from "@langchain/langgraph/prebuilt";

    const maxIterations = 3;
    // highlight-next-line
    const recursionLimit = 2 * maxIterations + 1;
    const agent = createReactAgent({
        llm: new ChatAnthropic({ model: "claude-3-5-haiku-latest" }),
        tools: [getWeather]
    });

    try {
        const response = await agent.invoke(
            {"messages": [{"role": "user", "content": "what's the weather in sf"}]},
            // highlight-next-line
            { recursionLimit }
        );
    } catch (error) {
        if (error instanceof GraphRecursionError) {
            console.log("Agent stopped due to max iterations.");
        }
    }
    ```

=== "`.withConfig()`"

    ```typescript
    import { GraphRecursionError } from "@langchain/langgraph";
    import { ChatAnthropic } from "@langchain/langgraph/prebuilt";
    import { createReactAgent } from "@langchain/langgraph/prebuilt";

    const maxIterations = 3;
    // highlight-next-line
    const recursionLimit = 2 * maxIterations + 1;
    const agent = createReactAgent({
        llm: new ChatAnthropic({ model: "claude-3-5-haiku-latest" }),
        tools: [getWeather]
    });
    // highlight-next-line
    const agentWithRecursionLimit = agent.withConfig({ recursionLimit });

    try {
        const response = await agentWithRecursionLimit.invoke(
            {"messages": [{"role": "user", "content": "what's the weather in sf"}]},
        );
    } catch (error) {
        if (error instanceof GraphRecursionError) {
            console.log("Agent stopped due to max iterations.");
        }
    }
    ```

:::

:::python

## Additional Resources

- [Async programming in LangChain](https://python.langchain.com/docs/concepts/async)
  :::
