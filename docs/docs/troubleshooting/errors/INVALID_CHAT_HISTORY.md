# INVALID_CHAT_HISTORY

:::python
이 에러는 prebuilt @[create_react_agent][create_react_agent]에서 `call_model` 그래프 노드가 잘못된 형식의 메시지 목록을 받을 때 발생합니다. 특히 `tool_calls`가 있는 `AIMessages`(LLM이 도구 호출을 요청)에 대응하는 `ToolMessage`(LLM에 반환할 도구 호출 결과)가 없을 때 잘못된 형식입니다.
:::

:::js
이 에러는 prebuilt @[createReactAgent][create_react_agent]에서 `callModel` 그래프 노드가 잘못된 형식의 메시지 목록을 받을 때 발생합니다. 특히 `tool_calls`가 있는 `AIMessage`s(LLM이 도구 호출을 요청)에 대응하는 `ToolMessage`(LLM에 반환할 도구 호출 결과)가 없을 때 잘못된 형식입니다.
:::

이 에러가 발생하는 몇 가지 이유가 있을 수 있습니다:

:::python

1. 그래프를 호출할 때 잘못된 형식의 메시지 목록을 수동으로 전달한 경우, 예: `graph.invoke({'messages': [AIMessage(..., tool_calls=[...])]})`
2. `tools` 노드(즉, ToolMessages 목록)로부터 업데이트를 받기 전에 그래프가 중단되었고
   None이나 ToolMessage가 아닌 입력으로 호출한 경우,
   예: `graph.invoke({'messages': [HumanMessage(...)]}, config)`.

   이 중단은 다음 방법 중 하나로 트리거되었을 수 있습니다:

   - `create_react_agent`에서 `interrupt_before = ['tools']`를 수동으로 설정
   - 도구 중 하나가 @[ToolNode][ToolNode] (`"tools"`)에서 처리되지 않은 에러를 발생시킨 경우

:::

:::js

1. 그래프를 호출할 때 잘못된 형식의 메시지 목록을 수동으로 전달한 경우, 예: `graph.invoke({messages: [new AIMessage({..., tool_calls: [...]})]})`
2. `tools` 노드(즉, ToolMessages 목록)로부터 업데이트를 받기 전에 그래프가 중단되었고
   null이나 ToolMessage가 아닌 입력으로 호출한 경우,
   예: `graph.invoke({messages: [new HumanMessage(...)]}, config)`.

   이 중단은 다음 방법 중 하나로 트리거되었을 수 있습니다:

   - `createReactAgent`에서 `interruptBefore: ['tools']`를 수동으로 설정
   - 도구 중 하나가 @[ToolNode][ToolNode] (`"tools"`)에서 처리되지 않은 에러를 발생시킨 경우

:::

## 트러블슈팅

이 문제를 해결하려면 다음 중 하나를 수행할 수 있습니다:

1. 잘못된 형식의 메시지 목록으로 그래프를 호출하지 마세요
2. 중단(수동 또는 에러로 인한) 의 경우 다음을 수행할 수 있습니다:

:::python

- 기존 tool call과 일치하는 ToolMessages를 제공하고 `graph.invoke({'messages': [ToolMessage(...)]})` 를 호출합니다.
  **참고**: 이렇게 하면 메시지가 히스토리에 추가되고 START 노드에서 그래프가 실행됩니다.

  - 상태를 수동으로 업데이트하고 중단에서 그래프를 재개합니다:

          1. `graph.get_state(config)`로 그래프 상태에서 가장 최근 메시지 목록을 가져옵니다
          2. 메시지 목록을 수정하여 AIMessages에서 응답되지 않은 tool call을 제거하거나

응답되지 않은 tool call과 일치하는 tool_call_ids를 가진 ToolMessages를 추가합니다 3. 수정된 메시지 목록으로 `graph.update_state(config, {'messages': ...})`를 호출합니다 4. 그래프를 재개합니다, 예: `graph.invoke(None, config)` 호출
:::

:::js

- 기존 tool call과 일치하는 ToolMessages를 제공하고 `graph.invoke({messages: [new ToolMessage(...)]})`를 호출합니다.
  **참고**: 이렇게 하면 메시지가 히스토리에 추가되고 START 노드에서 그래프가 실행됩니다.

  - 상태를 수동으로 업데이트하고 중단에서 그래프를 재개합니다:

          1. `graph.getState(config)`로 그래프 상태에서 가장 최근 메시지 목록을 가져옵니다
          2. 메시지 목록을 수정하여 AIMessages에서 응답되지 않은 tool call을 제거하거나

응답되지 않은 tool call과 일치하는 `toolCallId`를 가진 ToolMessages를 추가합니다 3. 수정된 메시지 목록으로 `graph.updateState(config, {messages: ...})`를 호출합니다 4. 그래프를 재개합니다, 예: `graph.invoke(null, config)` 호출
:::
