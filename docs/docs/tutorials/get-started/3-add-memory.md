# 메모리 추가

챗봇은 이제 [도구를 사용](./2-add-tools.md)하여 사용자 질문에 답변할 수 있지만 이전 상호작용의 컨텍스트를 기억하지 못합니다. 이는 일관된 다회차 대화를 진행하는 능력을 제한합니다.

LangGraph는 **영구 체크포인팅**을 통해 이 문제를 해결합니다. 그래프를 컴파일할 때 `checkpointer`를 제공하고 그래프를 호출할 때 `thread_id`를 제공하면 LangGraph는 각 단계 후 자동으로 상태를 저장합니다. 동일한 `thread_id`를 사용하여 그래프를 다시 호출하면 그래프가 저장된 상태를 로드하여 챗봇이 중단한 지점에서 계속할 수 있습니다.

**체크포인팅**은 단순한 채팅 메모리보다 _훨씬_ 더 강력하다는 것을 나중에 보게 될 것입니다. 오류 복구, human-in-the-loop 워크플로우, 시간 여행 상호작용 등을 위해 언제든지 복잡한 상태를 저장하고 재개할 수 있습니다. 하지만 먼저 다회차 대화를 활성화하기 위해 체크포인팅을 추가해 보겠습니다.

!!! note

    이 튜토리얼은 [도구 추가](./2-add-tools.md)를 기반으로 합니다.

## 1. `MemorySaver` 체크포인터 생성

`MemorySaver` 체크포인터를 생성합니다:

:::python

```python
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()
```

:::

:::js

```typescript
import { MemorySaver } from "@langchain/langgraph";

const memory = new MemorySaver();
```

:::

이것은 인메모리 체크포인터로 튜토리얼에 편리합니다. 그러나 프로덕션 애플리케이션에서는 `SqliteSaver` 또는 `PostgresSaver`를 사용하고 데이터베이스에 연결하도록 변경할 가능성이 높습니다.

## 2. 그래프 컴파일

제공된 체크포인터로 그래프를 컴파일합니다. 이는 그래프가 각 노드를 통해 작업할 때 `State`를 체크포인트합니다:

:::python

```python
graph = graph_builder.compile(checkpointer=memory)
```

:::

:::js

```typescript hl_lines="7"
const graph = new StateGraph(State)
  .addNode("chatbot", chatbot)
  .addNode("tools", new ToolNode(tools))
  .addConditionalEdges("chatbot", toolsCondition, ["tools", END])
  .addEdge("tools", "chatbot")
  .addEdge(START, "chatbot")
  .compile({ checkpointer: memory });
```

:::

## 3. 챗봇과 상호작용

이제 봇과 상호작용할 수 있습니다!

1.  이 대화의 키로 사용할 thread를 선택합니다.

    :::python

    ```python
    config = {"configurable": {"thread_id": "1"}}
    ```

    :::

    :::js

    ```typescript
    const config = { configurable: { thread_id: "1" } };
    ```

    :::

2.  챗봇을 호출합니다:

    :::python

    ```python
    user_input = "Hi there! My name is Will."

    # config는 stream() 또는 invoke()의 **두 번째 위치 인수**입니다!
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()
    ```

    ```
    ================================ Human Message =================================

    Hi there! My name is Will.
    ================================== Ai Message ==================================

    Hello Will! It's nice to meet you. How can I assist you today? Is there anything specific you'd like to know or discuss?
    ```

    !!! note

        config는 그래프를 호출할 때 **두 번째 위치 인수**로 제공되었습니다. 중요한 것은 그래프 입력(`{'messages': []}`) 내에 중첩되어 있지 _않다는_ 것입니다.

    :::

    :::js

    ```typescript
    const userInput = "Hi there! My name is Will.";

    const events = await graph.stream(
      { messages: [{ type: "human", content: userInput }] },
      { configurable: { thread_id: "1" }, streamMode: "values" }
    );

    for await (const event of events) {
      const lastMessage = event.messages.at(-1);
      console.log(`${lastMessage?.getType()}: ${lastMessage?.text}`);
    }
    ```

    ```
    human: Hi there! My name is Will.
    ai: Hello Will! It's nice to meet you. How can I assist you today? Is there anything specific you'd like to know or discuss?
    ```

    !!! note

        config는 그래프를 호출할 때 **두 번째 매개변수**로 제공되었습니다. 중요한 것은 그래프 입력(`{"messages": []}`) 내에 중첩되어 있지 _않다는_ 것입니다.

    :::

## 4. 후속 질문하기

후속 질문을 합니다:

:::python

```python
user_input = "Remember my name?"

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()
```

```
================================ Human Message =================================

Remember my name?
================================== Ai Message ==================================

Of course, I remember your name, Will. I always try to pay attention to important details that users share with me. Is there anything else you'd like to talk about or any questions you have? I'm here to help with a wide range of topics or tasks.
```

:::

:::js

```typescript
const userInput2 = "Remember my name?";

const events2 = await graph.stream(
  { messages: [{ type: "human", content: userInput2 }] },
  { configurable: { thread_id: "1" }, streamMode: "values" }
);

for await (const event of events2) {
  const lastMessage = event.messages.at(-1);
  console.log(`${lastMessage?.getType()}: ${lastMessage?.text}`);
}
```

```
human: Remember my name?
ai: Yes, your name is Will. How can I help you today?
```

:::

메모리에 외부 목록을 사용하지 않는다는 점에 **주목하세요**: 모든 것이 체크포인터에 의해 처리됩니다! 무슨 일이 일어나고 있는지 확인하려면 이 [LangSmith 추적](https://smith.langchain.com/public/29ba22b5-6d40-4fbe-8d27-b369e3329c84/r)에서 전체 실행을 검사할 수 있습니다.

믿지 못하겠나요? 다른 config를 사용하여 시도해 보세요.

:::python

```python
# 유일한 차이점은 여기서 `thread_id`를 "1" 대신 "2"로 변경한다는 것입니다
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    # highlight-next-line
    {"configurable": {"thread_id": "2"}},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()
```

```
================================ Human Message =================================

Remember my name?
================================== Ai Message ==================================

I apologize, but I don't have any previous context or memory of your name. As an AI assistant, I don't retain information from past conversations. Each interaction starts fresh. Could you please tell me your name so I can address you properly in this conversation?
```

:::

:::js

```typescript hl_lines="3-4"
const events3 = await graph.stream(
  { messages: [{ type: "human", content: userInput2 }] },
  // 유일한 차이점은 여기서 `thread_id`를 "1" 대신 "2"로 변경한다는 것입니다
  { configurable: { thread_id: "2" }, streamMode: "values" }
);

for await (const event of events3) {
  const lastMessage = event.messages.at(-1);
  console.log(`${lastMessage?.getType()}: ${lastMessage?.text}`);
}
```

```
human: Remember my name?
ai: I don't have the ability to remember personal information about users between interactions. However, I'm here to help you with any questions or topics you want to discuss!
```

:::

**주목할 점**은 config에서 `thread_id`를 수정한 것이 **유일한** 변경 사항이라는 것입니다. 비교를 위해 이 호출의 [LangSmith 추적](https://smith.langchain.com/public/51a62351-2f0a-4058-91cc-9996c5561428/r)을 참조하세요.

## 5. 상태 검사

:::python

지금까지 두 개의 서로 다른 thread에서 몇 가지 체크포인트를 만들었습니다. 그런데 체크포인트에는 무엇이 들어갈까요? 주어진 config에 대한 그래프의 `state`를 언제든지 검사하려면 `get_state(config)`를 호출하세요.

```python
snapshot = graph.get_state(config)
snapshot
```

```
StateSnapshot(values={'messages': [HumanMessage(content='Hi there! My name is Will.', additional_kwargs={}, response_metadata={}, id='8c1ca919-c553-4ebf-95d4-b59a2d61e078'), AIMessage(content="Hello Will! It's nice to meet you. How can I assist you today? Is there anything specific you'd like to know or discuss?", additional_kwargs={}, response_metadata={'id': 'msg_01WTQebPhNwmMrmmWojJ9KXJ', 'model': 'claude-3-5-sonnet-20240620', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 405, 'output_tokens': 32}}, id='run-58587b77-8c82-41e6-8a90-d62c444a261d-0', usage_metadata={'input_tokens': 405, 'output_tokens': 32, 'total_tokens': 437}), HumanMessage(content='Remember my name?', additional_kwargs={}, response_metadata={}, id='daba7df6-ad75-4d6b-8057-745881cea1ca'), AIMessage(content="Of course, I remember your name, Will. I always try to pay attention to important details that users share with me. Is there anything else you'd like to talk about or any questions you have? I'm here to help with a wide range of topics or tasks.", additional_kwargs={}, response_metadata={'id': 'msg_01E41KitY74HpENRgXx94vag', 'model': 'claude-3-5-sonnet-20240620', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 444, 'output_tokens': 58}}, id='run-ffeaae5c-4d2d-4ddb-bd59-5d5cbf2a5af8-0', usage_metadata={'input_tokens': 444, 'output_tokens': 58, 'total_tokens': 502})]}, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef7d06e-93e0-6acc-8004-f2ac846575d2'}}, metadata={'source': 'loop', 'writes': {'chatbot': {'messages': [AIMessage(content="Of course, I remember your name, Will. I always try to pay attention to important details that users share with me. Is there anything else you'd like to talk about or any questions you have? I'm here to help with a wide range of topics or tasks.", additional_kwargs={}, response_metadata={'id': 'msg_01E41KitY74HpENRgXx94vag', 'model': 'claude-3-5-sonnet-20240620', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 444, 'output_tokens': 58}}, id='run-ffeaae5c-4d2d-4ddb-bd59-5d5cbf2a5af8-0', usage_metadata={'input_tokens': 444, 'output_tokens': 58, 'total_tokens': 502})]}}, 'step': 4, 'parents': {}}, created_at='2024-09-27T19:30:10.820758+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef7d06e-859f-6206-8003-e1bd3c264b8f'}}, tasks=())
```

```
snapshot.next  # (그래프가 이 차례에 종료되었으므로 `next`는 비어 있습니다. 그래프 호출 내에서 상태를 가져오면 next는 다음에 실행될 노드를 알려줍니다)
```

:::

:::js

지금까지 두 개의 서로 다른 thread에서 몇 가지 체크포인트를 만들었습니다. 그런데 체크포인트에는 무엇이 들어갈까요? 주어진 config에 대한 그래프의 `state`를 언제든지 검사하려면 `getState(config)`를 호출하세요.

```typescript
await graph.getState({ configurable: { thread_id: "1" } });
```

```typescript
{
  values: {
    messages: [
      HumanMessage {
        "id": "32fabcef-b3b8-481f-8bcb-fd83399a5f8d",
        "content": "Hi there! My name is Will.",
        "additional_kwargs": {},
        "response_metadata": {}
      },
      AIMessage {
        "id": "chatcmpl-BrPbTsCJbVqBvXWySlYoTJvM75Kv8",
        "content": "Hello Will! How can I assist you today?",
        "additional_kwargs": {},
        "response_metadata": {},
        "tool_calls": [],
        "invalid_tool_calls": []
      },
      HumanMessage {
        "id": "561c3aad-f8fc-4fac-94a6-54269a220856",
        "content": "Remember my name?",
        "additional_kwargs": {},
        "response_metadata": {}
      },
      AIMessage {
        "id": "chatcmpl-BrPbU4BhhsUikGbW37hYuF5vvnnE2",
        "content": "Yes, I remember your name, Will! How can I help you today?",
        "additional_kwargs": {},
        "response_metadata": {},
        "tool_calls": [],
        "invalid_tool_calls": []
      }
    ]
  },
  next: [],
  tasks: [],
  metadata: {
    source: 'loop',
    step: 4,
    parents: {},
    thread_id: '1'
  },
  config: {
    configurable: {
      thread_id: '1',
      checkpoint_id: '1f05cccc-9bb6-6270-8004-1d2108bcec77',
      checkpoint_ns: ''
    }
  },
  createdAt: '2025-07-09T13:58:27.607Z',
  parentConfig: {
    configurable: {
      thread_id: '1',
      checkpoint_ns: '',
      checkpoint_id: '1f05cccc-78fa-68d0-8003-ffb01a76b599'
    }
  }
}
```

```typescript
import * as assert from "node:assert";

// 그래프가 이 차례에 종료되었으므로 `next`는 비어 있습니다.
// 그래프 호출 내에서 상태를 가져오면 next는 다음에 실행될 노드를 알려줍니다)
assert.deepEqual(snapshot.next, []);
```

:::

위의 스냅샷에는 현재 상태 값, 해당 config 및 처리할 `next` 노드가 포함되어 있습니다. 우리의 경우 그래프가 `END` 상태에 도달했으므로 `next`는 비어 있습니다.

**축하합니다!** 이제 챗봇은 LangGraph의 체크포인팅 시스템 덕분에 세션 간에 대화 상태를 유지할 수 있습니다. 이는 더 자연스럽고 컨텍스트를 인식하는 상호작용을 위한 흥미로운 가능성을 열어줍니다. LangGraph의 체크포인팅은 **임의로 복잡한 그래프 상태**까지 처리할 수 있어 단순한 채팅 메모리보다 훨씬 더 표현력이 풍부하고 강력합니다.

이 튜토리얼의 그래프를 검토하려면 아래 코드 스니펫을 확인하세요:

:::python

{% include-markdown "../../../snippets/chat_model_tabs.md" %}

<!---
```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
```
-->

```python hl_lines="36 37"
from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

tool = TavilySearch(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```

:::

:::js

```typescript hl_lines="16 26"
import { END, MessagesZodState, START } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { TavilySearch } from "@langchain/tavily";

import { MemorySaver } from "@langchain/langgraph";
import { StateGraph } from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { z } from "zod";

const State = z.object({
  messages: MessagesZodState.shape.messages,
});

const tools = [new TavilySearch({ maxResults: 2 })];
const llm = new ChatOpenAI({ model: "gpt-4o-mini" }).bindTools(tools);
const memory = new MemorySaver();

async function generateText(content: string) {

const graph = new StateGraph(State)
  .addNode("chatbot", async (state) => ({
    messages: [await llm.invoke(state.messages)],
  }))
  .addNode("tools", new ToolNode(tools))
  .addConditionalEdges("chatbot", toolsCondition, ["tools", END])
  .addEdge("tools", "chatbot")
  .addEdge(START, "chatbot")
  .compile({ checkpointer: memory });
```

:::

## 다음 단계

다음 튜토리얼에서는 챗봇이 계속 진행하기 전에 안내나 검증이 필요할 수 있는 상황을 처리하기 위해 [챗봇에 human-in-the-loop를 추가](./4-human-in-the-loop.md)합니다.

