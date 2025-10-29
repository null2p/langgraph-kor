# 사람이 개입하는 제어 추가

에이전트는 신뢰할 수 없을 수 있으며 작업을 성공적으로 수행하기 위해 사람의 입력이 필요할 수 있습니다. 마찬가지로, 일부 작업의 경우 모든 것이 의도한 대로 실행되고 있는지 확인하기 위해 실행 전에 사람의 승인이 필요할 수 있습니다.

LangGraph의 [지속성](../../concepts/persistence.md) 레이어는 **사람이 개입하는(human-in-the-loop)** 워크플로를 지원하여 사용자 피드백에 따라 실행을 일시 중지하고 재개할 수 있습니다. 이 기능의 주요 인터페이스는 [`interrupt`](../../how-tos/human_in_the_loop/add-human-in-the-loop.md) 함수입니다. 노드 내에서 `interrupt`를 호출하면 실행이 일시 중지됩니다. [Command](../../concepts/low_level.md#command)를 전달하여 사람의 새 입력과 함께 실행을 재개할 수 있습니다.

:::python
`interrupt`는 Python의 내장 `input()`과 인체공학적으로 유사하지만 [몇 가지 주의 사항](../../how-tos/human_in_the_loop/add-human-in-the-loop.md)이 있습니다.
:::

:::js
`interrupt`는 Node.js의 내장 `readline.question()` 함수와 인체공학적으로 유사하지만 [몇 가지 주의 사항](../../how-tos/human_in_the_loop/add-human-in-the-loop.md)이 있습니다.
:::

!!! note

    이 튜토리얼은 [메모리 추가](./3-add-memory.md)를 기반으로 합니다.

## 1. `human_assistance` 도구 추가

[챗봇에 메모리 추가](./3-add-memory.md) 튜토리얼의 기존 코드에서 시작하여 챗봇에 `human_assistance` 도구를 추가합니다. 이 도구는 `interrupt`를 사용하여 사람으로부터 정보를 받습니다.

먼저 챗 모델을 선택합니다:

:::python
{% include-markdown "../../../snippets/chat_model_tabs.md" %}

<!---
```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
```
-->

:::

:::js

```typescript
// 여기에 API 키를 추가하세요
process.env.ANTHROPIC_API_KEY = "YOUR_API_KEY";
```

:::

이제 추가 도구와 함께 `StateGraph`에 통합할 수 있습니다:

:::python

```python hl_lines="12 19 20 21 22 23"
from typing import Annotated

from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.types import Command, interrupt

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

@tool
def human_assistance(query: str) -> str:
    """사람에게 도움을 요청합니다."""
    human_response = interrupt({"query": query})
    return human_response["data"]

tool = TavilySearch(max_results=2)
tools = [tool, human_assistance]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # 도구 실행 중에 중단될 것이므로,
    # 재개할 때 도구 호출이 반복되지 않도록
    # 병렬 도구 호출을 비활성화합니다.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
```

:::

:::js

```typescript hl_lines="1 7-19"
import { interrupt, MessagesZodState } from "@langchain/langgraph";
import { ChatAnthropic } from "@langchain/anthropic";
import { TavilySearch } from "@langchain/tavily";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const humanAssistance = tool(
  async ({ query }) => {
    const humanResponse = interrupt({ query });
    return humanResponse.data;
  },
  {
    name: "humanAssistance",
    description: "사람에게 도움을 요청합니다.",
    schema: z.object({
      query: z.string().describe("사람을 위한 인간이 읽을 수 있는 질문"),
    }),
  }
);

const searchTool = new TavilySearch({ maxResults: 2 });
const searchTool = new TavilySearch({ maxResults: 2 });
const tools = [searchTool, humanAssistance];

const llmWithTools = new ChatAnthropic({
  model: "claude-3-5-sonnet-latest",
}).bindTools(tools);
const llmWithTools = new ChatAnthropic({
  model: "claude-3-5-sonnet-latest",
}).bindTools(tools);

async function chatbot(state: z.infer<typeof MessagesZodState>) {
async function chatbot(state: z.infer<typeof MessagesZodState>) {
  const message = await llmWithTools.invoke(state.messages);


  // 도구 실행 중에 중단될 것이므로,
  // 재개할 때 도구 호출이 반복되지 않도록
  // 병렬 도구 호출을 비활성화합니다.
  if (message.tool_calls && message.tool_calls.length > 1) {
    throw new Error("Multiple tool calls not supported with interrupts");
  }

  return { messages: message };
}
```

:::

!!! tip

    사람이 개입하는 워크플로에 대한 자세한 정보와 예제는 [사람이 개입하는 루프](../../concepts/human_in_the_loop.md)를 참조하세요.

## 2. 그래프 컴파일

이전과 마찬가지로 체크포인터로 그래프를 컴파일합니다:

:::python

```python
memory = InMemorySaver()

graph = graph_builder.compile(checkpointer=memory)
```

:::

:::js

```typescript hl_lines="3 11"
import { StateGraph, MemorySaver, START, END } from "@langchain/langgraph";

const memory = new MemorySaver();

const graph = new StateGraph(MessagesZodState)
  .addNode("chatbot", chatbot)
  .addNode("tools", new ToolNode(tools))
  .addConditionalEdges("chatbot", toolsCondition, ["tools", END])
  .addEdge("tools", "chatbot")
  .addEdge(START, "chatbot")
  .compile({ checkpointer: memory });
const graph = new StateGraph(MessagesZodState)
  .addNode("chatbot", chatbot)
  .addNode("tools", new ToolNode(tools))
  .addConditionalEdges("chatbot", toolsCondition, ["tools", END])
  .addEdge("tools", "chatbot")
  .addEdge(START, "chatbot")
  .compile({ checkpointer: memory });
```

:::

## 3. 그래프 시각화 (선택사항)

그래프를 시각화하면 추가된 도구와 함께 이전과 동일한 레이아웃을 얻을 수 있습니다!

:::python

```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # 이것은 추가 종속성이 필요하며 선택사항입니다
    pass
```

:::

:::js

```typescript
import * as fs from "node:fs/promises";
import * as fs from "node:fs/promises";

const drawableGraph = await graph.getGraphAsync();
const drawableGraph = await graph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("chatbot-with-tools.png", imageBuffer);
await fs.writeFile("chatbot-with-tools.png", imageBuffer);
```

:::

![chatbot-with-tools-diagram](chatbot-with-tools.png)

## 4. 챗봇에 프롬프트 전달

이제 새 `human_assistance` 도구를 사용하게 될 질문으로 챗봇에 프롬프트를 전달합니다:

:::python

```python
user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

```
================================ Human Message =================================

I need some expert guidance for building an AI agent. Could you request assistance for me?
================================== Ai Message ==================================

[{'text': "Certainly! I'd be happy to request expert assistance for you regarding building an AI agent. To do this, I'll use the human_assistance function to relay your request. Let me do that for you now.", 'type': 'text'}, {'id': 'toolu_01ABUqneqnuHNuo1vhfDFQCW', 'input': {'query': 'A user is requesting expert guidance for building an AI agent. Could you please provide some expert advice or resources on this topic?'}, 'name': 'human_assistance', 'type': 'tool_use'}]
Tool Calls:
  human_assistance (toolu_01ABUqneqnuHNuo1vhfDFQCW)
 Call ID: toolu_01ABUqneqnuHNuo1vhfDFQCW
  Args:
    query: A user is requesting expert guidance for building an AI agent. Could you please provide some expert advice or resources on this topic?
```

:::

:::js

```typescript
import { isAIMessage } from "@langchain/core/messages";

const userInput =
  "I need some expert guidance for building an AI agent. Could you request assistance for me?";

const events = await graph.stream(
  { messages: [{ role: "user", content: userInput }] },
  { configurable: { thread_id: "1" }, streamMode: "values" }
  { configurable: { thread_id: "1" }, streamMode: "values" }
);

for await (const event of events) {
  if ("messages" in event) {
    const lastMessage = event.messages.at(-1);
    console.log(`[${lastMessage?.getType()}]: ${lastMessage?.text}`);

    if (
      lastMessage &&
      isAIMessage(lastMessage) &&
      lastMessage.tool_calls?.length
    ) {
    const lastMessage = event.messages.at(-1);
    console.log(`[${lastMessage?.getType()}]: ${lastMessage?.text}`);

    if (
      lastMessage &&
      isAIMessage(lastMessage) &&
      lastMessage.tool_calls?.length
    ) {
      console.log("Tool calls:", lastMessage.tool_calls);
    }
  }
}
```

```
[human]: I need some expert guidance for building an AI agent. Could you request assistance for me?
[ai]: I'll help you request human assistance for guidance on building an AI agent.
[ai]: I'll help you request human assistance for guidance on building an AI agent.
Tool calls: [
  {
    name: 'humanAssistance',
    args: {
      query: 'I would like expert guidance on building an AI agent. Could you please provide assistance with this topic?'
      query: 'I would like expert guidance on building an AI agent. Could you please provide assistance with this topic?'
    },
    id: 'toolu_01Bpxc8rFVMhSaRosS6b85Ts',
    type: 'tool_call'
    id: 'toolu_01Bpxc8rFVMhSaRosS6b85Ts',
    type: 'tool_call'
  }
]
```

:::

챗봇이 도구 호출을 생성했지만 실행이 중단되었습니다. 그래프 상태를 검사하면 tools 노드에서 중지된 것을 볼 수 있습니다:

:::python

```python
snapshot = graph.get_state(config)
snapshot.next
```

```
('tools',)
```

:::

:::js

```typescript
const snapshot = await graph.getState({ configurable: { thread_id: "1" } });
snapshot.next;
const snapshot = await graph.getState({ configurable: { thread_id: "1" } });
snapshot.next;
```

```json
["tools"]
```

:::

!!! info 추가 정보

    :::python

    `human_assistance` 도구를 자세히 살펴보세요:

    ```python
    @tool
    def human_assistance(query: str) -> str:
        """사람에게 도움을 요청합니다."""
        human_response = interrupt({"query": query})
        return human_response["data"]
    ```

    Python의 내장 `input()` 함수와 유사하게, 도구 내에서 `interrupt`를 호출하면 실행이 일시 중지됩니다. 진행 상황은 [체크포인터](../../concepts/persistence.md#checkpointer-libraries)를 기반으로 저장됩니다. 따라서 Postgres로 저장하는 경우 데이터베이스가 살아있는 한 언제든지 재개할 수 있습니다. 이 예제에서는 인메모리 체크포인터로 저장하며 Python 커널이 실행 중이면 언제든지 재개할 수 있습니다.
    :::

    :::js

    `humanAssistance` 도구를 자세히 살펴보세요:

    ```typescript hl_lines="3"
    const humanAssistance = tool(
      async ({ query }) => {
        const humanResponse = interrupt({ query });
        return humanResponse.data;
      },
      {
        name: "humanAssistance",
        description: "Request assistance from a human.",
        schema: z.object({
          query: z.string().describe("Human readable question for the human"),
        }),
      },
    );

    Take a closer look at the `humanAssistance` tool:

    ```typescript hl_lines="3"
    const humanAssistance = tool(
      async ({ query }) => {
        const humanResponse = interrupt({ query });
        return humanResponse.data;
      },
      {
        name: "humanAssistance",
        description: "Request assistance from a human.",
        schema: z.object({
          query: z.string().describe("Human readable question for the human"),
        }),
      },
    );
    ```

    도구 내에서 `interrupt`를 호출하면 실행이 일시 중지됩니다. 진행 상황은 [체크포인터](../../concepts/persistence.md#checkpointer-libraries)를 기반으로 저장됩니다. 따라서 Postgres로 저장하는 경우 데이터베이스가 살아있는 한 언제든지 재개할 수 있습니다. 이 예제에서는 인메모리 체크포인터로 저장하며 JavaScript 런타임이 실행 중이면 언제든지 재개할 수 있습니다.
    :::

## 5. 실행 재개

실행을 재개하려면 도구에서 예상하는 데이터를 포함하는 [`Command`](../../concepts/low_level.md#command) 객체를 전달합니다. 이 데이터의 형식은 필요에 따라 사용자 정의할 수 있습니다.

:::python

이 예제에서는 키가 `"data"`인 딕셔너리를 사용합니다:

```python
human_response = (
    "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
    " It's much more reliable and extensible than simple autonomous agents."
)

human_command = Command(resume={"data": human_response})

events = graph.stream(human_command, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

```
================================== Ai Message ==================================

[{'text': "Certainly! I'd be happy to request expert assistance for you regarding building an AI agent. To do this, I'll use the human_assistance function to relay your request. Let me do that for you now.", 'type': 'text'}, {'id': 'toolu_01ABUqneqnuHNuo1vhfDFQCW', 'input': {'query': 'A user is requesting expert guidance for building an AI agent. Could you please provide some expert advice or resources on this topic?'}, 'name': 'human_assistance', 'type': 'tool_use'}]
Tool Calls:
  human_assistance (toolu_01ABUqneqnuHNuo1vhfDFQCW)
 Call ID: toolu_01ABUqneqnuHNuo1vhfDFQCW
  Args:
    query: A user is requesting expert guidance for building an AI agent. Could you please provide some expert advice or resources on this topic?
================================= Tool Message =================================
Name: human_assistance

We, the experts are here to help! We'd recommend you check out LangGraph to build your agent. It's much more reliable and extensible than simple autonomous agents.
================================== Ai Message ==================================

Thank you for your patience. I've received some expert advice regarding your request for guidance on building an AI agent. Here's what the experts have suggested:

The experts recommend that you look into LangGraph for building your AI agent. They mention that LangGraph is a more reliable and extensible option compared to simple autonomous agents.

LangGraph is likely a framework or library designed specifically for creating AI agents with advanced capabilities. Here are a few points to consider based on this recommendation:

1. Reliability: The experts emphasize that LangGraph is more reliable than simpler autonomous agent approaches. This could mean it has better stability, error handling, or consistent performance.

2. Extensibility: LangGraph is described as more extensible, which suggests that it probably offers a flexible architecture that allows you to easily add new features or modify existing ones as your agent's requirements evolve.

3. Advanced capabilities: Given that it's recommended over "simple autonomous agents," LangGraph likely provides more sophisticated tools and techniques for building complex AI agents.
...
2. Look for tutorials or guides specifically focused on building AI agents with LangGraph.
3. Check if there are any community forums or discussion groups where you can ask questions and get support from other developers using LangGraph.

If you'd like more specific information about LangGraph or have any questions about this recommendation, please feel free to ask, and I can request further assistance from the experts.
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```

:::

:::js
이 예제에서는 키가 `"data"`인 객체를 사용합니다:

```typescript
import { Command } from "@langchain/langgraph";

const humanResponse =
  "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent." +
  " It's much more reliable and extensible than simple autonomous agents.";
(" It's much more reliable and extensible than simple autonomous agents.");

const humanCommand = new Command({ resume: { data: humanResponse } });

const resumeEvents = await graph.stream(humanCommand, {
  configurable: { thread_id: "1" },
  streamMode: "values",
});
const resumeEvents = await graph.stream(humanCommand, {
  configurable: { thread_id: "1" },
  streamMode: "values",
});

for await (const event of resumeEvents) {
  if ("messages" in event) {
    const lastMessage = event.messages.at(-1);
    console.log(`[${lastMessage?.getType()}]: ${lastMessage?.text}`);
    const lastMessage = event.messages.at(-1);
    console.log(`[${lastMessage?.getType()}]: ${lastMessage?.text}`);
  }
}
```

```
[tool]: We, the experts are here to help! We'd recommend you check out LangGraph to build your agent. It's much more reliable and extensible than simple autonomous agents.
[ai]: Thank you for your patience. I've received some expert advice regarding your request for guidance on building an AI agent. Here's what the experts have suggested:

The experts recommend that you look into LangGraph for building your AI agent. They mention that LangGraph is a more reliable and extensible option compared to simple autonomous agents.

LangGraph is likely a framework or library designed specifically for creating AI agents with advanced capabilities. Here are a few points to consider based on this recommendation:

1. Reliability: The experts emphasize that LangGraph is more reliable than simpler autonomous agent approaches. This could mean it has better stability, error handling, or consistent performance.

2. Extensibility: LangGraph is described as more extensible, which suggests that it probably offers a flexible architecture that allows you to easily add new features or modify existing ones as your agent's requirements evolve.

3. Advanced capabilities: Given that it's recommended over "simple autonomous agents," LangGraph likely provides more sophisticated tools and techniques for building complex AI agents.

...
```

:::

입력이 수신되어 도구 메시지로 처리되었습니다. 이 호출의 [LangSmith trace](https://smith.langchain.com/public/9f0f87e3-56a7-4dde-9c76-b71675624e91/r)를 검토하여 위 호출에서 수행된 정확한 작업을 확인하세요. 챗봇이 중단된 지점에서 계속할 수 있도록 첫 번째 단계에서 상태가 로드되는 것을 확인하세요.

**축하합니다!** `interrupt`를 사용하여 챗봇에 사람이 개입하는 실행을 추가하여 필요할 때 사람의 감독과 개입을 허용했습니다. 이는 AI 시스템으로 만들 수 있는 잠재적인 UI의 가능성을 열어줍니다. 이미 **체크포인터**를 추가했으므로 기본 지속성 레이어가 실행 중인 한 그래프를 **무기한** 일시 중지하고 언제든지 아무 일도 없었던 것처럼 재개할 수 있습니다.

이 튜토리얼의 그래프를 검토하려면 아래 코드 스니펫을 확인하세요:

:::python

{% include-markdown "../../../snippets/chat_model_tabs.md" %}

```python
from typing import Annotated

from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

tool = TavilySearch(max_results=2)
tools = [tool, human_assistance]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert(len(message.tool_calls) <= 1)
    return {"messages": [message]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```

:::

:::js

```typescript
import {
  interrupt,
  MessagesZodState,
  StateGraph,
  MemorySaver,
  START,
  END,
} from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { isAIMessage } from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import { TavilySearch } from "@langchain/tavily";
import {
  interrupt,
  MessagesZodState,
  StateGraph,
  MemorySaver,
  START,
  END,
} from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { isAIMessage } from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import { TavilySearch } from "@langchain/tavily";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const humanAssistance = tool(
  async ({ query }) => {
    const humanResponse = interrupt({ query });
    return humanResponse.data;
  },
  {
    name: "humanAssistance",
    description: "사람에게 도움을 요청합니다.",
    schema: z.object({
      query: z.string().describe("사람을 위한 인간이 읽을 수 있는 질문"),
    }),
  }
);
const humanAssistance = tool(
  async ({ query }) => {
    const humanResponse = interrupt({ query });
    return humanResponse.data;
  },
  {
    name: "humanAssistance",
    description: "사람에게 도움을 요청합니다.",
    schema: z.object({
      query: z.string().describe("사람을 위한 인간이 읽을 수 있는 질문"),
    }),
  }
);

const searchTool = new TavilySearch({ maxResults: 2 });
const searchTool = new TavilySearch({ maxResults: 2 });
const tools = [searchTool, humanAssistance];

const llmWithTools = new ChatAnthropic({
  model: "claude-3-5-sonnet-latest",
}).bindTools(tools);
const llmWithTools = new ChatAnthropic({
  model: "claude-3-5-sonnet-latest",
}).bindTools(tools);

const chatbot = async (state: z.infer<typeof MessagesZodState>) => {
const chatbot = async (state: z.infer<typeof MessagesZodState>) => {
  const message = await llmWithTools.invoke(state.messages);

  // 도구 실행 중에 중단될 것이므로,
  // 재개할 때 도구 호출이 반복되지 않도록
  // 병렬 도구 호출을 비활성화합니다.

  // 도구 실행 중에 중단될 것이므로,
  // 재개할 때 도구 호출이 반복되지 않도록
  // 병렬 도구 호출을 비활성화합니다.
  if (message.tool_calls && message.tool_calls.length > 1) {
    throw new Error("Multiple tool calls not supported with interrupts");
  }

  return { messages: message };

  return { messages: message };
};

const memory = new MemorySaver();

const graph = new StateGraph(MessagesZodState)
  .addNode("chatbot", chatbot)
  .addNode("tools", new ToolNode(tools))
  .addConditionalEdges("chatbot", toolsCondition, ["tools", END])
  .addEdge("tools", "chatbot")
  .addEdge(START, "chatbot")
  .compile({ checkpointer: memory });

const graph = new StateGraph(MessagesZodState)
  .addNode("chatbot", chatbot)
  .addNode("tools", new ToolNode(tools))
  .addConditionalEdges("chatbot", toolsCondition, ["tools", END])
  .addEdge("tools", "chatbot")
  .addEdge(START, "chatbot")
  .compile({ checkpointer: memory });
```

:::

## 다음 단계

지금까지 튜토리얼 예제는 하나의 항목이 있는 간단한 상태(메시지 목록)에 의존했습니다. 이 간단한 상태로도 많은 것을 할 수 있지만 메시지 목록에 의존하지 않고 복잡한 동작을 정의하려는 경우 [상태에 추가 필드를 추가](./5-customize-state.md)할 수 있습니다.
