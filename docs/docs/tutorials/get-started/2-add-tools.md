# 도구 추가

챗봇이 "기억"만으로는 답할 수 없는 질문을 처리하려면 웹 검색 도구를 통합하세요. 챗봇은 이 도구를 사용하여 관련 정보를 찾고 더 나은 응답을 제공할 수 있습니다.

!!! note

    이 튜토리얼은 [기본 챗봇 만들기](./1-build-basic-chatbot.md)를 기반으로 합니다.

## 사전 요구 사항

이 튜토리얼을 시작하기 전에 다음이 있는지 확인하세요:

:::python

- [Tavily Search Engine](https://python.langchain.com/docs/integrations/tools/tavily_search/)용 API 키

:::

:::js

- [Tavily Search Engine](https://js.langchain.com/docs/integrations/tools/tavily_search/)용 API 키

:::

## 1. 검색 엔진 설치

:::python
[Tavily Search Engine](https://python.langchain.com/docs/integrations/tools/tavily_search/)을 사용하기 위한 요구 사항을 설치합니다:

```bash
pip install -U langchain-tavily
```

:::

:::js
[Tavily Search Engine](https://docs.tavily.com/)을 사용하기 위한 요구 사항을 설치합니다:

=== "npm"

    ```bash
    npm install @langchain/tavily
    ```

=== "yarn"

    ```bash
    yarn add @langchain/tavily
    ```

=== "pnpm"

    ```bash
    pnpm add @langchain/tavily
    ```

=== "bun"

    ```bash
    bun add @langchain/tavily
    ```

:::

## 2. 환경 구성

검색 엔진 API 키로 환경을 구성합니다:

:::python
```python
import os

os.environ["TAVILY_API_KEY"] = "tvly-..."
```
:::

:::js

```typescript
process.env.TAVILY_API_KEY = "tvly-...";
```

:::

## 3. 도구 정의

웹 검색 도구를 정의합니다:

:::python

```python
from langchain_tavily import TavilySearch

tool = TavilySearch(max_results=2)
tools = [tool]
tool.invoke("What's a 'node' in LangGraph?")
```

:::

:::js

```typescript
import { TavilySearch } from "@langchain/tavily";

const tool = new TavilySearch({ maxResults: 2 });
const tools = [tool];

await tool.invoke({ query: "What's a 'node' in LangGraph?" });
```

:::

결과는 챗봇이 질문에 답하는 데 사용할 수 있는 페이지 요약입니다:

:::python

```
{'query': "What's a 'node' in LangGraph?",
'follow_up_questions': None,
'answer': None,
'images': [],
'results': [{'title': "Introduction to LangGraph: A Beginner's Guide - Medium",
'url': 'https://medium.com/@cplog/introduction-to-langgraph-a-beginners-guide-14f9be027141',
'content': 'Stateful Graph: LangGraph revolves around the concept of a stateful graph, where each node in the graph represents a step in your computation, and the graph maintains a state that is passed around and updated as the computation progresses. LangGraph supports conditional edges, allowing you to dynamically determine the next node to execute based on the current state of the graph. We define nodes for classifying the input, handling greetings, and handling search queries. def classify_input_node(state): LangGraph is a versatile tool for building complex, stateful applications with LLMs. By understanding its core concepts and working through simple examples, beginners can start to leverage its power for their projects. Remember to pay attention to state management, conditional edges, and ensuring there are no dead-end nodes in your graph.',
'score': 0.7065353,
'raw_content': None},
{'title': 'LangGraph Tutorial: What Is LangGraph and How to Use It?',
'url': 'https://www.datacamp.com/tutorial/langgraph-tutorial',
'content': 'LangGraph is a library within the LangChain ecosystem that provides a framework for defining, coordinating, and executing multiple LLM agents (or chains) in a structured and efficient manner. By managing the flow of data and the sequence of operations, LangGraph allows developers to focus on the high-level logic of their applications rather than the intricacies of agent coordination. Whether you need a chatbot that can handle various types of user requests or a multi-agent system that performs complex tasks, LangGraph provides the tools to build exactly what you need. LangGraph significantly simplifies the development of complex LLM applications by providing a structured framework for managing state and coordinating agent interactions.',
'score': 0.5008063,
'raw_content': None}],
'response_time': 1.38}
```

:::

:::js

```json
{
  "query": "What's a 'node' in LangGraph?",
  "follow_up_questions": null,
  "answer": null,
  "images": [],
  "results": [
    {
      "url": "https://blog.langchain.dev/langgraph/",
      "title": "LangGraph - LangChain Blog",
      "content": "TL;DR: LangGraph is module built on top of LangChain to better enable creation of cyclical graphs, often needed for agent runtimes. This state is updated by nodes in the graph, which return operations to attributes of this state (in the form of a key-value store). After adding nodes, you can then add edges to create the graph. An example of this may be in the basic agent runtime, where we always want the model to be called after we call a tool. The state of this graph by default contains concepts that should be familiar to you if you've used LangChain agents: `input`, `chat_history`, `intermediate_steps` (and `agent_outcome` to represent the most recent agent outcome)",
      "score": 0.7407191,
      "raw_content": null
    },
    {
      "url": "https://medium.com/@cplog/introduction-to-langgraph-a-beginners-guide-14f9be027141",
      "title": "Introduction to LangGraph: A Beginner's Guide - Medium",
      "content": "*   **Stateful Graph:** LangGraph revolves around the concept of a stateful graph, where each node in the graph represents a step in your computation, and the graph maintains a state that is passed around and updated as the computation progresses. LangGraph supports conditional edges, allowing you to dynamically determine the next node to execute based on the current state of the graph. Image 10: Introduction to AI Agent with LangChain and LangGraph: A Beginner’s Guide Image 18: How to build LLM Agent with LangGraph — StateGraph and Reducer Image 20: Simplest Graphs using LangGraph Framework Image 24: Building a ReAct Agent with Langgraph: A Step-by-Step Guide Image 28: Building an Agentic RAG with LangGraph: A Step-by-Step Guide",
      "score": 0.65279555,
      "raw_content": null
    }
  ],
  "response_time": 1.34
}
```

:::

## 4. 그래프 정의

:::python
[첫 번째 튜토리얼](./1-build-basic-chatbot.md)에서 생성한 `StateGraph`에 대해 LLM에 `bind_tools`를 추가합니다. 이를 통해 LLM이 검색 엔진을 사용하려는 경우 사용할 올바른 JSON 형식을 알 수 있습니다.
:::

:::js
[첫 번째 튜토리얼](./1-build-basic-chatbot.md)에서 생성한 `StateGraph`에 대해 LLM에 `bindTools`를 추가합니다. 이를 통해 LLM이 검색 엔진을 사용하려는 경우 사용할 올바른 JSON 형식을 알 수 있습니다.
:::

먼저 LLM을 선택합니다:

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
import { ChatAnthropic } from "@langchain/anthropic";

const llm = new ChatAnthropic({ model: "claude-3-5-sonnet-latest" });
```

:::

이제 이를 `StateGraph`에 통합할 수 있습니다:

:::python

```python
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# 수정: LLM이 호출할 수 있는 도구를 알려줍니다
# highlight-next-line
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
```

:::

:::js

```typescript hl_lines="7-8"
import { StateGraph, MessagesZodState } from "@langchain/langgraph";
import { z } from "zod";

const State = z.object({ messages: MessagesZodState.shape.messages });

const chatbot = async (state: z.infer<typeof State>) => {
  // 수정: LLM이 호출할 수 있는 도구를 알려줍니다
  const llmWithTools = llm.bindTools(tools);

  return { messages: [await llmWithTools.invoke(state.messages)] };
};
```

:::

## 5. 도구를 실행하는 함수 생성

:::python

이제 도구가 호출되면 도구를 실행하는 함수를 생성합니다. 상태의 가장 최근 메시지를 확인하고 메시지에 `tool_calls`가 포함되어 있으면 도구를 호출하는 `BasicToolNode`라는 새 노드에 도구를 추가하여 이를 수행합니다. 이는 Anthropic, OpenAI, Google Gemini 및 기타 여러 LLM 제공업체에서 사용할 수 있는 LLM의 `tool_calling` 지원에 의존합니다.

```python
import json

from langchain_core.messages import ToolMessage


class BasicToolNode:
    """마지막 AIMessage에서 요청된 도구를 실행하는 노드입니다."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)
```

!!! note

    나중에 이를 직접 구축하고 싶지 않다면 LangGraph의 사전 빌드된 [ToolNode](https://langchain-ai.github.io/langgraph/reference/agents/#langgraph.prebuilt.tool_node.ToolNode)를 사용할 수 있습니다.

:::

:::js

이제 도구가 호출되면 도구를 실행하는 함수를 생성합니다. 상태의 가장 최근 메시지를 확인하고 메시지에 `tool_calls`가 포함되어 있으면 도구를 호출하는 `"tools"`라는 새 노드에 도구를 추가하여 이를 수행합니다. 이는 Anthropic, OpenAI, Google Gemini 및 기타 여러 LLM 제공업체에서 사용할 수 있는 LLM의 도구 호출 지원에 의존합니다.

```typescript
import type { StructuredToolInterface } from "@langchain/core/tools";
import { isAIMessage, ToolMessage } from "@langchain/core/messages";

function createToolNode(tools: StructuredToolInterface[]) {
  const toolByName: Record<string, StructuredToolInterface> = {};
  for (const tool of tools) {
    toolByName[tool.name] = tool;
  }

  return async (inputs: z.infer<typeof State>) => {
    const { messages } = inputs;
    if (!messages || messages.length === 0) {
      throw new Error("No message found in input");
    }

    const message = messages.at(-1);
    if (!message || !isAIMessage(message) || !message.tool_calls) {
      throw new Error("Last message is not an AI message with tool calls");
    }

    const outputs: ToolMessage[] = [];
    for (const toolCall of message.tool_calls) {
      if (!toolCall.id) throw new Error("Tool call ID is required");

      const tool = toolByName[toolCall.name];
      if (!tool) throw new Error(`Tool ${toolCall.name} not found`);

      const result = await tool.invoke(toolCall.args);

      outputs.push(
        new ToolMessage({
          content: JSON.stringify(result),
          name: toolCall.name,
          tool_call_id: toolCall.id,
        })
      );
    }

    return { messages: outputs };
  };
}
```

!!! note

    나중에 이를 직접 구축하고 싶지 않다면 LangGraph의 사전 빌드된 [ToolNode](https://langchain-ai.github.io/langgraphjs/reference/classes/langgraph_prebuilt.ToolNode.html)를 사용할 수 있습니다.

:::

## 6. `conditional_edges` 정의

도구 노드가 추가되면 이제 `conditional_edges`를 정의할 수 있습니다.

**Edges**는 한 노드에서 다음 노드로 제어 흐름을 라우팅합니다. **Conditional edges**는 단일 노드에서 시작하며 일반적으로 현재 그래프 상태에 따라 다른 노드로 라우팅하는 "if" 문을 포함합니다. 이러한 함수는 현재 그래프 `state`를 받아 다음에 호출할 노드를 나타내는 문자열 또는 문자열 목록을 반환합니다.

:::python
다음으로, 챗봇의 출력에서 `tool_calls`를 확인하는 `route_tools`라는 라우터 함수를 정의합니다. `add_conditional_edges`를 호출하여 그래프에 이 함수를 제공하면, `chatbot` 노드가 완료될 때마다 이 함수를 확인하여 다음에 어디로 갈지 결정하도록 그래프에 지시합니다.
:::

:::js
다음으로, 챗봇의 출력에서 `tool_calls`를 확인하는 `routeTools`라는 라우터 함수를 정의합니다. `addConditionalEdges`를 호출하여 그래프에 이 함수를 제공하면, `chatbot` 노드가 완료될 때마다 이 함수를 확인하여 다음에 어디로 갈지 결정하도록 그래프에 지시합니다.
:::

조건은 도구 호출이 있으면 `tools`로 라우팅하고 없으면 `END`로 라우팅합니다. 조건이 `END`를 반환할 수 있으므로 이번에는 명시적으로 `finish_point`를 설정할 필요가 없습니다.

:::python

```python
def route_tools(
    state: State,
):
    """
    마지막 메시지에 도구 호출이 있는 경우 ToolNode로 라우팅하기 위해
    conditional_edge에서 사용합니다. 그렇지 않으면 종료로 라우팅합니다.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# `tools_condition` 함수는 챗봇이 도구를 사용하려고 하면 "tools"를 반환하고,
# 직접 응답해도 괜찮으면 "END"를 반환합니다. 이 조건부 라우팅은 주요 에이전트 루프를 정의합니다.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # 다음 딕셔너리를 사용하면 조건의 출력을 특정 노드로 해석하도록 그래프에 지시할 수 있습니다
    # 기본값은 identity 함수이지만,
    # "tools"가 아닌 다른 이름의 노드를 사용하려면
    # 딕셔너리의 값을 다른 것으로 업데이트할 수 있습니다
    # 예: "tools": "my_tools"
    {"tools": "tools", END: END},
)
# 도구가 호출될 때마다 챗봇으로 돌아가서 다음 단계를 결정합니다
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()
```

!!! note

    더 간결하게 하려면 사전 빌드된 [tools_condition](https://langchain-ai.github.io/langgraph/reference/prebuilt/#tools_condition)으로 대체할 수 있습니다.

:::

:::js

```typescript
import { END, START } from "@langchain/langgraph";

const routeTools = (state: z.infer<typeof State>) => {
  /**
   * 마지막 메시지에 도구 호출이 있는 경우 ToolNode로 라우팅하기 위해
   * 조건부 엣지로 사용합니다.
   */
  const lastMessage = state.messages.at(-1);
  if (
    lastMessage &&
    isAIMessage(lastMessage) &&
    lastMessage.tool_calls?.length
  ) {
    return "tools";
  }

  /** 그렇지 않으면 종료로 라우팅합니다. */
  return END;
};

const graph = new StateGraph(State)
  .addNode("chatbot", chatbot)

  // `routeTools` 함수는 챗봇이 도구를 사용하려고 하면 "tools"를 반환하고,
  // 직접 응답해도 괜찮으면 "END"를 반환합니다. 이 조건부 라우팅은 주요 에이전트 루프를 정의합니다.
  .addNode("tools", createToolNode(tools))

  // 챗봇으로 그래프를 시작합니다
  .addEdge(START, "chatbot")

  // `routeTools` 함수는 챗봇이 도구를 사용하려고 하면 "tools"를 반환하고,
  // 직접 응답해도 괜찮으면 "END"를 반환합니다.
  .addConditionalEdges("chatbot", routeTools, ["tools", END])

  // 도구가 호출될 때마다 챗봇으로 돌아가야 합니다
  .addEdge("tools", "chatbot")
  .compile();
```

!!! note

    더 간결하게 하려면 사전 빌드된 [toolsCondition](https://langchain-ai.github.io/langgraphjs/reference/functions/langgraph_prebuilt.toolsCondition.html)으로 대체할 수 있습니다.

:::

## 7. 그래프 시각화 (선택사항)

:::python
`get_graph` 메서드와 `draw_ascii` 또는 `draw_png`와 같은 "draw" 메서드 중 하나를 사용하여 그래프를 시각화할 수 있습니다. `draw` 메서드는 각각 추가 종속성이 필요합니다.

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
`getGraph` 메서드를 사용하여 그래프를 시각화하고 `drawMermaidPng` 메서드로 그래프를 렌더링할 수 있습니다.

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await graph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("chatbot-with-tools.png", imageBuffer);
```

:::

![chatbot-with-tools-diagram](chatbot-with-tools.png)

## 8. 봇에게 질문하기

이제 챗봇에게 훈련 데이터 외부의 질문을 할 수 있습니다:

:::python

```python
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # input()을 사용할 수 없는 경우 대체
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
```

```
Assistant: [{'text': "To provide you with accurate and up-to-date information about LangGraph, I'll need to search for the latest details. Let me do that for you.", 'type': 'text'}, {'id': 'toolu_01Q588CszHaSvvP2MxRq9zRD', 'input': {'query': 'LangGraph AI tool information'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
Assistant: [{"url": "https://www.langchain.com/langgraph", "content": "LangGraph sets the foundation for how we can build and scale AI workloads \u2014 from conversational agents, complex task automation, to custom LLM-backed experiences that 'just work'. The next chapter in building complex production-ready features with LLMs is agentic, and with LangGraph and LangSmith, LangChain delivers an out-of-the-box solution ..."}, {"url": "https://github.com/langchain-ai/langgraph", "content": "Overview. LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. Compared to other LLM frameworks, it offers these core benefits: cycles, controllability, and persistence. LangGraph allows you to define flows that involve cycles, essential for most agentic architectures ..."}]
Assistant: Based on the search results, I can provide you with information about LangGraph:

1. Purpose:
   LangGraph is a library designed for building stateful, multi-actor applications with Large Language Models (LLMs). It's particularly useful for creating agent and multi-agent workflows.

2. Developer:
   LangGraph is developed by LangChain, a company known for its tools and frameworks in the AI and LLM space.

3. Key Features:
   - Cycles: LangGraph allows the definition of flows that involve cycles, which is essential for most agentic architectures.
   - Controllability: It offers enhanced control over the application flow.
   - Persistence: The library provides ways to maintain state and persistence in LLM-based applications.

4. Use Cases:
   LangGraph can be used for various applications, including:
   - Conversational agents
   - Complex task automation
   - Custom LLM-backed experiences

5. Integration:
   LangGraph works in conjunction with LangSmith, another tool by LangChain, to provide an out-of-the-box solution for building complex, production-ready features with LLMs.

6. Significance:
...
   LangGraph is noted to offer unique benefits compared to other LLM frameworks, particularly in its ability to handle cycles, provide controllability, and maintain persistence.

LangGraph appears to be a significant tool in the evolving landscape of LLM-based application development, offering developers new ways to create more complex, stateful, and interactive AI systems.
Goodbye!
```

:::

:::js

```typescript
import readline from "node:readline/promises";

const prompt = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

async function generateText(content: string) {
  const stream = await graph.stream(
    { messages: [{ type: "human", content }] },
    { streamMode: "values" }
  );

  for await (const event of stream) {
    const lastMessage = event.messages.at(-1);

    if (lastMessage?.getType() === "ai" || lastMessage?.getType() === "tool") {
      console.log(`Assistant: ${lastMessage?.text}`);
    }
  }
}

while (true) {
  const human = await prompt.question("User: ");
  if (["quit", "exit", "q"].includes(human.trim())) break;
  await generateText(human || "What do you know about LangGraph?");
}

prompt.close();
```

```
User: What do you know about LangGraph?
Assistant: I'll search for the latest information about LangGraph for you.
Assistant: [{"title":"Introduction to LangGraph: A Beginner's Guide - Medium","url":"https://medium.com/@cplog/introduction-to-langgraph-a-beginners-guide-14f9be027141","content":"..."}]
Assistant: Based on the search results, I can provide you with information about LangGraph:

LangGraph is a library within the LangChain ecosystem designed for building stateful, multi-actor applications with Large Language Models (LLMs). Here are the key aspects:

**Core Purpose:**
- LangGraph is specifically designed for creating agent and multi-agent workflows
- It provides a framework for defining, coordinating, and executing multiple LLM agents in a structured manner

**Key Features:**
1. **Stateful Graph Architecture**: LangGraph revolves around a stateful graph where each node represents a step in computation, and the graph maintains state that is passed around and updated as the computation progresses

2. **Conditional Edges**: It supports conditional edges, allowing you to dynamically determine the next node to execute based on the current state of the graph

3. **Cycles**: Unlike other LLM frameworks, LangGraph allows you to define flows that involve cycles, which is essential for most agentic architectures

4. **Controllability**: It offers enhanced control over the application flow

5. **Persistence**: The library provides ways to maintain state and persistence in LLM-based applications

**Use Cases:**
- Conversational agents
- Complex task automation
- Custom LLM-backed experiences
- Multi-agent systems that perform complex tasks

**Benefits:**
LangGraph allows developers to focus on the high-level logic of their applications rather than the intricacies of agent coordination, making it easier to build complex, production-ready features with LLMs.

This makes LangGraph a significant tool in the evolving landscape of LLM-based application development.
```

:::

## 9. 사전 빌드 사용

사용 편의성을 위해 코드를 조정하여 다음을 LangGraph 사전 빌드 컴포넌트로 교체합니다. 이들은 병렬 API 실행과 같은 내장 기능을 가지고 있습니다.

:::python

- `BasicToolNode`는 사전 빌드된 [ToolNode](https://langchain-ai.github.io/langgraph/reference/prebuilt/#toolnode)로 대체됩니다
- `route_tools`는 사전 빌드된 [tools_condition](https://langchain-ai.github.io/langgraph/reference/prebuilt/#tools_condition)으로 대체됩니다

{% include-markdown "../../../snippets/chat_model_tabs.md" %}

<!---
```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
```
-->

<!---
```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
```
-->

```python hl_lines="25 30"
from typing import Annotated

from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
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
# 도구가 호출될 때마다 챗봇으로 돌아가서 다음 단계를 결정합니다
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()
```

:::

:::js

- `createToolNode`는 사전 빌드된 [ToolNode](https://langchain-ai.github.io/langgraphjs/reference/classes/langgraph_prebuilt.ToolNode.html)로 대체됩니다
- `routeTools`는 사전 빌드된 [toolsCondition](https://langchain-ai.github.io/langgraphjs/reference/functions/langgraph_prebuilt.toolsCondition.html)으로 대체됩니다

```typescript
import { TavilySearch } from "@langchain/tavily";
import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, START, MessagesZodState, END } from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { z } from "zod";

const State = z.object({ messages: MessagesZodState.shape.messages });

const tools = [new TavilySearch({ maxResults: 2 })];

const llm = new ChatOpenAI({ model: "gpt-4o-mini" }).bindTools(tools);

const graph = new StateGraph(State)
  .addNode("chatbot", async (state) => ({
    messages: [await llm.invoke(state.messages)],
  }))
  .addNode("tools", new ToolNode(tools))
  .addConditionalEdges("chatbot", toolsCondition, ["tools", END])
  .addEdge("tools", "chatbot")
  .addEdge(START, "chatbot")
  .compile();
```

:::

**축하합니다!** 필요할 때 검색 엔진을 사용하여 업데이트된 정보를 검색할 수 있는 LangGraph 대화형 에이전트를 만들었습니다. 이제 더 광범위한 사용자 쿼리를 처리할 수 있습니다.

:::python

에이전트가 방금 수행한 모든 단계를 검사하려면 이 [LangSmith trace](https://smith.langchain.com/public/4fbd7636-25af-4638-9587-5a02fdbb0172/r)를 확인하세요.

:::

## 다음 단계

챗봇은 자체적으로 과거 상호작용을 기억할 수 없으므로 일관된 다중 턴 대화를 나누는 능력이 제한됩니다. 다음 파트에서는 이를 해결하기 위해 [**메모리**를 추가](./3-add-memory.md)합니다.
