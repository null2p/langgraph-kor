# 시간 여행

일반적인 챗봇 워크플로우에서 사용자는 작업을 수행하기 위해 봇과 한 번 이상 상호작용합니다. [메모리](./3-add-memory.md)와 [human-in-the-loop](./4-human-in-the-loop.md)는 그래프 상태에서 체크포인트를 활성화하고 향후 응답을 제어합니다.

사용자가 이전 응답에서 시작하여 다른 결과를 탐색할 수 있도록 하고 싶다면 어떻게 해야 할까요? 또는 자율 소프트웨어 엔지니어와 같은 애플리케이션에서 일반적으로 발생하는 실수를 수정하거나 다른 전략을 시도하기 위해 챗봇의 작업을 되감을 수 있도록 하려면 어떻게 해야 할까요?

LangGraph의 내장 **시간 여행** 기능을 사용하여 이러한 유형의 경험을 만들 수 있습니다.

!!! note

    이 튜토리얼은 [상태 커스터마이즈](./5-customize-state.md)를 기반으로 합니다.

## 1. 그래프 되감기

:::python
그래프의 `get_state_history` 메서드를 사용하여 체크포인트를 가져와서 그래프를 되감습니다. 그런 다음 이전 시점에서 실행을 재개할 수 있습니다.
:::

:::js
그래프의 `getStateHistory` 메서드를 사용하여 체크포인트를 가져와서 그래프를 되감습니다. 그런 다음 이전 시점에서 실행을 재개할 수 있습니다.
:::

:::python

{% include-markdown "../../../snippets/chat_model_tabs.md" %}

<!---
```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
```
-->

```python
from typing import Annotated

from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
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
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```

:::

:::js

```typescript
import {
  StateGraph,
  START,
  END,
  MessagesZodState,
  MemorySaver,
} from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { TavilySearch } from "@langchain/tavily";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

const State = z.object({ messages: MessagesZodState.shape.messages });

const tools = [new TavilySearch({ maxResults: 2 })];
const llmWithTools = new ChatOpenAI({ model: "gpt-4o-mini" }).bindTools(tools);
const memory = new MemorySaver();

const graph = new StateGraph(State)
  .addNode("chatbot", async (state) => ({
    messages: [await llmWithTools.invoke(state.messages)],
  }))
  .addNode("tools", new ToolNode(tools))
  .addConditionalEdges("chatbot", toolsCondition, ["tools", END])
  .addEdge("tools", "chatbot")
  .addEdge(START, "chatbot")
  .compile({ checkpointer: memory });
```

:::

## 2. 단계 추가

그래프에 단계를 추가합니다. 모든 단계는 상태 히스토리에 체크포인트됩니다:

:::python

```python
config = {"configurable": {"thread_id": "1"}}
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm learning LangGraph. "
                    "Could you do some research on it for me?"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

```
================================ Human Message =================================

I'm learning LangGraph. Could you do some research on it for me?
================================== Ai Message ==================================

[{'text': "Certainly! I'd be happy to research LangGraph for you. To get the most up-to-date and accurate information, I'll use the Tavily search engine to look this up. Let me do that for you now.", 'type': 'text'}, {'id': 'toolu_01BscbfJJB9EWJFqGrN6E54e', 'input': {'query': 'LangGraph latest information and features'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
Tool Calls:
  tavily_search_results_json (toolu_01BscbfJJB9EWJFqGrN6E54e)
 Call ID: toolu_01BscbfJJB9EWJFqGrN6E54e
  Args:
    query: LangGraph latest information and features
================================= Tool Message =================================
Name: tavily_search_results_json

[{"url": "https://blockchain.news/news/langchain-new-features-upcoming-events-update", "content": "LangChain, a leading platform in the AI development space, has released its latest updates, showcasing new use cases and enhancements across its ecosystem. According to the LangChain Blog, the updates cover advancements in LangGraph Platform, LangSmith's self-improving evaluators, and revamped documentation for LangGraph."}, {"url": "https://blog.langchain.dev/langgraph-platform-announce/", "content": "With these learnings under our belt, we decided to couple some of our latest offerings under LangGraph Platform. LangGraph Platform today includes LangGraph Server, LangGraph Studio, plus the CLI and SDK. ... we added features in LangGraph Server to deliver on a few key value areas. Below, we'll focus on these aspects of LangGraph Platform."}]
================================== Ai Message ==================================

Thank you for your patience. I've found some recent information about LangGraph for you. Let me summarize the key points:

1. LangGraph is part of the LangChain ecosystem, which is a leading platform in AI development.

2. Recent updates and features of LangGraph include:

   a. LangGraph Platform: This seems to be a cloud-based version of LangGraph, though specific details weren't provided in the search results.
...
3. Keep an eye on LangGraph Platform developments, as cloud-based solutions often provide an easier starting point for learners.
4. Consider how LangGraph fits into the broader LangChain ecosystem, especially its interaction with tools like LangSmith.

Is there any specific aspect of LangGraph you'd like to know more about? I'd be happy to do a more focused search on particular features or use cases.
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```

```python
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Ya that's helpful. Maybe I'll "
                    "build an autonomous agent with it!"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

```
================================ Human Message =================================

Ya that's helpful. Maybe I'll build an autonomous agent with it!
================================== Ai Message ==================================

[{'text': "That's an exciting idea! Building an autonomous agent with LangGraph is indeed a great application of this technology. LangGraph is particularly well-suited for creating complex, multi-step AI workflows, which is perfect for autonomous agents. Let me gather some more specific information about using LangGraph for building autonomous agents.", 'type': 'text'}, {'id': 'toolu_01QWNHhUaeeWcGXvA4eHT7Zo', 'input': {'query': 'Building autonomous agents with LangGraph examples and tutorials'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
Tool Calls:
  tavily_search_results_json (toolu_01QWNHhUaeeWcGXvA4eHT7Zo)
 Call ID: toolu_01QWNHhUaeeWcGXvA4eHT7Zo
  Args:
    query: Building autonomous agents with LangGraph examples and tutorials
================================= Tool Message =================================
Name: tavily_search_results_json

[{"url": "https://towardsdatascience.com/building-autonomous-multi-tool-agents-with-gemini-2-0-and-langgraph-ad3d7bd5e79d", "content": "Building Autonomous Multi-Tool Agents with Gemini 2.0 and LangGraph | by Youness Mansar | Jan, 2025 | Towards Data Science Building Autonomous Multi-Tool Agents with Gemini 2.0 and LangGraph A practical tutorial with full code examples for building and running multi-tool agents Towards Data Science LLMs are remarkable — they can memorize vast amounts of information, answer general knowledge questions, write code, generate stories, and even fix your grammar. In this tutorial, we are going to build a simple LLM agent that is equipped with four tools that it can use to answer a user's question. This Agent will have the following specifications: Follow Published in Towards Data Science --------------------------------- Your home for data science and AI. Follow Follow Follow"}, {"url": "https://github.com/anmolaman20/Tools_and_Agents", "content": "GitHub - anmolaman20/Tools_and_Agents: This repository provides resources for building AI agents using Langchain and Langgraph. This repository provides resources for building AI agents using Langchain and Langgraph. This repository provides resources for building AI agents using Langchain and Langgraph. This repository serves as a comprehensive guide for building AI-powered agents using Langchain and Langgraph. It provides hands-on examples, practical tutorials, and resources for developers and AI enthusiasts to master building intelligent systems and workflows. AI Agent Development: Gain insights into creating intelligent systems that think, reason, and adapt in real time. This repository is ideal for AI practitioners, developers exploring language models, or anyone interested in building intelligent systems. This repository provides resources for building AI agents using Langchain and Langgraph."}]
================================== Ai Message ==================================

Great idea! Building an autonomous agent with LangGraph is definitely an exciting project. Based on the latest information I've found, here are some insights and tips for building autonomous agents with LangGraph:

1. Multi-Tool Agents: LangGraph is particularly well-suited for creating autonomous agents that can use multiple tools. This allows your agent to have a diverse set of capabilities and choose the right tool for each task.

2. Integration with Large Language Models (LLMs): You can combine LangGraph with powerful LLMs like Gemini 2.0 to create more intelligent and capable agents. The LLM can serve as the "brain" of your agent, making decisions and generating responses.

3. Workflow Management: LangGraph excels at managing complex, multi-step AI workflows. This is crucial for autonomous agents that need to break down tasks into smaller steps and execute them in the right order.
...
6. Pay attention to how you structure the agent's decision-making process and workflow.
7. Don't forget to implement proper error handling and safety measures, especially if your agent will be interacting with external systems or making important decisions.

Building an autonomous agent is an iterative process, so be prepared to refine and improve your agent over time. Good luck with your project! If you need any more specific information as you progress, feel free to ask.
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```

:::

:::js

```typescript
import { randomUUID } from "node:crypto";
const threadId = randomUUID();

let iter = 0;

for (const userInput of [
  "I'm learning LangGraph. Could you do some research on it for me?",
  "Ya that's helpful. Maybe I'll build an autonomous agent with it!",
]) {
  iter += 1;

  console.log(`\n--- Conversation Turn ${iter} ---\n`);
  const events = await graph.stream(
    { messages: [{ role: "user", content: userInput }] },
    { configurable: { thread_id: threadId }, streamMode: "values" }
  );

  for await (const event of events) {
    if ("messages" in event) {
      const lastMessage = event.messages.at(-1);

      console.log(
        "=".repeat(32),
        `${lastMessage?.getType()} Message`,
        "=".repeat(32)
      );
      console.log(lastMessage?.text);
    }
  }
}
```

```
--- Conversation Turn 1 ---

================================ human Message ================================
I'm learning LangGraph.js. Could you do some research on it for me?
================================ ai Message ================================
I'll search for information about LangGraph.js for you.
================================ tool Message ================================
{
  "query": "LangGraph.js framework TypeScript langchain what is it tutorial guide",
  "follow_up_questions": null,
  "answer": null,
  "images": [],
  "results": [
    {
      "url": "https://techcommunity.microsoft.com/blog/educatordeveloperblog/an-absolute-beginners-guide-to-langgraph-js/4212496",
      "title": "An Absolute Beginner's Guide to LangGraph.js",
      "content": "(...)",
      "score": 0.79369855,
      "raw_content": null
    },
    {
      "url": "https://langchain-ai.github.io/langgraphjs/",
      "title": "LangGraph.js",
      "content": "(...)",
      "score": 0.78154784,
      "raw_content": null
    }
  ],
  "response_time": 2.37
}
================================ ai Message ================================
Let me provide you with an overview of LangGraph.js based on the search results:

LangGraph.js is a JavaScript/TypeScript library that's part of the LangChain ecosystem, specifically designed for creating and managing complex LLM (Large Language Model) based workflows. Here are the key points about LangGraph.js:

1. Purpose:
- It's a low-level orchestration framework for building controllable agents
- Particularly useful for creating agentic workflows where LLMs decide the course of action based on current state
- Helps model workflows as graphs with nodes and edges

(...)

--- Conversation Turn 2 ---

================================ human Message ================================
Ya that's helpful. Maybe I'll build an autonomous agent with it!
================================ ai Message ================================
Let me search for specific information about building autonomous agents with LangGraph.js.
================================ tool Message ================================
{
  "query": "how to build autonomous agents with LangGraph.js examples tutorial react agent",
  "follow_up_questions": null,
  "answer": null,
  "images": [],
  "results": [
    {
      "url": "https://ai.google.dev/gemini-api/docs/langgraph-example",
      "title": "ReAct agent from scratch with Gemini 2.5 and LangGraph",
      "content": "(...)",
      "score": 0.7602419,
      "raw_content": null
    },
    {
      "url": "https://www.youtube.com/watch?v=ZfjaIshGkmk",
      "title": "Build Autonomous AI Agents with ReAct and LangGraph Tools",
      "content": "(...)",
      "score": 0.7471924,
      "raw_content": null
    }
  ],
  "response_time": 1.98
}
================================ ai Message ================================
Based on the search results, I can provide you with a practical overview of how to build an autonomous agent with LangGraph.js. Here's what you need to know:

1. Basic Structure for Building an Agent:
- LangGraph.js provides a ReAct (Reason + Act) pattern implementation
- The basic components include:
  - State management for conversation history
  - Nodes for different actions
  - Edges for decision-making flow
  - Tools for specific functionalities

(...)

```

:::

## 3. 전체 상태 히스토리 재생

챗봇에 단계를 추가했으므로 이제 전체 상태 히스토리를 `재생`하여 발생한 모든 것을 볼 수 있습니다.

:::python

```python
to_replay = None
for state in graph.get_state_history(config):
    print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
    print("-" * 80)
    if len(state.values["messages"]) == 6:
        # 상태의 채팅 메시지 수를 기준으로 특정 상태를 다소 임의로 선택하고 있습니다.
        to_replay = state
```

```
Num Messages:  8 Next:  ()
--------------------------------------------------------------------------------
Num Messages:  7 Next:  ('chatbot',)
--------------------------------------------------------------------------------
Num Messages:  6 Next:  ('tools',)
--------------------------------------------------------------------------------
Num Messages:  5 Next:  ('chatbot',)
--------------------------------------------------------------------------------
Num Messages:  4 Next:  ('__start__',)
--------------------------------------------------------------------------------
Num Messages:  4 Next:  ()
--------------------------------------------------------------------------------
Num Messages:  3 Next:  ('chatbot',)
--------------------------------------------------------------------------------
Num Messages:  2 Next:  ('tools',)
--------------------------------------------------------------------------------
Num Messages:  1 Next:  ('chatbot',)
--------------------------------------------------------------------------------
Num Messages:  0 Next:  ('__start__',)
--------------------------------------------------------------------------------
```

:::

:::js

```typescript
import type { StateSnapshot } from "@langchain/langgraph";

let toReplay: StateSnapshot | undefined;
for await (const state of graph.getStateHistory({
  configurable: { thread_id: threadId },
})) {
  console.log(
    `Num Messages: ${state.values.messages.length}, Next: ${JSON.stringify(
      state.next
    )}`
  );
  console.log("-".repeat(80));
  if (state.values.messages.length === 6) {
    // We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
    toReplay = state;
  }
}
```

```
Num Messages: 8 Next:  []
--------------------------------------------------------------------------------
Num Messages: 7 Next:  ["chatbot"]
--------------------------------------------------------------------------------
Num Messages: 6 Next:  ["tools"]
--------------------------------------------------------------------------------
Num Messages: 7, Next: ["chatbot"]
--------------------------------------------------------------------------------
Num Messages: 6, Next: ["tools"]
--------------------------------------------------------------------------------
Num Messages: 5, Next: ["chatbot"]
--------------------------------------------------------------------------------
Num Messages: 4, Next: ["__start__"]
--------------------------------------------------------------------------------
Num Messages: 4, Next: []
--------------------------------------------------------------------------------
Num Messages: 3, Next: ["chatbot"]
--------------------------------------------------------------------------------
Num Messages: 2, Next: ["tools"]
--------------------------------------------------------------------------------
Num Messages: 1, Next: ["chatbot"]
--------------------------------------------------------------------------------
Num Messages: 0, Next: ["__start__"]
--------------------------------------------------------------------------------
```

:::

체크포인트는 그래프의 모든 단계에 대해 저장됩니다. 이는 **여러 호출에 걸쳐** 적용되므로 전체 thread의 히스토리를 되감을 수 있습니다.

## 체크포인트에서 재개

:::python

두 번째 그래프 호출에서 `chatbot` 노드 이후의 `to_replay` 상태에서 재개합니다. 이 지점에서 재개하면 다음으로 **action** 노드가 호출됩니다.
:::

:::js
그래프 호출 중 하나에서 특정 노드 이후의 `toReplay` 상태에서 재개합니다. 이 지점에서 재개하면 다음으로 예약된 노드가 호출됩니다.
:::

:::python

```python
print(to_replay.next)
print(to_replay.config)
```

```
('tools',)
{'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1efd43e3-0c1f-6c4e-8006-891877d65740'}}
```

:::

:::js

Resume from the `toReplay` state, which is after the `chatbot` node in one of the graph invocations. Resuming from this point will call the next scheduled node.

```typescript
console.log(toReplay.next);
console.log(toReplay.config);
```

```
["tools"]
{
  configurable: {
    thread_id: "007708b8-ea9b-4ff7-a7ad-3843364dbf75",
    checkpoint_ns: "",
    checkpoint_id: "1efd43e3-0c1f-6c4e-8006-891877d65740"
  }
}
```

:::

## 4. 특정 시점의 상태 로드

:::python

체크포인트의 `to_replay.config`에는 `checkpoint_id` 타임스탬프가 포함되어 있습니다. 이 `checkpoint_id` 값을 제공하면 LangGraph의 체크포인터가 해당 시점의 상태를 **로드**하도록 지시합니다.

```python
# `to_replay.config`의 `checkpoint_id`는 체크포인터에 저장한 상태에 해당합니다.
for event in graph.stream(None, to_replay.config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

```
================================== Ai Message ==================================

[{'text': "That's an exciting idea! Building an autonomous agent with LangGraph is indeed a great application of this technology. LangGraph is particularly well-suited for creating complex, multi-step AI workflows, which is perfect for autonomous agents. Let me gather some more specific information about using LangGraph for building autonomous agents.", 'type': 'text'}, {'id': 'toolu_01QWNHhUaeeWcGXvA4eHT7Zo', 'input': {'query': 'Building autonomous agents with LangGraph examples and tutorials'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
Tool Calls:
  tavily_search_results_json (toolu_01QWNHhUaeeWcGXvA4eHT7Zo)
 Call ID: toolu_01QWNHhUaeeWcGXvA4eHT7Zo
  Args:
    query: Building autonomous agents with LangGraph examples and tutorials
================================= Tool Message =================================
Name: tavily_search_results_json

[{"url": "https://towardsdatascience.com/building-autonomous-multi-tool-agents-with-gemini-2-0-and-langgraph-ad3d7bd5e79d", "content": "Building Autonomous Multi-Tool Agents with Gemini 2.0 and LangGraph | by Youness Mansar | Jan, 2025 | Towards Data Science Building Autonomous Multi-Tool Agents with Gemini 2.0 and LangGraph A practical tutorial with full code examples for building and running multi-tool agents Towards Data Science LLMs are remarkable — they can memorize vast amounts of information, answer general knowledge questions, write code, generate stories, and even fix your grammar. In this tutorial, we are going to build a simple LLM agent that is equipped with four tools that it can use to answer a user's question. This Agent will have the following specifications: Follow Published in Towards Data Science --------------------------------- Your home for data science and AI. Follow Follow Follow"}, {"url": "https://github.com/anmolaman20/Tools_and_Agents", "content": "GitHub - anmolaman20/Tools_and_Agents: This repository provides resources for building AI agents using Langchain and Langgraph. This repository provides resources for building AI agents using Langchain and Langgraph. This repository provides resources for building AI agents using Langchain and Langgraph. This repository serves as a comprehensive guide for building AI-powered agents using Langchain and Langgraph. It provides hands-on examples, practical tutorials, and resources for developers and AI enthusiasts to master building intelligent systems and workflows. AI Agent Development: Gain insights into creating intelligent systems that think, reason, and adapt in real time. This repository is ideal for AI practitioners, developers exploring language models, or anyone interested in building intelligent systems. This repository provides resources for building AI agents using Langchain and Langgraph."}]
================================== Ai Message ==================================

Great idea! Building an autonomous agent with LangGraph is definitely an exciting project. Based on the latest information I've found, here are some insights and tips for building autonomous agents with LangGraph:

1. Multi-Tool Agents: LangGraph is particularly well-suited for creating autonomous agents that can use multiple tools. This allows your agent to have a diverse set of capabilities and choose the right tool for each task.

2. Integration with Large Language Models (LLMs): You can combine LangGraph with powerful LLMs like Gemini 2.0 to create more intelligent and capable agents. The LLM can serve as the "brain" of your agent, making decisions and generating responses.

3. Workflow Management: LangGraph excels at managing complex, multi-step AI workflows. This is crucial for autonomous agents that need to break down tasks into smaller steps and execute them in the right order.
...

Remember, building an autonomous agent is an iterative process. Start simple and gradually increase complexity as you become more comfortable with LangGraph and its capabilities.

Would you like more information on any specific aspect of building your autonomous agent with LangGraph?
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```

그래프는 `tools` 노드에서 실행을 재개했습니다. 위에 출력된 첫 번째 값이 검색 엔진 도구의 응답이기 때문에 이를 알 수 있습니다.
:::

:::js

체크포인트의 `toReplay.config`에는 `checkpoint_id` 타임스탬프가 포함되어 있습니다. 이 `checkpoint_id` 값을 제공하면 LangGraph의 체크포인터가 해당 시점의 상태를 **로드**하도록 지시합니다.

```typescript
// `toReplay.config`의 `checkpoint_id`는 체크포인터에 저장한 상태에 해당합니다.
for await (const event of await graph.stream(null, {
  ...toReplay?.config,
  streamMode: "values",
})) {
  if ("messages" in event) {
    const lastMessage = event.messages.at(-1);

    console.log(
      "=".repeat(32),
      `${lastMessage?.getType()} Message`,
      "=".repeat(32)
    );
    console.log(lastMessage?.text);
  }
}
```

```
================================ ai Message ================================
Let me search for specific information about building autonomous agents with LangGraph.js.
================================ tool Message ================================
{
  "query": "how to build autonomous agents with LangGraph.js examples tutorial",
  "follow_up_questions": null,
  "answer": null,
  "images": [],
  "results": [
    {
      "url": "https://www.mongodb.com/developer/languages/typescript/build-javascript-ai-agent-langgraphjs-mongodb/",
      "title": "Build a JavaScript AI Agent With LangGraph.js and MongoDB",
      "content": "(...)",
      "score": 0.7672197,
      "raw_content": null
    },
    {
      "url": "https://medium.com/@lorevanoudenhove/how-to-build-ai-agents-with-langgraph-a-step-by-step-guide-5d84d9c7e832",
      "title": "How to Build AI Agents with LangGraph: A Step-by-Step Guide",
      "content": "(...)",
      "score": 0.7407191,
      "raw_content": null
    }
  ],
  "response_time": 0.82
}
================================ ai Message ================================
Based on the search results, I can share some practical information about building autonomous agents with LangGraph.js. Here are some concrete examples and approaches:

1. Example HR Assistant Agent:
- Can handle HR-related queries using employee information
- Features include:
  - Starting and continuing conversations
  - Looking up information using vector search
  - Persisting conversation state using checkpoints
  - Managing threaded conversations

2. Energy Savings Calculator Agent:
- Functions as a lead generation tool for solar panel sales
- Capabilities include:
  - Calculating potential energy savings
  - Handling multi-step conversations
  - Processing user inputs for personalized estimates
  - Managing conversation state

(...)
```

그래프는 `tools` 노드에서 실행을 재개했습니다. 위에 출력된 첫 번째 값이 검색 엔진 도구의 응답이기 때문에 이를 알 수 있습니다.
:::

**축하합니다!** 이제 LangGraph에서 시간 여행 체크포인트 탐색을 사용해 보았습니다. 되감기를 통해 대안 경로를 탐색할 수 있는 기능은 디버깅, 실험 및 대화형 애플리케이션에 무한한 가능성을 열어줍니다.

## 더 알아보기

배포 및 고급 기능을 탐색하여 LangGraph 여정을 더 발전시키세요:

- **[LangGraph Server 빠른 시작](../../tutorials/langgraph-platform/local-server.md)**: LangGraph 서버를 로컬에서 시작하고 REST API 및 LangGraph Studio Web UI를 사용하여 상호작용합니다.
- **[LangGraph Platform 빠른 시작](../../cloud/quick_start.md)**: LangGraph Platform을 사용하여 LangGraph 앱을 배포합니다.
- **[LangGraph Platform 개념](../../concepts/langgraph_platform.md)**: LangGraph Platform의 기본 개념을 이해합니다.
