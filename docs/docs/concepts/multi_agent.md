# 멀티 에이전트 시스템

[에이전트](./agentic_concepts.md#agent-architectures)는 _LLM을 사용하여 애플리케이션의 제어 흐름을 결정하는 시스템_입니다. 이러한 시스템을 개발하면서 시간이 지남에 따라 더 복잡해져 관리 및 확장이 어려워질 수 있습니다. 예를 들어, 다음과 같은 문제가 발생할 수 있습니다:

- 에이전트가 사용할 수 있는 도구가 너무 많아 다음에 호출할 도구에 대해 잘못된 결정을 내립니다
- 컨텍스트가 너무 복잡해져 단일 에이전트가 추적하기 어렵습니다
- 시스템에 여러 전문 분야가 필요합니다(예: 계획자, 연구원, 수학 전문가 등)

이러한 문제를 해결하기 위해 애플리케이션을 여러 개의 작고 독립적인 에이전트로 나누고 **멀티 에이전트 시스템**으로 구성하는 것을 고려할 수 있습니다. 이러한 독립 에이전트는 프롬프트와 LLM 호출만큼 간단하거나 [ReAct](./agentic_concepts.md#tool-calling-agent) 에이전트만큼 복잡할 수 있습니다(그 이상도 가능합니다!).

멀티 에이전트 시스템 사용의 주요 이점은 다음과 같습니다:

- **모듈성**: 별도의 에이전트는 에이전트 시스템을 개발, 테스트 및 유지 관리하기 쉽게 만듭니다.
- **전문화**: 특정 도메인에 집중하는 전문 에이전트를 생성할 수 있어 전체 시스템 성능에 도움이 됩니다.
- **제어**: 함수 호출에 의존하는 대신 에이전트가 통신하는 방식을 명시적으로 제어할 수 있습니다.

## 멀티 에이전트 아키텍처

![](./img/multi_agent/architectures.png)

멀티 에이전트 시스템에서 에이전트를 연결하는 방법에는 여러 가지가 있습니다:

- **네트워크**: 각 에이전트는 [다른 모든 에이전트](../tutorials/multi_agent/multi-agent-collaboration.ipynb/)와 통신할 수 있습니다. 모든 에이전트는 다음에 호출할 다른 에이전트를 결정할 수 있습니다.
- **슈퍼바이저**: 각 에이전트는 단일 [슈퍼바이저](../tutorials/multi_agent/agent_supervisor.md/) 에이전트와 통신합니다. 슈퍼바이저 에이전트는 다음에 어떤 에이전트를 호출해야 하는지 결정합니다.
- **슈퍼바이저 (도구 호출)**: 이것은 슈퍼바이저 아키텍처의 특수한 경우입니다. 개별 에이전트는 도구로 표현될 수 있습니다. 이 경우 슈퍼바이저 에이전트는 도구 호출 LLM을 사용하여 호출할 에이전트 도구와 해당 에이전트에 전달할 인수를 결정합니다.
- **계층적**: [슈퍼바이저의 슈퍼바이저](../tutorials/multi_agent/hierarchical_agent_teams.ipynb/)가 있는 멀티 에이전트 시스템을 정의할 수 있습니다. 이것은 슈퍼바이저 아키텍처의 일반화이며 더 복잡한 제어 흐름을 허용합니다.
- **커스텀 멀티 에이전트 워크플로우**: 각 에이전트는 에이전트의 하위 집합과만 통신합니다. 흐름의 일부는 결정론적이며 일부 에이전트만 다음에 호출할 다른 에이전트를 결정할 수 있습니다.

### 핸드오프

멀티 에이전트 아키텍처에서 에이전트는 그래프 노드로 표현될 수 있습니다. 각 에이전트 노드는 단계를 실행하고 실행을 완료할지 다른 에이전트로 라우팅할지 결정하며, 자기 자신으로 라우팅할 수도 있습니다(예: 루프에서 실행). 멀티 에이전트 상호 작용의 일반적인 패턴은 **핸드오프**로, 한 에이전트가 다른 에이전트에게 제어를 _넘겨주는_ 것입니다. 핸드오프를 사용하면 다음을 지정할 수 있습니다:

- **destination**: 이동할 대상 에이전트(예: 이동할 노드의 이름)
- **payload**: [해당 에이전트에 전달할 정보](#communication-and-state-management)(예: 상태 업데이트)

LangGraph에서 핸드오프를 구현하기 위해 에이전트 노드는 제어 흐름과 상태 업데이트를 모두 결합할 수 있는 [`Command`](./low_level.md#command) 객체를 반환할 수 있습니다:

:::python

```python
def agent(state) -> Command[Literal["agent", "another_agent"]]:
    # the condition for routing/halting can be anything, e.g. LLM tool call / structured output, etc.
    goto = get_next_agent(...)  # 'agent' / 'another_agent'
    return Command(
        # Specify which agent to call next
        goto=goto,
        # Update the graph state
        update={"my_state_key": "my_state_value"}
    )
```

:::

:::js

```typescript
graph.addNode((state) => {
    // the condition for routing/halting can be anything, e.g. LLM tool call / structured output, etc.
    const goto = getNextAgent(...); // 'agent' / 'another_agent'
    return new Command({
      // Specify which agent to call next
      goto,
      // Update the graph state
      update: { myStateKey: "myStateValue" }
    });
})
```

:::

:::python
각 에이전트 노드가 자체적으로 그래프인([서브그래프](./subgraphs.md)) 더 복잡한 시나리오에서는 에이전트 서브그래프 중 하나의 노드가 다른 에이전트로 이동하려고 할 수 있습니다. 예를 들어, 두 개의 에이전트 `alice`와 `bob`(부모 그래프의 서브그래프 노드)이 있고 `alice`가 `bob`으로 이동해야 하는 경우 `Command` 객체에서 `graph=Command.PARENT`를 설정할 수 있습니다:

```python
def some_node_inside_alice(state):
    return Command(
        goto="bob",
        update={"my_state_key": "my_state_value"},
        # specify which graph to navigate to (defaults to the current graph)
        graph=Command.PARENT,
    )
```

:::

:::js
각 에이전트 노드가 자체적으로 그래프인([서브그래프](./subgraphs.md)) 더 복잡한 시나리오에서는 에이전트 서브그래프 중 하나의 노드가 다른 에이전트로 이동하려고 할 수 있습니다. 예를 들어, 두 개의 에이전트 `alice`와 `bob`(부모 그래프의 서브그래프 노드)이 있고 `alice`가 `bob`으로 이동해야 하는 경우 `Command` 객체에서 `graph: Command.PARNT`를 설정할 수 있습니다:

```typescript
alice.addNode((state) => {
  return new Command({
    goto: "bob",
    update: { myStateKey: "myStateValue" },
    // specify which graph to navigate to (defaults to the current graph)
    graph: Command.PARENT,
  });
});
```

:::

!!! note

    :::python

    `Command(graph=Command.PARENT)`를 사용하여 통신하는 서브그래프의 시각화를 지원해야 하는 경우 `Command` 어노테이션이 있는 노드 함수로 래핑해야 합니다:
    이렇게 하는 대신:

    ```python
    builder.add_node(alice)
    ```

    다음과 같이 해야 합니다:

    ```python
    def call_alice(state) -> Command[Literal["bob"]]:
        return alice.invoke(state)

    builder.add_node("alice", call_alice)
    ```

    :::

    :::js
    `Command({ graph: Command.PARENT })`를 사용하여 통신하는 서브그래프의 시각화를 지원해야 하는 경우 `Command` 어노테이션이 있는 노드 함수로 래핑해야 합니다:

    이렇게 하는 대신:

    ```typescript
    builder.addNode("alice", alice);
    ```

    다음과 같이 해야 합니다:

    ```typescript
    builder.addNode("alice", (state) => alice.invoke(state), { ends: ["bob"] });
    ```

    :::

#### 도구로서의 핸드오프

가장 일반적인 에이전트 유형 중 하나는 [도구 호출 에이전트](../agents/overview.md)입니다. 이러한 유형의 에이전트의 경우 일반적인 패턴은 핸드오프를 도구 호출로 래핑하는 것입니다:

:::python

```python
from langchain_core.tools import tool

@tool
def transfer_to_bob():
    """Transfer to bob."""
    return Command(
        # name of the agent (node) to go to
        goto="bob",
        # data to send to the agent
        update={"my_state_key": "my_state_value"},
        # indicate to LangGraph that we need to navigate to
        # agent node in a parent graph
        graph=Command.PARENT,
    )
```

:::

:::js

```typescript
import { tool } from "@langchain/core/tools";
import { Command } from "@langchain/langgraph";
import { z } from "zod";

const transferToBob = tool(
  async () => {
    return new Command({
      // name of the agent (node) to go to
      goto: "bob",
      // data to send to the agent
      update: { myStateKey: "myStateValue" },
      // indicate to LangGraph that we need to navigate to
      // agent node in a parent graph
      graph: Command.PARENT,
    });
  },
  {
    name: "transfer_to_bob",
    description: "Transfer to bob.",
    schema: z.object({}),
  }
);
```

:::

이것은 도구에서 그래프 상태를 업데이트하는 특수한 경우로, 상태 업데이트 외에 제어 흐름도 포함됩니다.

!!! important

      :::python
      `Command`를 반환하는 도구를 사용하려면 사전 구축된 @[`create_react_agent`][create_react_agent] / @[`ToolNode`][ToolNode] 컴포넌트를 사용하거나 자체 로직을 구현할 수 있습니다:

      ```python
      def call_tools(state):
          ...
          commands = [tools_by_name[tool_call["name"]].invoke(tool_call) for tool_call in tool_calls]
          return commands
      ```
      :::

      :::js
      `Command`를 반환하는 도구를 사용하려면 사전 구축된 @[`createReactAgent`][create_react_agent] / @[ToolNode] 컴포넌트를 사용하거나 자체 로직을 구현할 수 있습니다:

      ```typescript
      graph.addNode("call_tools", async (state) => {
        // ... tool execution logic
        const commands = toolCalls.map((toolCall) =>
          toolsByName[toolCall.name].invoke(toolCall)
        );
        return commands;
      });
      ```
      :::

이제 다양한 멀티 에이전트 아키텍처를 자세히 살펴보겠습니다.

### 네트워크

이 아키텍처에서 에이전트는 그래프 노드로 정의됩니다. 각 에이전트는 다른 모든 에이전트와 통신할 수 있으며(다대다 연결) 다음에 호출할 에이전트를 결정할 수 있습니다. 이 아키텍처는 에이전트의 명확한 계층 구조나 에이전트를 호출해야 하는 특정 순서가 없는 문제에 적합합니다.

:::python

```python
from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langgraph.graph import StateGraph, MessagesState, START, END

model = ChatOpenAI()

def agent_1(state: MessagesState) -> Command[Literal["agent_2", "agent_3", END]]:
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # to determine which agent to call next. a common pattern is to call the model
    # with a structured output (e.g. force it to return an output with a "next_agent" field)
    response = model.invoke(...)
    # route to one of the agents or exit based on the LLM's decision
    # if the LLM returns "__end__", the graph will finish execution
    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )

def agent_2(state: MessagesState) -> Command[Literal["agent_1", "agent_3", END]]:
    response = model.invoke(...)
    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )

def agent_3(state: MessagesState) -> Command[Literal["agent_1", "agent_2", END]]:
    ...
    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )

builder = StateGraph(MessagesState)
builder.add_node(agent_1)
builder.add_node(agent_2)
builder.add_node(agent_3)

builder.add_edge(START, "agent_1")
network = builder.compile()
```

:::

:::js

```typescript
import { StateGraph, MessagesZodState, START, END } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { Command } from "@langchain/langgraph";
import { z } from "zod";

const model = new ChatOpenAI();

const agent1 = async (state: z.infer<typeof MessagesZodState>) => {
  // you can pass relevant parts of the state to the LLM (e.g., state.messages)
  // to determine which agent to call next. a common pattern is to call the model
  // with a structured output (e.g. force it to return an output with a "next_agent" field)
  const response = await model.invoke(...);
  // route to one of the agents or exit based on the LLM's decision
  // if the LLM returns "__end__", the graph will finish execution
  return new Command({
    goto: response.nextAgent,
    update: { messages: [response.content] },
  });
};

const agent2 = async (state: z.infer<typeof MessagesZodState>) => {
  const response = await model.invoke(...);
  return new Command({
    goto: response.nextAgent,
    update: { messages: [response.content] },
  });
};

const agent3 = async (state: z.infer<typeof MessagesZodState>) => {
  // ...
  return new Command({
    goto: response.nextAgent,
    update: { messages: [response.content] },
  });
};

const builder = new StateGraph(MessagesZodState)
  .addNode("agent1", agent1, {
    ends: ["agent2", "agent3", END]
  })
  .addNode("agent2", agent2, {
    ends: ["agent1", "agent3", END]
  })
  .addNode("agent3", agent3, {
    ends: ["agent1", "agent2", END]
  })
  .addEdge(START, "agent1");

const network = builder.compile();
```

:::

### 슈퍼바이저

이 아키텍처에서는 에이전트를 노드로 정의하고 다음에 호출해야 하는 에이전트 노드를 결정하는 슈퍼바이저 노드(LLM)를 추가합니다. 슈퍼바이저의 결정에 따라 적절한 에이전트 노드로 실행을 라우팅하기 위해 [`Command`](./low_level.md#command)를 사용합니다. 이 아키텍처는 여러 에이전트를 병렬로 실행하거나 [map-reduce](../how-tos/graph-api.md#map-reduce-and-the-send-api) 패턴을 사용하는 데 적합합니다.

:::python

```python
from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langgraph.graph import StateGraph, MessagesState, START, END

model = ChatOpenAI()

def supervisor(state: MessagesState) -> Command[Literal["agent_1", "agent_2", END]]:
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # to determine which agent to call next. a common pattern is to call the model
    # with a structured output (e.g. force it to return an output with a "next_agent" field)
    response = model.invoke(...)
    # route to one of the agents or exit based on the supervisor's decision
    # if the supervisor returns "__end__", the graph will finish execution
    return Command(goto=response["next_agent"])

def agent_1(state: MessagesState) -> Command[Literal["supervisor"]]:
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # and add any additional logic (different models, custom prompts, structured output, etc.)
    response = model.invoke(...)
    return Command(
        goto="supervisor",
        update={"messages": [response]},
    )

def agent_2(state: MessagesState) -> Command[Literal["supervisor"]]:
    response = model.invoke(...)
    return Command(
        goto="supervisor",
        update={"messages": [response]},
    )

builder = StateGraph(MessagesState)
builder.add_node(supervisor)
builder.add_node(agent_1)
builder.add_node(agent_2)

builder.add_edge(START, "supervisor")

supervisor = builder.compile()
```

:::

:::js

```typescript
import { StateGraph, MessagesZodState, Command, START, END } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

const model = new ChatOpenAI();

const supervisor = async (state: z.infer<typeof MessagesZodState>) => {
  // you can pass relevant parts of the state to the LLM (e.g., state.messages)
  // to determine which agent to call next. a common pattern is to call the model
  // with a structured output (e.g. force it to return an output with a "next_agent" field)
  const response = await model.invoke(...);
  // route to one of the agents or exit based on the supervisor's decision
  // if the supervisor returns "__end__", the graph will finish execution
  return new Command({ goto: response.nextAgent });
};

const agent1 = async (state: z.infer<typeof MessagesZodState>) => {
  // you can pass relevant parts of the state to the LLM (e.g., state.messages)
  // and add any additional logic (different models, custom prompts, structured output, etc.)
  const response = await model.invoke(...);
  return new Command({
    goto: "supervisor",
    update: { messages: [response] },
  });
};

const agent2 = async (state: z.infer<typeof MessagesZodState>) => {
  const response = await model.invoke(...);
  return new Command({
    goto: "supervisor",
    update: { messages: [response] },
  });
};

const builder = new StateGraph(MessagesZodState)
  .addNode("supervisor", supervisor, {
    ends: ["agent1", "agent2", END]
  })
  .addNode("agent1", agent1, {
    ends: ["supervisor"]
  })
  .addNode("agent2", agent2, {
    ends: ["supervisor"]
  })
  .addEdge(START, "supervisor");

const supervisorGraph = builder.compile();
```

:::

:::js

```typescript
import { StateGraph, MessagesZodState, Command, START, END } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

const model = new ChatOpenAI();

const supervisor = async (state: z.infer<typeof MessagesZodState>) => {
  // you can pass relevant parts of the state to the LLM (e.g., state.messages)
  // to determine which agent to call next. a common pattern is to call the model
  // with a structured output (e.g. force it to return an output with a "next_agent" field)
  const response = await model.invoke(...);
  // route to one of the agents or exit based on the supervisor's decision
  // if the supervisor returns "__end__", the graph will finish execution
  return new Command({ goto: response.nextAgent });
};

const agent1 = async (state: z.infer<typeof MessagesZodState>) => {
  // you can pass relevant parts of the state to the LLM (e.g., state.messages)
  // and add any additional logic (different models, custom prompts, structured output, etc.)
  const response = await model.invoke(...);
  return new Command({
    goto: "supervisor",
    update: { messages: [response] },
  });
};

const agent2 = async (state: z.infer<typeof MessagesZodState>) => {
  const response = await model.invoke(...);
  return new Command({
    goto: "supervisor",
    update: { messages: [response] },
  });
};

const builder = new StateGraph(MessagesZodState)
  .addNode("supervisor", supervisor, {
    ends: ["agent1", "agent2", END]
  })
  .addNode("agent1", agent1, {
    ends: ["supervisor"]
  })
  .addNode("agent2", agent2, {
    ends: ["supervisor"]
  })
  .addEdge(START, "supervisor");

const supervisorGraph = builder.compile();
```

:::

슈퍼바이저 멀티 에이전트 아키텍처의 예는 이 [튜토리얼](../tutorials/multi_agent/agent_supervisor.md)을 확인하세요.

### 슈퍼바이저 (도구 호출)

[슈퍼바이저](#supervisor) 아키텍처의 이 변형에서는 하위 에이전트를 호출하는 역할을 하는 슈퍼바이저 [에이전트](./agentic_concepts.md#agent-architectures)를 정의합니다. 하위 에이전트는 슈퍼바이저에게 도구로 노출되며 슈퍼바이저 에이전트는 다음에 호출할 도구를 결정합니다. 슈퍼바이저 에이전트는 중지하기로 결정할 때까지 while 루프에서 도구를 호출하는 LLM으로 실행되는 [표준 구현](./agentic_concepts.md#tool-calling-agent)을 따릅니다.

:::python

```python
from typing import Annotated
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState, create_react_agent

model = ChatOpenAI()

# this is the agent function that will be called as tool
# notice that you can pass the state to the tool via InjectedState annotation
def agent_1(state: Annotated[dict, InjectedState]):
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # and add any additional logic (different models, custom prompts, structured output, etc.)
    response = model.invoke(...)
    # return the LLM response as a string (expected tool response format)
    # this will be automatically turned to ToolMessage
    # by the prebuilt create_react_agent (supervisor)
    return response.content

def agent_2(state: Annotated[dict, InjectedState]):
    response = model.invoke(...)
    return response.content

tools = [agent_1, agent_2]
# the simplest way to build a supervisor w/ tool-calling is to use prebuilt ReAct agent graph
# that consists of a tool-calling LLM node (i.e. supervisor) and a tool-executing node
supervisor = create_react_agent(model, tools)
```

:::

:::js

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const model = new ChatOpenAI();

// this is the agent function that will be called as tool
// notice that you can pass the state to the tool via config parameter
const agent1 = tool(
  async (_, config) => {
    const state = config.configurable?.state;
    // you can pass relevant parts of the state to the LLM (e.g., state.messages)
    // and add any additional logic (different models, custom prompts, structured output, etc.)
    const response = await model.invoke(...);
    // return the LLM response as a string (expected tool response format)
    // this will be automatically turned to ToolMessage
    // by the prebuilt createReactAgent (supervisor)
    return response.content;
  },
  {
    name: "agent1",
    description: "Agent 1 description",
    schema: z.object({}),
  }
);

const agent2 = tool(
  async (_, config) => {
    const state = config.configurable?.state;
    const response = await model.invoke(...);
    return response.content;
  },
  {
    name: "agent2",
    description: "Agent 2 description",
    schema: z.object({}),
  }
);

const tools = [agent1, agent2];
// the simplest way to build a supervisor w/ tool-calling is to use prebuilt ReAct agent graph
// that consists of a tool-calling LLM node (i.e. supervisor) and a tool-executing node
const supervisor = createReactAgent({ llm: model, tools });
```

:::

### 계층적

시스템에 더 많은 에이전트를 추가함에 따라 슈퍼바이저가 모두 관리하기 어려워질 수 있습니다. 슈퍼바이저는 다음에 호출할 에이전트에 대해 잘못된 결정을 내리기 시작하거나 컨텍스트가 단일 슈퍼바이저가 추적하기에 너무 복잡해질 수 있습니다. 다시 말해, 처음에 멀티 에이전트 아키텍처를 동기 부여한 것과 동일한 문제가 발생합니다.

이를 해결하기 위해 시스템을 _계층적으로_ 설계할 수 있습니다. 예를 들어, 개별 슈퍼바이저가 관리하는 별도의 전문 에이전트 팀과 팀을 관리하는 최상위 슈퍼바이저를 만들 수 있습니다.

:::python

```python
from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
model = ChatOpenAI()

# define team 1 (same as the single supervisor example above)

def team_1_supervisor(state: MessagesState) -> Command[Literal["team_1_agent_1", "team_1_agent_2", END]]:
    response = model.invoke(...)
    return Command(goto=response["next_agent"])

def team_1_agent_1(state: MessagesState) -> Command[Literal["team_1_supervisor"]]:
    response = model.invoke(...)
    return Command(goto="team_1_supervisor", update={"messages": [response]})

def team_1_agent_2(state: MessagesState) -> Command[Literal["team_1_supervisor"]]:
    response = model.invoke(...)
    return Command(goto="team_1_supervisor", update={"messages": [response]})

team_1_builder = StateGraph(Team1State)
team_1_builder.add_node(team_1_supervisor)
team_1_builder.add_node(team_1_agent_1)
team_1_builder.add_node(team_1_agent_2)
team_1_builder.add_edge(START, "team_1_supervisor")
team_1_graph = team_1_builder.compile()

# define team 2 (same as the single supervisor example above)
class Team2State(MessagesState):
    next: Literal["team_2_agent_1", "team_2_agent_2", "__end__"]

def team_2_supervisor(state: Team2State):
    ...

def team_2_agent_1(state: Team2State):
    ...

def team_2_agent_2(state: Team2State):
    ...

team_2_builder = StateGraph(Team2State)
...
team_2_graph = team_2_builder.compile()


# define top-level supervisor

builder = StateGraph(MessagesState)
def top_level_supervisor(state: MessagesState) -> Command[Literal["team_1_graph", "team_2_graph", END]]:
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # to determine which team to call next. a common pattern is to call the model
    # with a structured output (e.g. force it to return an output with a "next_team" field)
    response = model.invoke(...)
    # route to one of the teams or exit based on the supervisor's decision
    # if the supervisor returns "__end__", the graph will finish execution
    return Command(goto=response["next_team"])

builder = StateGraph(MessagesState)
builder.add_node(top_level_supervisor)
builder.add_node("team_1_graph", team_1_graph)
builder.add_node("team_2_graph", team_2_graph)
builder.add_edge(START, "top_level_supervisor")
builder.add_edge("team_1_graph", "top_level_supervisor")
builder.add_edge("team_2_graph", "top_level_supervisor")
graph = builder.compile()
```

:::

:::js

```typescript
import { StateGraph, MessagesZodState, Command, START, END } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

const model = new ChatOpenAI();

// define team 1 (same as the single supervisor example above)

const team1Supervisor = async (state: z.infer<typeof MessagesZodState>) => {
  const response = await model.invoke(...);
  return new Command({ goto: response.nextAgent });
};

const team1Agent1 = async (state: z.infer<typeof MessagesZodState>) => {
  const response = await model.invoke(...);
  return new Command({
    goto: "team1Supervisor",
    update: { messages: [response] }
  });
};

const team1Agent2 = async (state: z.infer<typeof MessagesZodState>) => {
  const response = await model.invoke(...);
  return new Command({
    goto: "team1Supervisor",
    update: { messages: [response] }
  });
};

const team1Builder = new StateGraph(MessagesZodState)
  .addNode("team1Supervisor", team1Supervisor, {
    ends: ["team1Agent1", "team1Agent2", END]
  })
  .addNode("team1Agent1", team1Agent1, {
    ends: ["team1Supervisor"]
  })
  .addNode("team1Agent2", team1Agent2, {
    ends: ["team1Supervisor"]
  })
  .addEdge(START, "team1Supervisor");
const team1Graph = team1Builder.compile();

// define team 2 (same as the single supervisor example above)
const team2Supervisor = async (state: z.infer<typeof MessagesZodState>) => {
  // ...
};

const team2Agent1 = async (state: z.infer<typeof MessagesZodState>) => {
  // ...
};

const team2Agent2 = async (state: z.infer<typeof MessagesZodState>) => {
  // ...
};

const team2Builder = new StateGraph(MessagesZodState);
// ... build team2Graph
const team2Graph = team2Builder.compile();

// define top-level supervisor

const topLevelSupervisor = async (state: z.infer<typeof MessagesZodState>) => {
  // you can pass relevant parts of the state to the LLM (e.g., state.messages)
  // to determine which team to call next. a common pattern is to call the model
  // with a structured output (e.g. force it to return an output with a "next_team" field)
  const response = await model.invoke(...);
  // route to one of the teams or exit based on the supervisor's decision
  // if the supervisor returns "__end__", the graph will finish execution
  return new Command({ goto: response.nextTeam });
};

const builder = new StateGraph(MessagesZodState)
  .addNode("topLevelSupervisor", topLevelSupervisor, {
    ends: ["team1Graph", "team2Graph", END]
  })
  .addNode("team1Graph", team1Graph)
  .addNode("team2Graph", team2Graph)
  .addEdge(START, "topLevelSupervisor")
  .addEdge("team1Graph", "topLevelSupervisor")
  .addEdge("team2Graph", "topLevelSupervisor");

const graph = builder.compile();
```

:::

### 커스텀 멀티 에이전트 워크플로우

이 아키텍처에서는 개별 에이전트를 그래프 노드로 추가하고 커스텀 워크플로우에서 에이전트가 호출되는 순서를 미리 정의합니다. LangGraph에서 워크플로우는 두 가지 방법으로 정의할 수 있습니다:

- **명시적 제어 흐름 (일반 엣지)**: LangGraph를 사용하면 [일반 그래프 엣지](./low_level.md#normal-edges)를 통해 애플리케이션의 제어 흐름(즉, 에이전트가 통신하는 순서)을 명시적으로 정의할 수 있습니다. 이것은 위 아키텍처의 가장 결정론적인 변형입니다 — 항상 다음에 호출될 에이전트를 미리 알 수 있습니다.

- **동적 제어 흐름 (Command)**: LangGraph에서는 LLM이 애플리케이션 제어 흐름의 일부를 결정하도록 허용할 수 있습니다. 이는 [`Command`](./low_level.md#command)를 사용하여 달성할 수 있습니다. 이것의 특수한 경우가 [슈퍼바이저 도구 호출](#supervisor-tool-calling) 아키텍처입니다. 이 경우 슈퍼바이저 에이전트를 구동하는 도구 호출 LLM이 도구(에이전트)가 호출되는 순서에 대한 결정을 내립니다.

:::python

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START

model = ChatOpenAI()

def agent_1(state: MessagesState):
    response = model.invoke(...)
    return {"messages": [response]}

def agent_2(state: MessagesState):
    response = model.invoke(...)
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node(agent_1)
builder.add_node(agent_2)
# define the flow explicitly
builder.add_edge(START, "agent_1")
builder.add_edge("agent_1", "agent_2")
```

:::

:::js

```typescript
import { StateGraph, MessagesZodState, START } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

const model = new ChatOpenAI();

const agent1 = async (state: z.infer<typeof MessagesZodState>) => {
  const response = await model.invoke(...);
  return { messages: [response] };
};

const agent2 = async (state: z.infer<typeof MessagesZodState>) => {
  const response = await model.invoke(...);
  return { messages: [response] };
};

const builder = new StateGraph(MessagesZodState)
  .addNode("agent1", agent1)
  .addNode("agent2", agent2)
  // define the flow explicitly
  .addEdge(START, "agent1")
  .addEdge("agent1", "agent2");
```

:::

## 통신 및 상태 관리

멀티 에이전트 시스템을 구축할 때 가장 중요한 것은 에이전트가 통신하는 방법을 파악하는 것입니다.

에이전트가 통신하는 일반적이고 보편적인 방법은 메시지 목록을 통하는 것입니다. 이것은 다음 질문을 제기합니다:

- 에이전트는 [**핸드오프를 통해 또는 도구 호출을 통해**](#handoffs-vs-tool-calls) 통신합니까?
- [**한 에이전트에서 다음 에이전트로 전달되는**](#message-passing-between-agents) 메시지는 무엇입니까?
- [**핸드오프가 메시지 목록에서 어떻게 표현**](#representing-handoffs-in-message-history)됩니까?
- [**하위 에이전트의 상태를 어떻게 관리**](#state-management-for-subagents)합니까?

또한 더 복잡한 에이전트를 다루거나 개별 에이전트 상태를 멀티 에이전트 시스템 상태와 별도로 유지하려는 경우 [**다른 상태 스키마**](#using-different-state-schemas)를 사용해야 할 수 있습니다.

### 핸드오프 vs 도구 호출

에이전트 간에 전달되는 "페이로드"는 무엇입니까? 위에서 논의한 대부분의 아키텍처에서 에이전트는 [핸드오프](#handoffs)를 통해 통신하고 핸드오프 페이로드의 일부로 [그래프 상태](./low_level.md#state)를 전달합니다. 특히 에이전트는 그래프 상태의 일부로 메시지 목록을 전달합니다. [도구 호출을 사용하는 슈퍼바이저](#supervisor-tool-calling)의 경우 페이로드는 도구 호출 인수입니다.

![](./img/multi_agent/request.png)

### 에이전트 간 메시지 전달

에이전트가 통신하는 가장 일반적인 방법은 공유 상태 채널, 일반적으로 메시지 목록을 통하는 것입니다. 이것은 에이전트가 공유하는 상태에 항상 적어도 하나의 채널(키)이 있다고 가정합니다(예: `messages`). 공유 메시지 목록을 통해 통신할 때 추가 고려 사항이 있습니다: 에이전트가 사고 프로세스의 [전체 기록을 공유](#sharing-full-thought-process)해야 합니까, 아니면 [최종 결과만](#sharing-only-final-results) 공유해야 합니까?

![](./img/multi_agent/response.png)

#### 전체 사고 프로세스 공유

에이전트는 사고 프로세스의 **전체 기록**(즉, "스크래치패드")을 다른 모든 에이전트와 공유할 수 있습니다. 이 "스크래치패드"는 일반적으로 [메시지 목록](./low_level.md#why-use-messages)처럼 보일 것입니다. 전체 사고 프로세스를 공유하는 이점은 다른 에이전트가 더 나은 결정을 내리고 시스템 전체의 추론 능력을 향상시키는 데 도움이 될 수 있다는 것입니다. 단점은 에이전트의 수와 복잡성이 증가함에 따라 "스크래치패드"가 빠르게 증가하여 [메모리 관리](../how-tos/memory/add-memory.md)를 위한 추가 전략이 필요할 수 있다는 것입니다.

#### 최종 결과만 공유

에이전트는 자체 개인 "스크래치패드"를 가질 수 있으며 나머지 에이전트와 **최종 결과만 공유**할 수 있습니다. 이 접근 방식은 많은 에이전트가 있거나 더 복잡한 에이전트가 있는 시스템에서 더 잘 작동할 수 있습니다. 이 경우 [다른 상태 스키마](#using-different-state-schemas)로 에이전트를 정의해야 합니다.

도구로 호출되는 에이전트의 경우 슈퍼바이저는 도구 스키마를 기반으로 입력을 결정합니다. 또한 LangGraph는 런타임에 개별 도구에 [상태를 전달](../how-tos/tool-calling.md#short-term-memory)할 수 있으므로 필요한 경우 하위 에이전트가 부모 상태에 액세스할 수 있습니다.

#### 메시지에 에이전트 이름 표시

특히 긴 메시지 기록의 경우 특정 AI 메시지가 어떤 에이전트에서 온 것인지 표시하는 것이 유용할 수 있습니다. 일부 LLM 제공자(예: OpenAI)는 메시지에 `name` 매개변수 추가를 지원합니다 — 이를 사용하여 에이전트 이름을 메시지에 첨부할 수 있습니다. 지원되지 않는 경우 에이전트 이름을 메시지 콘텐츠에 수동으로 삽입하는 것을 고려할 수 있습니다(예: `<agent>alice</agent><message>message from alice</message>`).

### 메시지 기록에서 핸드오프 표현

:::python
핸드오프는 일반적으로 LLM이 전용 [핸드오프 도구](#handoffs-as-tools)를 호출하여 수행됩니다. 이것은 다음 에이전트(LLM)에 전달되는 도구 호출이 있는 [AI 메시지](https://python.langchain.com/docs/concepts/messages/#aimessage)로 표현됩니다. 대부분의 LLM 제공자는 해당 도구 메시지 **없이** 도구 호출이 있는 AI 메시지 수신을 지원하지 않습니다.
:::

:::js
핸드오프는 일반적으로 LLM이 전용 [핸드오프 도구](#handoffs-as-tools)를 호출하여 수행됩니다. 이것은 다음 에이전트(LLM)에 전달되는 도구 호출이 있는 [AI 메시지](https://js.langchain.com/docs/concepts/messages/#aimessage)로 표현됩니다. 대부분의 LLM 제공자는 해당 도구 메시지 **없이** 도구 호출이 있는 AI 메시지 수신을 지원하지 않습니다.
:::

따라서 두 가지 옵션이 있습니다:

:::python

1. 메시지 목록에 추가 [도구 메시지](https://python.langchain.com/docs/concepts/messages/#toolmessage)를 추가합니다(예: "에이전트 X로 성공적으로 전송됨").
2. 도구 호출이 있는 AI 메시지를 제거합니다.
   :::

:::js

1. 메시지 목록에 추가 [도구 메시지](https://js.langchain.com/docs/concepts/messages/#toolmessage)를 추가합니다(예: "에이전트 X로 성공적으로 전송됨").
2. 도구 호출이 있는 AI 메시지를 제거합니다.
:::

실제로 대부분의 개발자는 옵션 (1)을 선택합니다.

### 하위 에이전트의 상태 관리

일반적인 관행은 공유 메시지 목록에서 여러 에이전트가 통신하지만 [목록에 최종 메시지만 추가](#sharing-only-final-results)하는 것입니다. 이것은 중간 메시지(예: 도구 호출)가 이 목록에 저장되지 않음을 의미합니다.

이 특정 하위 에이전트가 나중에 호출될 때 이러한 메시지를 전달할 수 있도록 저장하려면 어떻게 해야 합니까?

이를 달성하는 두 가지 상위 수준 접근 방식이 있습니다:

:::python

1. 이러한 메시지를 공유 메시지 목록에 저장하되 하위 에이전트 LLM에 전달하기 전에 목록을 필터링합니다. 예를 들어 **다른** 에이전트의 모든 도구 호출을 필터링하도록 선택할 수 있습니다.
2. 하위 에이전트의 그래프 상태에서 각 에이전트에 대한 별도의 메시지 목록(예: `alice_messages`)을 저장합니다. 이것은 메시지 기록이 어떻게 보이는지에 대한 그들의 "뷰"가 될 것입니다.
:::

:::js

1. 이러한 메시지를 공유 메시지 목록에 저장하되 하위 에이전트 LLM에 전달하기 전에 목록을 필터링합니다. 예를 들어 **다른** 에이전트의 모든 도구 호출을 필터링하도록 선택할 수 있습니다.
2. 하위 에이전트의 그래프 상태에서 각 에이전트에 대한 별도의 메시지 목록(예: `aliceMessages`)을 저장합니다. 이것은 메시지 기록이 어떻게 보이는지에 대한 그들의 "뷰"가 될 것입니다.
:::

### 다른 상태 스키마 사용

에이전트는 나머지 에이전트와 다른 상태 스키마를 가져야 할 수 있습니다. 예를 들어, 검색 에이전트는 쿼리와 검색된 문서만 추적하면 될 수 있습니다. LangGraph에서 이를 달성하는 두 가지 방법이 있습니다:

- 별도의 상태 스키마로 [서브그래프](./subgraphs.md) 에이전트를 정의합니다. 서브그래프와 부모 그래프 간에 공유 상태 키(채널)가 없는 경우 부모 그래프가 서브그래프와 통신하는 방법을 알 수 있도록 [입력 / 출력 변환을 추가](../how-tos/subgraph.md#different-state-schemas)하는 것이 중요합니다.
- 전체 그래프 상태 스키마와 구별되는 [개인 입력 상태 스키마](../how-tos/graph-api.md#pass-private-state-between-nodes)로 에이전트 노드 함수를 정의합니다. 이를 통해 특정 에이전트를 실행하는 데만 필요한 정보를 전달할 수 있습니다.
