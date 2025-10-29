---
search:
  boost: 2
tags:
  - agent
hide:
  - tags
---

# 멀티-에이전트

단일 에이전트는 여러 도메인을 전문화하거나 많은 도구를 관리해야 하는 경우 어려움을 겪을 수 있습니다. 이를 해결하기 위해 에이전트를 더 작고 독립적인 에이전트로 분해하고 이를 [멀티-에이전트 시스템](../concepts/multi_agent.md)으로 구성할 수 있습니다.

멀티-에이전트 시스템에서 에이전트는 서로 통신해야 합니다. 이는 [handoff](#handoffs)를 통해 이루어지며, 이는 어떤 에이전트에게 제어권을 넘길지와 그 에이전트에게 전달할 페이로드를 설명하는 프리미티브입니다.

가장 인기 있는 두 가지 멀티-에이전트 아키텍처는 다음과 같습니다:

- [supervisor](#supervisor) — 개별 에이전트가 중앙 supervisor 에이전트에 의해 조정됩니다. supervisor는 모든 통신 흐름과 작업 위임을 제어하며, 현재 컨텍스트와 작업 요구 사항을 기반으로 어떤 에이전트를 호출할지 결정합니다.
- [swarm](#swarm) — 에이전트가 전문성에 따라 서로에게 동적으로 제어권을 넘깁니다. 시스템은 마지막으로 활성화된 에이전트를 기억하여, 후속 상호작용에서 해당 에이전트와 대화가 재개되도록 합니다.

## Supervisor

![Supervisor](./assets/supervisor.png)

:::python
[`langgraph-supervisor`](https://github.com/langchain-ai/langgraph-supervisor-py) 라이브러리를 사용하여 supervisor 멀티-에이전트 시스템을 만드세요:

```bash
pip install langgraph-supervisor
```

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
# highlight-next-line
from langgraph_supervisor import create_supervisor

def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."

def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."

flight_assistant = create_react_agent(
    model="openai:gpt-4o",
    tools=[book_flight],
    prompt="You are a flight booking assistant",
    # highlight-next-line
    name="flight_assistant"
)

hotel_assistant = create_react_agent(
    model="openai:gpt-4o",
    tools=[book_hotel],
    prompt="You are a hotel booking assistant",
    # highlight-next-line
    name="hotel_assistant"
)

# highlight-next-line
supervisor = create_supervisor(
    agents=[flight_assistant, hotel_assistant],
    model=ChatOpenAI(model="gpt-4o"),
    prompt=(
        "You manage a hotel booking assistant and a"
        "flight booking assistant. Assign work to them."
    )
).compile()

for chunk in supervisor.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "book a flight from BOS to JFK and a stay at McKittrick Hotel"
            }
        ]
    }
):
    print(chunk)
    print("\n")
```

:::

:::js
[`@langchain/langgraph-supervisor`](https://github.com/langchain-ai/langgraphjs/tree/main/libs/langgraph-supervisor) 라이브러리를 사용하여 supervisor 멀티-에이전트 시스템을 만드세요:

```bash
npm install @langchain/langgraph-supervisor
```

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
// highlight-next-line
import { createSupervisor } from "langgraph-supervisor";

function bookHotel(hotelName: string) {
  /**Book a hotel*/
  return `Successfully booked a stay at ${hotelName}.`;
}

function bookFlight(fromAirport: string, toAirport: string) {
  /**Book a flight*/
  return `Successfully booked a flight from ${fromAirport} to ${toAirport}.`;
}

const flightAssistant = createReactAgent({
  llm: "openai:gpt-4o",
  tools: [bookFlight],
  stateModifier: "You are a flight booking assistant",
  // highlight-next-line
  name: "flight_assistant",
});

const hotelAssistant = createReactAgent({
  llm: "openai:gpt-4o",
  tools: [bookHotel],
  stateModifier: "You are a hotel booking assistant",
  // highlight-next-line
  name: "hotel_assistant",
});

// highlight-next-line
const supervisor = createSupervisor({
  agents: [flightAssistant, hotelAssistant],
  llm: new ChatOpenAI({ model: "gpt-4o" }),
  systemPrompt:
    "You manage a hotel booking assistant and a " +
    "flight booking assistant. Assign work to them.",
});

for await (const chunk of supervisor.stream({
  messages: [
    {
      role: "user",
      content: "book a flight from BOS to JFK and a stay at McKittrick Hotel",
    },
  ],
})) {
  console.log(chunk);
  console.log("\n");
}
```

:::

## Swarm

![Swarm](./assets/swarm.png)

:::python
[`langgraph-swarm`](https://github.com/langchain-ai/langgraph-swarm-py) 라이브러리를 사용하여 swarm 멀티-에이전트 시스템을 만드세요:

```bash
pip install langgraph-swarm
```

```python
from langgraph.prebuilt import create_react_agent
# highlight-next-line
from langgraph_swarm import create_swarm, create_handoff_tool

transfer_to_hotel_assistant = create_handoff_tool(
    agent_name="hotel_assistant",
    description="Transfer user to the hotel-booking assistant.",
)
transfer_to_flight_assistant = create_handoff_tool(
    agent_name="flight_assistant",
    description="Transfer user to the flight-booking assistant.",
)

flight_assistant = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    # highlight-next-line
    tools=[book_flight, transfer_to_hotel_assistant],
    prompt="You are a flight booking assistant",
    # highlight-next-line
    name="flight_assistant"
)
hotel_assistant = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    # highlight-next-line
    tools=[book_hotel, transfer_to_flight_assistant],
    prompt="You are a hotel booking assistant",
    # highlight-next-line
    name="hotel_assistant"
)

# highlight-next-line
swarm = create_swarm(
    agents=[flight_assistant, hotel_assistant],
    default_active_agent="flight_assistant"
).compile()

for chunk in swarm.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "book a flight from BOS to JFK and a stay at McKittrick Hotel"
            }
        ]
    }
):
    print(chunk)
    print("\n")
```

:::

:::js
[`@langchain/langgraph-swarm`](https://github.com/langchain-ai/langgraphjs/tree/main/libs/langgraph-swarm) 라이브러리를 사용하여 swarm 멀티-에이전트 시스템을 만드세요:

```bash
npm install @langchain/langgraph-swarm
```

```typescript
import { createReactAgent } from "@langchain/langgraph/prebuilt";
// highlight-next-line
import { createSwarm, createHandoffTool } from "@langchain/langgraph-swarm";

const transferToHotelAssistant = createHandoffTool({
  agentName: "hotel_assistant",
  description: "Transfer user to the hotel-booking assistant.",
});

const transferToFlightAssistant = createHandoffTool({
  agentName: "flight_assistant",
  description: "Transfer user to the flight-booking assistant.",
});

const flightAssistant = createReactAgent({
  llm: "anthropic:claude-3-5-sonnet-latest",
  // highlight-next-line
  tools: [bookFlight, transferToHotelAssistant],
  stateModifier: "You are a flight booking assistant",
  // highlight-next-line
  name: "flight_assistant",
});

const hotelAssistant = createReactAgent({
  llm: "anthropic:claude-3-5-sonnet-latest",
  // highlight-next-line
  tools: [bookHotel, transferToFlightAssistant],
  stateModifier: "You are a hotel booking assistant",
  // highlight-next-line
  name: "hotel_assistant",
});

// highlight-next-line
const swarm = createSwarm({
  agents: [flightAssistant, hotelAssistant],
  defaultActiveAgent: "flight_assistant",
});

for await (const chunk of swarm.stream({
  messages: [
    {
      role: "user",
      content: "book a flight from BOS to JFK and a stay at McKittrick Hotel",
    },
  ],
})) {
  console.log(chunk);
  console.log("\n");
}
```

:::

## Handoffs

멀티-에이전트 상호작용의 일반적인 패턴은 **handoff**로, 한 에이전트가 다른 에이전트에게 제어권을 _넘기는_ 것입니다. handoff를 사용하면 다음을 지정할 수 있습니다:

- **destination**: 이동할 대상 에이전트
- **payload**: 해당 에이전트에게 전달할 정보

:::python
이는 `langgraph-supervisor`(supervisor가 개별 에이전트에게 handoff)와 `langgraph-swarm`(개별 에이전트가 다른 에이전트에게 handoff 가능) 모두에서 사용됩니다.

`create_react_agent`로 handoff를 구현하려면 다음이 필요합니다:

1.  다른 에이전트로 제어권을 전송할 수 있는 특별한 도구를 만듭니다

    ```python
    def transfer_to_bob():
        """Transfer to bob."""
        return Command(
            # name of the agent (node) to go to
            # highlight-next-line
            goto="bob",
            # data to send to the agent
            # highlight-next-line
            update={"messages": [...]},
            # indicate to LangGraph that we need to navigate to
            # agent node in a parent graph
            # highlight-next-line
            graph=Command.PARENT,
        )
    ```

2.  handoff 도구에 접근할 수 있는 개별 에이전트를 만듭니다:

    ```python
    flight_assistant = create_react_agent(
        ..., tools=[book_flight, transfer_to_hotel_assistant]
    )
    hotel_assistant = create_react_agent(
        ..., tools=[book_hotel, transfer_to_flight_assistant]
    )
    ```

3.  개별 에이전트를 노드로 포함하는 부모 그래프를 정의합니다:

    ```python
    from langgraph.graph import StateGraph, MessagesState
    multi_agent_graph = (
        StateGraph(MessagesState)
        .add_node(flight_assistant)
        .add_node(hotel_assistant)
        ...
    )
    ```

:::

:::js
이는 `@langchain/langgraph-supervisor`(supervisor가 개별 에이전트에게 handoff)와 `@langchain/langgraph-swarm`(개별 에이전트가 다른 에이전트에게 handoff 가능) 모두에서 사용됩니다.

`createReactAgent`로 handoff를 구현하려면 다음이 필요합니다:

1.  다른 에이전트로 제어권을 전송할 수 있는 특별한 도구를 만듭니다

    ```typescript
    function transferToBob() {
      /**Transfer to bob.*/
      return new Command({
        // name of the agent (node) to go to
        // highlight-next-line
        goto: "bob",
        // data to send to the agent
        // highlight-next-line
        update: { messages: [...] },
        // indicate to LangGraph that we need to navigate to
        // agent node in a parent graph
        // highlight-next-line
        graph: Command.PARENT,
      });
    }
    ```

2.  handoff 도구에 접근할 수 있는 개별 에이전트를 만듭니다:

    ```typescript
    const flightAssistant = createReactAgent({
      ..., tools: [bookFlight, transferToHotelAssistant]
    });
    const hotelAssistant = createReactAgent({
      ..., tools: [bookHotel, transferToFlightAssistant]
    });
    ```

3.  개별 에이전트를 노드로 포함하는 부모 그래프를 정의합니다:

    ```typescript
    import { StateGraph, MessagesZodState } from "@langchain/langgraph";
    const multiAgentGraph = new StateGraph(MessagesZodState)
      .addNode("flight_assistant", flightAssistant)
      .addNode("hotel_assistant", hotelAssistant)
      // ...
    ```

    :::

이를 종합하여, 다음은 항공편 예약 도우미와 호텔 예약 도우미라는 두 에이전트가 있는 간단한 멀티-에이전트 시스템을 구현하는 방법입니다:

:::python

```python
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command

def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Transfer to {agent_name}"

    @tool(name, description=description)
    def handoff_tool(
        # highlight-next-line
        state: Annotated[MessagesState, InjectedState], # (1)!
        # highlight-next-line
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(  # (2)!
            # highlight-next-line
            goto=agent_name,  # (3)!
            # highlight-next-line
            update={"messages": state["messages"] + [tool_message]},  # (4)!
            # highlight-next-line
            graph=Command.PARENT,  # (5)!
        )
    return handoff_tool

# Handoffs
transfer_to_hotel_assistant = create_handoff_tool(
    agent_name="hotel_assistant",
    description="Transfer user to the hotel-booking assistant.",
)
transfer_to_flight_assistant = create_handoff_tool(
    agent_name="flight_assistant",
    description="Transfer user to the flight-booking assistant.",
)

# Simple agent tools
def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."

def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."

# Define agents
flight_assistant = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    # highlight-next-line
    tools=[book_flight, transfer_to_hotel_assistant],
    prompt="You are a flight booking assistant",
    # highlight-next-line
    name="flight_assistant"
)
hotel_assistant = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    # highlight-next-line
    tools=[book_hotel, transfer_to_flight_assistant],
    prompt="You are a hotel booking assistant",
    # highlight-next-line
    name="hotel_assistant"
)

# Define multi-agent graph
multi_agent_graph = (
    StateGraph(MessagesState)
    .add_node(flight_assistant)
    .add_node(hotel_assistant)
    .add_edge(START, "flight_assistant")
    .compile()
)

# Run the multi-agent graph
for chunk in multi_agent_graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "book a flight from BOS to JFK and a stay at McKittrick Hotel"
            }
        ]
    }
):
    print(chunk)
    print("\n")
```

1. 에이전트의 state에 접근
2. `Command` 프리미티브를 사용하면 state 업데이트와 노드 전환을 단일 작업으로 지정할 수 있어 handoff를 구현하는 데 유용합니다.
3. handoff할 에이전트 또는 노드의 이름
4. 에이전트의 메시지를 가져와서 handoff의 일부로 부모의 **state**에 **추가**합니다. 다음 에이전트는 부모 state를 보게 됩니다.
5. **부모** 멀티-에이전트 그래프의 에이전트 노드로 이동해야 한다는 것을 LangGraph에 표시합니다.
   :::

:::js

```typescript
import { tool } from "@langchain/core/tools";
import { ChatAnthropic } from "@langchain/anthropic";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import {
  StateGraph,
  START,
  MessagesZodState,
  Command,
} from "@langchain/langgraph";
import { z } from "zod";

function createHandoffTool({
  agentName,
  description,
}: {
  agentName: string;
  description?: string;
}) {
  const name = `transfer_to_${agentName}`;
  const toolDescription = description || `Transfer to ${agentName}`;

  return tool(
    async (_, config) => {
      const toolMessage = {
        role: "tool" as const,
        content: `Successfully transferred to ${agentName}`,
        name: name,
        tool_call_id: config.toolCall?.id!,
      };
      return new Command({
        // (2)!
        // highlight-next-line
        goto: agentName, // (3)!
        // highlight-next-line
        update: { messages: [toolMessage] }, // (4)!
        // highlight-next-line
        graph: Command.PARENT, // (5)!
      });
    },
    {
      name,
      description: toolDescription,
      schema: z.object({}),
    }
  );
}

// Handoffs
const transferToHotelAssistant = createHandoffTool({
  agentName: "hotel_assistant",
  description: "Transfer user to the hotel-booking assistant.",
});

const transferToFlightAssistant = createHandoffTool({
  agentName: "flight_assistant",
  description: "Transfer user to the flight-booking assistant.",
});

// Simple agent tools
const bookHotel = tool(
  async ({ hotelName }) => {
    /**Book a hotel*/
    return `Successfully booked a stay at ${hotelName}.`;
  },
  {
    name: "book_hotel",
    description: "Book a hotel",
    schema: z.object({
      hotelName: z.string().describe("Name of the hotel to book"),
    }),
  }
);

const bookFlight = tool(
  async ({ fromAirport, toAirport }) => {
    /**Book a flight*/
    return `Successfully booked a flight from ${fromAirport} to ${toAirport}.`;
  },
  {
    name: "book_flight",
    description: "Book a flight",
    schema: z.object({
      fromAirport: z.string().describe("Departure airport code"),
      toAirport: z.string().describe("Arrival airport code"),
    }),
  }
);

// Define agents
const flightAssistant = createReactAgent({
  llm: new ChatAnthropic({ model: "anthropic:claude-3-5-sonnet-latest" }),
  // highlight-next-line
  tools: [bookFlight, transferToHotelAssistant],
  stateModifier: "You are a flight booking assistant",
  // highlight-next-line
  name: "flight_assistant",
});

const hotelAssistant = createReactAgent({
  llm: new ChatAnthropic({ model: "anthropic:claude-3-5-sonnet-latest" }),
  // highlight-next-line
  tools: [bookHotel, transferToFlightAssistant],
  stateModifier: "You are a hotel booking assistant",
  // highlight-next-line
  name: "hotel_assistant",
});

// Define multi-agent graph
const multiAgentGraph = new StateGraph(MessagesZodState)
  .addNode("flight_assistant", flightAssistant)
  .addNode("hotel_assistant", hotelAssistant)
  .addEdge(START, "flight_assistant")
  .compile();

// Run the multi-agent graph
for await (const chunk of multiAgentGraph.stream({
  messages: [
    {
      role: "user",
      content: "book a flight from BOS to JFK and a stay at McKittrick Hotel",
    },
  ],
})) {
  console.log(chunk);
  console.log("\n");
}
```

1. 에이전트의 state에 접근
2. `Command` 프리미티브를 사용하면 state 업데이트와 노드 전환을 단일 작업으로 지정할 수 있어 handoff를 구현하는 데 유용합니다.
3. handoff할 에이전트 또는 노드의 이름
4. 에이전트의 메시지를 가져와서 handoff의 일부로 부모의 **state**에 **추가**합니다. 다음 에이전트는 부모 state를 보게 됩니다.
5. **부모** 멀티-에이전트 그래프의 에이전트 노드로 이동해야 한다는 것을 LangGraph에 표시합니다.

:::

!!! Note

    이 handoff 구현은 다음을 가정합니다:

    - 각 에이전트는 멀티-에이전트 시스템의 전체 메시지 기록(모든 에이전트 전체)을 입력으로 받습니다
    - 각 에이전트는 내부 메시지 기록을 멀티-에이전트 시스템의 전체 메시지 기록으로 출력합니다

:::python
handoff를 커스터마이징하는 방법을 알아보려면 LangGraph [supervisor](https://github.com/langchain-ai/langgraph-supervisor-py#customizing-handoff-tools) 및 [swarm](https://github.com/langchain-ai/langgraph-swarm-py#customizing-handoff-tools) 문서를 확인하세요.
:::

:::js
handoff를 커스터마이징하는 방법을 알아보려면 LangGraph [supervisor](https://github.com/langchain-ai/langgraphjs/tree/main/libs/langgraph-supervisor#customizing-handoff-tools) 및 [swarm](https://github.com/langchain-ai/langgraphjs/tree/main/libs/langgraph-swarm#customizing-handoff-tools) 문서를 확인하세요.
:::
