# 다중 에이전트 시스템 구축

단일 에이전트는 여러 도메인에 특화되거나 많은 도구를 관리해야 하는 경우 어려움을 겪을 수 있습니다. 이를 해결하기 위해 에이전트를 더 작고 독립적인 에이전트로 나누고 이를 [다중 에이전트 시스템](../concepts/multi_agent.md)으로 구성할 수 있습니다.

다중 에이전트 시스템에서 에이전트는 서로 통신해야 합니다. 이들은 [핸드오프](#handoffs)를 통해 통신합니다 — 어떤 에이전트에게 제어권을 넘길지와 해당 에이전트에 보낼 페이로드를 설명하는 프리미티브입니다.

이 가이드는 다음을 다룹니다:

* 에이전트 간 [핸드오프](#handoffs) 구현
* 핸드오프와 사전 빌드된 [에이전트](../agents/agents.md)를 사용하여 [커스텀 다중 에이전트 시스템 구축](#build-a-multi-agent-system)

다중 에이전트 시스템 구축을 시작하려면 가장 인기 있는 두 가지 다중 에이전트 아키텍처인 [supervisor](../agents/multi-agent.md#supervisor)와 [swarm](../agents/multi-agent.md#swarm)의 LangGraph [사전 빌드 구현](#prebuilt-implementations)을 확인하세요.

## 핸드오프

다중 에이전트 시스템에서 에이전트 간 통신을 설정하려면 [**핸드오프**](../concepts/multi_agent.md#handoffs)를 사용할 수 있습니다 — 한 에이전트가 다른 에이전트에게 제어권을 *넘기는* 패턴입니다. 핸드오프를 사용하면 다음을 지정할 수 있습니다:

- **destination**: 탐색할 대상 에이전트 (예: 이동할 LangGraph 노드의 이름)
- **payload**: 해당 에이전트에 전달할 정보 (예: 상태 업데이트)

### 핸드오프 생성

핸드오프를 구현하려면 에이전트 노드 또는 도구에서 `Command` 객체를 반환할 수 있습니다:

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
```

1. @[InjectedState] 어노테이션을 사용하여 핸드오프 도구를 호출하는 에이전트의 [상태](../concepts/low_level.md#state)에 액세스합니다.
2. `Command` 프리미티브는 상태 업데이트와 노드 전환을 단일 작업으로 지정할 수 있어 핸드오프 구현에 유용합니다.
3. 핸드오프할 에이전트 또는 노드의 이름입니다.
4. 에이전트의 메시지를 가져와 핸드오프의 일부로 부모의 **상태**에 **추가**합니다. 다음 에이전트는 부모 상태를 볼 수 있습니다.
5. **부모** 다중 에이전트 그래프의 에이전트 노드로 이동해야 함을 LangGraph에 나타냅니다.

!!! tip

    If you want to use tools that return `Command`, you can either use prebuilt @[`create_react_agent`][create_react_agent] / @[`ToolNode`][ToolNode] components, or implement your own tool-executing node that collects `Command` objects returned by the tools and returns a list of them, e.g.:
    
    ```python
    def call_tools(state):
        ...
        commands = [tools_by_name[tool_call["name"]].invoke(tool_call) for tool_call in tool_calls]
        return commands
    ```
:::

:::js
```typescript
import { tool } from "@langchain/core/tools";
import { Command, MessagesZodState } from "@langchain/langgraph";
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
      // (1)!
      const state = config.state;
      const toolCallId = config.toolCall.id;

      const toolMessage = {
        role: "tool" as const,
        content: `Successfully transferred to ${agentName}`,
        name: name,
        tool_call_id: toolCallId,
      };

      return new Command({
        // (3)!
        goto: agentName,
        // (4)!
        update: { messages: [...state.messages, toolMessage] },
        // (5)!
        graph: Command.PARENT,
      });
    },
    {
      name,
      description: toolDescription,
      schema: z.object({}),
    }
  );
}
```

1. Access the [state](../concepts/low_level.md#state) of the agent that is calling the handoff tool through the `config` parameter.
2. The `Command` primitive allows specifying a state update and a node transition as a single operation, making it useful for implementing handoffs.
3. Name of the agent or node to hand off to.
4. Take the agent's messages and **add** them to the parent's **state** as part of the handoff. The next agent will see the parent state.
5. Indicate to LangGraph that we need to navigate to agent node in a **parent** multi-agent graph.

!!! tip

    If you want to use tools that return `Command`, you can either use prebuilt @[`create_react_agent`][create_react_agent] / @[`ToolNode`][ToolNode] components, or implement your own tool-executing node that collects `Command` objects returned by the tools and returns a list of them, e.g.:
    
    ```typescript
    const callTools = async (state) => {
      // ...
      const commands = await Promise.all(
        toolCalls.map(toolCall => toolsByName[toolCall.name].invoke(toolCall))
      );
      return commands;
    };
    ```
:::

!!! Important

    This handoff implementation assumes that:
    
    - each agent receives overall message history (across all agents) in the multi-agent system as its input. If you want more control over agent inputs, see [this section](#control-agent-inputs)
    - each agent outputs its internal messages history to the overall message history of the multi-agent system. If you want more control over **how agent outputs are added**, wrap the agent in a separate node function:

      :::python
      ```python
      def call_hotel_assistant(state):
          # return agent's final response,
          # excluding inner monologue
          response = hotel_assistant.invoke(state)
          # highlight-next-line
          return {"messages": response["messages"][-1]}
      ```
      :::

      :::js
      ```typescript
      const callHotelAssistant = async (state) => {
        // return agent's final response,
        // excluding inner monologue
        const response = await hotelAssistant.invoke(state);
        // highlight-next-line
        return { messages: [response.messages.at(-1)] };
      };
      ```
      :::

### 에이전트 입력 제어

:::python
@[`Send()`][Send] 프리미티브를 사용하여 핸드오프 중에 작업자 에이전트에 직접 데이터를 보낼 수 있습니다. 예를 들어, 호출하는 에이전트가 다음 에이전트를 위한 작업 설명을 채우도록 요청할 수 있습니다:

```python

from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState
# highlight-next-line
from langgraph.types import Command, Send

def create_task_description_handoff_tool(
    *, agent_name: str, description: str | None = None
):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        # this is populated by the calling agent
        task_description: Annotated[
            str,
            "Description of what the next agent should do, including all of the relevant context.",
        ],
        # these parameters are ignored by the LLM
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        task_description_message = {"role": "user", "content": task_description}
        agent_input = {**state, "messages": [task_description_message]}
        return Command(
            # highlight-next-line
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )

    return handoff_tool
```
:::

:::js
You can use the @[`Send()`][Send] primitive to directly send data to the worker agents during the handoff. For example, you can request that the calling agent populate a task description for the next agent:

```typescript
import { tool } from "@langchain/core/tools";
import { Command, Send, MessagesZodState } from "@langchain/langgraph";
import { z } from "zod";

function createTaskDescriptionHandoffTool({
  agentName,
  description,
}: {
  agentName: string;
  description?: string;
}) {
  const name = `transfer_to_${agentName}`;
  const toolDescription = description || `Ask ${agentName} for help.`;

  return tool(
    async (
      { taskDescription },
      config
    ) => {
      const state = config.state;
      
      const taskDescriptionMessage = {
        role: "user" as const,
        content: taskDescription,
      };
      const agentInput = {
        ...state,
        messages: [taskDescriptionMessage],
      };
      
      return new Command({
        // highlight-next-line
        goto: [new Send(agentName, agentInput)],
        graph: Command.PARENT,
      });
    },
    {
      name,
      description: toolDescription,
      schema: z.object({
        taskDescription: z
          .string()
          .describe(
            "Description of what the next agent should do, including all of the relevant context."
          ),
      }),
    }
  );
}
```
:::

See the multi-agent [supervisor](../tutorials/multi_agent/agent_supervisor.md#4-create-delegation-tasks) example for a full example of using @[`Send()`][Send] in handoffs.

## 다중 에이전트 시스템 구축

LangGraph로 구축된 모든 에이전트에서 핸드오프를 사용할 수 있습니다. `Command`를 반환하는 핸드오프 도구를 기본적으로 지원하는 사전 빌드된 [에이전트](../agents/overview.md) 또는 [`ToolNode`](./tool-calling.md#toolnode) 사용을 권장합니다. 다음은 핸드오프를 사용하여 여행 예약을 위한 다중 에이전트 시스템을 구현하는 방법의 예입니다:

:::python
```python
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, MessagesState

def create_handoff_tool(*, agent_name: str, description: str | None = None):
    # same implementation as above
    ...
    return Command(...)

# Handoffs
transfer_to_hotel_assistant = create_handoff_tool(agent_name="hotel_assistant")
transfer_to_flight_assistant = create_handoff_tool(agent_name="flight_assistant")

# Define agents
flight_assistant = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    # highlight-next-line
    tools=[..., transfer_to_hotel_assistant],
    # highlight-next-line
    name="flight_assistant"
)
hotel_assistant = create_react_agent(
    model="anthropic:claude-3-5-sonnet-latest",
    # highlight-next-line
    tools=[..., transfer_to_flight_assistant],
    # highlight-next-line
    name="hotel_assistant"
)

# Define multi-agent graph
multi_agent_graph = (
    StateGraph(MessagesState)
    # highlight-next-line
    .add_node(flight_assistant)
    # highlight-next-line
    .add_node(hotel_assistant)
    .add_edge(START, "flight_assistant")
    .compile()
)
```
:::

:::js
```typescript
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { StateGraph, START, MessagesZodState } from "@langchain/langgraph";
import { z } from "zod";

function createHandoffTool({
  agentName,
  description,
}: {
  agentName: string;
  description?: string;
}) {
  // same implementation as above
  // ...
  return new Command(/* ... */);
}

// Handoffs
const transferToHotelAssistant = createHandoffTool({
  agentName: "hotel_assistant",
});
const transferToFlightAssistant = createHandoffTool({
  agentName: "flight_assistant",
});

// Define agents
const flightAssistant = createReactAgent({
  llm: model,
  // highlight-next-line
  tools: [/* ... */, transferToHotelAssistant],
  // highlight-next-line
  name: "flight_assistant",
});

const hotelAssistant = createReactAgent({
  llm: model,
  // highlight-next-line
  tools: [/* ... */, transferToFlightAssistant],
  // highlight-next-line
  name: "hotel_assistant",
});

// Define multi-agent graph
const multiAgentGraph = new StateGraph(MessagesZodState)
  // highlight-next-line
  .addNode("flight_assistant", flightAssistant)
  // highlight-next-line
  .addNode("hotel_assistant", hotelAssistant)
  .addEdge(START, "flight_assistant")
  .compile();
```
:::

??? example "Full example: Multi-agent system for booking travel"

    :::python
    ```python
    from typing import Annotated
    from langchain_core.messages import convert_to_messages
    from langchain_core.tools import tool, InjectedToolCallId
    from langgraph.prebuilt import create_react_agent, InjectedState
    from langgraph.graph import StateGraph, START, MessagesState
    from langgraph.types import Command
    
    # We'll use `pretty_print_messages` helper to render the streamed agent outputs nicely
    
    def pretty_print_message(message, indent=False):
        pretty_message = message.pretty_repr(html=True)
        if not indent:
            print(pretty_message)
            return
    
        indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
        print(indented)
    
    
    def pretty_print_messages(update, last_message=False):
        is_subgraph = False
        if isinstance(update, tuple):
            ns, update = update
            # skip parent graph updates in the printouts
            if len(ns) == 0:
                return
    
            graph_id = ns[-1].split(":")[0]
            print(f"Update from subgraph {graph_id}:")
            print("\n")
            is_subgraph = True
    
        for node_name, node_update in update.items():
            update_label = f"Update from node {node_name}:"
            if is_subgraph:
                update_label = "\t" + update_label
    
            print(update_label)
            print("\n")
    
            messages = convert_to_messages(node_update["messages"])
            if last_message:
                messages = messages[-1:]
    
            for m in messages:
                pretty_print_message(m, indent=is_subgraph)
            print("\n")


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
        },
        # highlight-next-line
        subgraphs=True
    ):
        pretty_print_messages(chunk)
    ```

    1. Access agent's state
    2. The `Command` primitive allows specifying a state update and a node transition as a single operation, making it useful for implementing handoffs.
    3. Name of the agent or node to hand off to.
    4. Take the agent's messages and **add** them to the parent's **state** as part of the handoff. The next agent will see the parent state.
    5. Indicate to LangGraph that we need to navigate to agent node in a **parent** multi-agent graph.
    :::

    :::js
    ```typescript
    import { tool } from "@langchain/core/tools";
    import { createReactAgent } from "@langchain/langgraph/prebuilt";
    import { StateGraph, START, MessagesZodState, Command } from "@langchain/langgraph";
    import { ChatAnthropic } from "@langchain/anthropic";
    import { isBaseMessage } from "@langchain/core/messages";
    import { z } from "zod";

    // We'll use a helper to render the streamed agent outputs nicely
    const prettyPrintMessages = (update: Record<string, any>) => {
      // Handle tuple case with namespace
      if (Array.isArray(update)) {
        const [ns, updateData] = update;
        // Skip parent graph updates in the printouts
        if (ns.length === 0) {
          return;
        }

        const graphId = ns[ns.length - 1].split(":")[0];
        console.log(`Update from subgraph ${graphId}:\n`);
        update = updateData;
      }

      for (const [nodeName, updateValue] of Object.entries(update)) {
        console.log(`Update from node ${nodeName}:\n`);

        const messages = updateValue.messages || [];
        for (const message of messages) {
          if (isBaseMessage(message)) {
            const textContent =
              typeof message.content === "string"
                ? message.content
                : JSON.stringify(message.content);
            console.log(`${message.getType()}: ${textContent}`);
          }
        }
        console.log("\n");
      }
    };

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
          // highlight-next-line
          const state = config.state; // (1)!
          const toolCallId = config.toolCall.id;

          const toolMessage = {
            role: "tool" as const,
            content: `Successfully transferred to ${agentName}`,
            name: name,
            tool_call_id: toolCallId,
          };

          return new Command({
            // highlight-next-line
            goto: agentName, // (3)!
            // highlight-next-line
            update: { messages: [...state.messages, toolMessage] }, // (4)!
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
        return `Successfully booked a stay at ${hotelName}.`;
      },
      {
        name: "book_hotel",
        description: "Book a hotel",
        schema: z.object({
          hotelName: z.string(),
        }),
      }
    );

    const bookFlight = tool(
      async ({ fromAirport, toAirport }) => {
        return `Successfully booked a flight from ${fromAirport} to ${toAirport}.`;
      },
      {
        name: "book_flight",
        description: "Book a flight",
        schema: z.object({
          fromAirport: z.string(),
          toAirport: z.string(),
        }),
      }
    );

    const model = new ChatAnthropic({
      model: "claude-3-5-sonnet-latest",
    });

    // Define agents
    const flightAssistant = createReactAgent({
      llm: model,
      // highlight-next-line
      tools: [bookFlight, transferToHotelAssistant],
      prompt: "You are a flight booking assistant",
      // highlight-next-line
      name: "flight_assistant",
    });

    const hotelAssistant = createReactAgent({
      llm: model,
      // highlight-next-line
      tools: [bookHotel, transferToFlightAssistant],
      prompt: "You are a hotel booking assistant",
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
    const stream = await multiAgentGraph.stream(
      {
        messages: [
          {
            role: "user",
            content: "book a flight from BOS to JFK and a stay at McKittrick Hotel",
          },
        ],
      },
      // highlight-next-line
      { subgraphs: true }
    );

    for await (const chunk of stream) {
      prettyPrintMessages(chunk);
    }
    ```

    1. Access agent's state
    2. The `Command` primitive allows specifying a state update and a node transition as a single operation, making it useful for implementing handoffs.
    3. Name of the agent or node to hand off to.
    4. Take the agent's messages and **add** them to the parent's **state** as part of the handoff. The next agent will see the parent state.
    5. Indicate to LangGraph that we need to navigate to agent node in a **parent** multi-agent graph.
    :::

## 다중 턴 대화

사용자는 하나 이상의 에이전트와 *다중 턴 대화*를 원할 수 있습니다. 이를 처리할 수 있는 시스템을 구축하려면 @[`interrupt`][interrupt]를 사용하여 사용자 입력을 수집하고 **활성** 에이전트로 다시 라우팅하는 노드를 생성할 수 있습니다.

그런 다음 에이전트는 에이전트 단계를 실행하고 다음 작업을 결정하는 그래프의 노드로 구현될 수 있습니다:

1. 대화를 계속하기 위해 **사용자 입력을 기다리거나**, 또는
2. [핸드오프](#handoffs)를 통해 **다른 에이전트로 라우팅** (또는 루프처럼 자신에게 다시 라우팅)

:::python
```python
def human(state) -> Command[Literal["agent", "another_agent"]]:
    """A node for collecting user input."""
    user_input = interrupt(value="Ready for user input.")

    # Determine the active agent.
    active_agent = ...

    ...
    return Command(
        update={
            "messages": [{
                "role": "human",
                "content": user_input,
            }]
        },
        goto=active_agent
    )

def agent(state) -> Command[Literal["agent", "another_agent", "human"]]:
    # The condition for routing/halting can be anything, e.g. LLM tool call / structured output, etc.
    goto = get_next_agent(...)  # 'agent' / 'another_agent'
    if goto:
        return Command(goto=goto, update={"my_state_key": "my_state_value"})
    else:
        return Command(goto="human") # Go to human node
```
:::

:::js
```typescript
import { interrupt, Command } from "@langchain/langgraph";

function human(state: MessagesState): Command {
  const userInput: string = interrupt("Ready for user input.");

  // Determine the active agent
  const activeAgent = /* ... */;

  return new Command({
    update: {
      messages: [{
        role: "human",
        content: userInput,
      }]
    },
    goto: activeAgent,
  });
}

function agent(state: MessagesState): Command {
  // The condition for routing/halting can be anything, e.g. LLM tool call / structured output, etc.
  const goto = getNextAgent(/* ... */); // 'agent' / 'anotherAgent'

  if (goto) {
    return new Command({
      goto,
      update: { myStateKey: "myStateValue" }
    });
  }

  return new Command({ goto: "human" });
}
```
:::

??? example "Full example: multi-agent system for travel recommendations"

    In this example, we will build a team of travel assistant agents that can communicate with each other via handoffs.
    
    We will create 2 agents:
    
    * travel_advisor: can help with travel destination recommendations. Can ask hotel_advisor for help.
    * hotel_advisor: can help with hotel recommendations. Can ask travel_advisor for help.

    :::python
    ```python
    from langchain_anthropic import ChatAnthropic
    from langgraph.graph import MessagesState, StateGraph, START
    from langgraph.prebuilt import create_react_agent, InjectedState
    from langgraph.types import Command, interrupt
    from langgraph.checkpoint.memory import InMemorySaver
    
    
    model = ChatAnthropic(model="claude-3-5-sonnet-latest")

    class MultiAgentState(MessagesState):
        last_active_agent: str
    
    
    # Define travel advisor tools and ReAct agent
    travel_advisor_tools = [
        get_travel_recommendations,
        make_handoff_tool(agent_name="hotel_advisor"),
    ]
    travel_advisor = create_react_agent(
        model,
        travel_advisor_tools,
        prompt=(
            "You are a general travel expert that can recommend travel destinations (e.g. countries, cities, etc). "
            "If you need hotel recommendations, ask 'hotel_advisor' for help. "
            "You MUST include human-readable response before transferring to another agent."
        ),
    )
    
    
    def call_travel_advisor(
        state: MultiAgentState,
    ) -> Command[Literal["hotel_advisor", "human"]]:
        # You can also add additional logic like changing the input to the agent / output from the agent, etc.
        # NOTE: we're invoking the ReAct agent with the full history of messages in the state
        response = travel_advisor.invoke(state)
        update = {**response, "last_active_agent": "travel_advisor"}
        return Command(update=update, goto="human")
    
    
    # Define hotel advisor tools and ReAct agent
    hotel_advisor_tools = [
        get_hotel_recommendations,
        make_handoff_tool(agent_name="travel_advisor"),
    ]
    hotel_advisor = create_react_agent(
        model,
        hotel_advisor_tools,
        prompt=(
            "You are a hotel expert that can provide hotel recommendations for a given destination. "
            "If you need help picking travel destinations, ask 'travel_advisor' for help."
            "You MUST include human-readable response before transferring to another agent."
        ),
    )
    
    
    def call_hotel_advisor(
        state: MultiAgentState,
    ) -> Command[Literal["travel_advisor", "human"]]:
        response = hotel_advisor.invoke(state)
        update = {**response, "last_active_agent": "hotel_advisor"}
        return Command(update=update, goto="human")
    
    
    def human_node(
        state: MultiAgentState, config
    ) -> Command[Literal["hotel_advisor", "travel_advisor", "human"]]:
        """A node for collecting user input."""
    
        user_input = interrupt(value="Ready for user input.")
        active_agent = state["last_active_agent"]
    
        return Command(
            update={
                "messages": [
                    {
                        "role": "human",
                        "content": user_input,
                    }
                ]
            },
            goto=active_agent,
        )
    
    
    builder = StateGraph(MultiAgentState)
    builder.add_node("travel_advisor", call_travel_advisor)
    builder.add_node("hotel_advisor", call_hotel_advisor)
    
    # This adds a node to collect human input, which will route
    # back to the active agent.
    builder.add_node("human", human_node)
    
    # We'll always start with a general travel advisor.
    builder.add_edge(START, "travel_advisor")
    
    
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    ```
    
    Let's test a multi turn conversation with this application.

    ```python
    import uuid
    
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    inputs = [
        # 1st round of conversation,
        {
            "messages": [
                {"role": "user", "content": "i wanna go somewhere warm in the caribbean"}
            ]
        },
        # Since we're using `interrupt`, we'll need to resume using the Command primitive.
        # 2nd round of conversation,
        Command(
            resume="could you recommend a nice hotel in one of the areas and tell me which area it is."
        ),
        # 3rd round of conversation,
        Command(
            resume="i like the first one. could you recommend something to do near the hotel?"
        ),
    ]
    
    for idx, user_input in enumerate(inputs):
        print()
        print(f"--- Conversation Turn {idx + 1} ---")
        print()
        print(f"User: {user_input}")
        print()
        for update in graph.stream(
            user_input,
            config=thread_config,
            stream_mode="updates",
        ):
            for node_id, value in update.items():
                if isinstance(value, dict) and value.get("messages", []):
                    last_message = value["messages"][-1]
                    if isinstance(last_message, dict) or last_message.type != "ai":
                        continue
                    print(f"{node_id}: {last_message.content}")
    ```
    
    ```
    --- Conversation Turn 1 ---
    
    User: {'messages': [{'role': 'user', 'content': 'i wanna go somewhere warm in the caribbean'}]}
    
    travel_advisor: Based on the recommendations, Aruba would be an excellent choice for your Caribbean getaway! Aruba is known as "One Happy Island" and offers:
    - Year-round warm weather with consistent temperatures around 82°F (28°C)
    - Beautiful white sand beaches like Eagle Beach and Palm Beach
    - Clear turquoise waters perfect for swimming and snorkeling
    - Minimal rainfall and location outside the hurricane belt
    - A blend of Caribbean and Dutch culture
    - Great dining options and nightlife
    - Various water sports and activities
    
    Would you like me to get some specific hotel recommendations in Aruba for your stay? I can transfer you to our hotel advisor who can help with accommodations.
    
    --- Conversation Turn 2 ---
    
    User: Command(resume='could you recommend a nice hotel in one of the areas and tell me which area it is.')
    
    hotel_advisor: Based on the recommendations, I can suggest two excellent options:
    
    1. The Ritz-Carlton, Aruba - Located in Palm Beach
    - This luxury resort is situated in the vibrant Palm Beach area
    - Known for its exceptional service and amenities
    - Perfect if you want to be close to dining, shopping, and entertainment
    - Features multiple restaurants, a casino, and a world-class spa
    - Located on a pristine stretch of Palm Beach
    
    2. Bucuti & Tara Beach Resort - Located in Eagle Beach
    - An adults-only boutique resort on Eagle Beach
    - Known for being more intimate and peaceful
    - Award-winning for its sustainability practices
    - Perfect for a romantic getaway or peaceful vacation
    - Located on one of the most beautiful beaches in the Caribbean
    
    Would you like more specific information about either of these properties or their locations?
    
    --- Conversation Turn 3 ---
    
    User: Command(resume='i like the first one. could you recommend something to do near the hotel?')
    
    travel_advisor: Near the Ritz-Carlton in Palm Beach, here are some highly recommended activities:
    
    1. Visit the Palm Beach Plaza Mall - Just a short walk from the hotel, featuring shopping, dining, and entertainment
    2. Try your luck at the Stellaris Casino - It's right in the Ritz-Carlton
    3. Take a sunset sailing cruise - Many depart from the nearby pier
    4. Visit the California Lighthouse - A scenic landmark just north of Palm Beach
    5. Enjoy water sports at Palm Beach:
       - Jet skiing
       - Parasailing
       - Snorkeling
       - Stand-up paddleboarding
    
    Would you like more specific information about any of these activities or would you like to know about other options in the area?
    ```
    :::

    :::js
    ```typescript
    import { ChatAnthropic } from "@langchain/anthropic";
    import { StateGraph, START, MessagesZodState, Command, interrupt, MemorySaver } from "@langchain/langgraph";
    import { createReactAgent } from "@langchain/langgraph/prebuilt";
    import { tool } from "@langchain/core/tools";
    import { z } from "zod";

    const model = new ChatAnthropic({ model: "claude-3-5-sonnet-latest" });

    const MultiAgentState = MessagesZodState.extend({
      lastActiveAgent: z.string().optional(),
    });

    // Define travel advisor tools
    const getTravelRecommendations = tool(
      async () => {
        // Placeholder implementation
        return "Based on current trends, I recommend visiting Japan, Portugal, or New Zealand.";
      },
      {
        name: "get_travel_recommendations",
        description: "Get current travel destination recommendations",
        schema: z.object({}),
      }
    );

    const makeHandoffTool = (agentName: string) => {
      return tool(
        async (_, config) => {
          const state = config.state;
          const toolCallId = config.toolCall.id;

          const toolMessage = {
            role: "tool" as const,
            content: `Successfully transferred to ${agentName}`,
            name: `transfer_to_${agentName}`,
            tool_call_id: toolCallId,
          };

          return new Command({
            goto: agentName,
            update: { messages: [...state.messages, toolMessage] },
            graph: Command.PARENT,
          });
        },
        {
          name: `transfer_to_${agentName}`,
          description: `Transfer to ${agentName}`,
          schema: z.object({}),
        }
      );
    };

    const travelAdvisorTools = [
      getTravelRecommendations,
      makeHandoffTool("hotel_advisor"),
    ];

    const travelAdvisor = createReactAgent({
      llm: model,
      tools: travelAdvisorTools,
      prompt: [
        "You are a general travel expert that can recommend travel destinations (e.g. countries, cities, etc). ",
        "If you need hotel recommendations, ask 'hotel_advisor' for help. ",
        "You MUST include human-readable response before transferring to another agent."
      ].join("")
    });

    const callTravelAdvisor = async (
      state: z.infer<typeof MultiAgentState>
    ): Promise<Command> => {
      const response = await travelAdvisor.invoke(state);
      const update = { ...response, lastActiveAgent: "travel_advisor" };
      return new Command({ update, goto: "human" });
    };

    // Define hotel advisor tools
    const getHotelRecommendations = tool(
      async () => {
        // Placeholder implementation
        return "I recommend the Ritz-Carlton for luxury stays or boutique hotels for unique experiences.";
      },
      {
        name: "get_hotel_recommendations",
        description: "Get hotel recommendations for destinations",
        schema: z.object({}),
      }
    );

    const hotelAdvisorTools = [
      getHotelRecommendations,
      makeHandoffTool("travel_advisor"),
    ];

    const hotelAdvisor = createReactAgent({
      llm: model,
      tools: hotelAdvisorTools,
      prompt: [
        "You are a hotel expert that can provide hotel recommendations for a given destination. ",
        "If you need help picking travel destinations, ask 'travel_advisor' for help.",
        "You MUST include human-readable response before transferring to another agent."
      ].join("")
    });

    const callHotelAdvisor = async (
      state: z.infer<typeof MultiAgentState>
    ): Promise<Command> => {
      const response = await hotelAdvisor.invoke(state);
      const update = { ...response, lastActiveAgent: "hotel_advisor" };
      return new Command({ update, goto: "human" });
    };

    const humanNode = async (
      state: z.infer<typeof MultiAgentState>
    ): Promise<Command> => {
      const userInput: string = interrupt("Ready for user input.");
      const activeAgent = state.lastActiveAgent || "travel_advisor";

      return new Command({
        update: {
          messages: [
            {
              role: "human",
              content: userInput,
            }
          ]
        },
        goto: activeAgent,
      });
    };

    const builder = new StateGraph(MultiAgentState)
      .addNode("travel_advisor", callTravelAdvisor)
      .addNode("hotel_advisor", callHotelAdvisor)
      .addNode("human", humanNode)
      .addEdge(START, "travel_advisor");

    const checkpointer = new MemorySaver();
    const graph = builder.compile({ checkpointer });
    ```
    
    Let's test a multi turn conversation with this application.

    ```typescript
    import { v4 as uuidv4 } from "uuid";
    import { Command } from "@langchain/langgraph";

    const threadConfig = { configurable: { thread_id: uuidv4() } };

    const inputs = [
      // 1st round of conversation
      {
        messages: [
          { role: "user", content: "i wanna go somewhere warm in the caribbean" }
        ]
      },
      // Since we're using `interrupt`, we'll need to resume using the Command primitive.
      // 2nd round of conversation
      new Command({
        resume: "could you recommend a nice hotel in one of the areas and tell me which area it is."
      }),
      // 3rd round of conversation
      new Command({
        resume: "i like the first one. could you recommend something to do near the hotel?"
      }),
    ];

    for (const [idx, userInput] of inputs.entries()) {
      console.log();
      console.log(`--- Conversation Turn ${idx + 1} ---`);
      console.log();
      console.log(`User: ${JSON.stringify(userInput)}`);
      console.log();
      
      for await (const update of await graph.stream(
        userInput,
        { ...threadConfig, streamMode: "updates" }
      )) {
        for (const [nodeId, value] of Object.entries(update)) {
          if (value?.messages?.length) {
            const lastMessage = value.messages.at(-1);
            if (lastMessage?.getType?.() === "ai") {
              console.log(`${nodeId}: ${lastMessage.content}`);
            }
          }
        }
      }
    }
    ```
    
    ```
    --- Conversation Turn 1 ---
    
    User: {"messages":[{"role":"user","content":"i wanna go somewhere warm in the caribbean"}]}
    
    travel_advisor: Based on the recommendations, Aruba would be an excellent choice for your Caribbean getaway! Aruba is known as "One Happy Island" and offers:
    - Year-round warm weather with consistent temperatures around 82°F (28°C)
    - Beautiful white sand beaches like Eagle Beach and Palm Beach
    - Clear turquoise waters perfect for swimming and snorkeling
    - Minimal rainfall and location outside the hurricane belt
    - A blend of Caribbean and Dutch culture
    - Great dining options and nightlife
    - Various water sports and activities
    
    Would you like me to get some specific hotel recommendations in Aruba for your stay? I can transfer you to our hotel advisor who can help with accommodations.
    
    --- Conversation Turn 2 ---
    
    User: Command { resume: 'could you recommend a nice hotel in one of the areas and tell me which area it is.' }
    
    hotel_advisor: Based on the recommendations, I can suggest two excellent options:
    
    1. The Ritz-Carlton, Aruba - Located in Palm Beach
    - This luxury resort is situated in the vibrant Palm Beach area
    - Known for its exceptional service and amenities
    - Perfect if you want to be close to dining, shopping, and entertainment
    - Features multiple restaurants, a casino, and a world-class spa
    - Located on a pristine stretch of Palm Beach
    
    2. Bucuti & Tara Beach Resort - Located in Eagle Beach
    - An adults-only boutique resort on Eagle Beach
    - Known for being more intimate and peaceful
    - Award-winning for its sustainability practices
    - Perfect for a romantic getaway or peaceful vacation
    - Located on one of the most beautiful beaches in the Caribbean
    
    Would you like more specific information about either of these properties or their locations?
    
    --- Conversation Turn 3 ---
    
    User: Command { resume: 'i like the first one. could you recommend something to do near the hotel?' }
    
    travel_advisor: Near the Ritz-Carlton in Palm Beach, here are some highly recommended activities:
    
    1. Visit the Palm Beach Plaza Mall - Just a short walk from the hotel, featuring shopping, dining, and entertainment
    2. Try your luck at the Stellaris Casino - It's right in the Ritz-Carlton
    3. Take a sunset sailing cruise - Many depart from the nearby pier
    4. Visit the California Lighthouse - A scenic landmark just north of Palm Beach
    5. Enjoy water sports at Palm Beach:
       - Jet skiing
       - Parasailing
       - Snorkeling
       - Stand-up paddleboarding
    
    Would you like more specific information about any of these activities or would you like to know about other options in the area?
    ```
    :::

## 사전 빌드 구현

LangGraph에는 가장 인기 있는 두 가지 다중 에이전트 아키텍처의 사전 빌드 구현이 함께 제공됩니다:

:::python
- [supervisor](../agents/multi-agent.md#supervisor) — 개별 에이전트가 중앙 슈퍼바이저 에이전트에 의해 조정됩니다. 슈퍼바이저는 모든 통신 흐름과 작업 위임을 제어하며, 현재 컨텍스트와 작업 요구 사항에 따라 호출할 에이전트를 결정합니다. [`langgraph-supervisor`](https://github.com/langchain-ai/langgraph-supervisor-py) 라이브러리를 사용하여 슈퍼바이저 다중 에이전트 시스템을 만들 수 있습니다.
- [swarm](../agents/multi-agent.md#supervisor) — 에이전트가 특화 분야에 따라 서로에게 동적으로 제어권을 넘깁니다. 시스템은 마지막으로 활성화된 에이전트를 기억하여 후속 상호작용에서 해당 에이전트와 대화를 재개할 수 있도록 합니다. [`langgraph-swarm`](https://github.com/langchain-ai/langgraph-swarm-py) 라이브러리를 사용하여 스웜 다중 에이전트 시스템을 만들 수 있습니다.
:::

:::js
- [supervisor](../agents/multi-agent.md#supervisor) — 개별 에이전트가 중앙 슈퍼바이저 에이전트에 의해 조정됩니다. 슈퍼바이저는 모든 통신 흐름과 작업 위임을 제어하며, 현재 컨텍스트와 작업 요구 사항에 따라 호출할 에이전트를 결정합니다. [`langgraph-supervisor`](https://github.com/langchain-ai/langgraph-supervisor-js) 라이브러리를 사용하여 슈퍼바이저 다중 에이전트 시스템을 만들 수 있습니다.
- [swarm](../agents/multi-agent.md#supervisor) — 에이전트가 특화 분야에 따라 서로에게 동적으로 제어권을 넘깁니다. 시스템은 마지막으로 활성화된 에이전트를 기억하여 후속 상호작용에서 해당 에이전트와 대화를 재개할 수 있도록 합니다. [`langgraph-swarm`](https://github.com/langchain-ai/langgraph-swarm-js) 라이브러리를 사용하여 스웜 다중 에이전트 시스템을 만들 수 있습니다.
:::