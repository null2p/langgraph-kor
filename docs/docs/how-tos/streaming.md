# 출력 스트리밍

LangGraph 에이전트 또는 워크플로에서 [출력을 스트리밍](../concepts/streaming.md)할 수 있습니다.

## 지원되는 스트림 모드 {#supported-stream-modes}

:::python
다음 스트림 모드 중 하나 이상을 리스트로 @[`stream()`][CompiledStateGraph.stream] 또는 @[`astream()`][CompiledStateGraph.astream] 메서드에 전달하세요:
:::

:::js
다음 스트림 모드 중 하나 이상을 리스트로 @[`stream()`][CompiledStateGraph.stream] 메서드에 전달하세요:
:::

| 모드       | 설명                                                                                                                                                                         |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `values`   | 그래프의 각 단계 후 상태의 전체 값을 스트리밍합니다.                                                                                                                   |
| `updates`  | 그래프의 각 단계 후 상태 업데이트를 스트리밍합니다. 동일한 단계에서 여러 업데이트가 발생하면(예: 여러 노드가 실행됨), 해당 업데이트는 개별적으로 스트리밍됩니다. |
| `custom`   | 그래프 노드 내부에서 커스텀 데이터를 스트리밍합니다.                                                                                                                                   |
| `messages` | LLM이 호출되는 모든 그래프 노드에서 2-튜플(LLM 토큰, 메타데이터)을 스트리밍합니다.                                                                                                |
| `debug`    | 그래프 실행 전반에 걸쳐 가능한 많은 정보를 스트리밍합니다.                                                                                                      |

## 에이전트에서 스트리밍

### 에이전트 진행 상황

:::python
에이전트 진행 상황을 스트리밍하려면 `stream_mode="updates"`와 함께 @[`stream()`][CompiledStateGraph.stream] 또는 @[`astream()`][CompiledStateGraph.astream] 메서드를 사용하세요. 이것은 모든 에이전트 단계 후에 이벤트를 발생시킵니다.
:::

:::js
에이전트 진행 상황을 스트리밍하려면 `streamMode: "updates"`와 함께 @[`stream()`][CompiledStateGraph.stream] 메서드를 사용하세요. 이것은 모든 에이전트 단계 후에 이벤트를 발생시킵니다.
:::

예를 들어, 도구를 한 번 호출하는 에이전트가 있다면 다음 업데이트를 볼 수 있습니다:

- **LLM 노드**: 도구 호출 요청이 포함된 AI 메시지
- **도구 노드**: 실행 결과가 포함된 도구 메시지
- **LLM 노드**: 최종 AI 응답

:::python
=== "Sync"

    ```python
    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_weather],
    )
    # highlight-next-line
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
        # highlight-next-line
        stream_mode="updates"
    ):
        print(chunk)
        print("\n")
    ```

=== "Async"

    ```python
    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_weather],
    )
    # highlight-next-line
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
        # highlight-next-line
        stream_mode="updates"
    ):
        print(chunk)
        print("\n")
    ```

:::

:::js

```typescript
const agent = createReactAgent({
  llm: model,
  tools: [getWeather],
});

for await (const chunk of await agent.stream(
  { messages: [{ role: "user", content: "what is the weather in sf" }] },
  { streamMode: "updates" }
)) {
  console.log(chunk);
  console.log("\n");
}
```

:::

### LLM 토큰

:::python
LLM이 생성하는 토큰을 스트리밍하려면 `stream_mode="messages"`를 사용하세요:

=== "Sync"

    ```python
    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_weather],
    )
    # highlight-next-line
    for token, metadata in agent.stream(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
        # highlight-next-line
        stream_mode="messages"
    ):
        print("Token", token)
        print("Metadata", metadata)
        print("\n")
    ```

=== "Async"

    ```python
    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_weather],
    )
    # highlight-next-line
    async for token, metadata in agent.astream(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
        # highlight-next-line
        stream_mode="messages"
    ):
        print("Token", token)
        print("Metadata", metadata)
        print("\n")
    ```

:::

:::js
LLM이 생성하는 토큰을 스트리밍하려면 `streamMode: "messages"`를 사용하세요:

```typescript
const agent = createReactAgent({
  llm: model,
  tools: [getWeather],
});

for await (const [token, metadata] of await agent.stream(
  { messages: [{ role: "user", content: "what is the weather in sf" }] },
  { streamMode: "messages" }
)) {
  console.log("Token", token);
  console.log("Metadata", metadata);
  console.log("\n");
}
```

:::

### 도구 업데이트

:::python
도구가 실행될 때 업데이트를 스트리밍하려면 @[get_stream_writer][get_stream_writer]를 사용할 수 있습니다.

=== "Sync"

    ```python
    # highlight-next-line
    from langgraph.config import get_stream_writer

    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        # highlight-next-line
        writer = get_stream_writer()
        # stream any arbitrary data
        # highlight-next-line
        writer(f"Looking up data for city: {city}")
        return f"It's always sunny in {city}!"

    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_weather],
    )

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
        # highlight-next-line
        stream_mode="custom"
    ):
        print(chunk)
        print("\n")
    ```

=== "Async"

    ```python
    # highlight-next-line
    from langgraph.config import get_stream_writer

    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        # highlight-next-line
        writer = get_stream_writer()
        # stream any arbitrary data
        # highlight-next-line
        writer(f"Looking up data for city: {city}")
        return f"It's always sunny in {city}!"

    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_weather],
    )

    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
        # highlight-next-line
        stream_mode="custom"
    ):
        print(chunk)
        print("\n")
    ```

!!! Note

      If you add `get_stream_writer` inside your tool, you won't be able to invoke the tool outside of a LangGraph execution context.

:::

:::js
도구가 실행될 때 업데이트를 스트리밍하려면 설정에서 `writer` 파라미터를 사용할 수 있습니다.

```typescript
import { LangGraphRunnableConfig } from "@langchain/langgraph";

const getWeather = tool(
  async (input, config: LangGraphRunnableConfig) => {
    // Stream any arbitrary data
    config.writer?.("Looking up data for city: " + input.city);
    return `It's always sunny in ${input.city}!`;
  },
  {
    name: "get_weather",
    description: "Get weather for a given city.",
    schema: z.object({
      city: z.string().describe("The city to get weather for."),
    }),
  }
);

const agent = createReactAgent({
  llm: model,
  tools: [getWeather],
});

for await (const chunk of await agent.stream(
  { messages: [{ role: "user", content: "what is the weather in sf" }] },
  { streamMode: "custom" }
)) {
  console.log(chunk);
  console.log("\n");
}
```

!!! Note
      If you add the `writer` parameter to your tool, you won't be able to invoke the tool outside of a LangGraph execution context without providing a writer function.
:::

### 여러 모드 스트리밍 {#stream-multiple-modes}

:::python
스트림 모드를 리스트로 전달하여 여러 스트리밍 모드를 지정할 수 있습니다: `stream_mode=["updates", "messages", "custom"]`:

=== "Sync"

    ```python
    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_weather],
    )

    for stream_mode, chunk in agent.stream(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
        # highlight-next-line
        stream_mode=["updates", "messages", "custom"]
    ):
        print(chunk)
        print("\n")
    ```

=== "Async"

    ```python
    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_weather],
    )

    async for stream_mode, chunk in agent.astream(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
        # highlight-next-line
        stream_mode=["updates", "messages", "custom"]
    ):
        print(chunk)
        print("\n")
    ```

:::

:::js
streamMode를 배열로 전달하여 여러 스트리밍 모드를 지정할 수 있습니다: `streamMode: ["updates", "messages", "custom"]`:

```typescript
const agent = createReactAgent({
  llm: model,
  tools: [getWeather],
});

for await (const chunk of await agent.stream(
  { messages: [{ role: "user", content: "what is the weather in sf" }] },
  { streamMode: ["updates", "messages", "custom"] }
)) {
  console.log(chunk);
  console.log("\n");
}
```

:::

### 스트리밍 비활성화

일부 애플리케이션에서는 특정 모델에 대해 개별 토큰의 스트리밍을 비활성화해야 할 수 있습니다. 이는 [멀티 에이전트](../agents/multi-agent.md) 시스템에서 어떤 에이전트가 출력을 스트리밍할지 제어하는 데 유용합니다.

스트리밍을 비활성화하는 방법은 [모델](../agents/models.md#disable-streaming) 가이드를 참조하세요.

## 워크플로에서 스트리밍

### 기본 사용 예제

:::python
LangGraph 그래프는 스트리밍된 출력을 이터레이터로 생성하기 위해 @[`.stream()`][Pregel.stream] (동기) 및 @[`.astream()`][Pregel.astream] (비동기) 메서드를 제공합니다.

=== "Sync"

    ```python
    for chunk in graph.stream(inputs, stream_mode="updates"):
        print(chunk)
    ```

=== "Async"

    ```python
    async for chunk in graph.astream(inputs, stream_mode="updates"):
        print(chunk)
    ```

:::

:::js
LangGraph 그래프는 스트리밍된 출력을 이터레이터로 생성하기 위해 @[`.stream()`][Pregel.stream] 메서드를 제공합니다.

```typescript
for await (const chunk of await graph.stream(inputs, {
  streamMode: "updates",
})) {
  console.log(chunk);
}
```

:::

??? example "Extended example: streaming updates"

      :::python
      ```python
      from typing import TypedDict
      from langgraph.graph import StateGraph, START, END

      class State(TypedDict):
          topic: str
          joke: str

      def refine_topic(state: State):
          return {"topic": state["topic"] + " and cats"}

      def generate_joke(state: State):
          return {"joke": f"This is a joke about {state['topic']}"}

      graph = (
          StateGraph(State)
          .add_node(refine_topic)
          .add_node(generate_joke)
          .add_edge(START, "refine_topic")
          .add_edge("refine_topic", "generate_joke")
          .add_edge("generate_joke", END)
          .compile()
      )

      # highlight-next-line
      for chunk in graph.stream( # (1)!
          {"topic": "ice cream"},
          # highlight-next-line
          stream_mode="updates", # (2)!
      ):
          print(chunk)
      ```

      1. The `stream()` method returns an iterator that yields streamed outputs.
      2. Set `stream_mode="updates"` to stream only the updates to the graph state after each node. Other stream modes are also available. See [supported stream modes](#supported-stream-modes) for details.
      :::

      :::js
      ```typescript
      import { StateGraph, START, END } from "@langchain/langgraph";
      import { z } from "zod";

      const State = z.object({
        topic: z.string(),
        joke: z.string(),
      });

      const graph = new StateGraph(State)
        .addNode("refineTopic", (state) => {
          return { topic: state.topic + " and cats" };
        })
        .addNode("generateJoke", (state) => {
          return { joke: `This is a joke about ${state.topic}` };
        })
        .addEdge(START, "refineTopic")
        .addEdge("refineTopic", "generateJoke")
        .addEdge("generateJoke", END)
        .compile();

      for await (const chunk of await graph.stream(
        { topic: "ice cream" },
        { streamMode: "updates" } // (1)!
      )) {
        console.log(chunk);
      }
      ```

      1. Set `streamMode: "updates"` to stream only the updates to the graph state after each node. Other stream modes are also available. See [supported stream modes](#supported-stream-modes) for details.
      :::

      ```output
      {'refineTopic': {'topic': 'ice cream and cats'}}
      {'generateJoke': {'joke': 'This is a joke about ice cream and cats'}}
      ```                                                                                                   |

### 여러 모드 스트리밍

:::python
한 번에 여러 모드를 스트리밍하기 위해 `stream_mode` 파라미터에 리스트를 전달할 수 있습니다.

스트리밍된 출력은 `(mode, chunk)` 튜플이며, 여기서 `mode`는 스트림 모드의 이름이고 `chunk`는 해당 모드에서 스트리밍된 데이터입니다.

=== "Sync"

    ```python
    for mode, chunk in graph.stream(inputs, stream_mode=["updates", "custom"]):
        print(chunk)
    ```

=== "Async"

    ```python
    async for mode, chunk in graph.astream(inputs, stream_mode=["updates", "custom"]):
        print(chunk)
    ```

:::

:::js
한 번에 여러 모드를 스트리밍하기 위해 `streamMode` 파라미터에 배열을 전달할 수 있습니다.

스트리밍된 출력은 `[mode, chunk]` 튜플이며, 여기서 `mode`는 스트림 모드의 이름이고 `chunk`는 해당 모드에서 스트리밍된 데이터입니다.

```typescript
for await (const [mode, chunk] of await graph.stream(inputs, {
  streamMode: ["updates", "custom"],
})) {
  console.log(chunk);
}
```

:::

### 그래프 상태 스트리밍 {#stream-graph-state}

그래프가 실행될 때 상태를 스트리밍하려면 `updates` 및 `values` 스트림 모드를 사용하세요.

- `updates`는 그래프의 각 단계 후 상태 **업데이트**를 스트리밍합니다.
- `values`는 그래프의 각 단계 후 상태의 **전체 값**을 스트리밍합니다.

:::python

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
  topic: str
  joke: str


def refine_topic(state: State):
    return {"topic": state["topic"] + " and cats"}


def generate_joke(state: State):
    return {"joke": f"This is a joke about {state['topic']}"}

graph = (
  StateGraph(State)
  .add_node(refine_topic)
  .add_node(generate_joke)
  .add_edge(START, "refine_topic")
  .add_edge("refine_topic", "generate_joke")
  .add_edge("generate_joke", END)
  .compile()
)
```

:::

:::js

```typescript
import { StateGraph, START, END } from "@langchain/langgraph";
import { z } from "zod";

const State = z.object({
  topic: z.string(),
  joke: z.string(),
});

const graph = new StateGraph(State)
  .addNode("refineTopic", (state) => {
    return { topic: state.topic + " and cats" };
  })
  .addNode("generateJoke", (state) => {
    return { joke: `This is a joke about ${state.topic}` };
  })
  .addEdge(START, "refineTopic")
  .addEdge("refineTopic", "generateJoke")
  .addEdge("generateJoke", END)
  .compile();
```

:::

=== "updates"

    각 단계 후 노드에서 반환된 **상태 업데이트**만 스트리밍하려면 이것을 사용하세요. 스트리밍된 출력에는 노드 이름과 업데이트가 포함됩니다.

    :::python
    ```python
    for chunk in graph.stream(
        {"topic": "ice cream"},
        # highlight-next-line
        stream_mode="updates",
    ):
        print(chunk)
    ```
    :::

    :::js
    ```typescript
    for await (const chunk of await graph.stream(
      { topic: "ice cream" },
      { streamMode: "updates" }
    )) {
      console.log(chunk);
    }
    ```
    :::

=== "values"

    각 단계 후 그래프의 **전체 상태**를 스트리밍하려면 이것을 사용하세요.

    :::python
    ```python
    for chunk in graph.stream(
        {"topic": "ice cream"},
        # highlight-next-line
        stream_mode="values",
    ):
        print(chunk)
    ```
    :::

    :::js
    ```typescript
    for await (const chunk of await graph.stream(
      { topic: "ice cream" },
      { streamMode: "values" }
    )) {
      console.log(chunk);
    }
    ```
    :::

### 서브그래프 출력 스트리밍 {#stream-subgraph-outputs}

:::python
스트리밍된 출력에 [서브그래프](../concepts/subgraphs.md)의 출력을 포함하려면 부모 그래프의 `.stream()` 메서드에서 `subgraphs=True`를 설정할 수 있습니다. 이렇게 하면 부모 그래프와 모든 서브그래프의 출력이 스트리밍됩니다.

출력은 `(namespace, data)` 튜플로 스트리밍되며, 여기서 `namespace`는 서브그래프가 호출되는 노드의 경로를 가진 튜플입니다(예: `("parent_node:<task_id>", "child_node:<task_id>")`).

```python
for chunk in graph.stream(
    {"foo": "foo"},
    # highlight-next-line
    subgraphs=True, # (1)!
    stream_mode="updates",
):
    print(chunk)
```

1. Set `subgraphs=True` to stream outputs from subgraphs.
   :::

:::js
스트리밍된 출력에 [서브그래프](../concepts/subgraphs.md)의 출력을 포함하려면 부모 그래프의 `.stream()` 메서드에서 `subgraphs: true`를 설정할 수 있습니다. 이렇게 하면 부모 그래프와 모든 서브그래프의 출력이 스트리밍됩니다.

출력은 `[namespace, data]` 튜플로 스트리밍되며, 여기서 `namespace`는 서브그래프가 호출되는 노드의 경로를 가진 튜플입니다(예: `["parent_node:<task_id>", "child_node:<task_id>"]`).

```typescript
for await (const chunk of await graph.stream(
  { foo: "foo" },
  {
    subgraphs: true, // (1)!
    streamMode: "updates",
  }
)) {
  console.log(chunk);
}
```

1. Set `subgraphs: true` to stream outputs from subgraphs.
   :::

??? example "Extended example: streaming from subgraphs"

      :::python
      ```python
      from langgraph.graph import START, StateGraph
      from typing import TypedDict

      # Define subgraph
      class SubgraphState(TypedDict):
          foo: str  # note that this key is shared with the parent graph state
          bar: str

      def subgraph_node_1(state: SubgraphState):
          return {"bar": "bar"}

      def subgraph_node_2(state: SubgraphState):
          return {"foo": state["foo"] + state["bar"]}

      subgraph_builder = StateGraph(SubgraphState)
      subgraph_builder.add_node(subgraph_node_1)
      subgraph_builder.add_node(subgraph_node_2)
      subgraph_builder.add_edge(START, "subgraph_node_1")
      subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
      subgraph = subgraph_builder.compile()

      # Define parent graph
      class ParentState(TypedDict):
          foo: str

      def node_1(state: ParentState):
          return {"foo": "hi! " + state["foo"]}

      builder = StateGraph(ParentState)
      builder.add_node("node_1", node_1)
      builder.add_node("node_2", subgraph)
      builder.add_edge(START, "node_1")
      builder.add_edge("node_1", "node_2")
      graph = builder.compile()

      for chunk in graph.stream(
          {"foo": "foo"},
          stream_mode="updates",
          # highlight-next-line
          subgraphs=True, # (1)!
      ):
          print(chunk)
      ```

      1. Set `subgraphs=True` to stream outputs from subgraphs.
      :::

      :::js
      ```typescript
      import { StateGraph, START } from "@langchain/langgraph";
      import { z } from "zod";

      // Define subgraph
      const SubgraphState = z.object({
        foo: z.string(), // note that this key is shared with the parent graph state
        bar: z.string(),
      });

      const subgraphBuilder = new StateGraph(SubgraphState)
        .addNode("subgraphNode1", (state) => {
          return { bar: "bar" };
        })
        .addNode("subgraphNode2", (state) => {
          return { foo: state.foo + state.bar };
        })
        .addEdge(START, "subgraphNode1")
        .addEdge("subgraphNode1", "subgraphNode2");
      const subgraph = subgraphBuilder.compile();

      // Define parent graph
      const ParentState = z.object({
        foo: z.string(),
      });

      const builder = new StateGraph(ParentState)
        .addNode("node1", (state) => {
          return { foo: "hi! " + state.foo };
        })
        .addNode("node2", subgraph)
        .addEdge(START, "node1")
        .addEdge("node1", "node2");
      const graph = builder.compile();

      for await (const chunk of await graph.stream(
        { foo: "foo" },
        {
          streamMode: "updates",
          subgraphs: true, // (1)!
        }
      )) {
        console.log(chunk);
      }
      ```

      1. Set `subgraphs: true` to stream outputs from subgraphs.
      :::

      :::python
      ```
      ((), {'node_1': {'foo': 'hi! foo'}})
      (('node_2:dfddc4ba-c3c5-6887-5012-a243b5b377c2',), {'subgraph_node_1': {'bar': 'bar'}})
      (('node_2:dfddc4ba-c3c5-6887-5012-a243b5b377c2',), {'subgraph_node_2': {'foo': 'hi! foobar'}})
      ((), {'node_2': {'foo': 'hi! foobar'}})
      ```
      :::

      :::js
      ```
      [[], {'node1': {'foo': 'hi! foo'}}]
      [['node2:dfddc4ba-c3c5-6887-5012-a243b5b377c2'], {'subgraphNode1': {'bar': 'bar'}}]
      [['node2:dfddc4ba-c3c5-6887-5012-a243b5b377c2'], {'subgraphNode2': {'foo': 'hi! foobar'}}]
      [[], {'node2': {'foo': 'hi! foobar'}}]
      ```
      :::

      **Note** that we are receiving not just the node updates, but we also the namespaces which tell us what graph (or subgraph) we are streaming from.

### 디버깅 {#debug}

그래프 실행 전반에 걸쳐 가능한 많은 정보를 스트리밍하려면 `debug` 스트리밍 모드를 사용하세요. 스트리밍된 출력에는 노드 이름과 전체 상태가 포함됩니다.

:::python

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    # highlight-next-line
    stream_mode="debug",
):
    print(chunk)
```

:::

:::js

```typescript
for await (const chunk of await graph.stream(
  { topic: "ice cream" },
  { streamMode: "debug" }
)) {
  console.log(chunk);
}
```

:::

### LLM 토큰 {#messages}

노드, 도구, 서브그래프 또는 작업을 포함한 그래프의 모든 부분에서 대형 언어 모델(LLM) 출력을 **토큰 단위로** 스트리밍하려면 `messages` 스트리밍 모드를 사용하세요.

:::python
[`messages` 모드](#supported-stream-modes)의 스트리밍된 출력은 `(message_chunk, metadata)` 튜플이며, 여기서:

- `message_chunk`: LLM의 토큰 또는 메시지 세그먼트입니다.
- `metadata`: 그래프 노드 및 LLM 호출에 대한 세부 정보가 포함된 딕셔너리입니다.

> LLM이 LangChain 통합으로 사용할 수 없는 경우 대신 `custom` 모드를 사용하여 출력을 스트리밍할 수 있습니다. 자세한 내용은 [모든 LLM과 사용](#use-with-any-llm)을 참조하세요.

!!! warning "Manual config required for async in Python < 3.11"

    When using Python < 3.11 with async code, you must explicitly pass `RunnableConfig` to `ainvoke()` to enable proper streaming. See [Async with Python < 3.11](#async) for details or upgrade to Python 3.11+.

```python
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START


@dataclass
class MyState:
    topic: str
    joke: str = ""


llm = init_chat_model(model="openai:gpt-4o-mini")

def call_model(state: MyState):
    """Call the LLM to generate a joke about a topic"""
    # highlight-next-line
    llm_response = llm.invoke( # (1)!
        [
            {"role": "user", "content": f"Generate a joke about {state.topic}"}
        ]
    )
    return {"joke": llm_response.content}

graph = (
    StateGraph(MyState)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .compile()
)

for message_chunk, metadata in graph.stream( # (2)!
    {"topic": "ice cream"},
    # highlight-next-line
    stream_mode="messages",
):
    if message_chunk.content:
        print(message_chunk.content, end="|", flush=True)
```

1. Note that the message events are emitted even when the LLM is run using `.invoke` rather than `.stream`.
2. The "messages" stream mode returns an iterator of tuples `(message_chunk, metadata)` where `message_chunk` is the token streamed by the LLM and `metadata` is a dictionary with information about the graph node where the LLM was called and other information.
   :::

:::js
[`messages` 모드](#supported-stream-modes)의 스트리밍된 출력은 `[message_chunk, metadata]` 튜플이며, 여기서:

- `message_chunk`: LLM의 토큰 또는 메시지 세그먼트입니다.
- `metadata`: 그래프 노드 및 LLM 호출에 대한 세부 정보가 포함된 딕셔너리입니다.

> LLM이 LangChain 통합으로 사용할 수 없는 경우 대신 `custom` 모드를 사용하여 출력을 스트리밍할 수 있습니다. 자세한 내용은 [모든 LLM과 사용](#use-with-any-llm)을 참조하세요.

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, START } from "@langchain/langgraph";
import { z } from "zod";

const MyState = z.object({
  topic: z.string(),
  joke: z.string().default(""),
});

const llm = new ChatOpenAI({ model: "gpt-4o-mini" });

const callModel = async (state: z.infer<typeof MyState>) => {
  // Call the LLM to generate a joke about a topic
  const llmResponse = await llm.invoke([
    { role: "user", content: `Generate a joke about ${state.topic}` },
  ]); // (1)!
  return { joke: llmResponse.content };
};

const graph = new StateGraph(MyState)
  .addNode("callModel", callModel)
  .addEdge(START, "callModel")
  .compile();

for await (const [messageChunk, metadata] of await graph.stream(
  // (2)!
  { topic: "ice cream" },
  { streamMode: "messages" }
)) {
  if (messageChunk.content) {
    console.log(messageChunk.content + "|");
  }
}
```

1. Note that the message events are emitted even when the LLM is run using `.invoke` rather than `.stream`.
2. The "messages" stream mode returns an iterator of tuples `[messageChunk, metadata]` where `messageChunk` is the token streamed by the LLM and `metadata` is a dictionary with information about the graph node where the LLM was called and other information.
   :::

#### LLM 호출별 필터링 {#filter-by-llm-invocation}

LLM 호출과 `tags`를 연결하여 LLM 호출별로 스트리밍된 토큰을 필터링할 수 있습니다.

:::python

```python
from langchain.chat_models import init_chat_model

llm_1 = init_chat_model(model="openai:gpt-4o-mini", tags=['joke']) # (1)!
llm_2 = init_chat_model(model="openai:gpt-4o-mini", tags=['poem']) # (2)!

graph = ... # define a graph that uses these LLMs

async for msg, metadata in graph.astream(  # (3)!
    {"topic": "cats"},
    # highlight-next-line
    stream_mode="messages",
):
    if metadata["tags"] == ["joke"]: # (4)!
        print(msg.content, end="|", flush=True)
```

1. llm_1 is tagged with "joke".
2. llm_2 is tagged with "poem".
3. The `stream_mode` is set to "messages" to stream LLM tokens. The `metadata` contains information about the LLM invocation, including the tags.
4. Filter the streamed tokens by the `tags` field in the metadata to only include the tokens from the LLM invocation with the "joke" tag.
   :::

:::js

```typescript
import { ChatOpenAI } from "@langchain/openai";

const llm1 = new ChatOpenAI({
  model: "gpt-4o-mini",
  tags: ['joke'] // (1)!
});
const llm2 = new ChatOpenAI({
  model: "gpt-4o-mini",
  tags: ['poem'] // (2)!
});

const graph = // ... define a graph that uses these LLMs

for await (const [msg, metadata] of await graph.stream( // (3)!
  { topic: "cats" },
  { streamMode: "messages" }
)) {
  if (metadata.tags?.includes("joke")) { // (4)!
    console.log(msg.content + "|");
  }
}
```

1. llm1 is tagged with "joke".
2. llm2 is tagged with "poem".
3. The `streamMode` is set to "messages" to stream LLM tokens. The `metadata` contains information about the LLM invocation, including the tags.
4. Filter the streamed tokens by the `tags` field in the metadata to only include the tokens from the LLM invocation with the "joke" tag.
   :::

??? example "Extended example: filtering by tags"

      :::python
      ```python
      from typing import TypedDict

      from langchain.chat_models import init_chat_model
      from langgraph.graph import START, StateGraph

      joke_model = init_chat_model(model="openai:gpt-4o-mini", tags=["joke"]) # (1)!
      poem_model = init_chat_model(model="openai:gpt-4o-mini", tags=["poem"]) # (2)!


      class State(TypedDict):
            topic: str
            joke: str
            poem: str


      async def call_model(state, config):
            topic = state["topic"]
            print("Writing joke...")
            # Note: Passing the config through explicitly is required for python < 3.11
            # Since context var support wasn't added before then: https://docs.python.org/3/library/asyncio-task.html#creating-tasks
            joke_response = await joke_model.ainvoke(
                  [{"role": "user", "content": f"Write a joke about {topic}"}],
                  config, # (3)!
            )
            print("\n\nWriting poem...")
            poem_response = await poem_model.ainvoke(
                  [{"role": "user", "content": f"Write a short poem about {topic}"}],
                  config, # (3)!
            )
            return {"joke": joke_response.content, "poem": poem_response.content}


      graph = (
            StateGraph(State)
            .add_node(call_model)
            .add_edge(START, "call_model")
            .compile()
      )

      async for msg, metadata in graph.astream(
            {"topic": "cats"},
            # highlight-next-line
            stream_mode="messages", # (4)!
      ):
          if metadata["tags"] == ["joke"]: # (4)!
              print(msg.content, end="|", flush=True)
      ```

      1. The `joke_model` is tagged with "joke".
      2. The `poem_model` is tagged with "poem".
      3. The `config` is passed through explicitly to ensure the context vars are propagated correctly. This is required for Python < 3.11 when using async code. Please see the [async section](#async) for more details.
      4. The `stream_mode` is set to "messages" to stream LLM tokens. The `metadata` contains information about the LLM invocation, including the tags.
      :::

      :::js
      ```typescript
      import { ChatOpenAI } from "@langchain/openai";
      import { StateGraph, START } from "@langchain/langgraph";
      import { z } from "zod";

      const jokeModel = new ChatOpenAI({
        model: "gpt-4o-mini",
        tags: ["joke"] // (1)!
      });
      const poemModel = new ChatOpenAI({
        model: "gpt-4o-mini",
        tags: ["poem"] // (2)!
      });

      const State = z.object({
        topic: z.string(),
        joke: z.string(),
        poem: z.string(),
      });

      const graph = new StateGraph(State)
        .addNode("callModel", (state) => {
          const topic = state.topic;
          console.log("Writing joke...");

          const jokeResponse = await jokeModel.invoke([
            { role: "user", content: `Write a joke about ${topic}` }
          ]);

          console.log("\n\nWriting poem...");
          const poemResponse = await poemModel.invoke([
            { role: "user", content: `Write a short poem about ${topic}` }
          ]);

          return {
            joke: jokeResponse.content,
            poem: poemResponse.content
          };
        })
        .addEdge(START, "callModel")
        .compile();

      for await (const [msg, metadata] of await graph.stream(
        { topic: "cats" },
        { streamMode: "messages" } // (3)!
      )) {
        if (metadata.tags?.includes("joke")) { // (4)!
          console.log(msg.content + "|");
        }
      }
      ```

      1. The `jokeModel` is tagged with "joke".
      2. The `poemModel` is tagged with "poem".
      3. The `streamMode` is set to "messages" to stream LLM tokens. The `metadata` contains information about the LLM invocation, including the tags.
      4. Filter the streamed tokens by the `tags` field in the metadata to only include the tokens from the LLM invocation with the "joke" tag.
      :::

#### 노드별 필터링 {#filter-by-node}

특정 노드에서만 토큰을 스트리밍하려면 `stream_mode="messages"`를 사용하고 스트리밍된 메타데이터의 `langgraph_node` 필드로 출력을 필터링하세요:

:::python

```python
for msg, metadata in graph.stream( # (1)!
    inputs,
    # highlight-next-line
    stream_mode="messages",
):
    # highlight-next-line
    if msg.content and metadata["langgraph_node"] == "some_node_name": # (2)!
        ...
```

1. The "messages" stream mode returns a tuple of `(message_chunk, metadata)` where `message_chunk` is the token streamed by the LLM and `metadata` is a dictionary with information about the graph node where the LLM was called and other information.
2. Filter the streamed tokens by the `langgraph_node` field in the metadata to only include the tokens from the `write_poem` node.
   :::

:::js

```typescript
for await (const [msg, metadata] of await graph.stream(
  // (1)!
  inputs,
  { streamMode: "messages" }
)) {
  if (msg.content && metadata.langgraph_node === "some_node_name") {
    // (2)!
    // ...
  }
}
```

1. The "messages" stream mode returns a tuple of `[messageChunk, metadata]` where `messageChunk` is the token streamed by the LLM and `metadata` is a dictionary with information about the graph node where the LLM was called and other information.
2. Filter the streamed tokens by the `langgraph_node` field in the metadata to only include the tokens from the `writePoem` node.
   :::

??? example "Extended example: streaming LLM tokens from specific nodes"

      :::python
      ```python
      from typing import TypedDict
      from langgraph.graph import START, StateGraph
      from langchain_openai import ChatOpenAI

      model = ChatOpenAI(model="gpt-4o-mini")


      class State(TypedDict):
            topic: str
            joke: str
            poem: str


      def write_joke(state: State):
            topic = state["topic"]
            joke_response = model.invoke(
                  [{"role": "user", "content": f"Write a joke about {topic}"}]
            )
            return {"joke": joke_response.content}


      def write_poem(state: State):
            topic = state["topic"]
            poem_response = model.invoke(
                  [{"role": "user", "content": f"Write a short poem about {topic}"}]
            )
            return {"poem": poem_response.content}


      graph = (
            StateGraph(State)
            .add_node(write_joke)
            .add_node(write_poem)
            # write both the joke and the poem concurrently
            .add_edge(START, "write_joke")
            .add_edge(START, "write_poem")
            .compile()
      )

      # highlight-next-line
      for msg, metadata in graph.stream( # (1)!
          {"topic": "cats"},
          stream_mode="messages",
      ):
          # highlight-next-line
          if msg.content and metadata["langgraph_node"] == "write_poem": # (2)!
              print(msg.content, end="|", flush=True)
      ```

      1. The "messages" stream mode returns a tuple of `(message_chunk, metadata)` where `message_chunk` is the token streamed by the LLM and `metadata` is a dictionary with information about the graph node where the LLM was called and other information.
      2. Filter the streamed tokens by the `langgraph_node` field in the metadata to only include the tokens from the `write_poem` node.
      :::

      :::js
      ```typescript
      import { ChatOpenAI } from "@langchain/openai";
      import { StateGraph, START } from "@langchain/langgraph";
      import { z } from "zod";

      const model = new ChatOpenAI({ model: "gpt-4o-mini" });

      const State = z.object({
        topic: z.string(),
        joke: z.string(),
        poem: z.string(),
      });

      const graph = new StateGraph(State)
        .addNode("writeJoke", async (state) => {
          const topic = state.topic;
          const jokeResponse = await model.invoke([
            { role: "user", content: `Write a joke about ${topic}` }
          ]);
          return { joke: jokeResponse.content };
        })
        .addNode("writePoem", async (state) => {
          const topic = state.topic;
          const poemResponse = await model.invoke([
            { role: "user", content: `Write a short poem about ${topic}` }
          ]);
          return { poem: poemResponse.content };
        })
        // write both the joke and the poem concurrently
        .addEdge(START, "writeJoke")
        .addEdge(START, "writePoem")
        .compile();

      for await (const [msg, metadata] of await graph.stream( // (1)!
        { topic: "cats" },
        { streamMode: "messages" }
      )) {
        if (msg.content && metadata.langgraph_node === "writePoem") { // (2)!
          console.log(msg.content + "|");
        }
      }
      ```

      1. The "messages" stream mode returns a tuple of `[messageChunk, metadata]` where `messageChunk` is the token streamed by the LLM and `metadata` is a dictionary with information about the graph node where the LLM was called and other information.
      2. Filter the streamed tokens by the `langgraph_node` field in the metadata to only include the tokens from the `writePoem` node.
      :::

### 커스텀 데이터 스트리밍 {#stream-custom-data}

:::python
LangGraph 노드 또는 도구 내부에서 **사용자 정의 커스텀 데이터**를 전송하려면 다음 단계를 따르세요:

1. `get_stream_writer()`를 사용하여 스트림 writer에 접근하고 커스텀 데이터를 발생시킵니다.
2. `.stream()` 또는 `.astream()`을 호출할 때 `stream_mode="custom"`을 설정하여 스트림에서 커스텀 데이터를 가져옵니다. 여러 모드를 결합할 수 있지만(예: `["updates", "custom"]`), 최소한 하나는 `"custom"`이어야 합니다.

!!! warning "No `get_stream_writer()` in async for Python < 3.11"

    In async code running on Python < 3.11, `get_stream_writer()` will not work.
    Instead, add a `writer` parameter to your node or tool and pass it manually.
    See [Async with Python < 3.11](#async) for usage examples.

=== "node"

      ```python
      from typing import TypedDict
      from langgraph.config import get_stream_writer
      from langgraph.graph import StateGraph, START

      class State(TypedDict):
          query: str
          answer: str

      def node(state: State):
          writer = get_stream_writer()  # (1)!
          writer({"custom_key": "Generating custom data inside node"}) # (2)!
          return {"answer": "some data"}

      graph = (
          StateGraph(State)
          .add_node(node)
          .add_edge(START, "node")
          .compile()
      )

      inputs = {"query": "example"}

      # Usage
      for chunk in graph.stream(inputs, stream_mode="custom"):  # (3)!
          print(chunk)
      ```

      1. Get the stream writer to send custom data.
      2. Emit a custom key-value pair (e.g., progress update).
      3. Set `stream_mode="custom"` to receive the custom data in the stream.

=== "tool"

      ```python
      from langchain_core.tools import tool
      from langgraph.config import get_stream_writer

      @tool
      def query_database(query: str) -> str:
          """Query the database."""
          writer = get_stream_writer() # (1)!
          # highlight-next-line
          writer({"data": "Retrieved 0/100 records", "type": "progress"}) # (2)!
          # perform query
          # highlight-next-line
          writer({"data": "Retrieved 100/100 records", "type": "progress"}) # (3)!
          return "some-answer"


      graph = ... # define a graph that uses this tool

      for chunk in graph.stream(inputs, stream_mode="custom"): # (4)!
          print(chunk)
      ```

      1. Access the stream writer to send custom data.
      2. Emit a custom key-value pair (e.g., progress update).
      3. Emit another custom key-value pair.
      4. Set `stream_mode="custom"` to receive the custom data in the stream.

:::

:::js
LangGraph 노드 또는 도구 내부에서 **사용자 정의 커스텀 데이터**를 전송하려면 다음 단계를 따르세요:

1. `LangGraphRunnableConfig`의 `writer` 파라미터를 사용하여 커스텀 데이터를 발생시킵니다.
2. `.stream()`을 호출할 때 `streamMode: "custom"`을 설정하여 스트림에서 커스텀 데이터를 가져옵니다. 여러 모드를 결합할 수 있지만(예: `["updates", "custom"]`), 최소한 하나는 `"custom"`이어야 합니다.

=== "node"

      ```typescript
      import { StateGraph, START, LangGraphRunnableConfig } from "@langchain/langgraph";
      import { z } from "zod";

      const State = z.object({
        query: z.string(),
        answer: z.string(),
      });

      const graph = new StateGraph(State)
        .addNode("node", async (state, config) => {
          config.writer({ custom_key: "Generating custom data inside node" }); // (1)!
          return { answer: "some data" };
        })
        .addEdge(START, "node")
        .compile();

      const inputs = { query: "example" };

      // Usage
      for await (const chunk of await graph.stream(inputs, { streamMode: "custom" })) { // (2)!
        console.log(chunk);
      }
      ```

      1. Use the writer to emit a custom key-value pair (e.g., progress update).
      2. Set `streamMode: "custom"` to receive the custom data in the stream.

=== "tool"

      ```typescript
      import { tool } from "@langchain/core/tools";
      import { LangGraphRunnableConfig } from "@langchain/langgraph";
      import { z } from "zod";

      const queryDatabase = tool(
        async (input, config: LangGraphRunnableConfig) => {
          config.writer({ data: "Retrieved 0/100 records", type: "progress" }); // (1)!
          // perform query
          config.writer({ data: "Retrieved 100/100 records", type: "progress" }); // (2)!
          return "some-answer";
        },
        {
          name: "query_database",
          description: "Query the database.",
          schema: z.object({
            query: z.string().describe("The query to execute."),
          }),
        }
      );

      const graph = // ... define a graph that uses this tool

      for await (const chunk of await graph.stream(inputs, { streamMode: "custom" })) { // (3)!
        console.log(chunk);
      }
      ```

      1. Use the writer to emit a custom key-value pair (e.g., progress update).
      2. Emit another custom key-value pair.
      3. Set `streamMode: "custom"` to receive the custom data in the stream.

:::

### 모든 LLM과 사용 {#use-with-any-llm}

:::python
**모든 LLM API**에서 데이터를 스트리밍하기 위해 `stream_mode="custom"`을 사용할 수 있습니다 — 해당 API가 LangChain 채팅 모델 인터페이스를 구현하지 **않더라도** 말입니다.

이를 통해 자체 스트리밍 인터페이스를 제공하는 원시 LLM 클라이언트 또는 외부 서비스를 통합할 수 있어 LangGraph를 커스텀 설정에 매우 유연하게 사용할 수 있습니다.

```python
from langgraph.config import get_stream_writer

def call_arbitrary_model(state):
    """Example node that calls an arbitrary model and streams the output"""
    # highlight-next-line
    writer = get_stream_writer() # (1)!
    # Assume you have a streaming client that yields chunks
    for chunk in your_custom_streaming_client(state["topic"]): # (2)!
        # highlight-next-line
        writer({"custom_llm_chunk": chunk}) # (3)!
    return {"result": "completed"}

graph = (
    StateGraph(State)
    .add_node(call_arbitrary_model)
    # Add other nodes and edges as needed
    .compile()
)

for chunk in graph.stream(
    {"topic": "cats"},
    # highlight-next-line
    stream_mode="custom", # (4)!
):
    # The chunk will contain the custom data streamed from the llm
    print(chunk)
```

1. Get the stream writer to send custom data.
2. Generate LLM tokens using your custom streaming client.
3. Use the writer to send custom data to the stream.
4. Set `stream_mode="custom"` to receive the custom data in the stream.
   :::

:::js
**모든 LLM API**에서 데이터를 스트리밍하기 위해 `streamMode: "custom"`을 사용할 수 있습니다 — 해당 API가 LangChain 채팅 모델 인터페이스를 구현하지 **않더라도** 말입니다.

이를 통해 자체 스트리밍 인터페이스를 제공하는 원시 LLM 클라이언트 또는 외부 서비스를 통합할 수 있어 LangGraph를 커스텀 설정에 매우 유연하게 사용할 수 있습니다.

```typescript
import { LangGraphRunnableConfig } from "@langchain/langgraph";

const callArbitraryModel = async (
  state: any,
  config: LangGraphRunnableConfig
) => {
  // Example node that calls an arbitrary model and streams the output
  // Assume you have a streaming client that yields chunks
  for await (const chunk of yourCustomStreamingClient(state.topic)) {
    // (1)!
    config.writer({ custom_llm_chunk: chunk }); // (2)!
  }
  return { result: "completed" };
};

const graph = new StateGraph(State)
  .addNode("callArbitraryModel", callArbitraryModel)
  // Add other nodes and edges as needed
  .compile();

for await (const chunk of await graph.stream(
  { topic: "cats" },
  { streamMode: "custom" } // (3)!
)) {
  // The chunk will contain the custom data streamed from the llm
  console.log(chunk);
}
```

1. Generate LLM tokens using your custom streaming client.
2. Use the writer to send custom data to the stream.
3. Set `streamMode: "custom"` to receive the custom data in the stream.
   :::

??? example "Extended example: streaming arbitrary chat model"

      :::python
      ```python
      import operator
      import json

      from typing import TypedDict
      from typing_extensions import Annotated
      from langgraph.graph import StateGraph, START

      from openai import AsyncOpenAI

      openai_client = AsyncOpenAI()
      model_name = "gpt-4o-mini"


      async def stream_tokens(model_name: str, messages: list[dict]):
          response = await openai_client.chat.completions.create(
              messages=messages, model=model_name, stream=True
          )
          role = None
          async for chunk in response:
              delta = chunk.choices[0].delta

              if delta.role is not None:
                  role = delta.role

              if delta.content:
                  yield {"role": role, "content": delta.content}


      # this is our tool
      async def get_items(place: str) -> str:
          """Use this tool to list items one might find in a place you're asked about."""
          writer = get_stream_writer()
          response = ""
          async for msg_chunk in stream_tokens(
              model_name,
              [
                  {
                      "role": "user",
                      "content": (
                          "Can you tell me what kind of items "
                          f"i might find in the following place: '{place}'. "
                          "List at least 3 such items separating them by a comma. "
                          "And include a brief description of each item."
                      ),
                  }
              ],
          ):
              response += msg_chunk["content"]
              writer(msg_chunk)

          return response


      class State(TypedDict):
          messages: Annotated[list[dict], operator.add]


      # this is the tool-calling graph node
      async def call_tool(state: State):
          ai_message = state["messages"][-1]
          tool_call = ai_message["tool_calls"][-1]

          function_name = tool_call["function"]["name"]
          if function_name != "get_items":
              raise ValueError(f"Tool {function_name} not supported")

          function_arguments = tool_call["function"]["arguments"]
          arguments = json.loads(function_arguments)

          function_response = await get_items(**arguments)
          tool_message = {
              "tool_call_id": tool_call["id"],
              "role": "tool",
              "name": function_name,
              "content": function_response,
          }
          return {"messages": [tool_message]}


      graph = (
          StateGraph(State)
          .add_node(call_tool)
          .add_edge(START, "call_tool")
          .compile()
      )
      ```

      Let's invoke the graph with an AI message that includes a tool call:

      ```python
      inputs = {
          "messages": [
              {
                  "content": None,
                  "role": "assistant",
                  "tool_calls": [
                      {
                          "id": "1",
                          "function": {
                              "arguments": '{"place":"bedroom"}',
                              "name": "get_items",
                          },
                          "type": "function",
                      }
                  ],
              }
          ]
      }

      async for chunk in graph.astream(
          inputs,
          stream_mode="custom",
      ):
          print(chunk["content"], end="|", flush=True)
      ```
      :::

      :::js
      ```typescript
      import { StateGraph, START, LangGraphRunnableConfig } from "@langchain/langgraph";
      import { z } from "zod";
      import OpenAI from "openai";

      const openaiClient = new OpenAI();
      const modelName = "gpt-4o-mini";

      async function* streamTokens(modelName: string, messages: any[]) {
        const response = await openaiClient.chat.completions.create({
          messages,
          model: modelName,
          stream: true,
        });

        let role: string | null = null;
        for await (const chunk of response) {
          const delta = chunk.choices[0]?.delta;

          if (delta?.role) {
            role = delta.role;
          }

          if (delta?.content) {
            yield { role, content: delta.content };
          }
        }
      }

      // this is our tool
      const getItems = tool(
        async (input, config: LangGraphRunnableConfig) => {
          let response = "";
          for await (const msgChunk of streamTokens(
            modelName,
            [
              {
                role: "user",
                content: `Can you tell me what kind of items i might find in the following place: '${input.place}'. List at least 3 such items separating them by a comma. And include a brief description of each item.`,
              },
            ]
          )) {
            response += msgChunk.content;
            config.writer?.(msgChunk);
          }
          return response;
        },
        {
          name: "get_items",
          description: "Use this tool to list items one might find in a place you're asked about.",
          schema: z.object({
            place: z.string().describe("The place to look up items for."),
          }),
        }
      );

      const State = z.object({
        messages: z.array(z.any()),
      });

      const graph = new StateGraph(State)
        // this is the tool-calling graph node
        .addNode("callTool", async (state) => {
          const aiMessage = state.messages.at(-1);
          const toolCall = aiMessage.tool_calls?.at(-1);

          const functionName = toolCall?.function?.name;
          if (functionName !== "get_items") {
            throw new Error(`Tool ${functionName} not supported`);
          }

          const functionArguments = toolCall?.function?.arguments;
          const args = JSON.parse(functionArguments);

          const functionResponse = await getItems.invoke(args);
          const toolMessage = {
            tool_call_id: toolCall.id,
            role: "tool",
            name: functionName,
            content: functionResponse,
          };
          return { messages: [toolMessage] };
        })
        .addEdge(START, "callTool")
        .compile();
      ```

      Let's invoke the graph with an AI message that includes a tool call:

      ```typescript
      const inputs = {
        messages: [
          {
            content: null,
            role: "assistant",
            tool_calls: [
              {
                id: "1",
                function: {
                  arguments: '{"place":"bedroom"}',
                  name: "get_items",
                },
                type: "function",
              }
            ],
          }
        ]
      };

      for await (const chunk of await graph.stream(
        inputs,
        { streamMode: "custom" }
      )) {
        console.log(chunk.content + "|");
      }
      ```
      :::

### 특정 채팅 모델에 대한 스트리밍 비활성화

애플리케이션에서 스트리밍을 지원하는 모델과 그렇지 않은 모델을 혼합하여 사용하는 경우 스트리밍을 지원하지 않는 모델에 대해 명시적으로 스트리밍을 비활성화해야 할 수 있습니다.

:::python
모델을 초기화할 때 `disable_streaming=True`를 설정하세요.

=== "init_chat_model"

      ```python
      from langchain.chat_models import init_chat_model

      model = init_chat_model(
          "anthropic:claude-3-7-sonnet-latest",
          # highlight-next-line
          disable_streaming=True # (1)!
      )
      ```

      1. Set `disable_streaming=True` to disable streaming for the chat model.

=== "chat model interface"

      ```python
      from langchain_openai import ChatOpenAI

      llm = ChatOpenAI(model="o1-preview", disable_streaming=True) # (1)!
      ```

      1. Set `disable_streaming=True` to disable streaming for the chat model.

:::

:::js
모델을 초기화할 때 `streaming: false`를 설정하세요.

```typescript
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({
  model: "o1-preview",
  streaming: false, // (1)!
});
```

:::

:::python

### Python < 3.11에서 비동기 사용 { #async }

Python 버전 < 3.11에서는 [asyncio 작업](https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task)이 `context` 파라미터를 지원하지 않습니다.
이는 LangGraph가 컨텍스트를 자동으로 전파하는 능력을 제한하며 LangGraph의 스트리밍 메커니즘에 두 가지 주요 방식으로 영향을 미칩니다:

1. 콜백이 자동으로 전파되지 않으므로 비동기 LLM 호출(예: `ainvoke()`)에 [`RunnableConfig`](https://python.langchain.com/docs/concepts/runnables/#runnableconfig)를 명시적으로 전달**해야 합니다**.
2. 비동기 노드 또는 도구에서 `get_stream_writer()`를 사용할 수 **없으며** — `writer` 인수를 직접 전달해야 합니다.

??? example "Extended example: async LLM call with manual config"

      ```python
      from typing import TypedDict
      from langgraph.graph import START, StateGraph
      from langchain.chat_models import init_chat_model

      llm = init_chat_model(model="openai:gpt-4o-mini")

      class State(TypedDict):
          topic: str
          joke: str

      async def call_model(state, config): # (1)!
          topic = state["topic"]
          print("Generating joke...")
          joke_response = await llm.ainvoke(
              [{"role": "user", "content": f"Write a joke about {topic}"}],
              # highlight-next-line
              config, # (2)!
          )
          return {"joke": joke_response.content}

      graph = (
          StateGraph(State)
          .add_node(call_model)
          .add_edge(START, "call_model")
          .compile()
      )

      async for chunk, metadata in graph.astream(
          {"topic": "ice cream"},
          # highlight-next-line
          stream_mode="messages", # (3)!
      ):
          if chunk.content:
              print(chunk.content, end="|", flush=True)
      ```

      1. Accept `config` as an argument in the async node function.
      2. Pass `config` to `llm.ainvoke()` to ensure proper context propagation.
      3. Set `stream_mode="messages"` to stream LLM tokens.

??? example "Extended example: async custom streaming with stream writer"

      ```python
      from typing import TypedDict
      from langgraph.types import StreamWriter

      class State(TypedDict):
            topic: str
            joke: str

      # highlight-next-line
      async def generate_joke(state: State, writer: StreamWriter): # (1)!
            writer({"custom_key": "Streaming custom data while generating a joke"})
            return {"joke": f"This is a joke about {state['topic']}"}

      graph = (
            StateGraph(State)
            .add_node(generate_joke)
            .add_edge(START, "generate_joke")
            .compile()
      )

      async for chunk in graph.astream(
            {"topic": "ice cream"},
            # highlight-next-line
            stream_mode="custom", # (2)!
      ):
            print(chunk)
      ```

      1. Add `writer` as an argument in the function signature of the async node or tool. LangGraph will automatically pass the stream writer to the function.
      2. Set `stream_mode="custom"` to receive the custom data in the stream.

:::
