# 그래프 API 사용 방법

이 가이드는 LangGraph의 Graph API의 기초를 보여줍니다. [상태](#define-and-update-state)뿐만 아니라 [시퀀스](#create-a-sequence-of-steps), [브랜치](#create-branches), [루프](#create-and-control-loops)와 같은 일반적인 그래프 구조를 구성하는 방법을 안내합니다. 또한 map-reduce 워크플로를 위한 [Send API](#map-reduce-and-the-send-api)와 노드 간 "홉"과 상태 업데이트를 결합하는 [Command API](#combine-control-flow-and-state-updates-with-command)를 포함한 LangGraph의 제어 기능을 다룹니다.

## 설정

:::python
`langgraph` 설치:

```bash
pip install -U langgraph
```
:::

:::js
`langgraph` 설치:

```bash
npm install @langchain/langgraph
```
:::

!!! tip "더 나은 디버깅을 위한 LangSmith 설정"

    [LangSmith](https://smith.langchain.com)에 가입하여 LangGraph 프로젝트의 문제를 빠르게 발견하고 성능을 개선하세요. LangSmith를 사용하면 추적 데이터를 활용하여 LangGraph로 구축한 LLM 앱을 디버그, 테스트 및 모니터링할 수 있습니다 — 시작하는 방법에 대한 자세한 내용은 [문서](https://docs.smith.langchain.com)를 참조하세요.

## 상태 정의 및 업데이트

여기서는 LangGraph에서 [상태](../concepts/low_level.md#state)를 정의하고 업데이트하는 방법을 보여줍니다. 다음을 보여줍니다:

1. 상태를 사용하여 그래프의 [스키마](../concepts/low_level.md#schema)를 정의하는 방법
2. [리듀서](../concepts/low_level.md#reducers)를 사용하여 상태 업데이트가 처리되는 방법을 제어하는 방법.

### 상태 정의

:::python
LangGraph의 [상태](../concepts/low_level.md#state)는 `TypedDict`, `Pydantic` 모델 또는 데이터클래스일 수 있습니다. 아래에서는 `TypedDict`를 사용합니다. Pydantic 사용에 대한 자세한 내용은 [이 섹션](#use-pydantic-models-for-graph-state)을 참조하세요.
:::

:::js
LangGraph의 [상태](../concepts/low_level.md#state)는 Zod 스키마를 사용하여 정의할 수 있습니다. 아래에서는 Zod를 사용합니다. 대체 접근 방식 사용에 대한 자세한 내용은 [이 섹션](#alternative-state-definitions)을 참조하세요.
:::

기본적으로 그래프는 동일한 입력 및 출력 스키마를 가지며 상태가 해당 스키마를 결정합니다. 별도의 입력 및 출력 스키마를 정의하는 방법은 [이 섹션](#define-input-and-output-schemas)을 참조하세요.

[메시지](../concepts/low_level.md#working-with-messages-in-graph-state)를 사용하는 간단한 예제를 고려해봅시다. 이것은 많은 LLM 애플리케이션에 대한 다양한 상태 공식을 나타냅니다. 자세한 내용은 [개념 페이지](../concepts/low_level.md#working-with-messages-in-graph-state)를 참조하세요.

:::python
```python
from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict

class State(TypedDict):
    messages: list[AnyMessage]
    extra_field: int
```

이 상태는 [메시지](https://python.langchain.com/docs/concepts/messages/) 객체 목록과 추가 정수 필드를 추적합니다.
:::

:::js
```typescript
import { BaseMessage } from "@langchain/core/messages";
import { z } from "zod";

const State = z.object({
  messages: z.array(z.custom<BaseMessage>()),
  extraField: z.number(),
});
```

이 상태는 [메시지](https://js.langchain.com/docs/concepts/messages/) 객체 목록과 추가 정수 필드를 추적합니다.
:::

### 상태 업데이트

:::python
단일 노드가 있는 예제 그래프를 만들어봅시다. [노드](../concepts/low_level.md#nodes)는 그래프의 상태를 읽고 업데이트하는 Python 함수일 뿐입니다. 이 함수의 첫 번째 인수는 항상 상태입니다:

```python
from langchain_core.messages import AIMessage

def node(state: State):
    messages = state["messages"]
    new_message = AIMessage("Hello!")
    return {"messages": messages + [new_message], "extra_field": 10}
```

이 노드는 단순히 메시지 목록에 메시지를 추가하고 추가 필드를 채웁니다.
:::

:::js
단일 노드가 있는 예제 그래프를 만들어봅시다. [노드](../concepts/low_level.md#nodes)는 그래프의 상태를 읽고 업데이트하는 TypeScript 함수일 뿐입니다. 이 함수의 첫 번째 인수는 항상 상태입니다:

```typescript
import { AIMessage } from "@langchain/core/messages";

const node = (state: z.infer<typeof State>) => {
  const messages = state.messages;
  const newMessage = new AIMessage("Hello!");
  return { messages: messages.concat([newMessage]), extraField: 10 };
};
```

This node simply appends a message to our message list, and populates an extra field.
:::

!!! important

    Nodes should return updates to the state directly, instead of mutating the state.

:::python
Let's next define a simple graph containing this node. We use [StateGraph](../concepts/low_level.md#stategraph) to define a graph that operates on this state. We then use [add_node](../concepts/low_level.md#nodes) populate our graph.

```python
from langgraph.graph import StateGraph

builder = StateGraph(State)
builder.add_node(node)
builder.set_entry_point("node")
graph = builder.compile()
```
:::

:::js
Let's next define a simple graph containing this node. We use [StateGraph](../concepts/low_level.md#stategraph) to define a graph that operates on this state. We then use [addNode](../concepts/low_level.md#nodes) populate our graph.

```typescript
import { StateGraph } from "@langchain/langgraph";

const graph = new StateGraph(State)
  .addNode("node", node)
  .addEdge("__start__", "node")
  .compile();
```
:::

LangGraph provides built-in utilities for visualizing your graph. Let's inspect our graph. See [this section](#visualize-your-graph) for detail on visualization.

:::python
```python
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

![Simple graph with single node](assets/graph_api_image_1.png)
:::

:::js
```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await graph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```
:::

In this case, our graph just executes a single node. Let's proceed with a simple invocation:

:::python
```python
from langchain_core.messages import HumanMessage

result = graph.invoke({"messages": [HumanMessage("Hi")]})
result
```

```
{'messages': [HumanMessage(content='Hi'), AIMessage(content='Hello!')], 'extra_field': 10}
```
:::

:::js
```typescript
import { HumanMessage } from "@langchain/core/messages";

const result = await graph.invoke({ messages: [new HumanMessage("Hi")], extraField: 0 });
console.log(result);
```

```
{ messages: [HumanMessage { content: 'Hi' }, AIMessage { content: 'Hello!' }], extraField: 10 }
```
:::

Note that:

- We kicked off invocation by updating a single key of the state.
- We receive the entire state in the invocation result.

:::python
For convenience, we frequently inspect the content of [message objects](https://python.langchain.com/docs/concepts/messages/) via pretty-print:

```python
for message in result["messages"]:
    message.pretty_print()
```

```
================================ Human Message ================================

Hi
================================== Ai Message ==================================

Hello!
```
:::

:::js
For convenience, we frequently inspect the content of [message objects](https://js.langchain.com/docs/concepts/messages/) via logging:

```typescript
for (const message of result.messages) {
  console.log(`${message.getType()}: ${message.content}`);
}
```

```
human: Hi
ai: Hello!
```
:::

### Process state updates with reducers

Each key in the state can have its own independent [reducer](../concepts/low_level.md#reducers) function, which controls how updates from nodes are applied. If no reducer function is explicitly specified then it is assumed that all updates to the key should override it.

:::python
For `TypedDict` state schemas, we can define reducers by annotating the corresponding field of the state with a reducer function.

In the earlier example, our node updated the `"messages"` key in the state by appending a message to it. Below, we add a reducer to this key, such that updates are automatically appended:

```python
from typing_extensions import Annotated

def add(left, right):
    """Can also import `add` from the `operator` built-in."""
    return left + right

class State(TypedDict):
    # highlight-next-line
    messages: Annotated[list[AnyMessage], add]
    extra_field: int
```

Now our node can be simplified:

```python
def node(state: State):
    new_message = AIMessage("Hello!")
    # highlight-next-line
    return {"messages": [new_message], "extra_field": 10}
```
:::

:::js
For Zod state schemas, we can define reducers by using the special `.langgraph.reducer()` method on the schema field.

In the earlier example, our node updated the `"messages"` key in the state by appending a message to it. Below, we add a reducer to this key, such that updates are automatically appended:

```typescript
import "@langchain/langgraph/zod";

const State = z.object({
  // highlight-next-line
  messages: z.array(z.custom<BaseMessage>()).langgraph.reducer((x, y) => x.concat(y)),
  extraField: z.number(),
});
```

Now our node can be simplified:

```typescript
const node = (state: z.infer<typeof State>) => {
  const newMessage = new AIMessage("Hello!");
  // highlight-next-line
  return { messages: [newMessage], extraField: 10 };
};
```
:::

:::python
```python
from langgraph.graph import START

graph = StateGraph(State).add_node(node).add_edge(START, "node").compile()

result = graph.invoke({"messages": [HumanMessage("Hi")]})

for message in result["messages"]:
    message.pretty_print()
```

```
================================ Human Message ================================

Hi
================================== Ai Message ==================================

Hello!
```
:::

:::js
```typescript
import { START } from "@langchain/langgraph";

const graph = new StateGraph(State)
  .addNode("node", node)
  .addEdge(START, "node")
  .compile();

const result = await graph.invoke({ messages: [new HumanMessage("Hi")] });

for (const message of result.messages) {
  console.log(`${message.getType()}: ${message.content}`);
}
```

```
human: Hi
ai: Hello!
```
:::

#### MessagesState

In practice, there are additional considerations for updating lists of messages:

- We may wish to update an existing message in the state.
- We may want to accept short-hands for [message formats](../concepts/low_level.md#using-messages-in-your-graph), such as [OpenAI format](https://python.langchain.com/docs/concepts/messages/#openai-format).

:::python
LangGraph includes a built-in reducer `add_messages` that handles these considerations:

```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    # highlight-next-line
    messages: Annotated[list[AnyMessage], add_messages]
    extra_field: int

def node(state: State):
    new_message = AIMessage("Hello!")
    return {"messages": [new_message], "extra_field": 10}

graph = StateGraph(State).add_node(node).set_entry_point("node").compile()
```

```python
# highlight-next-line
input_message = {"role": "user", "content": "Hi"}

result = graph.invoke({"messages": [input_message]})

for message in result["messages"]:
    message.pretty_print()
```

```
================================ Human Message ================================

Hi
================================== Ai Message ==================================

Hello!
```

This is a versatile representation of state for applications involving [chat models](https://python.langchain.com/docs/concepts/chat_models/). LangGraph includes a pre-built `MessagesState` for convenience, so that we can have:

```python
from langgraph.graph import MessagesState

class State(MessagesState):
    extra_field: int
```
:::

:::js
LangGraph includes a built-in `MessagesZodState` that handles these considerations:

```typescript
import { MessagesZodState } from "@langchain/langgraph";

const State = z.object({
  // highlight-next-line
  messages: MessagesZodState.shape.messages,
  extraField: z.number(),
});

const graph = new StateGraph(State)
  .addNode("node", (state) => {
    const newMessage = new AIMessage("Hello!");
    return { messages: [newMessage], extraField: 10 };
  })
  .addEdge(START, "node")
  .compile();
```

```typescript
// highlight-next-line
const inputMessage = { role: "user", content: "Hi" };

const result = await graph.invoke({ messages: [inputMessage] });

for (const message of result.messages) {
  console.log(`${message.getType()}: ${message.content}`);
}
```

```
human: Hi
ai: Hello!
```

This is a versatile representation of state for applications involving [chat models](https://js.langchain.com/docs/concepts/chat_models/). LangGraph includes this pre-built `MessagesZodState` for convenience, so that we can have:

```typescript
import { MessagesZodState } from "@langchain/langgraph";

const State = MessagesZodState.extend({
  extraField: z.number(),
});
```
:::

### Define input and output schemas

By default, `StateGraph` operates with a single schema, and all nodes are expected to communicate using that schema. However, it's also possible to define distinct input and output schemas for a graph.

When distinct schemas are specified, an internal schema will still be used for communication between nodes. The input schema ensures that the provided input matches the expected structure, while the output schema filters the internal data to return only the relevant information according to the defined output schema.

Below, we'll see how to define distinct input and output schema.

:::python
```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# Define the schema for the input
class InputState(TypedDict):
    question: str

# Define the schema for the output
class OutputState(TypedDict):
    answer: str

# Define the overall schema, combining both input and output
class OverallState(InputState, OutputState):
    pass

# Define the node that processes the input and generates an answer
def answer_node(state: InputState):
    # Example answer and an extra key
    return {"answer": "bye", "question": state["question"]}

# Build the graph with input and output schemas specified
builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
builder.add_node(answer_node)  # Add the answer node
builder.add_edge(START, "answer_node")  # Define the starting edge
builder.add_edge("answer_node", END)  # Define the ending edge
graph = builder.compile()  # Compile the graph

# Invoke the graph with an input and print the result
print(graph.invoke({"question": "hi"}))
```

```
{'answer': 'bye'}
```
:::

:::js
```typescript
import { StateGraph, START, END } from "@langchain/langgraph";
import { z } from "zod";

// Define the schema for the input
const InputState = z.object({
  question: z.string(),
});

// Define the schema for the output
const OutputState = z.object({
  answer: z.string(),
});

// Define the overall schema, combining both input and output
const OverallState = InputState.merge(OutputState);

// Build the graph with input and output schemas specified
const graph = new StateGraph({
  input: InputState,
  output: OutputState,
  state: OverallState,
})
  .addNode("answerNode", (state) => {
    // Example answer and an extra key
    return { answer: "bye", question: state.question };
  })
  .addEdge(START, "answerNode")
  .addEdge("answerNode", END)
  .compile();

// Invoke the graph with an input and print the result
console.log(await graph.invoke({ question: "hi" }));
```

```
{ answer: 'bye' }
```
:::

Notice that the output of invoke only includes the output schema.

### Pass private state between nodes

In some cases, you may want nodes to exchange information that is crucial for intermediate logic but doesn't need to be part of the main schema of the graph. This private data is not relevant to the overall input/output of the graph and should only be shared between certain nodes.

Below, we'll create an example sequential graph consisting of three nodes (node_1, node_2 and node_3), where private data is passed between the first two steps (node_1 and node_2), while the third step (node_3) only has access to the public overall state.

:::python
```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# The overall state of the graph (this is the public state shared across nodes)
class OverallState(TypedDict):
    a: str

# Output from node_1 contains private data that is not part of the overall state
class Node1Output(TypedDict):
    private_data: str

# The private data is only shared between node_1 and node_2
def node_1(state: OverallState) -> Node1Output:
    output = {"private_data": "set by node_1"}
    print(f"Entered node `node_1`:\n\tInput: {state}.\n\tReturned: {output}")
    return output

# Node 2 input only requests the private data available after node_1
class Node2Input(TypedDict):
    private_data: str

def node_2(state: Node2Input) -> OverallState:
    output = {"a": "set by node_2"}
    print(f"Entered node `node_2`:\n\tInput: {state}.\n\tReturned: {output}")
    return output

# Node 3 only has access to the overall state (no access to private data from node_1)
def node_3(state: OverallState) -> OverallState:
    output = {"a": "set by node_3"}
    print(f"Entered node `node_3`:\n\tInput: {state}.\n\tReturned: {output}")
    return output

# Connect nodes in a sequence
# node_2 accepts private data from node_1, whereas
# node_3 does not see the private data.
builder = StateGraph(OverallState).add_sequence([node_1, node_2, node_3])
builder.add_edge(START, "node_1")
graph = builder.compile()

# Invoke the graph with the initial state
response = graph.invoke(
    {
        "a": "set at start",
    }
)

print()
print(f"Output of graph invocation: {response}")
```

```
Entered node `node_1`:
	Input: {'a': 'set at start'}.
	Returned: {'private_data': 'set by node_1'}
Entered node `node_2`:
	Input: {'private_data': 'set by node_1'}.
	Returned: {'a': 'set by node_2'}
Entered node `node_3`:
	Input: {'a': 'set by node_2'}.
	Returned: {'a': 'set by node_3'}

Output of graph invocation: {'a': 'set by node_3'}
```
:::

:::js
```typescript
import { StateGraph, START, END } from "@langchain/langgraph";
import { z } from "zod";

// The overall state of the graph (this is the public state shared across nodes)
const OverallState = z.object({
  a: z.string(),
});

// Output from node1 contains private data that is not part of the overall state
const Node1Output = z.object({
  privateData: z.string(),
});

// The private data is only shared between node1 and node2
const node1 = (state: z.infer<typeof OverallState>): z.infer<typeof Node1Output> => {
  const output = { privateData: "set by node1" };
  console.log(`Entered node 'node1':\n\tInput: ${JSON.stringify(state)}.\n\tReturned: ${JSON.stringify(output)}`);
  return output;
};

// Node 2 input only requests the private data available after node1
const Node2Input = z.object({
  privateData: z.string(),
});

const node2 = (state: z.infer<typeof Node2Input>): z.infer<typeof OverallState> => {
  const output = { a: "set by node2" };
  console.log(`Entered node 'node2':\n\tInput: ${JSON.stringify(state)}.\n\tReturned: ${JSON.stringify(output)}`);
  return output;
};

// Node 3 only has access to the overall state (no access to private data from node1)
const node3 = (state: z.infer<typeof OverallState>): z.infer<typeof OverallState> => {
  const output = { a: "set by node3" };
  console.log(`Entered node 'node3':\n\tInput: ${JSON.stringify(state)}.\n\tReturned: ${JSON.stringify(output)}`);
  return output;
};

// Connect nodes in a sequence
// node2 accepts private data from node1, whereas
// node3 does not see the private data.
const graph = new StateGraph({
  state: OverallState,
  nodes: {
    node1: { action: node1, output: Node1Output },
    node2: { action: node2, input: Node2Input },
    node3: { action: node3 },
  }
})
  .addEdge(START, "node1")
  .addEdge("node1", "node2")
  .addEdge("node2", "node3")
  .addEdge("node3", END)
  .compile();

// Invoke the graph with the initial state
const response = await graph.invoke({ a: "set at start" });

console.log(`\nOutput of graph invocation: ${JSON.stringify(response)}`);
```

```
Entered node 'node1':
	Input: {"a":"set at start"}.
	Returned: {"privateData":"set by node1"}
Entered node 'node2':
	Input: {"privateData":"set by node1"}.
	Returned: {"a":"set by node2"}
Entered node 'node3':
	Input: {"a":"set by node2"}.
	Returned: {"a":"set by node3"}

Output of graph invocation: {"a":"set by node3"}
```
:::

:::python

### Use Pydantic models for graph state

A [StateGraph](https://langchain-ai.github.io/langgraph/reference/graphs.md#langgraph.graph.StateGraph) accepts a `state_schema` argument on initialization that specifies the "shape" of the state that the nodes in the graph can access and update.

In our examples, we typically use a python-native `TypedDict` or [`dataclass`](https://docs.python.org/3/library/dataclasses.html) for `state_schema`, but `state_schema` can be any [type](https://docs.python.org/3/library/stdtypes.html#type-objects).

Here, we'll see how a [Pydantic BaseModel](https://docs.pydantic.dev/latest/api/base_model/) can be used for `state_schema` to add run-time validation on **inputs**.

!!! note "Known Limitations" 

    - Currently, the output of the graph will **NOT** be an instance of a pydantic model. 
    - Run-time validation only occurs on inputs into nodes, not on the outputs. 
    - The validation error trace from pydantic does not show which node the error arises in. 
    - Pydantic's recursive validation can be slow. For performance-sensitive applications, you may want to consider using a `dataclass` instead.

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from pydantic import BaseModel

# The overall state of the graph (this is the public state shared across nodes)
class OverallState(BaseModel):
    a: str

def node(state: OverallState):
    return {"a": "goodbye"}

# Build the state graph
builder = StateGraph(OverallState)
builder.add_node(node)  # node_1 is the first node
builder.add_edge(START, "node")  # Start the graph with node_1
builder.add_edge("node", END)  # End the graph after node_1
graph = builder.compile()

# Test the graph with a valid input
graph.invoke({"a": "hello"})
```

Invoke the graph with an **invalid** input

```python
try:
    graph.invoke({"a": 123})  # Should be a string
except Exception as e:
    print("An exception was raised because `a` is an integer rather than a string.")
    print(e)
```

```
An exception was raised because `a` is an integer rather than a string.
1 validation error for OverallState
a
  Input should be a valid string [type=string_type, input_value=123, input_type=int]
    For further information visit https://errors.pydantic.dev/2.9/v/string_type
```

See below for additional features of Pydantic model state:

??? example "Serialization Behavior"

    When using Pydantic models as state schemas, it's important to understand how serialization works, especially when:
    - Passing Pydantic objects as inputs
    - Receiving outputs from the graph
    - Working with nested Pydantic models

    Let's see these behaviors in action.

    ```python
    from langgraph.graph import StateGraph, START, END
    from pydantic import BaseModel

    class NestedModel(BaseModel):
        value: str

    class ComplexState(BaseModel):
        text: str
        count: int
        nested: NestedModel

    def process_node(state: ComplexState):
        # Node receives a validated Pydantic object
        print(f"Input state type: {type(state)}")
        print(f"Nested type: {type(state.nested)}")
        # Return a dictionary update
        return {"text": state.text + " processed", "count": state.count + 1}

    # Build the graph
    builder = StateGraph(ComplexState)
    builder.add_node("process", process_node)
    builder.add_edge(START, "process")
    builder.add_edge("process", END)
    graph = builder.compile()

    # Create a Pydantic instance for input
    input_state = ComplexState(text="hello", count=0, nested=NestedModel(value="test"))
    print(f"Input object type: {type(input_state)}")

    # Invoke graph with a Pydantic instance
    result = graph.invoke(input_state)
    print(f"Output type: {type(result)}")
    print(f"Output content: {result}")

    # Convert back to Pydantic model if needed
    output_model = ComplexState(**result)
    print(f"Converted back to Pydantic: {type(output_model)}")
    ```

??? example "Runtime Type Coercion"

    Pydantic performs runtime type coercion for certain data types. This can be helpful but also lead to unexpected behavior if you're not aware of it.

    ```python
    from langgraph.graph import StateGraph, START, END
    from pydantic import BaseModel

    class CoercionExample(BaseModel):
        # Pydantic will coerce string numbers to integers
        number: int
        # Pydantic will parse string booleans to bool
        flag: bool

    def inspect_node(state: CoercionExample):
        print(f"number: {state.number} (type: {type(state.number)})")
        print(f"flag: {state.flag} (type: {type(state.flag)})")
        return {}

    builder = StateGraph(CoercionExample)
    builder.add_node("inspect", inspect_node)
    builder.add_edge(START, "inspect")
    builder.add_edge("inspect", END)
    graph = builder.compile()

    # Demonstrate coercion with string inputs that will be converted
    result = graph.invoke({"number": "42", "flag": "true"})

    # This would fail with a validation error
    try:
        graph.invoke({"number": "not-a-number", "flag": "true"})
    except Exception as e:
        print(f"\nExpected validation error: {e}")
    ```

??? example "Working with Message Models"

    When working with LangChain message types in your state schema, there are important considerations for serialization. You should use `AnyMessage` (rather than `BaseMessage`) for proper serialization/deserialization when using message objects over the wire.

    ```python
    from langgraph.graph import StateGraph, START, END
    from pydantic import BaseModel
    from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
    from typing import List

    class ChatState(BaseModel):
        messages: List[AnyMessage]
        context: str

    def add_message(state: ChatState):
        return {"messages": state.messages + [AIMessage(content="Hello there!")]}

    builder = StateGraph(ChatState)
    builder.add_node("add_message", add_message)
    builder.add_edge(START, "add_message")
    builder.add_edge("add_message", END)
    graph = builder.compile()

    # Create input with a message
    initial_state = ChatState(
        messages=[HumanMessage(content="Hi")], context="Customer support chat"
    )

    result = graph.invoke(initial_state)
    print(f"Output: {result}")

    # Convert back to Pydantic model to see message types
    output_model = ChatState(**result)
    for i, msg in enumerate(output_model.messages):
        print(f"Message {i}: {type(msg).__name__} - {msg.content}")
    ```
:::

:::js
### Alternative state definitions

While Zod schemas are the recommended approach, LangGraph also supports other ways to define state schemas:

```typescript
import { BaseMessage } from "@langchain/core/messages";
import { StateGraph } from "@langchain/langgraph";

interface WorkflowChannelsState {
  messages: BaseMessage[];
  question: string;
  answer: string;
}

const workflowWithChannels = new StateGraph<WorkflowChannelsState>({
  channels: {
    messages: {
      reducer: (currentState, updateValue) => currentState.concat(updateValue),
      default: () => [],
    },
    question: null,
    answer: null,
  },
});
```
:::

## 런타임 구성 추가

때때로 그래프를 호출할 때 구성할 수 있기를 원할 수 있습니다. 예를 들어, _그래프 상태를 이러한 매개변수로 오염시키지 않고_ 런타임에 사용할 LLM이나 시스템 프롬프트를 지정할 수 있기를 원할 수 있습니다.

런타임 구성을 추가하려면:

1. 구성에 대한 스키마를 지정합니다
2. 노드 또는 조건부 엣지의 함수 시그니처에 구성을 추가합니다
3. 그래프에 구성을 전달합니다.

간단한 예제는 아래를 참조하세요:

:::python
```python
from langgraph.graph import END, StateGraph, START
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

# 1. Specify config schema
class ContextSchema(TypedDict):
    my_runtime_value: str

# 2. Define a graph that accesses the config in a node
class State(TypedDict):
    my_state_value: str

# highlight-next-line
def node(state: State, runtime: Runtime[ContextSchema]):
    # highlight-next-line
    if runtime.context["my_runtime_value"] == "a":
        return {"my_state_value": 1}
        # highlight-next-line
    elif runtime.context["my_runtime_value"] == "b":
        return {"my_state_value": 2}
    else:
        raise ValueError("Unknown values.")

# highlight-next-line
builder = StateGraph(State, context_schema=ContextSchema)
builder.add_node(node)
builder.add_edge(START, "node")
builder.add_edge("node", END)

graph = builder.compile()

# 3. Pass in configuration at runtime:
# highlight-next-line
print(graph.invoke({}, context={"my_runtime_value": "a"}))
# highlight-next-line
print(graph.invoke({}, context={"my_runtime_value": "b"}))
```

```
{'my_state_value': 1}
{'my_state_value': 2}
```
:::

:::js
```typescript
import { StateGraph, END, START } from "@langchain/langgraph";
import { RunnableConfig } from "@langchain/core/runnables";
import { z } from "zod";

// 1. Specify config schema
const ConfigurableSchema = z.object({
  myRuntimeValue: z.string(),
});

// 2. Define a graph that accesses the config in a node
const State = z.object({
  myStateValue: z.number(),
});

const graph = new StateGraph(State)
  .addNode("node", (state, config) => {
    // highlight-next-line
    if (config?.configurable?.myRuntimeValue === "a") {
      return { myStateValue: 1 };
      // highlight-next-line
    } else if (config?.configurable?.myRuntimeValue === "b") {
      return { myStateValue: 2 };
    } else {
      throw new Error("Unknown values.");
    }
  })
  .addEdge(START, "node")
  .addEdge("node", END)
  .compile();

// 3. Pass in configuration at runtime:
// highlight-next-line
console.log(await graph.invoke({}, { configurable: { myRuntimeValue: "a" } }));
// highlight-next-line
console.log(await graph.invoke({}, { configurable: { myRuntimeValue: "b" } }));
```

```
{ myStateValue: 1 }
{ myStateValue: 2 }
```
:::

??? example "Extended example: specifying LLM at runtime"

    :::python
    Below we demonstrate a practical example in which we configure what LLM to use at runtime. We will use both OpenAI and Anthropic models.

    ```python
    from dataclasses import dataclass

    from langchain.chat_models import init_chat_model
    from langgraph.graph import MessagesState, END, StateGraph, START
    from langgraph.runtime import Runtime
    from typing_extensions import TypedDict

    @dataclass
    class ContextSchema:
        model_provider: str = "anthropic"

    MODELS = {
        "anthropic": init_chat_model("anthropic:claude-3-5-haiku-latest"),
        "openai": init_chat_model("openai:gpt-4.1-mini"),
    }

    def call_model(state: MessagesState, runtime: Runtime[ContextSchema]):
        model = MODELS[runtime.context.model_provider]
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(MessagesState, context_schema=ContextSchema)
    builder.add_node("model", call_model)
    builder.add_edge(START, "model")
    builder.add_edge("model", END)

    graph = builder.compile()

    # Usage
    input_message = {"role": "user", "content": "hi"}
    # With no configuration, uses default (Anthropic)
    response_1 = graph.invoke({"messages": [input_message]}, context=ContextSchema())["messages"][-1]
    # Or, can set OpenAI
    response_2 = graph.invoke({"messages": [input_message]}, context={"model_provider": "openai"})["messages"][-1]

    print(response_1.response_metadata["model_name"])
    print(response_2.response_metadata["model_name"])
    ```
    ```
    claude-3-5-haiku-20241022
    gpt-4.1-mini-2025-04-14
    ```
    :::

    :::js
    Below we demonstrate a practical example in which we configure what LLM to use at runtime. We will use both OpenAI and Anthropic models.

    ```typescript
    import { ChatOpenAI } from "@langchain/openai";
    import { ChatAnthropic } from "@langchain/anthropic";
    import { MessagesZodState, StateGraph, START, END } from "@langchain/langgraph";
    import { RunnableConfig } from "@langchain/core/runnables";
    import { z } from "zod";

    const ConfigSchema = z.object({
      modelProvider: z.string().default("anthropic"),
    });

    const MODELS = {
      anthropic: new ChatAnthropic({ model: "claude-3-5-haiku-latest" }),
      openai: new ChatOpenAI({ model: "gpt-4o-mini" }),
    };

    const graph = new StateGraph(MessagesZodState)
      .addNode("model", async (state, config) => {
        const modelProvider = config?.configurable?.modelProvider || "anthropic";
        const model = MODELS[modelProvider as keyof typeof MODELS];
        const response = await model.invoke(state.messages);
        return { messages: [response] };
      })
      .addEdge(START, "model")
      .addEdge("model", END)
      .compile();

    // Usage
    const inputMessage = { role: "user", content: "hi" };
    // With no configuration, uses default (Anthropic)
    const response1 = await graph.invoke({ messages: [inputMessage] });
    // Or, can set OpenAI
    const response2 = await graph.invoke(
      { messages: [inputMessage] },
      { configurable: { modelProvider: "openai" } }
    );

    console.log(response1.messages.at(-1)?.response_metadata?.model);
    console.log(response2.messages.at(-1)?.response_metadata?.model);
    ```
    ```
    claude-3-5-haiku-20241022
    gpt-4o-mini-2024-07-18
    ```
    :::

??? example "Extended example: specifying model and system message at runtime"

    :::python
    Below we demonstrate a practical example in which we configure two parameters: the LLM and system message to use at runtime.

    ```python
    from dataclasses import dataclass
    from typing import Optional
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import SystemMessage
    from langgraph.graph import END, MessagesState, StateGraph, START
    from langgraph.runtime import Runtime
    from typing_extensions import TypedDict

    @dataclass
    class ContextSchema:
        model_provider: str = "anthropic"
        system_message: str | None = None

    MODELS = {
        "anthropic": init_chat_model("anthropic:claude-3-5-haiku-latest"),
        "openai": init_chat_model("openai:gpt-4.1-mini"),
    }

    def call_model(state: MessagesState, runtime: Runtime[ContextSchema]):
        model = MODELS[runtime.context.model_provider]
        messages = state["messages"]
        if (system_message := runtime.context.system_message):
            messages = [SystemMessage(system_message)] + messages
        response = model.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState, context_schema=ContextSchema)
    builder.add_node("model", call_model)
    builder.add_edge(START, "model")
    builder.add_edge("model", END)

    graph = builder.compile()

    # Usage
    input_message = {"role": "user", "content": "hi"}
    response = graph.invoke({"messages": [input_message]}, context={"model_provider": "openai", "system_message": "Respond in Italian."})
    for message in response["messages"]:
        message.pretty_print()
    ```
    ```
    ================================ Human Message ================================

    hi
    ================================== Ai Message ==================================

    Ciao! Come posso aiutarti oggi?
    ```
    :::

    :::js
    Below we demonstrate a practical example in which we configure two parameters: the LLM and system message to use at runtime.

    ```typescript
    import { ChatOpenAI } from "@langchain/openai";
    import { ChatAnthropic } from "@langchain/anthropic";
    import { SystemMessage } from "@langchain/core/messages";
    import { MessagesZodState, StateGraph, START, END } from "@langchain/langgraph";
    import { z } from "zod";

    const ConfigSchema = z.object({
      modelProvider: z.string().default("anthropic"),
      systemMessage: z.string().optional(),
    });

    const MODELS = {
      anthropic: new ChatAnthropic({ model: "claude-3-5-haiku-latest" }),
      openai: new ChatOpenAI({ model: "gpt-4o-mini" }),
    };

    const graph = new StateGraph(MessagesZodState)
      .addNode("model", async (state, config) => {
        const modelProvider = config?.configurable?.modelProvider || "anthropic";
        const systemMessage = config?.configurable?.systemMessage;
        
        const model = MODELS[modelProvider as keyof typeof MODELS];
        let messages = state.messages;
        
        if (systemMessage) {
          messages = [new SystemMessage(systemMessage), ...messages];
        }
        
        const response = await model.invoke(messages);
        return { messages: [response] };
      })
      .addEdge(START, "model")
      .addEdge("model", END)
      .compile();

    // Usage
    const inputMessage = { role: "user", content: "hi" };
    const response = await graph.invoke(
      { messages: [inputMessage] },
      {
        configurable: {
          modelProvider: "openai",
          systemMessage: "Respond in Italian."
        }
      }
    );
    
    for (const message of response.messages) {
      console.log(`${message.getType()}: ${message.content}`);
    }
    ```
    ```
    human: hi
    ai: Ciao! Come posso aiutarti oggi?
    ```
    :::

## 재시도 정책 추가

API를 호출하거나, 데이터베이스를 쿼리하거나, LLM을 호출하는 등 노드에 커스텀 재시도 정책을 원하는 많은 사용 사례가 있습니다. LangGraph를 사용하면 노드에 재시도 정책을 추가할 수 있습니다.

:::python
재시도 정책을 구성하려면 [add_node](../reference/graphs.md#langgraph.graph.state.StateGraph.add_node)에 `retry_policy` 매개변수를 전달합니다. `retry_policy` 매개변수는 `RetryPolicy` 명명된 튜플 객체를 받습니다. 아래에서는 기본 매개변수로 `RetryPolicy` 객체를 인스턴스화하고 노드와 연결합니다:

```python
from langgraph.types import RetryPolicy

builder.add_node(
    "node_name",
    node_function,
    retry_policy=RetryPolicy(),
)
```

기본적으로 `retry_on` 매개변수는 `default_retry_on` 함수를 사용하며, 다음을 제외한 모든 예외에 대해 재시도합니다:

- `ValueError`
- `TypeError`
- `ArithmeticError`
- `ImportError`
- `LookupError`
- `NameError`
- `SyntaxError`
- `RuntimeError`
- `ReferenceError`
- `StopIteration`
- `StopAsyncIteration`
- `OSError`

또한 `requests` 및 `httpx`와 같은 인기 있는 http 요청 라이브러리의 예외에 대해서는 5xx 상태 코드에서만 재시도합니다.
:::

:::js
재시도 정책을 구성하려면 [addNode](../reference/graphs.md#langgraph.graph.state.StateGraph.add_node)에 `retryPolicy` 매개변수를 전달합니다. `retryPolicy` 매개변수는 `RetryPolicy` 객체를 받습니다. 아래에서는 기본 매개변수로 `RetryPolicy` 객체를 인스턴스화하고 노드와 연결합니다:

```typescript
import { RetryPolicy } from "@langchain/langgraph";

const graph = new StateGraph(State)
  .addNode("nodeName", nodeFunction, { retryPolicy: {} })
  .compile();
```

기본적으로 재시도 정책은 다음을 제외한 모든 예외에 대해 재시도합니다:

- `TypeError`
- `SyntaxError`
- `ReferenceError`
:::

??? example "Extended example: customizing retry policies"

    :::python
    Consider an example in which we are reading from a SQL database. Below we pass two different retry policies to nodes:

    ```python
    import sqlite3
    from typing_extensions import TypedDict
    from langchain.chat_models import init_chat_model
    from langgraph.graph import END, MessagesState, StateGraph, START
    from langgraph.types import RetryPolicy
    from langchain_community.utilities import SQLDatabase
    from langchain_core.messages import AIMessage

    db = SQLDatabase.from_uri("sqlite:///:memory:")
    model = init_chat_model("anthropic:claude-3-5-haiku-latest")

    def query_database(state: MessagesState):
        query_result = db.run("SELECT * FROM Artist LIMIT 10;")
        return {"messages": [AIMessage(content=query_result)]}

    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    # Define a new graph
    builder = StateGraph(MessagesState)
    builder.add_node(
        "query_database",
        query_database,
        retry_policy=RetryPolicy(retry_on=sqlite3.OperationalError),
    )
    builder.add_node("model", call_model, retry_policy=RetryPolicy(max_attempts=5))
    builder.add_edge(START, "model")
    builder.add_edge("model", "query_database")
    builder.add_edge("query_database", END)
    graph = builder.compile()
    ```
    :::

    :::js
    Consider an example in which we are reading from a SQL database. Below we pass two different retry policies to nodes:

    ```typescript
    import Database from "better-sqlite3";
    import { ChatAnthropic } from "@langchain/anthropic";
    import { StateGraph, START, END, MessagesZodState } from "@langchain/langgraph";
    import { AIMessage } from "@langchain/core/messages";
    import { z } from "zod";

    // Create an in-memory database
    const db: typeof Database.prototype = new Database(":memory:");

    const model = new ChatAnthropic({ model: "claude-3-5-sonnet-20240620" });

    const callModel = async (state: z.infer<typeof MessagesZodState>) => {
      const response = await model.invoke(state.messages);
      return { messages: [response] };
    };

    const queryDatabase = async (state: z.infer<typeof MessagesZodState>) => {
      const queryResult: string = JSON.stringify(
        db.prepare("SELECT * FROM Artist LIMIT 10;").all(),
      );

      return { messages: [new AIMessage({ content: "queryResult" })] };
    };

    const workflow = new StateGraph(MessagesZodState)
      // Define the two nodes we will cycle between
      .addNode("call_model", callModel, { retryPolicy: { maxAttempts: 5 } })
      .addNode("query_database", queryDatabase, {
        retryPolicy: {
          retryOn: (e: any): boolean => {
            if (e instanceof Database.SqliteError) {
              // Retry on "SQLITE_BUSY" error
              return e.code === "SQLITE_BUSY";
            }
            return false; // Don't retry on other errors
          },
        },
      })
      .addEdge(START, "call_model")
      .addEdge("call_model", "query_database")
      .addEdge("query_database", END);

    const graph = workflow.compile();
    ```
    :::

:::python

## 노드 캐싱 추가

노드 캐싱은 비용이 많이 드는(시간 또는 비용 측면에서) 작업을 수행할 때처럼 반복 작업을 피하고 싶은 경우에 유용합니다. LangGraph를 사용하면 그래프의 노드에 개별화된 캐싱 정책을 추가할 수 있습니다.

캐시 정책을 구성하려면 [add_node](https://langchain-ai.github.io/langgraph/reference/graphs.md#langgraph.graph.state.StateGraph.add_node) 함수에 `cache_policy` 매개변수를 전달합니다. 다음 예제에서는 120초의 time to live와 기본 `key_func` 생성기로 [`CachePolicy`](https://langchain-ai.github.io/langgraph/reference/types/?h=cachepolicy#langgraph.types.CachePolicy) 객체를 인스턴스화합니다. 그런 다음 노드와 연결합니다:

```python
from langgraph.types import CachePolicy

builder.add_node(
    "node_name",
    node_function,
    cache_policy=CachePolicy(ttl=120),
)
```

그런 다음 그래프에 대한 노드 레벨 캐싱을 활성화하려면 그래프를 컴파일할 때 `cache` 인수를 설정합니다. 아래 예제는 `InMemoryCache`를 사용하여 인메모리 캐시로 그래프를 설정하지만 `SqliteCache`도 사용할 수 있습니다.

```python
from langgraph.cache.memory import InMemoryCache

graph = builder.compile(cache=InMemoryCache())
```
:::

## 단계 시퀀스 생성

!!! info "필수 조건"

    이 가이드는 위의 [상태](#define-and-update-state) 섹션에 익숙하다고 가정합니다.

여기서는 간단한 단계 시퀀스를 구성하는 방법을 보여줍니다. 다음을 보여줍니다:

1. 순차 그래프를 구축하는 방법
2. 유사한 그래프를 구성하기 위한 내장 단축 방법.

:::python
노드 시퀀스를 추가하려면 [그래프](../concepts/low_level.md#stategraph)의 `.add_node` 및 `.add_edge` 메서드를 사용합니다:

```python
from langgraph.graph import START, StateGraph

builder = StateGraph(State)

# Add nodes
builder.add_node(step_1)
builder.add_node(step_2)
builder.add_node(step_3)

# Add edges
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
```

We can also use the built-in shorthand `.add_sequence`:

```python
builder = StateGraph(State).add_sequence([step_1, step_2, step_3])
builder.add_edge(START, "step_1")
```
:::

:::js
To add a sequence of nodes, we use the `.addNode` and `.addEdge` methods of our [graph](../concepts/low_level.md#stategraph):

```typescript
import { START, StateGraph } from "@langchain/langgraph";

const builder = new StateGraph(State)
  .addNode("step1", step1)
  .addNode("step2", step2)
  .addNode("step3", step3)
  .addEdge(START, "step1")
  .addEdge("step1", "step2")
  .addEdge("step2", "step3");
```
:::

??? info "Why split application steps into a sequence with LangGraph?"
    LangGraph makes it easy to add an underlying persistence layer to your application.
    This allows state to be checkpointed in between the execution of nodes, so your LangGraph nodes govern:

- How state updates are [checkpointed](../concepts/persistence.md)
- How interruptions are resumed in [human-in-the-loop](../concepts/human_in_the_loop.md) workflows
- How we can "rewind" and branch-off executions using LangGraph's [time travel](../concepts/time-travel.md) features

They also determine how execution steps are [streamed](../concepts/streaming.md), and how your application is visualized
and debugged using [LangGraph Studio](../concepts/langgraph_studio.md).

Let's demonstrate an end-to-end example. We will create a sequence of three steps:

1. Populate a value in a key of the state
2. Update the same value
3. Populate a different value

Let's first define our [state](../concepts/low_level.md#state). This governs the [schema of the graph](../concepts/low_level.md#schema), and can also specify how to apply updates. See [this section](#process-state-updates-with-reducers) for more detail.

In our case, we will just keep track of two values:

:::python
```python
from typing_extensions import TypedDict

class State(TypedDict):
    value_1: str
    value_2: int
```
:::

:::js
```typescript
import { z } from "zod";

const State = z.object({
  value1: z.string(),
  value2: z.number(),
});
```
:::

:::python
Our [nodes](../concepts/low_level.md#nodes) are just Python functions that read our graph's state and make updates to it. The first argument to this function will always be the state:

```python
def step_1(state: State):
    return {"value_1": "a"}

def step_2(state: State):
    current_value_1 = state["value_1"]
    return {"value_1": f"{current_value_1} b"}

def step_3(state: State):
    return {"value_2": 10}
```
:::

:::js
Our [nodes](../concepts/low_level.md#nodes) are just TypeScript functions that read our graph's state and make updates to it. The first argument to this function will always be the state:

```typescript
const step1 = (state: z.infer<typeof State>) => {
  return { value1: "a" };
};

const step2 = (state: z.infer<typeof State>) => {
  const currentValue1 = state.value1;
  return { value1: `${currentValue1} b` };
};

const step3 = (state: z.infer<typeof State>) => {
  return { value2: 10 };
};
```
:::

!!! note

    Note that when issuing updates to the state, each node can just specify the value of the key it wishes to update.

    By default, this will **overwrite** the value of the corresponding key. You can also use [reducers](../concepts/low_level.md#reducers) to control how updates are processed— for example, you can append successive updates to a key instead. See [this section](#process-state-updates-with-reducers) for more detail.

Finally, we define the graph. We use [StateGraph](../concepts/low_level.md#stategraph) to define a graph that operates on this state.

:::python
We will then use [add_node](../concepts/low_level.md#messagesstate) and [add_edge](../concepts/low_level.md#edges) to populate our graph and define its control flow.

```python
from langgraph.graph import START, StateGraph

builder = StateGraph(State)

# Add nodes
builder.add_node(step_1)
builder.add_node(step_2)
builder.add_node(step_3)

# Add edges
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
```
:::

:::js
We will then use [addNode](../concepts/low_level.md#nodes) and [addEdge](../concepts/low_level.md#edges) to populate our graph and define its control flow.

```typescript
import { START, StateGraph } from "@langchain/langgraph";

const graph = new StateGraph(State)
  .addNode("step1", step1)
  .addNode("step2", step2)
  .addNode("step3", step3)
  .addEdge(START, "step1")
  .addEdge("step1", "step2")
  .addEdge("step2", "step3")
  .compile();
```
:::

:::python
!!! tip "Specifying custom names"

    You can specify custom names for nodes using `.add_node`:

    ```python
    builder.add_node("my_node", step_1)
    ```
:::

:::js
!!! tip "Specifying custom names"

    You can specify custom names for nodes using `.addNode`:

    ```typescript
    const graph = new StateGraph(State)
      .addNode("myNode", step1)
      .compile();
    ```
:::

Note that:

:::python
- `.add_edge` takes the names of nodes, which for functions defaults to `node.__name__`.
- We must specify the entry point of the graph. For this we add an edge with the [START node](../concepts/low_level.md#start-node).
- The graph halts when there are no more nodes to execute.

We next [compile](../concepts/low_level.md#compiling-your-graph) our graph. This provides a few basic checks on the structure of the graph (e.g., identifying orphaned nodes). If we were adding persistence to our application via a [checkpointer](../concepts/persistence.md), it would also be passed in here.

```python
graph = builder.compile()
```
:::

:::js
- `.addEdge` takes the names of nodes, which for functions defaults to `node.name`.
- We must specify the entry point of the graph. For this we add an edge with the [START node](../concepts/low_level.md#start-node).
- The graph halts when there are no more nodes to execute.

We next [compile](../concepts/low_level.md#compiling-your-graph) our graph. This provides a few basic checks on the structure of the graph (e.g., identifying orphaned nodes). If we were adding persistence to our application via a [checkpointer](../concepts/persistence.md), it would also be passed in here.
:::

LangGraph provides built-in utilities for visualizing your graph. Let's inspect our sequence. See [this guide](#visualize-your-graph) for detail on visualization.

:::python
```python
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

![Sequence of steps graph](assets/graph_api_image_2.png)
:::

:::js
```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await graph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```
:::

Let's proceed with a simple invocation:

:::python
```python
graph.invoke({"value_1": "c"})
```

```
{'value_1': 'a b', 'value_2': 10}
```
:::

:::js
```typescript
const result = await graph.invoke({ value1: "c" });
console.log(result);
```

```
{ value1: 'a b', value2: 10 }
```
:::

Note that:

- We kicked off invocation by providing a value for a single state key. We must always provide a value for at least one key.
- The value we passed in was overwritten by the first node.
- The second node updated the value.
- The third node populated a different value.

:::python
!!! tip "Built-in shorthand"

    `langgraph>=0.2.46` includes a built-in short-hand `add_sequence` for adding node sequences. You can compile the same graph as follows:

    ```python
    # highlight-next-line
    builder = StateGraph(State).add_sequence([step_1, step_2, step_3])
    builder.add_edge(START, "step_1")

    graph = builder.compile()

    graph.invoke({"value_1": "c"})
    ```
:::

## 브랜치 생성

노드의 병렬 실행은 전체 그래프 작업 속도를 높이는 데 필수적입니다. LangGraph는 노드의 병렬 실행을 기본적으로 지원하여 그래프 기반 워크플로의 성능을 크게 향상시킬 수 있습니다. 이 병렬화는 표준 엣지와 [conditional_edges](https://langchain-ai.github.io/langgraph/reference/graphs.md#langgraph.graph.MessageGraph.add_conditional_edges)를 모두 활용하여 팬아웃(fan-out) 및 팬인(fan-in) 메커니즘을 통해 달성됩니다. 아래는 작동하는 분기 데이터 플로우를 생성하는 방법을 보여주는 몇 가지 예제입니다.

### 그래프 노드를 병렬로 실행

이 예제에서는 `Node A`에서 `B와 C`로 팬아웃한 다음 `D`로 팬인합니다. 상태에서 [리듀서 추가 작업을 지정합니다](https://langchain-ai.github.io/langgraph/concepts/low_level.md#reducers). 이렇게 하면 기존 값을 단순히 덮어쓰는 대신 State의 특정 키에 대한 값을 결합하거나 누적합니다. 리스트의 경우 새 리스트를 기존 리스트에 연결합니다. 리듀서로 상태를 업데이트하는 방법에 대한 자세한 내용은 위의 [상태 리듀서](#process-state-updates-with-reducers) 섹션을 참조하세요.

:::python
```python
import operator
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    aggregate: Annotated[list, operator.add]

def a(state: State):
    print(f'Adding "A" to {state["aggregate"]}')
    return {"aggregate": ["A"]}

def b(state: State):
    print(f'Adding "B" to {state["aggregate"]}')
    return {"aggregate": ["B"]}

def c(state: State):
    print(f'Adding "C" to {state["aggregate"]}')
    return {"aggregate": ["C"]}

def d(state: State):
    print(f'Adding "D" to {state["aggregate"]}')
    return {"aggregate": ["D"]}

builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)
builder.add_node(c)
builder.add_node(d)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "d")
builder.add_edge("c", "d")
builder.add_edge("d", END)
graph = builder.compile()
```
:::

:::js
```typescript
import "@langchain/langgraph/zod";
import { StateGraph, START, END } from "@langchain/langgraph";
import { z } from "zod";

const State = z.object({
  // The reducer makes this append-only
  aggregate: z.array(z.string()).langgraph.reducer((x, y) => x.concat(y)),
});

const nodeA = (state: z.infer<typeof State>) => {
  console.log(`Adding "A" to ${state.aggregate}`);
  return { aggregate: ["A"] };
};

const nodeB = (state: z.infer<typeof State>) => {
  console.log(`Adding "B" to ${state.aggregate}`);
  return { aggregate: ["B"] };
};

const nodeC = (state: z.infer<typeof State>) => {
  console.log(`Adding "C" to ${state.aggregate}`);
  return { aggregate: ["C"] };
};

const nodeD = (state: z.infer<typeof State>) => {
  console.log(`Adding "D" to ${state.aggregate}`);
  return { aggregate: ["D"] };
};

const graph = new StateGraph(State)
  .addNode("a", nodeA)
  .addNode("b", nodeB)
  .addNode("c", nodeC)
  .addNode("d", nodeD)
  .addEdge(START, "a")
  .addEdge("a", "b")
  .addEdge("a", "c")
  .addEdge("b", "d")
  .addEdge("c", "d")
  .addEdge("d", END)
  .compile();
```
:::

:::python
```python
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

![Parallel execution graph](assets/graph_api_image_3.png)
:::

:::js
```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await graph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```
:::

With the reducer, you can see that the values added in each node are accumulated.

:::python
```python
graph.invoke({"aggregate": []}, {"configurable": {"thread_id": "foo"}})
```

```
Adding "A" to []
Adding "B" to ['A']
Adding "C" to ['A']
Adding "D" to ['A', 'B', 'C']
```
:::

:::js
```typescript
const result = await graph.invoke({
  aggregate: [],
});
console.log(result);
```

```
Adding "A" to []
Adding "B" to ['A']
Adding "C" to ['A']
Adding "D" to ['A', 'B', 'C']
{ aggregate: ['A', 'B', 'C', 'D'] }
```
:::

!!! note

    In the above example, nodes `"b"` and `"c"` are executed concurrently in the same [superstep](../concepts/low_level.md#graphs). Because they are in the same step, node `"d"` executes after both `"b"` and `"c"` are finished.

    Importantly, updates from a parallel superstep may not be ordered consistently. If you need a consistent, predetermined ordering of updates from a parallel superstep, you should write the outputs to a separate field in the state together with a value with which to order them.

??? note "Exception handling?"

    LangGraph executes nodes within [supersteps](../concepts/low_level.md#graphs), meaning that while parallel branches are executed in parallel, the entire superstep is **transactional**. If any of these branches raises an exception, **none** of the updates are applied to the state (the entire superstep errors).

    Importantly, when using a [checkpointer](../concepts/persistence.md), results from successful nodes within a superstep are saved, and don't repeat when resumed.

    If you have error-prone (perhaps want to handle flakey API calls), LangGraph provides two ways to address this:

    1. You can write regular python code within your node to catch and handle exceptions.
    2. You can set a **[retry_policy](../reference/types.md#langgraph.types.RetryPolicy)** to direct the graph to retry nodes that raise certain types of exceptions. Only failing branches are retried, so you needn't worry about performing redundant work.

    Together, these let you perform parallel execution and fully control exception handling.

:::python

### Defer node execution

Deferring node execution is useful when you want to delay the execution of a node until all other pending tasks are completed. This is particularly relevant when branches have different lengths, which is common in workflows like map-reduce flows.

The above example showed how to fan-out and fan-in when each path was only one step. But what if one branch had more than one step? Let's add a node `"b_2"` in the `"b"` branch:

```python
import operator
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    aggregate: Annotated[list, operator.add]

def a(state: State):
    print(f'Adding "A" to {state["aggregate"]}')
    return {"aggregate": ["A"]}

def b(state: State):
    print(f'Adding "B" to {state["aggregate"]}')
    return {"aggregate": ["B"]}

def b_2(state: State):
    print(f'Adding "B_2" to {state["aggregate"]}')
    return {"aggregate": ["B_2"]}

def c(state: State):
    print(f'Adding "C" to {state["aggregate"]}')
    return {"aggregate": ["C"]}

def d(state: State):
    print(f'Adding "D" to {state["aggregate"]}')
    return {"aggregate": ["D"]}

builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)
builder.add_node(b_2)
builder.add_node(c)
# highlight-next-line
builder.add_node(d, defer=True)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "b_2")
builder.add_edge("b_2", "d")
builder.add_edge("c", "d")
builder.add_edge("d", END)
graph = builder.compile()
```

```python
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

![Deferred execution graph](assets/graph_api_image_4.png)

```python
graph.invoke({"aggregate": []})
```

```
Adding "A" to []
Adding "B" to ['A']
Adding "C" to ['A']
Adding "B_2" to ['A', 'B', 'C']
Adding "D" to ['A', 'B', 'C', 'B_2']
```

In the above example, nodes `"b"` and `"c"` are executed concurrently in the same superstep. We set `defer=True` on node `d` so it will not execute until all pending tasks are finished. In this case, this means that `"d"` waits to execute until the entire `"b"` branch is finished.
:::

### Conditional branching

:::python
If your fan-out should vary at runtime based on the state, you can use [add_conditional_edges](https://langchain-ai.github.io/langgraph/reference/graphs.md#langgraph.graph.StateGraph.add_conditional_edges) to select one or more paths using the graph state. See example below, where node `a` generates a state update that determines the following node.

```python
import operator
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    aggregate: Annotated[list, operator.add]
    # Add a key to the state. We will set this key to determine
    # how we branch.
    which: str

def a(state: State):
    print(f'Adding "A" to {state["aggregate"]}')
    # highlight-next-line
    return {"aggregate": ["A"], "which": "c"}

def b(state: State):
    print(f'Adding "B" to {state["aggregate"]}')
    return {"aggregate": ["B"]}

def c(state: State):
    print(f'Adding "C" to {state["aggregate"]}')
    return {"aggregate": ["C"]}

builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)
builder.add_node(c)
builder.add_edge(START, "a")
builder.add_edge("b", END)
builder.add_edge("c", END)

def conditional_edge(state: State) -> Literal["b", "c"]:
    # Fill in arbitrary logic here that uses the state
    # to determine the next node
    return state["which"]

# highlight-next-line
builder.add_conditional_edges("a", conditional_edge)

graph = builder.compile()
```

```python
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

![Conditional branching graph](assets/graph_api_image_5.png)

```python
result = graph.invoke({"aggregate": []})
print(result)
```

```
Adding "A" to []
Adding "C" to ['A']
{'aggregate': ['A', 'C'], 'which': 'c'}
```
:::

:::js
If your fan-out should vary at runtime based on the state, you can use [addConditionalEdges](https://langchain-ai.github.io/langgraph/reference/graphs.md#langgraph.graph.StateGraph.addConditionalEdges) to select one or more paths using the graph state. See example below, where node `a` generates a state update that determines the following node.

```typescript
import "@langchain/langgraph/zod";
import { StateGraph, START, END } from "@langchain/langgraph";
import { z } from "zod";

const State = z.object({
  aggregate: z.array(z.string()).langgraph.reducer((x, y) => x.concat(y)),
  // Add a key to the state. We will set this key to determine
  // how we branch.
  which: z.string().langgraph.reducer((x, y) => y ?? x),
});

const nodeA = (state: z.infer<typeof State>) => {
  console.log(`Adding "A" to ${state.aggregate}`);
  // highlight-next-line
  return { aggregate: ["A"], which: "c" };
};

const nodeB = (state: z.infer<typeof State>) => {
  console.log(`Adding "B" to ${state.aggregate}`);
  return { aggregate: ["B"] };
};

const nodeC = (state: z.infer<typeof State>) => {
  console.log(`Adding "C" to ${state.aggregate}`);
  return { aggregate: ["C"] };
};

const conditionalEdge = (state: z.infer<typeof State>): "b" | "c" => {
  // Fill in arbitrary logic here that uses the state
  // to determine the next node
  return state.which as "b" | "c";
};

// highlight-next-line
const graph = new StateGraph(State)
  .addNode("a", nodeA)  
  .addNode("b", nodeB)
  .addNode("c", nodeC)
  .addEdge(START, "a")
  .addEdge("b", END)
  .addEdge("c", END)
  .addConditionalEdges("a", conditionalEdge)
  .compile();
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await graph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
const result = await graph.invoke({ aggregate: [] });
console.log(result);
```

```
Adding "A" to []
Adding "C" to ['A']
{ aggregate: ['A', 'C'], which: 'c' }
```
:::

!!! tip

    Your conditional edges can route to multiple destination nodes. For example:

    :::python
    ```python
    def route_bc_or_cd(state: State) -> Sequence[str]:
        if state["which"] == "cd":
            return ["c", "d"]
        return ["b", "c"]
    ```
    :::

    :::js
    ```typescript
    const routeBcOrCd = (state: z.infer<typeof State>): string[] => {
      if (state.which === "cd") {
        return ["c", "d"];
      }
      return ["b", "c"];
    };
    ```
    :::

## Map-Reduce와 Send API

LangGraph는 Send API를 사용하여 map-reduce 및 기타 고급 분기 패턴을 지원합니다. 사용 방법의 예제는 다음과 같습니다:

:::python
```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing_extensions import TypedDict, Annotated
import operator

class OverallState(TypedDict):
    topic: str
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]
    best_selected_joke: str

def generate_topics(state: OverallState):
    return {"subjects": ["lions", "elephants", "penguins"]}

def generate_joke(state: OverallState):
    joke_map = {
        "lions": "Why don't lions like fast food? Because they can't catch it!",
        "elephants": "Why don't elephants use computers? They're afraid of the mouse!",
        "penguins": "Why don't penguins like talking to strangers at parties? Because they find it hard to break the ice."
    }
    return {"jokes": [joke_map[state["subject"]]]}

def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

def best_joke(state: OverallState):
    return {"best_selected_joke": "penguins"}

builder = StateGraph(OverallState)
builder.add_node("generate_topics", generate_topics)
builder.add_node("generate_joke", generate_joke)
builder.add_node("best_joke", best_joke)
builder.add_edge(START, "generate_topics")
builder.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
builder.add_edge("generate_joke", "best_joke")
builder.add_edge("best_joke", END)
graph = builder.compile()
```

```python
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

![Map-reduce graph with fanout](assets/graph_api_image_6.png)

```python
# Call the graph: here we call it to generate a list of jokes
for step in graph.stream({"topic": "animals"}):
    print(step)
```

```
{'generate_topics': {'subjects': ['lions', 'elephants', 'penguins']}}
{'generate_joke': {'jokes': ["Why don't lions like fast food? Because they can't catch it!"]}}
{'generate_joke': {'jokes': ["Why don't elephants use computers? They're afraid of the mouse!"]}}
{'generate_joke': {'jokes': ['Why don't penguins like talking to strangers at parties? Because they find it hard to break the ice.']}}
{'best_joke': {'best_selected_joke': 'penguins'}}
```
:::

:::js
```typescript
import "@langchain/langgraph/zod";
import { StateGraph, START, END, Send } from "@langchain/langgraph";
import { z } from "zod";

const OverallState = z.object({
  topic: z.string(),
  subjects: z.array(z.string()),
  jokes: z.array(z.string()).langgraph.reducer((x, y) => x.concat(y)),
  bestSelectedJoke: z.string(),
});

const generateTopics = (state: z.infer<typeof OverallState>) => {
  return { subjects: ["lions", "elephants", "penguins"] };
};

const generateJoke = (state: { subject: string }) => {
  const jokeMap: Record<string, string> = {
    lions: "Why don't lions like fast food? Because they can't catch it!",
    elephants: "Why don't elephants use computers? They're afraid of the mouse!",
    penguins: "Why don't penguins like talking to strangers at parties? Because they find it hard to break the ice."
  };
  return { jokes: [jokeMap[state.subject]] };
};

const continueToJokes = (state: z.infer<typeof OverallState>) => {
  return state.subjects.map((subject) => new Send("generateJoke", { subject }));
};

const bestJoke = (state: z.infer<typeof OverallState>) => {
  return { bestSelectedJoke: "penguins" };
};

const graph = new StateGraph(OverallState)
  .addNode("generateTopics", generateTopics)
  .addNode("generateJoke", generateJoke)
  .addNode("bestJoke", bestJoke)
  .addEdge(START, "generateTopics")
  .addConditionalEdges("generateTopics", continueToJokes)
  .addEdge("generateJoke", "bestJoke")
  .addEdge("bestJoke", END)
  .compile();
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await graph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

```typescript
// Call the graph: here we call it to generate a list of jokes
for await (const step of await graph.stream({ topic: "animals" })) {
  console.log(step);
}
```

```
{ generateTopics: { subjects: [ 'lions', 'elephants', 'penguins' ] } }
{ generateJoke: { jokes: [ "Why don't lions like fast food? Because they can't catch it!" ] } }
{ generateJoke: { jokes: [ "Why don't elephants use computers? They're afraid of the mouse!" ] } }
{ generateJoke: { jokes: [ "Why don't penguins like talking to strangers at parties? Because they find it hard to break the ice." ] } }
{ bestJoke: { bestSelectedJoke: 'penguins' } }
```
:::

## 루프 생성 및 제어

루프가 있는 그래프를 생성할 때 실행을 종료하는 메커니즘이 필요합니다. 이는 일반적으로 종료 조건에 도달하면 [END](../concepts/low_level.md#end-node) 노드로 라우팅하는 [조건부 엣지](../concepts/low_level.md#conditional-edges)를 추가하여 수행됩니다.

그래프를 호출하거나 스트리밍할 때 그래프 재귀 제한을 설정할 수도 있습니다. 재귀 제한은 오류를 발생시키기 전에 그래프가 실행할 수 있는 [슈퍼스텝](../concepts/low_level.md#graphs) 수를 설정합니다. 재귀 제한 개념에 대한 자세한 내용은 [여기](../concepts/low_level.md#recursion-limit)를 참조하세요.

이러한 메커니즘이 작동하는 방식을 더 잘 이해하기 위해 루프가 있는 간단한 그래프를 고려해봅시다.

!!! tip

    재귀 제한 오류를 받는 대신 상태의 마지막 값을 반환하려면 [다음 섹션](#impose-a-recursion-limit)을 참조하세요.

루프를 생성할 때 종료 조건을 지정하는 조건부 엣지를 포함할 수 있습니다:

:::python
```python
builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)

def route(state: State) -> Literal["b", END]:
    if termination_condition(state):
        return END
    else:
        return "b"

builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)
builder.add_edge("b", "a")
graph = builder.compile()
```
:::

:::js
```typescript
const graph = new StateGraph(State)
  .addNode("a", nodeA)
  .addNode("b", nodeB)
  .addEdge(START, "a")
  .addConditionalEdges("a", route)
  .addEdge("b", "a")
  .compile();

const route = (state: z.infer<typeof State>): "b" | typeof END => {
  if (terminationCondition(state)) {
    return END;
  } else {
    return "b";
  }
};
```
:::

To control the recursion limit, specify `"recursionLimit"` in the config. This will raise a `GraphRecursionError`, which you can catch and handle:

:::python
```python
from langgraph.errors import GraphRecursionError

try:
    graph.invoke(inputs, {"recursion_limit": 3})
except GraphRecursionError:
    print("Recursion Error")
```
:::

:::js
```typescript
import { GraphRecursionError } from "@langchain/langgraph";

try {
  await graph.invoke(inputs, { recursionLimit: 3 });
} catch (error) {
  if (error instanceof GraphRecursionError) {
    console.log("Recursion Error");
  }
}
```
:::

Let's define a graph with a simple loop. Note that we use a conditional edge to implement a termination condition.

:::python
```python
import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    aggregate: Annotated[list, operator.add]

def a(state: State):
    print(f'Node A sees {state["aggregate"]}')
    return {"aggregate": ["A"]}

def b(state: State):
    print(f'Node B sees {state["aggregate"]}')
    return {"aggregate": ["B"]}

# Define nodes
builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)

# Define edges
def route(state: State) -> Literal["b", END]:
    if len(state["aggregate"]) < 7:
        return "b"
    else:
        return END

builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)
builder.add_edge("b", "a")
graph = builder.compile()
```

```python
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

![Simple loop graph](assets/graph_api_image_7.png)
:::

:::js
```typescript
import "@langchain/langgraph/zod";
import { StateGraph, START, END } from "@langchain/langgraph";
import { z } from "zod";

const State = z.object({
  // The reducer makes this append-only
  aggregate: z.array(z.string()).langgraph.reducer((x, y) => x.concat(y)),
});

const nodeA = (state: z.infer<typeof State>) => {
  console.log(`Node A sees ${state.aggregate}`);
  return { aggregate: ["A"] };
};

const nodeB = (state: z.infer<typeof State>) => {
  console.log(`Node B sees ${state.aggregate}`);
  return { aggregate: ["B"] };
};

// Define edges
const route = (state: z.infer<typeof State>): "b" | typeof END => {
  if (state.aggregate.length < 7) {
    return "b";
  } else {
    return END;
  }
};

const graph = new StateGraph(State)
  .addNode("a", nodeA)
  .addNode("b", nodeB)
  .addEdge(START, "a")
  .addConditionalEdges("a", route)
  .addEdge("b", "a")
  .compile();
```

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await graph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```
:::

This architecture is similar to a [React agent](../agents/overview.md) in which node `"a"` is a tool-calling model, and node `"b"` represents the tools.

In our `route` conditional edge, we specify that we should end after the `"aggregate"` list in the state passes a threshold length.

Invoking the graph, we see that we alternate between nodes `"a"` and `"b"` before terminating once we reach the termination condition.

:::python
```python
graph.invoke({"aggregate": []})
```

```
Node A sees []
Node B sees ['A']
Node A sees ['A', 'B']
Node B sees ['A', 'B', 'A']
Node A sees ['A', 'B', 'A', 'B']
Node B sees ['A', 'B', 'A', 'B', 'A']
Node A sees ['A', 'B', 'A', 'B', 'A', 'B']
```
:::

:::js
```typescript
const result = await graph.invoke({ aggregate: [] });
console.log(result);
```

```
Node A sees []
Node B sees ['A']
Node A sees ['A', 'B']
Node B sees ['A', 'B', 'A']
Node A sees ['A', 'B', 'A', 'B']
Node B sees ['A', 'B', 'A', 'B', 'A']
Node A sees ['A', 'B', 'A', 'B', 'A', 'B']
{ aggregate: ['A', 'B', 'A', 'B', 'A', 'B', 'A'] }
```
:::

### 재귀 제한 설정

일부 애플리케이션에서는 주어진 종료 조건에 도달할 것이라는 보장이 없을 수 있습니다. 이러한 경우 그래프의 [재귀 제한](../concepts/low_level.md#recursion-limit)을 설정할 수 있습니다. 이렇게 하면 주어진 수의 [슈퍼스텝](../concepts/low_level.md#graphs) 후에 `GraphRecursionError`가 발생합니다. 그런 다음 이 예외를 포착하고 처리할 수 있습니다:

:::python
```python
from langgraph.errors import GraphRecursionError

try:
    graph.invoke({"aggregate": []}, {"recursion_limit": 4})
except GraphRecursionError:
    print("Recursion Error")
```

```
Node A sees []
Node B sees ['A']
Node C sees ['A', 'B']
Node D sees ['A', 'B']
Node A sees ['A', 'B', 'C', 'D']
Recursion Error
```
:::

:::js
```typescript
import { GraphRecursionError } from "@langchain/langgraph";

try {
  await graph.invoke({ aggregate: [] }, { recursionLimit: 4 });
} catch (error) {
  if (error instanceof GraphRecursionError) {
    console.log("Recursion Error");
  }
}
```

```
Node A sees []
Node B sees ['A']
Node A sees ['A', 'B']
Node B sees ['A', 'B', 'A']
Node A sees ['A', 'B', 'A', 'B']
Recursion Error
```
:::


:::python
??? example "Extended example: return state on hitting recursion limit"

    Instead of raising `GraphRecursionError`, we can introduce a new key to the state that keeps track of the number of steps remaining until reaching the recursion limit. We can then use this key to determine if we should end the run.

    LangGraph implements a special `RemainingSteps` annotation. Under the hood, it creates a `ManagedValue` channel -- a state channel that will exist for the duration of our graph run and no longer.

    ```python
    import operator
    from typing import Annotated, Literal
    from typing_extensions import TypedDict
    from langgraph.graph import StateGraph, START, END
    from langgraph.managed.is_last_step import RemainingSteps

    class State(TypedDict):
        aggregate: Annotated[list, operator.add]
        remaining_steps: RemainingSteps

    def a(state: State):
        print(f'Node A sees {state["aggregate"]}')
        return {"aggregate": ["A"]}

    def b(state: State):
        print(f'Node B sees {state["aggregate"]}')
        return {"aggregate": ["B"]}

    # Define nodes
    builder = StateGraph(State)
    builder.add_node(a)
    builder.add_node(b)

    # Define edges
    def route(state: State) -> Literal["b", END]:
        if state["remaining_steps"] <= 2:
            return END
        else:
            return "b"

    builder.add_edge(START, "a")
    builder.add_conditional_edges("a", route)
    builder.add_edge("b", "a")
    graph = builder.compile()

    # Test it out
    result = graph.invoke({"aggregate": []}, {"recursion_limit": 4})
    print(result)
    ```
    ```
    Node A sees []
    Node B sees ['A']
    Node A sees ['A', 'B']
    {'aggregate': ['A', 'B', 'A']}
    ```
:::

:::python
??? example "Extended example: loops with branches"

    To better understand how the recursion limit works, let's consider a more complex example. Below we implement a loop, but one step fans out into two nodes:

    ```python
    import operator
    from typing import Annotated, Literal
    from typing_extensions import TypedDict
    from langgraph.graph import StateGraph, START, END

    class State(TypedDict):
        aggregate: Annotated[list, operator.add]

    def a(state: State):
        print(f'Node A sees {state["aggregate"]}')
        return {"aggregate": ["A"]}

    def b(state: State):
        print(f'Node B sees {state["aggregate"]}')
        return {"aggregate": ["B"]}

    def c(state: State):
        print(f'Node C sees {state["aggregate"]}')
        return {"aggregate": ["C"]}

    def d(state: State):
        print(f'Node D sees {state["aggregate"]}')
        return {"aggregate": ["D"]}

    # Define nodes
    builder = StateGraph(State)
    builder.add_node(a)
    builder.add_node(b)
    builder.add_node(c)
    builder.add_node(d)

    # Define edges
    def route(state: State) -> Literal["b", END]:
        if len(state["aggregate"]) < 7:
            return "b"
        else:
            return END

    builder.add_edge(START, "a")
    builder.add_conditional_edges("a", route)
    builder.add_edge("b", "c")
    builder.add_edge("b", "d")
    builder.add_edge(["c", "d"], "a")
    graph = builder.compile()
    ```

    ```python
    from IPython.display import Image, display

    display(Image(graph.get_graph().draw_mermaid_png()))
    ```

    ![Complex loop graph with branches](assets/graph_api_image_8.png)

    This graph looks complex, but can be conceptualized as loop of [supersteps](../concepts/low_level.md#graphs):

    1. Node A
    2. Node B
    3. Nodes C and D
    4. Node A
    5. ...

    We have a loop of four supersteps, where nodes C and D are executed concurrently.

    Invoking the graph as before, we see that we complete two full "laps" before hitting the termination condition:

    ```python
    result = graph.invoke({"aggregate": []})
    ```
    ```
    Node A sees []
    Node B sees ['A']
    Node D sees ['A', 'B']
    Node C sees ['A', 'B']
    Node A sees ['A', 'B', 'C', 'D']
    Node B sees ['A', 'B', 'C', 'D', 'A']
    Node D sees ['A', 'B', 'C', 'D', 'A', 'B']
    Node C sees ['A', 'B', 'C', 'D', 'A', 'B']
    Node A sees ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D']
    ```

    However, if we set the recursion limit to four, we only complete one lap because each lap is four supersteps:

    ```python
    from langgraph.errors import GraphRecursionError

    try:
        result = graph.invoke({"aggregate": []}, {"recursion_limit": 4})
    except GraphRecursionError:
        print("Recursion Error")
    ```
    ```
    Node A sees []
    Node B sees ['A']
    Node C sees ['A', 'B']
    Node D sees ['A', 'B']
    Node A sees ['A', 'B', 'C', 'D']
    Recursion Error
    ```
:::

:::python

## Async

Using the async programming paradigm can produce significant performance improvements when running [IO-bound](https://en.wikipedia.org/wiki/I/O_bound) code concurrently (e.g., making concurrent API requests to a chat model provider).

To convert a `sync` implementation of the graph to an `async` implementation, you will need to:

1. Update `nodes` use `async def` instead of `def`.
2. Update the code inside to use `await` appropriately.
3. Invoke the graph with `.ainvoke` or `.astream` as desired.

Because many LangChain objects implement the [Runnable Protocol](https://python.langchain.com/docs/expression_language/interface/) which has `async` variants of all the `sync` methods it's typically fairly quick to upgrade a `sync` graph to an `async` graph.

See example below. To demonstrate async invocations of underlying LLMs, we will include a chat model:

{% include-markdown "../../snippets/chat_model_tabs.md" %}

```python
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph

# highlight-next-line
async def node(state: MessagesState): # (1)!
    # highlight-next-line
    new_message = await llm.ainvoke(state["messages"]) # (2)!
    return {"messages": [new_message]}

builder = StateGraph(MessagesState).add_node(node).set_entry_point("node")
graph = builder.compile()

input_message = {"role": "user", "content": "Hello"}
# highlight-next-line
result = await graph.ainvoke({"messages": [input_message]}) # (3)!
```

1. Declare nodes to be async functions.
2. Use async invocations when available within the node.
3. Use async invocations on the graph object itself.

!!! tip "Async streaming"

    See the [streaming guide](./streaming.md) for examples of streaming with async.

:::

## Combine control flow and state updates with `Command`

It can be useful to combine control flow (edges) and state updates (nodes). For example, you might want to BOTH perform state updates AND decide which node to go to next in the SAME node. LangGraph provides a way to do so by returning a [Command](../reference/types.md#langgraph.types.Command) object from node functions:

:::python
```python
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    return Command(
        # state update
        update={"foo": "bar"},
        # control flow
        goto="my_other_node"
    )
```
:::

:::js
```typescript
import { Command } from "@langchain/langgraph";

const myNode = (state: State): Command => {
  return new Command({
    // state update
    update: { foo: "bar" },
    // control flow
    goto: "myOtherNode"
  });
};
```
:::

We show an end-to-end example below. Let's create a simple graph with 3 nodes: A, B and C. We will first execute node A, and then decide whether to go to Node B or Node C next based on the output of node A.

:::python
```python
import random
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START
from langgraph.types import Command

# Define graph state
class State(TypedDict):
    foo: str

# Define the nodes

def node_a(state: State) -> Command[Literal["node_b", "node_c"]]:
    print("Called A")
    value = random.choice(["b", "c"])
    # this is a replacement for a conditional edge function
    if value == "b":
        goto = "node_b"
    else:
        goto = "node_c"

    # note how Command allows you to BOTH update the graph state AND route to the next node
    return Command(
        # this is the state update
        update={"foo": value},
        # this is a replacement for an edge
        goto=goto,
    )

def node_b(state: State):
    print("Called B")
    return {"foo": state["foo"] + "b"}

def node_c(state: State):
    print("Called C")
    return {"foo": state["foo"] + "c"}
```

We can now create the `StateGraph` with the above nodes. Notice that the graph doesn't have [conditional edges](../concepts/low_level.md#conditional-edges) for routing! This is because control flow is defined with `Command` inside `node_a`.

```python
builder = StateGraph(State)
builder.add_edge(START, "node_a")
builder.add_node(node_a)
builder.add_node(node_b)
builder.add_node(node_c)
# NOTE: there are no edges between nodes A, B and C!

graph = builder.compile()
```

!!! important

    You might have noticed that we used `Command` as a return type annotation, e.g. `Command[Literal["node_b", "node_c"]]`. This is necessary for the graph rendering and tells LangGraph that `node_a` can navigate to `node_b` and `node_c`.

```python
from IPython.display import display, Image

display(Image(graph.get_graph().draw_mermaid_png()))
```

![Command-based graph navigation](assets/graph_api_image_11.png)

If we run the graph multiple times, we'd see it take different paths (A -> B or A -> C) based on the random choice in node A.

```python
graph.invoke({"foo": ""})
```

```
Called A
Called C
```
:::

:::js
```typescript
import { StateGraph, START, Command } from "@langchain/langgraph";
import { z } from "zod";

// Define graph state
const State = z.object({
  foo: z.string(),
});

// Define the nodes

const nodeA = (state: z.infer<typeof State>): Command => {
  console.log("Called A");
  const value = Math.random() > 0.5 ? "b" : "c";
  // this is a replacement for a conditional edge function  
  const goto = value === "b" ? "nodeB" : "nodeC";

  // note how Command allows you to BOTH update the graph state AND route to the next node
  return new Command({
    // this is the state update
    update: { foo: value },
    // this is a replacement for an edge
    goto,
  });
};

const nodeB = (state: z.infer<typeof State>) => {
  console.log("Called B");
  return { foo: state.foo + "b" };
};

const nodeC = (state: z.infer<typeof State>) => {
  console.log("Called C");
  return { foo: state.foo + "c" };
};
```

We can now create the `StateGraph` with the above nodes. Notice that the graph doesn't have [conditional edges](../concepts/low_level.md#conditional-edges) for routing! This is because control flow is defined with `Command` inside `nodeA`.

```typescript
const graph = new StateGraph(State)
  .addNode("nodeA", nodeA, {
    ends: ["nodeB", "nodeC"],
  })
  .addNode("nodeB", nodeB)
  .addNode("nodeC", nodeC)
  .addEdge(START, "nodeA")
  .compile();
```

!!! important

    You might have noticed that we used `ends` to specify which nodes `nodeA` can navigate to. This is necessary for the graph rendering and tells LangGraph that `nodeA` can navigate to `nodeB` and `nodeC`.

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await graph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

If we run the graph multiple times, we'd see it take different paths (A -> B or A -> C) based on the random choice in node A.

```typescript
const result = await graph.invoke({ foo: "" });
console.log(result);
```

```
Called A
Called C
{ foo: 'cc' }
```
:::

### 부모 그래프의 노드로 이동

[서브그래프](../concepts/subgraphs.md)를 사용하는 경우, 서브그래프 내의 노드에서 다른 서브그래프(즉, 부모 그래프의 다른 노드)로 이동하고 싶을 수 있습니다. 이렇게 하려면 `Command`에서 `graph=Command.PARENT`를 지정할 수 있습니다:

:::python
```python
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    return Command(
        update={"foo": "bar"},
        goto="other_subgraph",  # where `other_subgraph` is a node in the parent graph
        graph=Command.PARENT
    )
```
:::

:::js
```typescript
const myNode = (state: State): Command => {
  return new Command({
    update: { foo: "bar" },
    goto: "otherSubgraph",  // where `otherSubgraph` is a node in the parent graph
    graph: Command.PARENT
  });
};
```
:::

위의 예제를 사용하여 이를 시연해봅시다. 위 예제의 `nodeA`를 단일 노드 그래프로 변경하여 부모 그래프에 서브그래프로 추가하겠습니다.

!!! important "`Command.PARENT`를 사용한 상태 업데이트"

    부모 및 서브그래프 [상태 스키마](../concepts/low_level.md#schema) 모두에서 공유되는 키에 대해 서브그래프 노드에서 부모 그래프 노드로 업데이트를 보낼 때, 부모 그래프 상태에서 업데이트하는 키에 대한 [리듀서](../concepts/low_level.md#reducers)를 **반드시** 정의해야 합니다. 아래 예제를 참조하세요.

:::python
```python
import operator
from typing_extensions import Annotated

class State(TypedDict):
    # NOTE: we define a reducer here
    # highlight-next-line
    foo: Annotated[str, operator.add]

def node_a(state: State):
    print("Called A")
    value = random.choice(["a", "b"])
    # this is a replacement for a conditional edge function
    if value == "a":
        goto = "node_b"
    else:
        goto = "node_c"

    # note how Command allows you to BOTH update the graph state AND route to the next node
    return Command(
        update={"foo": value},
        goto=goto,
        # this tells LangGraph to navigate to node_b or node_c in the parent graph
        # NOTE: this will navigate to the closest parent graph relative to the subgraph
        # highlight-next-line
        graph=Command.PARENT,
    )

subgraph = StateGraph(State).add_node(node_a).add_edge(START, "node_a").compile()

def node_b(state: State):
    print("Called B")
    # NOTE: since we've defined a reducer, we don't need to manually append
    # new characters to existing 'foo' value. instead, reducer will append these
    # automatically (via operator.add)
    # highlight-next-line
    return {"foo": "b"}

def node_c(state: State):
    print("Called C")
    # highlight-next-line
    return {"foo": "c"}

builder = StateGraph(State)
builder.add_edge(START, "subgraph")
builder.add_node("subgraph", subgraph)
builder.add_node(node_b)
builder.add_node(node_c)

graph = builder.compile()
```

```python
graph.invoke({"foo": ""})
```

```
Called A
Called C
```
:::

:::js
```typescript
import "@langchain/langgraph/zod";
import { StateGraph, START, Command } from "@langchain/langgraph";
import { z } from "zod";

const State = z.object({
  // NOTE: we define a reducer here
  // highlight-next-line
  foo: z.string().langgraph.reducer((x, y) => x + y),
});

const nodeA = (state: z.infer<typeof State>) => {
  console.log("Called A");
  const value = Math.random() > 0.5 ? "nodeB" : "nodeC";
  
  // note how Command allows you to BOTH update the graph state AND route to the next node
  return new Command({
    update: { foo: "a" },
    goto: value,
    // this tells LangGraph to navigate to nodeB or nodeC in the parent graph
    // NOTE: this will navigate to the closest parent graph relative to the subgraph
    // highlight-next-line
    graph: Command.PARENT,
  });
};

const subgraph = new StateGraph(State)
  .addNode("nodeA", nodeA, { ends: ["nodeB", "nodeC"] })
  .addEdge(START, "nodeA")
  .compile();

const nodeB = (state: z.infer<typeof State>) => {
  console.log("Called B");
  // NOTE: since we've defined a reducer, we don't need to manually append
  // new characters to existing 'foo' value. instead, reducer will append these
  // automatically
  // highlight-next-line
  return { foo: "b" };
};

const nodeC = (state: z.infer<typeof State>) => {
  console.log("Called C");
  // highlight-next-line
  return { foo: "c" };
};

const graph = new StateGraph(State)
  .addNode("subgraph", subgraph, { ends: ["nodeB", "nodeC"] })
  .addNode("nodeB", nodeB)
  .addNode("nodeC", nodeC)
  .addEdge(START, "subgraph")
  .compile();
```

```typescript
const result = await graph.invoke({ foo: "" });
console.log(result);
```

```
Called A
Called C
{ foo: 'ac' }
```
:::

### 도구 내에서 사용

일반적인 사용 사례는 도구 내부에서 그래프 상태를 업데이트하는 것입니다. 예를 들어, 고객 지원 애플리케이션에서 대화 시작 시 계정 번호나 ID를 기반으로 고객 정보를 조회하고 싶을 수 있습니다. 도구에서 그래프 상태를 업데이트하려면 도구에서 `Command(update={"my_custom_key": "foo", "messages": [...]})`를 반환할 수 있습니다:

:::python
```python
@tool
def lookup_user_info(tool_call_id: Annotated[str, InjectedToolCallId], config: RunnableConfig):
    """Use this to look up user information to better assist them with their questions."""
    user_info = get_user_info(config.get("configurable", {}).get("user_id"))
    return Command(
        update={
            # update the state keys
            "user_info": user_info,
            # update the message history
            "messages": [ToolMessage("Successfully looked up user information", tool_call_id=tool_call_id)]
        }
    )
```
:::

:::js
```typescript
import { tool } from "@langchain/core/tools";
import { Command } from "@langchain/langgraph";
import { RunnableConfig } from "@langchain/core/runnables";
import { z } from "zod";

const lookupUserInfo = tool(
  async (input, config: RunnableConfig) => {
    const userId = config.configurable?.userId;
    const userInfo = getUserInfo(userId);
    return new Command({
      update: {
        // update the state keys
        userInfo: userInfo,
        // update the message history
        messages: [{
          role: "tool",
          content: "Successfully looked up user information",
          tool_call_id: config.toolCall.id
        }]
      }
    });
  },
  {
    name: "lookupUserInfo",
    description: "Use this to look up user information to better assist them with their questions.",
    schema: z.object({}),
  }
);
```
:::

!!! important

    You MUST include `messages` (or any state key used for the message history) in `Command.update` when returning `Command` from a tool and the list of messages in `messages` MUST contain a `ToolMessage`. This is necessary for the resulting message history to be valid (LLM providers require AI messages with tool calls to be followed by the tool result messages).

If you are using tools that update state via `Command`, we recommend using prebuilt [`ToolNode`](../reference/agents.md#langgraph.prebuilt.tool_node.ToolNode) which automatically handles tools returning `Command` objects and propagates them to the graph state. If you're writing a custom node that calls tools, you would need to manually propagate `Command` objects returned by the tools as the update from the node.

## Visualize your graph

Here we demonstrate how to visualize the graphs you create.

You can visualize any arbitrary [Graph](https://langchain-ai.github.io/langgraph/reference/graphs/), including [StateGraph](https://langchain-ai.github.io/langgraph/reference/graphs.md#langgraph.graph.state.StateGraph). 

:::python
Let's have some fun by drawing fractals :).

```python
import random
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

class MyNode:
    def __init__(self, name: str):
        self.name = name
    def __call__(self, state: State):
        return {"messages": [("assistant", f"Called node {self.name}")]}

def route(state) -> Literal["entry_node", "__end__"]:
    if len(state["messages"]) > 10:
        return "__end__"
    return "entry_node"

def add_fractal_nodes(builder, current_node, level, max_level):
    if level > max_level:
        return
    # Number of nodes to create at this level
    num_nodes = random.randint(1, 3)  # Adjust randomness as needed
    for i in range(num_nodes):
        nm = ["A", "B", "C"][i]
        node_name = f"node_{current_node}_{nm}"
        builder.add_node(node_name, MyNode(node_name))
        builder.add_edge(current_node, node_name)
        # Recursively add more nodes
        r = random.random()
        if r > 0.2 and level + 1 < max_level:
            add_fractal_nodes(builder, node_name, level + 1, max_level)
        elif r > 0.05:
            builder.add_conditional_edges(node_name, route, node_name)
        else:
            # End
            builder.add_edge(node_name, "__end__")

def build_fractal_graph(max_level: int):
    builder = StateGraph(State)
    entry_point = "entry_node"
    builder.add_node(entry_point, MyNode(entry_point))
    builder.add_edge(START, entry_point)
    add_fractal_nodes(builder, entry_point, 1, max_level)
    # Optional: set a finish point if required
    builder.add_edge(entry_point, END)  # or any specific node
    return builder.compile()

app = build_fractal_graph(3)
```
:::

:::js
Let's create a simple example graph to demonstrate visualization.

```typescript
import { StateGraph, START, END } from "@langchain/langgraph";
import { MessagesZodState } from "@langchain/langgraph";
import { z } from "zod";

const State = MessagesZodState.extend({
  value: z.number(),
});

const app = new StateGraph(State)
  .addNode("node1", (state) => {
    return { value: state.value + 1 };
  })
  .addNode("node2", (state) => {
    return { value: state.value * 2 };
  })
  .addEdge(START, "node1")
  .addConditionalEdges("node1", (state) => {
    if (state.value < 10) {
      return "node2";
    }
    return END;
  })
  .addEdge("node2", "node1")
  .compile();
```
:::

### Mermaid

그래프 클래스를 Mermaid 구문으로 변환할 수도 있습니다.

:::python
```python
print(app.get_graph().draw_mermaid())
```

```
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
	__start__([<p>__start__</p>]):::first
	entry_node(entry_node)
	node_entry_node_A(node_entry_node_A)
	node_entry_node_B(node_entry_node_B)
	node_node_entry_node_B_A(node_node_entry_node_B_A)
	node_node_entry_node_B_B(node_node_entry_node_B_B)
	node_node_entry_node_B_C(node_node_entry_node_B_C)
	__end__([<p>__end__</p>]):::last
	__start__ --> entry_node;
	entry_node --> __end__;
	entry_node --> node_entry_node_A;
	entry_node --> node_entry_node_B;
	node_entry_node_B --> node_node_entry_node_B_A;
	node_entry_node_B --> node_node_entry_node_B_B;
	node_entry_node_B --> node_node_entry_node_B_C;
	node_entry_node_A -.-> entry_node;
	node_entry_node_A -.-> __end__;
	node_node_entry_node_B_A -.-> entry_node;
	node_node_entry_node_B_A -.-> __end__;
	node_node_entry_node_B_B -.-> entry_node;
	node_node_entry_node_B_B -.-> __end__;
	node_node_entry_node_B_C -.-> entry_node;
	node_node_entry_node_B_C -.-> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
:::

:::js
```typescript
const drawableGraph = await app.getGraphAsync();
console.log(drawableGraph.drawMermaid());
```

```
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
	__start__([<p>__start__</p>]):::first
	node1(node1)
	node2(node2)
	__end__([<p>__end__</p>]):::last
	__start__ --> node1;
	node1 -.-> node2;
	node1 -.-> __end__;
	node2 --> node1;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
:::

### PNG

:::python
원한다면 그래프를 `.png`로 렌더링할 수 있습니다. 여기서 세 가지 옵션을 사용할 수 있습니다:

- Mermaid.ink API 사용 (추가 패키지 필요 없음)
- Mermaid + Pyppeteer 사용 (`pip install pyppeteer` 필요)
- graphviz 사용 (`pip install graphviz` 필요)

**Mermaid.Ink 사용**

기본적으로 `draw_mermaid_png()`는 Mermaid.Ink의 API를 사용하여 다이어그램을 생성합니다.

```python
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

display(Image(app.get_graph().draw_mermaid_png()))
```

![Fractal graph visualization](assets/graph_api_image_10.png)

**Mermaid + Pyppeteer 사용**

```python
import nest_asyncio

nest_asyncio.apply()  # Jupyter Notebook에서 비동기 함수를 실행하기 위해 필요

display(
    Image(
        app.get_graph().draw_mermaid_png(
            curve_style=CurveStyle.LINEAR,
            node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
            wrap_label_n_words=9,
            output_file_path=None,
            draw_method=MermaidDrawMethod.PYPPETEER,
            background_color="white",
            padding=10,
        )
    )
)
```

**Graphviz 사용**

```python
try:
    display(Image(app.get_graph().draw_png()))
except ImportError:
    print(
        "You likely need to install dependencies for pygraphviz, see more here https://github.com/pygraphviz/pygraphviz/blob/main/INSTALL.txt"
    )
```
:::

:::js
원한다면 그래프를 `.png`로 렌더링할 수 있습니다. 이는 Mermaid.ink API를 사용하여 다이어그램을 생성합니다.

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await app.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```
:::
