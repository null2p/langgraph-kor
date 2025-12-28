---
search:
  boost: 2
---

# 그래프 API 개념

## 그래프 {#graphs}

LangGraph의 핵심은 에이전트 워크플로우를 그래프로 모델링하는 것입니다. 세 가지 주요 구성 요소를 사용하여 에이전트의 동작을 정의합니다:

1. [`State`](#state): 애플리케이션의 현재 스냅샷을 나타내는 공유 데이터 구조입니다. 모든 데이터 타입이 될 수 있지만 일반적으로 공유 상태 스키마를 사용하여 정의됩니다.

2. [`Nodes`](#nodes): 에이전트의 로직을 인코딩하는 함수입니다. 현재 상태를 입력으로 받아 일부 계산 또는 부작용을 수행하고 업데이트된 상태를 반환합니다.

3. [`Edges`](#edges): 현재 상태를 기반으로 다음에 실행할 `Node`를 결정하는 함수입니다. 조건부 분기 또는 고정 전환이 될 수 있습니다.

`Nodes`와 `Edges`를 구성하여 시간이 지남에 따라 상태가 진화하는 복잡한 루프 워크플로우를 만들 수 있습니다. 하지만 진정한 힘은 LangGraph가 상태를 관리하는 방법에서 나옵니다. 강조하자면: `Nodes`와 `Edges`는 함수일 뿐입니다 - LLM을 포함할 수도 있고 일반적인 코드일 수도 있습니다.

간단히 말해: _노드는 작업을 수행하고, 엣지는 다음에 무엇을 할지 알려줍니다_.

LangGraph의 기본 그래프 알고리즘은 [메시지 전달](https://en.wikipedia.org/wiki/Message_passing)을 사용하여 일반 프로그램을 정의합니다. 노드가 작업을 완료하면 하나 이상의 엣지를 따라 다른 노드로 메시지를 보냅니다. 이러한 수신 노드는 함수를 실행하고 결과 메시지를 다음 노드 집합으로 전달하며 프로세스가 계속됩니다. Google의 [Pregel](https://research.google/pubs/pregel-a-system-for-large-scale-graph-processing/) 시스템에서 영감을 받아 프로그램은 개별 "super-step"으로 진행됩니다.

super-step은 그래프 노드에 대한 단일 반복으로 간주될 수 있습니다. 병렬로 실행되는 노드는 동일한 super-step의 일부이고, 순차적으로 실행되는 노드는 별도의 super-step에 속합니다. 그래프 실행이 시작되면 모든 노드는 `inactive` 상태에서 시작합니다. 노드는 들어오는 엣지(또는 "채널") 중 하나에서 새 메시지(상태)를 받으면 `active`가 됩니다. 활성 노드는 함수를 실행하고 업데이트로 응답합니다. 각 super-step이 끝나면 들어오는 메시지가 없는 노드는 자신을 `inactive`로 표시하여 `halt`에 투표합니다. 모든 노드가 `inactive`이고 전송 중인 메시지가 없으면 그래프 실행이 종료됩니다.

### StateGraph

`StateGraph` 클래스는 사용할 메인 그래프 클래스입니다. 사용자 정의 `State` 객체로 매개변수화됩니다.

### 그래프 컴파일하기 {#compiling-your-graph}

그래프를 빌드하려면 먼저 [state](#state)를 정의하고, [nodes](#nodes)와 [edges](#edges)를 추가한 다음 컴파일합니다. 그래프를 컴파일한다는 것은 정확히 무엇이며 왜 필요할까요?

컴파일은 상당히 간단한 단계입니다. 그래프 구조에 대한 몇 가지 기본 검사를 제공합니다 (고립된 노드가 없는지 등). 또한 [checkpointers](./persistence.md)나 breakpoints 같은 런타임 인자를 지정할 수 있는 곳이기도 합니다. `.compile` 메서드를 호출하기만 하면 그래프를 컴파일할 수 있습니다:

:::python

```python
graph = graph_builder.compile(...)
```

:::

:::js

```typescript
const graph = new StateGraph(StateAnnotation)
  .addNode("nodeA", nodeA)
  .addEdge(START, "nodeA")
  .addEdge("nodeA", END)
  .compile();
```

:::

그래프를 사용하기 전에 **반드시** 컴파일해야 합니다.

## State

:::python
그래프를 정의할 때 가장 먼저 하는 일은 그래프의 `State`를 정의하는 것입니다. `State`는 [그래프의 스키마](#schema)와 상태에 업데이트를 적용하는 방법을 지정하는 [`reducer` 함수](#reducers)로 구성됩니다. `State`의 스키마는 그래프의 모든 `Nodes`와 `Edges`의 입력 스키마가 되며, `TypedDict` 또는 `Pydantic` 모델일 수 있습니다. 모든 `Nodes`는 `State`에 대한 업데이트를 발행하며, 이는 지정된 `reducer` 함수를 사용하여 적용됩니다.
:::

:::js
그래프를 정의할 때 가장 먼저 하는 일은 그래프의 `State`를 정의하는 것입니다. `State`는 [그래프의 스키마](#schema)와 상태에 업데이트를 적용하는 방법을 지정하는 [`reducer` 함수](#reducers)로 구성됩니다. `State`의 스키마는 그래프의 모든 `Nodes`와 `Edges`의 입력 스키마가 되며, Zod 스키마 또는 `Annotation.Root`를 사용하여 빌드된 스키마일 수 있습니다. 모든 `Nodes`는 `State`에 대한 업데이트를 발행하며, 이는 지정된 `reducer` 함수를 사용하여 적용됩니다.
:::

### Schema

:::python
그래프의 스키마를 지정하는 주요 문서화된 방법은 [`TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict)를 사용하는 것입니다. 상태에 기본값을 제공하려면 [`dataclass`](https://docs.python.org/3/library/dataclasses.html)를 사용하세요. 재귀적 데이터 검증을 원하는 경우 그래프 상태로 Pydantic [BaseModel](../how-tos/graph-api.md#use-pydantic-models-for-graph-state)을 사용하는 것도 지원합니다 (단, pydantic은 `TypedDict`나 `dataclass`보다 성능이 낮습니다).

기본적으로 그래프는 동일한 입력 및 출력 스키마를 가집니다. 이를 변경하려면 명시적인 입력 및 출력 스키마를 직접 지정할 수도 있습니다. 이는 많은 키가 있고 일부는 명시적으로 입력용이고 다른 일부는 출력용인 경우에 유용합니다. 사용 방법은 [여기 가이드](../how-tos/graph-api.md#define-input-and-output-schemas)를 참조하세요.
:::

:::js
그래프의 스키마를 지정하는 주요 문서화된 방법은 Zod 스키마를 사용하는 것입니다. 그러나 `Annotation` API를 사용하여 그래프의 스키마를 정의하는 것도 지원합니다.

기본적으로 그래프는 동일한 입력 및 출력 스키마를 가집니다. 이를 변경하려면 명시적인 입력 및 출력 스키마를 직접 지정할 수도 있습니다. 이는 많은 키가 있고 일부는 명시적으로 입력용이고 다른 일부는 출력용인 경우에 유용합니다.
:::

#### Multiple schemas

일반적으로 모든 그래프 노드는 단일 스키마와 통신합니다. 즉, 동일한 상태 채널에서 읽고 쓴다는 의미입니다. 하지만 이에 대한 더 많은 제어가 필요한 경우가 있습니다:

- 내부 노드는 그래프의 입력/출력에 필요하지 않은 정보를 전달할 수 있습니다.
- 그래프에 대해 다른 입력/출력 스키마를 사용하고 싶을 수도 있습니다. 예를 들어, 출력에는 단일 관련 출력 키만 포함될 수 있습니다.

내부 노드 통신을 위해 그래프 내부의 비공개 상태 채널에 노드가 쓰도록 할 수 있습니다. 간단히 비공개 스키마인 `PrivateState`를 정의할 수 있습니다.

그래프에 대한 명시적인 입력 및 출력 스키마를 정의하는 것도 가능합니다. 이러한 경우, 그래프 작업과 관련된 _모든_ 키를 포함하는 "내부" 스키마를 정의합니다. 하지만 그래프의 입력과 출력을 제한하기 위해 "내부" 스키마의 하위 집합인 `input` 및 `output` 스키마도 정의합니다. 자세한 내용은 [이 가이드](../how-tos/graph-api.md#define-input-and-output-schemas)를 참조하세요.

예제를 살펴보겠습니다:

:::python

```python
class InputState(TypedDict):
    user_input: str

class OutputState(TypedDict):
    graph_output: str

class OverallState(TypedDict):
    foo: str
    user_input: str
    graph_output: str

class PrivateState(TypedDict):
    bar: str

def node_1(state: InputState) -> OverallState:
    # Write to OverallState
    return {"foo": state["user_input"] + " name"}

def node_2(state: OverallState) -> PrivateState:
    # Read from OverallState, write to PrivateState
    return {"bar": state["foo"] + " is"}

def node_3(state: PrivateState) -> OutputState:
    # Read from PrivateState, write to OutputState
    return {"graph_output": state["bar"] + " Lance"}

builder = StateGraph(OverallState,input_schema=InputState,output_schema=OutputState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

graph = builder.compile()
graph.invoke({"user_input":"My"})
# {'graph_output': 'My name is Lance'}
```

:::

:::js

```typescript
const InputState = z.object({
  userInput: z.string(),
});

const OutputState = z.object({
  graphOutput: z.string(),
});

const OverallState = z.object({
  foo: z.string(),
  userInput: z.string(),
  graphOutput: z.string(),
});

const PrivateState = z.object({
  bar: z.string(),
});

const graph = new StateGraph({
  state: OverallState,
  input: InputState,
  output: OutputState,
})
  .addNode("node1", (state) => {
    // Write to OverallState
    return { foo: state.userInput + " name" };
  })
  .addNode("node2", (state) => {
    // Read from OverallState, write to PrivateState
    return { bar: state.foo + " is" };
  })
  .addNode(
    "node3",
    (state) => {
      // Read from PrivateState, write to OutputState
      return { graphOutput: state.bar + " Lance" };
    },
    { input: PrivateState }
  )
  .addEdge(START, "node1")
  .addEdge("node1", "node2")
  .addEdge("node2", "node3")
  .addEdge("node3", END)
  .compile();

await graph.invoke({ userInput: "My" });
// { graphOutput: 'My name is Lance' }
```

:::

여기서 주목해야 할 미묘하지만 중요한 두 가지 사항이 있습니다:

:::python

1. `node_1`의 입력 스키마로 `state: InputState`를 전달합니다. 하지만 `OverallState`의 채널인 `foo`에 씁니다. 입력 스키마에 포함되지 않은 상태 채널에 어떻게 쓸 수 있을까요? 이는 노드가 _그래프 상태의 모든 상태 채널에 쓸 수 있기_ 때문입니다. 그래프 상태는 초기화 시 정의된 상태 채널의 합집합이며, 여기에는 `OverallState`와 필터 `InputState` 및 `OutputState`가 포함됩니다.

2. `StateGraph(OverallState,input_schema=InputState,output_schema=OutputState)`로 그래프를 초기화합니다. 그렇다면 `node_2`에서 `PrivateState`에 어떻게 쓸 수 있을까요? `StateGraph` 초기화에서 전달되지 않았는데 그래프가 이 스키마에 어떻게 액세스할까요? 상태 스키마 정의가 존재하는 한 _노드가 추가 상태 채널을 선언할 수도 있기_ 때문에 이것이 가능합니다. 이 경우 `PrivateState` 스키마가 정의되어 있으므로 그래프에 새 상태 채널로 `bar`를 추가하고 쓸 수 있습니다.
   :::

:::js

1. `node1`의 입력 스키마로 `state`를 전달합니다. 하지만 `OverallState`의 채널인 `foo`에 씁니다. 입력 스키마에 포함되지 않은 상태 채널에 어떻게 쓸 수 있을까요? 이는 노드가 _그래프 상태의 모든 상태 채널에 쓸 수 있기_ 때문입니다. 그래프 상태는 초기화 시 정의된 상태 채널의 합집합이며, 여기에는 `OverallState`와 필터 `InputState` 및 `OutputState`가 포함됩니다.

2. `StateGraph({ state: OverallState, input: InputState, output: OutputState })`로 그래프를 초기화합니다. 그렇다면 `node2`에서 `PrivateState`에 어떻게 쓸 수 있을까요? `StateGraph` 초기화에서 전달되지 않았는데 그래프가 이 스키마에 어떻게 액세스할까요? 상태 스키마 정의가 존재하는 한 _노드가 추가 상태 채널을 선언할 수도 있기_ 때문에 이것이 가능합니다. 이 경우 `PrivateState` 스키마가 정의되어 있으므로 그래프에 새 상태 채널로 `bar`를 추가하고 쓸 수 있습니다.
   :::

### Reducers

Reducer는 노드의 업데이트가 `State`에 어떻게 적용되는지 이해하는 데 핵심입니다. `State`의 각 키는 고유한 독립적인 reducer 함수를 가집니다. reducer 함수가 명시적으로 지정되지 않으면 해당 키에 대한 모든 업데이트가 이를 덮어써야 한다고 가정합니다. 기본 유형의 reducer부터 시작하여 몇 가지 다른 유형의 reducer가 있습니다:

#### Default Reducer

이 두 예제는 기본 reducer를 사용하는 방법을 보여줍니다:

**예제 A:**

:::python

```python
from typing_extensions import TypedDict

class State(TypedDict):
    foo: int
    bar: list[str]
```

:::

:::js

```typescript
const State = z.object({
  foo: z.number(),
  bar: z.array(z.string()),
});
```

:::

이 예제에서는 어떤 키에 대해서도 reducer 함수가 지정되지 않았습니다. 그래프에 대한 입력이 다음과 같다고 가정해 봅시다:

:::python
`{"foo": 1, "bar": ["hi"]}`. 그리고 첫 번째 `Node`가 `{"foo": 2}`를 반환한다고 가정해 봅시다. 이는 상태에 대한 업데이트로 처리됩니다. `Node`가 전체 `State` 스키마를 반환할 필요가 없고 업데이트만 반환하면 된다는 점에 주목하세요. 이 업데이트를 적용한 후 `State`는 `{"foo": 2, "bar": ["hi"]}`가 됩니다. 두 번째 노드가 `{"bar": ["bye"]}`를 반환하면 `State`는 `{"foo": 2, "bar": ["bye"]}`가 됩니다.
:::

:::js
`{ foo: 1, bar: ["hi"] }`. 그리고 첫 번째 `Node`가 `{ foo: 2 }`를 반환한다고 가정해 봅시다. 이는 상태에 대한 업데이트로 처리됩니다. `Node`가 전체 `State` 스키마를 반환할 필요가 없고 업데이트만 반환하면 된다는 점에 주목하세요. 이 업데이트를 적용한 후 `State`는 `{ foo: 2, bar: ["hi"] }`가 됩니다. 두 번째 노드가 `{ bar: ["bye"] }`를 반환하면 `State`는 `{ foo: 2, bar: ["bye"] }`가 됩니다.
:::

**예제 B:**

:::python

```python
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]
```

이 예제에서는 `Annotated` 타입을 사용하여 두 번째 키(`bar`)에 대한 reducer 함수(`operator.add`)를 지정했습니다. 첫 번째 키는 변경되지 않은 상태로 유지됩니다. 그래프에 대한 입력이 `{"foo": 1, "bar": ["hi"]}`라고 가정해 봅시다. 그리고 첫 번째 `Node`가 `{"foo": 2}`를 반환한다고 가정해 봅시다. 이는 상태에 대한 업데이트로 처리됩니다. `Node`가 전체 `State` 스키마를 반환할 필요가 없고 업데이트만 반환하면 된다는 점에 주목하세요. 이 업데이트를 적용한 후 `State`는 `{"foo": 2, "bar": ["hi"]}`가 됩니다. 두 번째 노드가 `{"bar": ["bye"]}`를 반환하면 `State`는 `{"foo": 2, "bar": ["hi", "bye"]}`가 됩니다. 여기서 `bar` 키는 두 리스트를 함께 추가하여 업데이트된다는 점에 주목하세요.
:::

:::js

```typescript
import { z } from "zod";
import { withLangGraph } from "@langchain/langgraph/zod";

const State = z.object({
  foo: z.number(),
  bar: withLangGraph(z.array(z.string()), {
    reducer: {
      fn: (x, y) => x.concat(y),
    },
  }),
});
```

이 예제에서는 `withLangGraph` 함수를 사용하여 두 번째 키(`bar`)에 대한 reducer 함수를 지정했습니다. 첫 번째 키는 변경되지 않은 상태로 유지됩니다. 그래프에 대한 입력이 `{ foo: 1, bar: ["hi"] }`라고 가정해 봅시다. 그리고 첫 번째 `Node`가 `{ foo: 2 }`를 반환한다고 가정해 봅시다. 이는 상태에 대한 업데이트로 처리됩니다. `Node`가 전체 `State` 스키마를 반환할 필요가 없고 업데이트만 반환하면 된다는 점에 주목하세요. 이 업데이트를 적용한 후 `State`는 `{ foo: 2, bar: ["hi"] }`가 됩니다. 두 번째 노드가 `{ bar: ["bye"] }`를 반환하면 `State`는 `{ foo: 2, bar: ["hi", "bye"] }`가 됩니다. 여기서 `bar` 키는 두 배열을 함께 추가하여 업데이트된다는 점에 주목하세요.
:::

### 그래프 상태에서 메시지 다루기 {#working-with-messages-in-graph-state}

#### 왜 메시지를 사용하나요? {#why-use-messages}

:::python
대부분의 최신 LLM 제공자는 메시지 리스트를 입력으로 받는 채팅 모델 인터페이스를 가지고 있습니다. 특히 LangChain의 [`ChatModel`](https://python.langchain.com/docs/concepts/#chat-models)은 `Message` 객체의 리스트를 입력으로 받습니다. 이러한 메시지는 `HumanMessage`(사용자 입력) 또는 `AIMessage`(LLM 응답)와 같은 다양한 형태로 제공됩니다. 메시지 객체에 대한 자세한 내용은 [이 개념 가이드](https://python.langchain.com/docs/concepts/#messages)를 참조하세요.
:::

:::js
대부분의 최신 LLM 제공자는 메시지 리스트를 입력으로 받는 채팅 모델 인터페이스를 가지고 있습니다. 특히 LangChain의 [`ChatModel`](https://js.langchain.com/docs/concepts/#chat-models)은 `Message` 객체의 리스트를 입력으로 받습니다. 이러한 메시지는 `HumanMessage`(사용자 입력) 또는 `AIMessage`(LLM 응답)와 같은 다양한 형태로 제공됩니다. 메시지 객체에 대한 자세한 내용은 [이 개념 가이드](https://js.langchain.com/docs/concepts/#messages)를 참조하세요.
:::

#### 그래프에서 메시지 사용하기 {#using-messages-in-your-graph}

:::python
많은 경우 이전 대화 기록을 그래프 상태에 메시지 리스트로 저장하는 것이 유용합니다. 이를 위해 `Message` 객체의 리스트를 저장하는 키(채널)를 그래프 상태에 추가하고 reducer 함수로 어노테이션할 수 있습니다(아래 예제의 `messages` 키 참조). Reducer 함수는 각 상태 업데이트(예: 노드가 업데이트를 보낼 때)마다 상태의 `Message` 객체 리스트를 어떻게 업데이트할지 그래프에 알려주는 데 필수적입니다. Reducer를 지정하지 않으면 모든 상태 업데이트가 가장 최근에 제공된 값으로 메시지 리스트를 덮어씁니다. 기존 리스트에 메시지를 단순히 추가하려면 `operator.add`를 reducer로 사용할 수 있습니다.

그러나 그래프 상태에서 메시지를 수동으로 업데이트하고 싶을 수도 있습니다(예: human-in-the-loop). `operator.add`를 사용하면 그래프에 보내는 수동 상태 업데이트가 기존 메시지를 업데이트하는 대신 기존 메시지 리스트에 추가됩니다. 이를 방지하려면 메시지 ID를 추적하고 업데이트된 경우 기존 메시지를 덮어쓸 수 있는 reducer가 필요합니다. 이를 위해 사전 빌드된 `add_messages` 함수를 사용할 수 있습니다. 새로운 메시지의 경우 단순히 기존 리스트에 추가하지만 기존 메시지에 대한 업데이트도 올바르게 처리합니다.
:::

:::js
많은 경우 이전 대화 기록을 그래프 상태에 메시지 리스트로 저장하는 것이 유용합니다. 이를 위해 `Message` 객체의 리스트를 저장하는 키(채널)를 그래프 상태에 추가하고 reducer 함수로 어노테이션할 수 있습니다(아래 예제의 `messages` 키 참조). Reducer 함수는 각 상태 업데이트(예: 노드가 업데이트를 보낼 때)마다 상태의 `Message` 객체 리스트를 어떻게 업데이트할지 그래프에 알려주는 데 필수적입니다. Reducer를 지정하지 않으면 모든 상태 업데이트가 가장 최근에 제공된 값으로 메시지 리스트를 덮어씁니다. 기존 리스트에 메시지를 단순히 추가하려면 배열을 연결하는 함수를 reducer로 사용할 수 있습니다.

그러나 그래프 상태에서 메시지를 수동으로 업데이트하고 싶을 수도 있습니다(예: human-in-the-loop). 단순한 연결 함수를 사용하면 그래프에 보내는 수동 상태 업데이트가 기존 메시지를 업데이트하는 대신 기존 메시지 리스트에 추가됩니다. 이를 방지하려면 메시지 ID를 추적하고 업데이트된 경우 기존 메시지를 덮어쓸 수 있는 reducer가 필요합니다. 이를 위해 사전 빌드된 `MessagesZodState` 스키마를 사용할 수 있습니다. 새로운 메시지의 경우 단순히 기존 리스트에 추가하지만 기존 메시지에 대한 업데이트도 올바르게 처리합니다.
:::

#### 직렬화

:::python
메시지 ID를 추적하는 것 외에도 `add_messages` 함수는 `messages` 채널에서 상태 업데이트가 수신될 때마다 메시지를 LangChain `Message` 객체로 역직렬화하려고 시도합니다. LangChain 직렬화/역직렬화에 대한 자세한 내용은 [여기](https://python.langchain.com/docs/how_to/serialization/)를 참조하세요. 이를 통해 다음 형식으로 그래프 입력/상태 업데이트를 보낼 수 있습니다:

```python
# this is supported
{"messages": [HumanMessage(content="message")]}

# and this is also supported
{"messages": [{"type": "human", "content": "message"}]}
```

`add_messages`를 사용할 때 상태 업데이트는 항상 LangChain `Messages`로 역직렬화되므로 `state["messages"][-1].content`와 같이 점 표기법을 사용하여 메시지 속성에 액세스해야 합니다. 다음은 `add_messages`를 reducer 함수로 사용하는 그래프의 예입니다.

```python
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

:::

:::js
메시지 ID를 추적하는 것 외에도 `MessagesZodState`는 `messages` 채널에서 상태 업데이트가 수신될 때마다 메시지를 LangChain `Message` 객체로 역직렬화하려고 시도합니다. 이를 통해 다음 형식으로 그래프 입력/상태 업데이트를 보낼 수 있습니다:

```typescript
// this is supported
{
  messages: [new HumanMessage("message")];
}

// and this is also supported
{
  messages: [{ role: "human", content: "message" }];
}
```

`MessagesZodState`를 사용할 때 상태 업데이트는 항상 LangChain `Messages`로 역직렬화되므로 `state.messages[state.messages.length - 1].content`와 같이 점 표기법을 사용하여 메시지 속성에 액세스해야 합니다. 다음은 `MessagesZodState`를 사용하는 그래프의 예입니다:

```typescript
import { StateGraph, MessagesZodState } from "@langchain/langgraph";

const graph = new StateGraph(MessagesZodState)
  ...
```

`MessagesZodState`는 `BaseMessage` 객체의 리스트인 단일 `messages` 키로 정의되며 적절한 reducer를 사용합니다. 일반적으로 메시지 외에도 추적해야 할 상태가 더 많으므로 다음과 같이 이 상태를 확장하고 더 많은 필드를 추가하는 것을 볼 수 있습니다:

```typescript
const State = z.object({
  messages: MessagesZodState.shape.messages,
  documents: z.array(z.string()),
});
```

:::

:::python

#### MessagesState

상태에 메시지 리스트를 포함하는 것이 매우 일반적이기 때문에 메시지를 쉽게 사용할 수 있도록 `MessagesState`라는 사전 빌드된 상태가 존재합니다. `MessagesState`는 `AnyMessage` 객체의 리스트인 단일 `messages` 키로 정의되며 `add_messages` reducer를 사용합니다. 일반적으로 메시지 외에도 추적해야 할 상태가 더 많으므로 다음과 같이 이 상태를 서브클래싱하고 더 많은 필드를 추가하는 것을 볼 수 있습니다:

```python
from langgraph.graph import MessagesState

class State(MessagesState):
    documents: list[str]
```

:::

## 노드 {#nodes}

:::python

LangGraph에서 노드는 다음 인수를 받는 Python 함수(동기 또는 비동기)입니다:

1. `state`: 그래프의 [state](#state)
2. `config`: `thread_id`와 같은 구성 정보 및 `tags`와 같은 추적 정보를 포함하는 `RunnableConfig` 객체
3. `runtime`: [runtime `context`](#runtime-context) 및 `store` 및 `stream_writer`와 같은 기타 정보를 포함하는 `Runtime` 객체

`NetworkX`와 유사하게 @[add_node][add_node] 메서드를 사용하여 이러한 노드를 그래프에 추가합니다:

```python
from dataclasses import dataclass
from typing_extensions import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

class State(TypedDict):
    input: str
    results: str

@dataclass
class Context:
    user_id: str

builder = StateGraph(State)

def plain_node(state: State):
    return state

def node_with_runtime(state: State, runtime: Runtime[Context]):
    print("In node: ", runtime.context.user_id)
    return {"results": f"Hello, {state['input']}!"}

def node_with_config(state: State, config: RunnableConfig):
    print("In node with thread_id: ", config["configurable"]["thread_id"])
    return {"results": f"Hello, {state['input']}!"}


builder.add_node("plain_node", plain_node)
builder.add_node("node_with_runtime", node_with_runtime)
builder.add_node("node_with_config", node_with_config)
...
```

:::

:::js

LangGraph에서 노드는 일반적으로 다음 인수를 받는 함수(동기 또는 비동기)입니다:

1. `state`: 그래프의 [state](#state)
2. `config`: `thread_id`와 같은 구성 정보 및 `tags`와 같은 추적 정보를 포함하는 `RunnableConfig` 객체

`addNode` 메서드를 사용하여 그래프에 노드를 추가할 수 있습니다.

```typescript
import { StateGraph } from "@langchain/langgraph";
import { RunnableConfig } from "@langchain/core/runnables";
import { z } from "zod";

const State = z.object({
  input: z.string(),
  results: z.string(),
});

const builder = new StateGraph(State);
  .addNode("myNode", (state, config) => {
    console.log("In node: ", config?.configurable?.user_id);
    return { results: `Hello, ${state.input}!` };
  })
  addNode("otherNode", (state) => {
    return state;
  })
  ...
```

:::

내부적으로 함수는 [RunnableLambda](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableLambda.html)로 변환되어 함수에 배치 및 비동기 지원과 함께 네이티브 추적 및 디버깅을 추가합니다.

이름을 지정하지 않고 그래프에 노드를 추가하면 함수 이름과 동일한 기본 이름이 지정됩니다.

:::python

```python
builder.add_node(my_node)
# You can then create edges to/from this node by referencing it as `"my_node"`
```

:::

:::js

```typescript
builder.addNode(myNode);
// You can then create edges to/from this node by referencing it as `"myNode"`
```

:::

### `START` 노드 {#start-node}

`START` 노드는 사용자 입력을 그래프에 보내는 노드를 나타내는 특수 노드입니다. 이 노드를 참조하는 주요 목적은 어떤 노드가 먼저 호출되어야 하는지 결정하는 것입니다.

:::python

```python
from langgraph.graph import START

graph.add_edge(START, "node_a")
```

:::

:::js

```typescript
import { START } from "@langchain/langgraph";

graph.addEdge(START, "nodeA");
```

:::

### `END` 노드 {#end-node}

`END` 노드는 터미널 노드를 나타내는 특수 노드입니다. 이 노드는 완료 후 어떤 엣지에도 액션이 없음을 나타내고 싶을 때 참조됩니다.

:::python

```python
from langgraph.graph import END

graph.add_edge("node_a", END)
```

:::

:::js

```typescript
import { END } from "@langchain/langgraph";

graph.addEdge("nodeA", END);
```

:::

### 노드 캐싱

:::python
LangGraph는 노드에 대한 입력을 기반으로 태스크/노드 캐싱을 지원합니다. 캐싱을 사용하려면:

- 그래프를 컴파일할 때(또는 진입점을 지정할 때) 캐시를 지정합니다
- 노드에 대한 캐시 정책을 지정합니다. 각 캐시 정책은 다음을 지원합니다:
  - `key_func`: 노드에 대한 입력을 기반으로 캐시 키를 생성하는 데 사용되며, 기본적으로 pickle을 사용한 입력의 `hash`입니다.
  - `ttl`: 초 단위의 캐시 유효 시간입니다. 지정하지 않으면 캐시가 만료되지 않습니다.

예를 들어:

```python
import time
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy


class State(TypedDict):
    x: int
    result: int


builder = StateGraph(State)


def expensive_node(state: State) -> dict[str, int]:
    # expensive computation
    time.sleep(2)
    return {"result": state["x"] * 2}


builder.add_node("expensive_node", expensive_node, cache_policy=CachePolicy(ttl=3))
builder.set_entry_point("expensive_node")
builder.set_finish_point("expensive_node")

graph = builder.compile(cache=InMemoryCache())

print(graph.invoke({"x": 5}, stream_mode='updates'))  # (1)!
[{'expensive_node': {'result': 10}}]
print(graph.invoke({"x": 5}, stream_mode='updates'))  # (2)!
[{'expensive_node': {'result': 10}, '__metadata__': {'cached': True}}]
```

1. 첫 번째 실행은 실행하는 데 2초가 걸립니다(모의 비용이 많이 드는 계산으로 인해).
2. 두 번째 실행은 캐시를 활용하여 빠르게 반환됩니다.
   :::

:::js
LangGraph는 노드에 대한 입력을 기반으로 태스크/노드 캐싱을 지원합니다. 캐싱을 사용하려면:

- 그래프를 컴파일할 때(또는 진입점을 지정할 때) 캐시를 지정합니다
- 노드에 대한 캐시 정책을 지정합니다. 각 캐시 정책은 다음을 지원합니다:
  - `keyFunc`: 노드에 대한 입력을 기반으로 캐시 키를 생성하는 데 사용됩니다.
  - `ttl`: 초 단위의 캐시 유효 시간입니다. 지정하지 않으면 캐시가 만료되지 않습니다.

```typescript
import { StateGraph, MessagesZodState } from "@langchain/langgraph";
import { InMemoryCache } from "@langchain/langgraph-checkpoint";

const graph = new StateGraph(MessagesZodState)
  .addNode(
    "expensive_node",
    async () => {
      // Simulate an expensive operation
      await new Promise((resolve) => setTimeout(resolve, 3000));
      return { result: 10 };
    },
    { cachePolicy: { ttl: 3 } }
  )
  .addEdge(START, "expensive_node")
  .compile({ cache: new InMemoryCache() });

await graph.invoke({ x: 5 }, { streamMode: "updates" }); // (1)!
// [{"expensive_node": {"result": 10}}]
await graph.invoke({ x: 5 }, { streamMode: "updates" }); // (2)!
// [{"expensive_node": {"result": 10}, "__metadata__": {"cached": true}}]
```

:::

## 엣지 {#edges}

엣지는 로직이 라우팅되는 방식과 그래프가 중지하는 방식을 정의합니다. 이것은 에이전트가 작동하는 방식과 서로 다른 노드가 서로 통신하는 방식의 큰 부분입니다. 몇 가지 주요 엣지 유형이 있습니다:

- 일반 엣지(Normal Edges): 한 노드에서 다음 노드로 직접 이동합니다.
- 조건부 엣지(Conditional Edges): 함수를 호출하여 다음에 이동할 노드를 결정합니다.
- 진입점(Entry Point): 사용자 입력이 도착할 때 먼저 호출할 노드입니다.
- 조건부 진입점(Conditional Entry Point): 함수를 호출하여 사용자 입력이 도착할 때 먼저 호출할 노드를 결정합니다.

노드는 여러 개의 나가는 엣지를 가질 수 있습니다. 노드에 여러 개의 나가는 엣지가 있는 경우, 해당 대상 노드 **모두**가 다음 슈퍼스텝의 일부로 병렬로 실행됩니다.

### 일반 엣지 {#normal-edges}

:::python
노드 A에서 노드 B로 **항상** 이동하려면 @[add_edge][add_edge] 메서드를 직접 사용할 수 있습니다.

```python
graph.add_edge("node_a", "node_b")
```

:::

:::js
노드 A에서 노드 B로 **항상** 이동하려면 @[`addEdge`][add_edge] 메서드를 직접 사용할 수 있습니다.

```typescript
graph.addEdge("nodeA", "nodeB");
```

:::

### 조건부 엣지 {#conditional-edges}

:::python
1개 이상의 엣지로 **선택적으로** 라우팅하거나 선택적으로 종료하려면 @[add_conditional_edges][add_conditional_edges] 메서드를 사용할 수 있습니다. 이 메서드는 노드 이름과 해당 노드가 실행된 후 호출할 "라우팅 함수"를 받습니다:

```python
graph.add_conditional_edges("node_a", routing_function)
```

노드와 유사하게 `routing_function`은 그래프의 현재 `state`를 받고 값을 반환합니다.

기본적으로 `routing_function`의 반환 값은 상태를 다음에 보낼 노드(또는 노드 리스트)의 이름으로 사용됩니다. 해당 노드 모두는 다음 슈퍼스텝의 일부로 병렬로 실행됩니다.

선택적으로 `routing_function`의 출력을 다음 노드의 이름에 매핑하는 딕셔너리를 제공할 수 있습니다.

```python
graph.add_conditional_edges("node_a", routing_function, {True: "node_b", False: "node_c"})
```

:::

:::js
1개 이상의 엣지로 **선택적으로** 라우팅하거나 선택적으로 종료하려면 @[`addConditionalEdges`][add_conditional_edges] 메서드를 사용할 수 있습니다. 이 메서드는 노드 이름과 해당 노드가 실행된 후 호출할 "라우팅 함수"를 받습니다:

```typescript
graph.addConditionalEdges("nodeA", routingFunction);
```

노드와 유사하게 `routingFunction`은 그래프의 현재 `state`를 받고 값을 반환합니다.

기본적으로 `routingFunction`의 반환 값은 상태를 다음에 보낼 노드(또는 노드 리스트)의 이름으로 사용됩니다. 해당 노드 모두는 다음 슈퍼스텝의 일부로 병렬로 실행됩니다.

선택적으로 `routingFunction`의 출력을 다음 노드의 이름에 매핑하는 객체를 제공할 수 있습니다.

```typescript
graph.addConditionalEdges("nodeA", routingFunction, {
  true: "nodeB",
  false: "nodeC",
});
```

:::

!!! tip

    상태 업데이트와 라우팅을 단일 함수에서 결합하려면 조건부 엣지 대신 [`Command`](#command)를 사용하세요.

### 진입점

:::python
진입점은 그래프가 시작될 때 실행되는 첫 번째 노드입니다. 가상 @[`START`][START] 노드에서 실행할 첫 번째 노드로 @[`add_edge`][add_edge] 메서드를 사용하여 그래프에 진입할 위치를 지정할 수 있습니다.

```python
from langgraph.graph import START

graph.add_edge(START, "node_a")
```

:::

:::js
진입점은 그래프가 시작될 때 실행되는 첫 번째 노드입니다. 가상 @[`START`][START] 노드에서 실행할 첫 번째 노드로 @[`addEdge`][add_edge] 메서드를 사용하여 그래프에 진입할 위치를 지정할 수 있습니다.

```typescript
import { START } from "@langchain/langgraph";

graph.addEdge(START, "nodeA");
```

:::

### 조건부 진입점

:::python
조건부 진입점을 사용하면 사용자 정의 로직에 따라 다른 노드에서 시작할 수 있습니다. 가상 @[`START`][START] 노드에서 @[`add_conditional_edges`][add_conditional_edges]를 사용하여 이를 수행할 수 있습니다.

```python
from langgraph.graph import START

graph.add_conditional_edges(START, routing_function)
```

선택적으로 `routing_function`의 출력을 다음 노드의 이름에 매핑하는 딕셔너리를 제공할 수 있습니다.

```python
graph.add_conditional_edges(START, routing_function, {True: "node_b", False: "node_c"})
```

:::

:::js
조건부 진입점을 사용하면 사용자 정의 로직에 따라 다른 노드에서 시작할 수 있습니다. 가상 @[`START`][START] 노드에서 @[`addConditionalEdges`][add_conditional_edges]를 사용하여 이를 수행할 수 있습니다.

```typescript
import { START } from "@langchain/langgraph";

graph.addConditionalEdges(START, routingFunction);
```

선택적으로 `routingFunction`의 출력을 다음 노드의 이름에 매핑하는 객체를 제공할 수 있습니다.

```typescript
graph.addConditionalEdges(START, routingFunction, {
  true: "nodeB",
  false: "nodeC",
});
```

:::

## `Send`

:::python
기본적으로 `Nodes`와 `Edges`는 미리 정의되며 동일한 공유 상태에서 작동합니다. 그러나 정확한 엣지를 미리 알 수 없거나 동시에 서로 다른 버전의 `State`가 존재하기를 원하는 경우가 있을 수 있습니다. 이에 대한 일반적인 예는 [map-reduce](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/) 디자인 패턴입니다. 이 디자인 패턴에서 첫 번째 노드가 객체 리스트를 생성할 수 있으며, 해당 모든 객체에 다른 노드를 적용하고 싶을 수 있습니다. 객체의 수를 미리 알 수 없을 수 있으며(즉, 엣지의 수를 알 수 없음) 다운스트림 `Node`에 대한 입력 `State`는 달라야 합니다(생성된 각 객체에 대해 하나씩).

이 디자인 패턴을 지원하기 위해 LangGraph는 조건부 엣지에서 @[`Send`][Send] 객체를 반환하는 것을 지원합니다. `Send`는 두 개의 인수를 받습니다: 첫 번째는 노드 이름이고 두 번째는 해당 노드에 전달할 상태입니다.

```python
def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state['subjects']]

graph.add_conditional_edges("node_a", continue_to_jokes)
```

:::

:::js
기본적으로 `Nodes`와 `Edges`는 미리 정의되며 동일한 공유 상태에서 작동합니다. 그러나 정확한 엣지를 미리 알 수 없거나 동시에 서로 다른 버전의 `State`가 존재하기를 원하는 경우가 있을 수 있습니다. 이에 대한 일반적인 예는 map-reduce 디자인 패턴입니다. 이 디자인 패턴에서 첫 번째 노드가 객체 리스트를 생성할 수 있으며, 해당 모든 객체에 다른 노드를 적용하고 싶을 수 있습니다. 객체의 수를 미리 알 수 없을 수 있으며(즉, 엣지의 수를 알 수 없음) 다운스트림 `Node`에 대한 입력 `State`는 달라야 합니다(생성된 각 객체에 대해 하나씩).

이 디자인 패턴을 지원하기 위해 LangGraph는 조건부 엣지에서 @[`Send`][Send] 객체를 반환하는 것을 지원합니다. `Send`는 두 개의 인수를 받습니다: 첫 번째는 노드 이름이고 두 번째는 해당 노드에 전달할 상태입니다.

```typescript
import { Send } from "@langchain/langgraph";

graph.addConditionalEdges("nodeA", (state) => {
  return state.subjects.map((subject) => new Send("generateJoke", { subject }));
});
```

:::

## `Command`

:::python
제어 흐름(엣지)과 상태 업데이트(노드)를 결합하는 것이 유용할 수 있습니다. 예를 들어, 동일한 노드에서 상태 업데이트를 수행하고 다음에 이동할 노드를 결정하는 것 모두를 원할 수 있습니다. LangGraph는 노드 함수에서 @[`Command`][Command] 객체를 반환하여 이를 수행하는 방법을 제공합니다:

```python
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    return Command(
        # state update
        update={"foo": "bar"},
        # control flow
        goto="my_other_node"
    )
```

`Command`를 사용하면 동적 제어 흐름 동작([조건부 엣지](#conditional-edges)와 동일)을 달성할 수도 있습니다:

```python
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    if state["foo"] == "bar":
        return Command(update={"foo": "baz"}, goto="my_other_node")
```

:::

:::js
제어 흐름(엣지)과 상태 업데이트(노드)를 결합하는 것이 유용할 수 있습니다. 예를 들어, 동일한 노드에서 상태 업데이트를 수행하고 다음에 이동할 노드를 결정하는 것 모두를 원할 수 있습니다. LangGraph는 노드 함수에서 `Command` 객체를 반환하여 이를 수행하는 방법을 제공합니다:

```typescript
import { Command } from "@langchain/langgraph";

graph.addNode("myNode", (state) => {
  return new Command({
    update: { foo: "bar" },
    goto: "myOtherNode",
  });
});
```

`Command`를 사용하면 동적 제어 흐름 동작([조건부 엣지](#conditional-edges)와 동일)을 달성할 수도 있습니다:

```typescript
import { Command } from "@langchain/langgraph";

graph.addNode("myNode", (state) => {
  if (state.foo === "bar") {
    return new Command({
      update: { foo: "baz" },
      goto: "myOtherNode",
    });
  }
});
```

노드 함수에서 `Command`를 사용할 때는 라우팅할 수 있는 노드를 지정하기 위해 노드를 추가할 때 `ends` 매개변수를 추가해야 합니다:

```typescript
builder.addNode("myNode", myNode, {
  ends: ["myOtherNode", END],
});
```

:::

!!! important

    노드 함수에서 `Command`를 반환할 때는 노드가 라우팅하는 노드 이름 리스트와 함께 반환 타입 어노테이션을 추가해야 합니다(예: `Command[Literal["my_other_node"]]`). 이는 그래프 렌더링에 필요하며 LangGraph에 `my_node`가 `my_other_node`로 이동할 수 있음을 알려줍니다.

`Command` 사용 방법에 대한 엔드투엔드 예제는 이 [how-to 가이드](../how-tos/graph-api.md#combine-control-flow-and-state-updates-with-command)를 확인하세요.

### 조건부 엣지 대신 Command를 언제 사용해야 하나요?

- 그래프 상태를 업데이트**하고** 다른 노드로 라우팅**해야 하는** 경우 `Command`를 사용합니다. 예를 들어, 다른 에이전트로 라우팅하고 해당 에이전트에게 일부 정보를 전달하는 것이 중요한 [multi-agent handoffs](./multi_agent.md#handoffs)를 구현할 때 사용합니다.
- 상태를 업데이트하지 않고 조건부로 노드 간에 라우팅하려면 [조건부 엣지](#conditional-edges)를 사용합니다.

### 부모 그래프의 노드로 이동하기

:::python
[서브그래프](./subgraphs.md)를 사용하는 경우 서브그래프 내의 노드에서 다른 서브그래프로 이동하고 싶을 수 있습니다(즉, 부모 그래프의 다른 노드). 이를 위해 `Command`에서 `graph=Command.PARENT`를 지정할 수 있습니다:

```python
def my_node(state: State) -> Command[Literal["other_subgraph"]]:
    return Command(
        update={"foo": "bar"},
        goto="other_subgraph",  # where `other_subgraph` is a node in the parent graph
        graph=Command.PARENT
    )
```

!!! note

    `graph`를 `Command.PARENT`로 설정하면 가장 가까운 부모 그래프로 이동합니다.

!!! important "`Command.PARENT`를 사용한 상태 업데이트"

    부모 및 서브그래프 [상태 스키마](#schema) 모두에서 공유되는 키에 대해 서브그래프 노드에서 부모 그래프 노드로 업데이트를 보낼 때는 부모 그래프 상태에서 업데이트하는 키에 대한 [reducer](#reducers)를 **반드시** 정의해야 합니다. 이 [예제](../how-tos/graph-api.md#navigate-to-a-node-in-a-parent-graph)를 참조하세요.

:::

:::js
[서브그래프](./subgraphs.md)를 사용하는 경우 서브그래프 내의 노드에서 다른 서브그래프로 이동하고 싶을 수 있습니다(즉, 부모 그래프의 다른 노드). 이를 위해 `Command`에서 `graph: Command.PARENT`를 지정할 수 있습니다:

```typescript
import { Command } from "@langchain/langgraph";

graph.addNode("myNode", (state) => {
  return new Command({
    update: { foo: "bar" },
    goto: "otherSubgraph", // where `otherSubgraph` is a node in the parent graph
    graph: Command.PARENT,
  });
});
```

!!! note

    `graph`를 `Command.PARENT`로 설정하면 가장 가까운 부모 그래프로 이동합니다.

!!! important "`Command.PARENT`를 사용한 상태 업데이트"

    부모 및 서브그래프 [상태 스키마](#schema) 모두에서 공유되는 키에 대해 서브그래프 노드에서 부모 그래프 노드로 업데이트를 보낼 때는 부모 그래프 상태에서 업데이트하는 키에 대한 [reducer](#reducers)를 **반드시** 정의해야 합니다.

:::

:::js
[서브그래프](./subgraphs.md)를 사용하는 경우 서브그래프 내의 노드에서 다른 서브그래프로 이동하고 싶을 수 있습니다(즉, 부모 그래프의 다른 노드). 이를 위해 `Command`에서 `graph: Command.PARENT`를 지정할 수 있습니다:

```typescript
import { Command } from "@langchain/langgraph";

graph.addNode("myNode", (state) => {
  return new Command({
    update: { foo: "bar" },
    goto: "otherSubgraph", // where `otherSubgraph` is a node in the parent graph
    graph: Command.PARENT,
  });
});
```

!!! note

    `graph`를 `Command.PARENT`로 설정하면 가장 가까운 부모 그래프로 이동합니다.

!!! important "`Command.PARENT`를 사용한 상태 업데이트"

    부모 및 서브그래프 [상태 스키마](#schema) 모두에서 공유되는 키에 대해 서브그래프 노드에서 부모 그래프 노드로 업데이트를 보낼 때는 부모 그래프 상태에서 업데이트하는 키에 대한 [reducer](#reducers)를 **반드시** 정의해야 합니다.

:::

이는 [multi-agent handoffs](./multi_agent.md#handoffs)를 구현할 때 특히 유용합니다.

자세한 내용은 [이 가이드](../how-tos/graph-api.md#navigate-to-a-node-in-a-parent-graph)를 확인하세요.

### 도구 내부에서 사용하기 {#use-inside-tools}

일반적인 사용 사례는 도구 내부에서 그래프 상태를 업데이트하는 것입니다. 예를 들어, 고객 지원 애플리케이션에서 대화 시작 시 계정 번호 또는 ID를 기반으로 고객 정보를 조회하고 싶을 수 있습니다.

자세한 내용은 [이 가이드](../how-tos/graph-api.md#use-inside-tools)를 참조하세요.

### Human-in-the-loop

:::python
`Command`는 human-in-the-loop 워크플로우의 중요한 부분입니다: `interrupt()`를 사용하여 사용자 입력을 수집할 때 `Command`는 입력을 제공하고 `Command(resume="User input")`를 통해 실행을 재개하는 데 사용됩니다. 자세한 내용은 [이 개념 가이드](./human_in_the_loop.md)를 확인하세요.
:::

:::js
`Command`는 human-in-the-loop 워크플로우의 중요한 부분입니다: `interrupt()`를 사용하여 사용자 입력을 수집할 때 `Command`는 입력을 제공하고 `new Command({ resume: "User input" })`를 통해 실행을 재개하는 데 사용됩니다. 자세한 내용은 [human-in-the-loop 개념 가이드](./human_in_the_loop.md)를 확인하세요.
:::

## 그래프 마이그레이션

LangGraph는 체크포인터를 사용하여 상태를 추적할 때도 그래프 정의(노드, 엣지 및 상태)의 마이그레이션을 쉽게 처리할 수 있습니다.

- 그래프의 끝에 있는 스레드(즉, 중단되지 않음)의 경우 그래프의 전체 토폴로지를 변경할 수 있습니다(즉, 모든 노드 및 엣지를 제거, 추가, 이름 변경 등)
- 현재 중단된 스레드의 경우 노드 이름 변경/제거를 제외한 모든 토폴로지 변경을 지원합니다(해당 스레드가 더 이상 존재하지 않는 노드에 진입하려고 할 수 있으므로) -- 이것이 문제가 되면 연락해 주시면 솔루션의 우선순위를 정할 수 있습니다.
- 상태 수정의 경우 키 추가 및 제거에 대한 완전한 하위 호환성 및 상위 호환성이 있습니다
- 이름이 변경된 상태 키는 기존 스레드에서 저장된 상태를 잃습니다
- 호환되지 않는 방식으로 타입이 변경된 상태 키는 현재 변경 이전의 상태를 가진 스레드에서 문제를 일으킬 수 있습니다 -- 이것이 문제가 되면 연락해 주시면 솔루션의 우선순위를 정할 수 있습니다.

:::python

## 런타임 컨텍스트 {#runtime-context}

그래프를 생성할 때 노드에 전달되는 런타임 컨텍스트에 대한 `context_schema`를 지정할 수 있습니다. 이는 그래프 상태의 일부가 아닌 정보를 노드에 전달하는 데 유용합니다. 예를 들어, 모델 이름이나 데이터베이스 연결과 같은 종속성을 전달하고 싶을 수 있습니다.

```python
@dataclass
class ContextSchema:
    llm_provider: str = "openai"

graph = StateGraph(State, context_schema=ContextSchema)
```

:::

:::js

그래프를 생성할 때 그래프의 특정 부분을 구성 가능하도록 표시할 수도 있습니다. 이는 일반적으로 모델이나 시스템 프롬프트 간에 쉽게 전환할 수 있도록 수행됩니다. 이를 통해 단일 "인지 아키텍처"(그래프)를 생성하면서도 여러 다른 인스턴스를 가질 수 있습니다.

그래프를 생성할 때 선택적으로 구성 스키마를 지정할 수 있습니다.

```typescript
import { z } from "zod";

const ConfigSchema = z.object({
  llm: z.string(),
});

const graph = new StateGraph(State, ConfigSchema);
```

:::

:::python
그런 다음 `invoke` 메서드의 `context` 매개변수를 사용하여 이 컨텍스트를 그래프에 전달할 수 있습니다.

```python
graph.invoke(inputs, context={"llm_provider": "anthropic"})
```

:::

:::js
그런 다음 `configurable` 구성 필드를 사용하여 이 구성을 그래프에 전달할 수 있습니다.

```typescript
const config = { configurable: { llm: "anthropic" } };

await graph.invoke(inputs, config);
```

:::

그런 다음 노드 또는 조건부 엣지 내부에서 이 컨텍스트에 액세스하여 사용할 수 있습니다:

```python
from langgraph.runtime import Runtime

def node_a(state: State, runtime: Runtime[ContextSchema]):
    llm = get_llm(runtime.context.llm_provider)
    ...
```

구성에 대한 전체 분석은 [이 가이드](../how-tos/graph-api.md#add-runtime-configuration)를 참조하세요.
:::

:::js

```typescript
graph.addNode("myNode", (state, config) => {
  const llmType = config?.configurable?.llm || "openai";
  const llm = getLlm(llmType);
  return { results: `Hello, ${state.input}!` };
});
```

:::

### 재귀 제한 {#recursion-limit}

:::python
재귀 제한은 단일 실행 중에 그래프가 실행할 수 있는 최대 [슈퍼스텝](#graphs) 수를 설정합니다. 제한에 도달하면 LangGraph는 `GraphRecursionError`를 발생시킵니다. 기본적으로 이 값은 25단계로 설정됩니다. 재귀 제한은 런타임에 모든 그래프에 설정할 수 있으며 구성 딕셔너리를 통해 `.invoke`/`.stream`에 전달됩니다. 중요한 점은 `recursion_limit`은 독립 실행형 `config` 키이며 다른 모든 사용자 정의 구성처럼 `configurable` 키 내부에 전달되어서는 안 된다는 것입니다. 아래 예제를 참조하세요:

```python
graph.invoke(inputs, config={"recursion_limit": 5}, context={"llm": "anthropic"})
```

재귀 제한이 작동하는 방식에 대해 자세히 알아보려면 [이 how-to](https://langchain-ai.github.io/langgraph/how-tos/recursion-limit/)를 읽어보세요.
:::

:::js
재귀 제한은 단일 실행 중에 그래프가 실행할 수 있는 최대 [슈퍼스텝](#graphs) 수를 설정합니다. 제한에 도달하면 LangGraph는 `GraphRecursionError`를 발생시킵니다. 기본적으로 이 값은 25단계로 설정됩니다. 재귀 제한은 런타임에 모든 그래프에 설정할 수 있으며 구성 객체를 통해 `.invoke`/`.stream`에 전달됩니다. 중요한 점은 `recursionLimit`은 독립 실행형 `config` 키이며 다른 모든 사용자 정의 구성처럼 `configurable` 키 내부에 전달되어서는 안 된다는 것입니다. 아래 예제를 참조하세요:

```typescript
await graph.invoke(inputs, {
  recursionLimit: 5,
  configurable: { llm: "anthropic" },
});
```

:::

## 시각화

특히 그래프가 더 복잡해질수록 그래프를 시각화할 수 있는 것이 유용합니다. LangGraph는 그래프를 시각화하는 여러 가지 내장 방법과 함께 제공됩니다. 자세한 내용은 [이 how-to 가이드](../how-tos/graph-api.md#visualize-your-graph)를 참조하세요.
