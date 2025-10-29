# INVALID_CONCURRENT_GRAPH_UPDATE

LangGraph [`StateGraph`](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph)가 여러 노드로부터 동시 상태 업데이트를 받았으나, 해당 상태 속성이 이를 지원하지 않습니다.

이러한 현상이 발생할 수 있는 한 가지 방법은 그래프에서 [fanout](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/) 또는 기타 병렬 실행을 사용하고 있으며 다음과 같이 그래프를 정의한 경우입니다:

:::python

```python hl_lines="2"
class State(TypedDict):
    some_key: str

def node(state: State):
    return {"some_key": "some_string_value"}

def other_node(state: State):
    return {"some_key": "some_string_value"}


builder = StateGraph(State)
builder.add_node(node)
builder.add_node(other_node)
builder.add_edge(START, "node")
builder.add_edge(START, "other_node")
graph = builder.compile()
```

:::

:::js

```typescript hl_lines="2"
import { StateGraph, Annotation, START } from "@langchain/langgraph";
import { z } from "zod";

const State = z.object({
  someKey: z.string(),
});

const builder = new StateGraph(State)
  .addNode("node", (state) => {
    return { someKey: "some_string_value" };
  })
  .addNode("otherNode", (state) => {
    return { someKey: "some_string_value" };
  })
  .addEdge(START, "node")
  .addEdge(START, "otherNode");

const graph = builder.compile();
```

:::

:::python
위 그래프의 노드가 `{ "some_key": "some_string_value" }`를 반환하면 `"some_key"`의 상태 값을 `"some_string_value"`로 덮어씁니다.
그러나 단일 단계 내의 fanout에서 여러 노드가 `"some_key"`에 대한 값을 반환하는 경우, 내부 상태를 업데이트하는 방법에 대한 불확실성이 있기 때문에 그래프가 이 에러를 발생시킵니다.
:::

:::js
위 그래프의 노드가 `{ someKey: "some_string_value" }`를 반환하면 `someKey`의 상태 값을 `"some_string_value"`로 덮어씁니다.
그러나 단일 단계 내의 fanout에서 여러 노드가 `someKey`에 대한 값을 반환하는 경우, 내부 상태를 업데이트하는 방법에 대한 불확실성이 있기 때문에 그래프가 이 에러를 발생시킵니다.
:::

이를 해결하려면 여러 값을 결합하는 reducer를 정의할 수 있습니다:

:::python

```python hl_lines="5-6"
import operator
from typing import Annotated

class State(TypedDict):
    # operator.add reducer fn은 이것을 append-only로 만듭니다
    some_key: Annotated[list, operator.add]
```

:::

:::js

```typescript hl_lines="4-7"
import { withLangGraph } from "@langchain/langgraph";
import { z } from "zod";

const State = z.object({
  someKey: withLangGraph(z.array(z.string()), {
    reducer: {
      fn: (existing, update) => existing.concat(update),
    },
    default: () => [],
  }),
});
```

:::

이를 통해 병렬로 실행되는 여러 노드에서 반환된 동일한 키를 처리하는 로직을 정의할 수 있습니다.

## 트러블슈팅

다음 사항이 이 에러를 해결하는 데 도움이 될 수 있습니다:

- 그래프가 병렬로 노드를 실행하는 경우, 관련 상태 키를 reducer와 함께 정의했는지 확인하세요.
