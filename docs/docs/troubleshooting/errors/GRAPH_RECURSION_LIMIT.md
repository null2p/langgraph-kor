# GRAPH_RECURSION_LIMIT

LangGraph [`StateGraph`](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph)가 중지 조건에 도달하기 전에 최대 단계 수에 도달했습니다.
이는 종종 아래 예제와 같은 코드로 인한 무한 루프 때문입니다:

:::python

```python
class State(TypedDict):
    some_key: str

builder = StateGraph(State)
builder.add_node("a", ...)
builder.add_node("b", ...)
builder.add_edge("a", "b")
builder.add_edge("b", "a")
...

graph = builder.compile()
```

:::

:::js

```typescript
import { StateGraph } from "@langchain/langgraph";
import { z } from "zod";

const State = z.object({
  someKey: z.string(),
});

const builder = new StateGraph(State)
  .addNode("a", ...)
  .addNode("b", ...)
  .addEdge("a", "b")
  .addEdge("b", "a")
  ...

const graph = builder.compile();
```

:::

그러나 복잡한 그래프는 자연스럽게 기본 제한에 도달할 수 있습니다.

## 트러블슈팅

- 그래프가 많은 반복을 거칠 것으로 예상하지 않는 경우, 사이클이 있을 가능성이 높습니다. 무한 루프가 있는지 로직을 확인하세요.

:::python

- 복잡한 그래프가 있는 경우, 그래프를 호출할 때 `config` 객체에 더 높은 `recursion_limit` 값을 다음과 같이 전달할 수 있습니다:

```python
graph.invoke({...}, {"recursion_limit": 100})
```

:::

:::js

- 복잡한 그래프가 있는 경우, 그래프를 호출할 때 `config` 객체에 더 높은 `recursionLimit` 값을 다음과 같이 전달할 수 있습니다:

```typescript
await graph.invoke({...}, { recursionLimit: 100 });
```

:::
