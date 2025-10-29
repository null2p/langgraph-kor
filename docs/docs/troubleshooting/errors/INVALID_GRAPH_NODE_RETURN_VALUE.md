# INVALID_GRAPH_NODE_RETURN_VALUE

:::python
LangGraph [`StateGraph`](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph)가
노드로부터 dict가 아닌 반환 타입을 받았습니다. 다음은 예제입니다:

```python
class State(TypedDict):
    some_key: str

def bad_node(state: State):
    # "some_key"에 대한 값을 가진 dict를 반환해야 하며, list가 아님
    return ["whoops"]

builder = StateGraph(State)
builder.add_node(bad_node)
...

graph = builder.compile()
```

위의 그래프를 호출하면 다음과 같은 에러가 발생합니다:

```python
graph.invoke({ "some_key": "someval" });
```

```
InvalidUpdateError: Expected dict, got ['whoops']
For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/INVALID_GRAPH_NODE_RETURN_VALUE
```

그래프의 노드는 상태에 정의된 하나 이상의 키를 포함하는 dict를 반환해야 합니다.
:::

:::js
LangGraph [`StateGraph`](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph)가
노드로부터 object가 아닌 반환 타입을 받았습니다. 다음은 예제입니다:

```typescript
import { z } from "zod";
import { StateGraph } from "@langchain/langgraph";

const State = z.object({
  someKey: z.string(),
});

const badNode = (state: z.infer<typeof State>) => {
  // "someKey"에 대한 값을 가진 object를 반환해야 하며, array가 아님
  return ["whoops"];
};

const builder = new StateGraph(State).addNode("badNode", badNode);
// ...

const graph = builder.compile();
```

위의 그래프를 호출하면 다음과 같은 에러가 발생합니다:

```typescript
await graph.invoke({ someKey: "someval" });
```

```
InvalidUpdateError: Expected object, got ['whoops']
For troubleshooting, visit: https://langchain-ai.github.io/langgraphjs/troubleshooting/errors/INVALID_GRAPH_NODE_RETURN_VALUE
```

그래프의 노드는 상태에 정의된 하나 이상의 키를 포함하는 object를 반환해야 합니다.
:::

## 트러블슈팅

다음 사항이 이 에러를 해결하는 데 도움이 될 수 있습니다:

:::python

- 노드에 복잡한 로직이 있는 경우, 모든 코드 경로가 정의된 상태에 적합한 dict를 반환하는지 확인하세요.
  :::

:::js

- 노드에 복잡한 로직이 있는 경우, 모든 코드 경로가 정의된 상태에 적합한 object를 반환하는지 확인하세요.
  :::
