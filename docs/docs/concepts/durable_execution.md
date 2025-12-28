---
search:
  boost: 2
---

# 내구성 실행

**내구성 실행(Durable execution)**은 프로세스 또는 워크플로우가 주요 지점에서 진행 상황을 저장하여 중단했다가 나중에 정확히 중단한 지점에서 재개할 수 있도록 하는 기술입니다. 이는 사용자가 계속하기 전에 프로세스를 검사, 검증 또는 수정할 수 있는 [human-in-the-loop](./human_in_the_loop.md)가 필요한 시나리오와 중단 또는 오류가 발생할 수 있는 장기 실행 작업(예: LLM 호출 시간 초과)에서 특히 유용합니다. 완료된 작업을 보존함으로써 내구성 실행은 이전 단계를 재처리하지 않고 프로세스를 재개할 수 있게 합니다 -- 상당한 지연(예: 일주일 후) 후에도 가능합니다.

LangGraph의 내장 [지속성](./persistence.md) 레이어는 워크플로우에 대한 내구성 실행을 제공하여 각 실행 단계의 상태가 내구성 스토어에 저장되도록 보장합니다. 이 기능은 시스템 장애로 인해든 [human-in-the-loop](./human_in_the_loop.md) 상호 작용을 위해든 워크플로우가 중단된 경우 마지막으로 기록된 상태에서 재개될 수 있음을 보장합니다.

!!! tip

    체크포인터와 함께 LangGraph를 사용하는 경우 이미 내구성 실행이 활성화되어 있습니다. 중단 또는 장애 후에도 언제든지 워크플로우를 일시 중지하고 재개할 수 있습니다.
    내구성 실행을 최대한 활용하려면 워크플로우가 [결정론적](#determinism-and-consistent-replay)이고 [멱등성](#determinism-and-consistent-replay)을 갖도록 설계되었는지 확인하고 모든 부작용이나 비결정론적 작업을 [tasks](./functional_api.md#task) 내부에 래핑하세요. [StateGraph (Graph API)](./low_level.md)와 [Functional API](./functional_api.md) 모두에서 [tasks](./functional_api.md#task)를 사용할 수 있습니다.

## 요구사항

LangGraph에서 내구성 실행을 활용하려면 다음이 필요합니다:

1. 워크플로우 진행 상황을 저장할 [체크포인터](./persistence.md#checkpointer-libraries)를 지정하여 워크플로우에서 [지속성](./persistence.md)을 활성화합니다.
2. 워크플로우를 실행할 때 [스레드 식별자](./persistence.md#threads)를 지정합니다. 이는 워크플로우의 특정 인스턴스에 대한 실행 기록을 추적합니다.

:::python

3. 비결정론적 작업(예: 난수 생성) 또는 부작용이 있는 작업(예: 파일 쓰기, API 호출)을 @[tasks][task] 내부에 래핑하여 워크플로우가 재개될 때 이러한 작업이 특정 실행에 대해 반복되지 않고 대신 지속성 레이어에서 결과를 검색하도록 합니다. 자세한 내용은 [결정론과 일관된 재생](#determinism-and-consistent-replay)을 참조하세요.

:::

:::js

3. 비결정론적 작업(예: 난수 생성) 또는 부작용이 있는 작업(예: 파일 쓰기, API 호출)을 @[tasks][task] 내부에 래핑하여 워크플로우가 재개될 때 이러한 작업이 특정 실행에 대해 반복되지 않고 대신 지속성 레이어에서 결과를 검색하도록 합니다. 자세한 내용은 [결정론과 일관된 재생](#determinism-and-consistent-replay)을 참조하세요.

:::

## 결정론과 일관된 재생 {#determinism-and-consistent-replay}

워크플로우 실행을 재개할 때 코드는 실행이 중지된 **동일한 코드 줄**에서 재개되지 **않습니다**. 대신 중단한 지점을 다시 시작할 적절한 [시작 지점](#starting-points-for-resuming-workflows)을 식별합니다. 이는 워크플로우가 [시작 지점](#starting-points-for-resuming-workflows)부터 중지된 지점에 도달할 때까지 모든 단계를 재생한다는 것을 의미합니다.

결과적으로 내구성 실행을 위한 워크플로우를 작성할 때는 모든 비결정론적 작업(예: 난수 생성) 및 부작용이 있는 모든 작업(예: 파일 쓰기, API 호출)을 [tasks](./functional_api.md#task) 또는 [nodes](./low_level.md#nodes) 내부에 래핑해야 합니다.

워크플로우가 결정론적이고 일관되게 재생될 수 있도록 하려면 다음 지침을 따르세요:

- **작업 반복 방지**: [노드](./low_level.md#nodes)에 부작용이 있는 여러 작업(예: 로깅, 파일 쓰기 또는 네트워크 호출)이 포함된 경우 각 작업을 별도의 **task**로 래핑합니다. 이렇게 하면 워크플로우가 재개될 때 작업이 반복되지 않고 지속성 레이어에서 결과가 검색됩니다.
- **비결정론적 작업 캡슐화:** 비결정론적 결과를 생성할 수 있는 모든 코드(예: 난수 생성)를 **tasks** 또는 **nodes** 내부에 래핑합니다. 이렇게 하면 재개 시 워크플로우가 동일한 결과로 정확하게 기록된 단계 순서를 따릅니다.
- **멱등 작업 사용**: 가능한 경우 부작용(예: API 호출, 파일 쓰기)이 멱등적인지 확인합니다. 이는 워크플로우에서 장애 후 작업이 재시도되는 경우 처음 실행했을 때와 동일한 효과를 가져야 함을 의미합니다. 이는 데이터 쓰기를 초래하는 작업에 특히 중요합니다. **task**가 시작되었지만 성공적으로 완료되지 못한 경우 워크플로우 재개 시 **task**를 다시 실행하며 기록된 결과에 의존하여 일관성을 유지합니다. 의도하지 않은 중복을 방지하고 원활하고 예측 가능한 워크플로우 실행을 보장하려면 멱등성 키를 사용하거나 기존 결과를 확인하세요.

:::python
피해야 할 함정의 몇 가지 예는 functional API의 [Common Pitfalls](./functional_api.md#common-pitfalls) 섹션을 참조하세요. 이 섹션은 **tasks**를 사용하여 이러한 문제를 방지하도록 코드를 구조화하는 방법을 보여줍니다. 동일한 원칙이 @[StateGraph (Graph API)][StateGraph]에도 적용됩니다.
:::

:::js
피해야 할 함정의 몇 가지 예는 functional API의 [Common Pitfalls](./functional_api.md#common-pitfalls) 섹션을 참조하세요. 이 섹션은 **tasks**를 사용하여 이러한 문제를 방지하도록 코드를 구조화하는 방법을 보여줍니다. 동일한 원칙이 @[StateGraph (Graph API)][StateGraph]에도 적용됩니다.
:::

## 내구성 모드

LangGraph는 애플리케이션 요구 사항에 따라 성능과 데이터 일관성의 균형을 맞출 수 있는 세 가지 내구성 모드를 지원합니다. 내구성이 낮은 것부터 높은 것까지 내구성 모드는 다음과 같습니다:

- [`"exit"`](#exit)
- [`"async"`](#async)
- [`"sync"`](#sync)

내구성 모드가 높을수록 워크플로우 실행에 더 많은 오버헤드가 추가됩니다.

!!! version-added "버전 0.6.0에서 추가됨"

    지속성 정책 관리를 위해 `checkpoint_during` (v0.6.0에서 더 이상 사용되지 않음) 대신 `durability` 매개변수를 사용하세요:

    * `durability="async"`는 `checkpoint_during=True`를 대체합니다
    * `durability="exit"`는 `checkpoint_during=False`를 대체합니다

    다음 매핑을 사용한 지속성 정책 관리:

    * `checkpoint_during=True` -> `durability="async"`
    * `checkpoint_during=False` -> `durability="exit"`

### `"exit"`

그래프 실행이 완료될 때(성공적으로 또는 오류와 함께)만 변경 사항이 지속됩니다. 이는 장기 실행 그래프에 최상의 성능을 제공하지만 중간 상태가 저장되지 않으므로 실행 중 장애로부터 복구하거나 그래프 실행을 중단할 수 없습니다.

### `"async"`

다음 단계가 실행되는 동안 변경 사항이 비동기적으로 지속됩니다. 이는 우수한 성능과 내구성을 제공하지만 실행 중 프로세스가 충돌하는 경우 체크포인트가 작성되지 않을 수 있는 작은 위험이 있습니다.

### `"sync"`

다음 단계가 시작되기 전에 변경 사항이 동기적으로 지속됩니다. 이는 실행을 계속하기 전에 모든 체크포인트가 작성되도록 보장하여 일부 성능 오버헤드를 대가로 높은 내구성을 제공합니다.

그래프 실행 메서드를 호출할 때 내구성 모드를 지정할 수 있습니다:

:::python

```python
graph.stream(
    {"input": "test"}, 
    durability="sync"
)
```

:::

## 노드에서 태스크 사용하기

[노드](./low_level.md#nodes)에 여러 작업이 포함된 경우 작업을 개별 노드로 리팩토링하는 대신 각 작업을 **task**로 변환하는 것이 더 쉬울 수 있습니다.

:::python
=== "Original"

    ```python
    from typing import NotRequired
    from typing_extensions import TypedDict
    import uuid

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import StateGraph, START, END
    import requests

    # Define a TypedDict to represent the state
    class State(TypedDict):
        url: str
        result: NotRequired[str]

    def call_api(state: State):
        """Example node that makes an API request."""
        # highlight-next-line
        result = requests.get(state['url']).text[:100]  # Side-effect
        return {
            "result": result
        }

    # Create a StateGraph builder and add a node for the call_api function
    builder = StateGraph(State)
    builder.add_node("call_api", call_api)

    # Connect the start and end nodes to the call_api node
    builder.add_edge(START, "call_api")
    builder.add_edge("call_api", END)

    # Specify a checkpointer
    checkpointer = InMemorySaver()

    # Compile the graph with the checkpointer
    graph = builder.compile(checkpointer=checkpointer)

    # Define a config with a thread ID.
    thread_id = uuid.uuid4()
    config = {"configurable": {"thread_id": thread_id}}

    # Invoke the graph
    graph.invoke({"url": "https://www.example.com"}, config)
    ```

=== "With task"

    ```python
    from typing import NotRequired
    from typing_extensions import TypedDict
    import uuid

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.func import task
    from langgraph.graph import StateGraph, START, END
    import requests

    # Define a TypedDict to represent the state
    class State(TypedDict):
        urls: list[str]
        result: NotRequired[list[str]]


    @task
    def _make_request(url: str):
        """Make a request."""
        # highlight-next-line
        return requests.get(url).text[:100]

    def call_api(state: State):
        """Example node that makes an API request."""
        # highlight-next-line
        requests = [_make_request(url) for url in state['urls']]
        results = [request.result() for request in requests]
        return {
            "results": results
        }

    # Create a StateGraph builder and add a node for the call_api function
    builder = StateGraph(State)
    builder.add_node("call_api", call_api)

    # Connect the start and end nodes to the call_api node
    builder.add_edge(START, "call_api")
    builder.add_edge("call_api", END)

    # Specify a checkpointer
    checkpointer = InMemorySaver()

    # Compile the graph with the checkpointer
    graph = builder.compile(checkpointer=checkpointer)

    # Define a config with a thread ID.
    thread_id = uuid.uuid4()
    config = {"configurable": {"thread_id": thread_id}}

    # Invoke the graph
    graph.invoke({"urls": ["https://www.example.com"]}, config)
    ```

:::

:::js
=== "Original"

    ```typescript
    import { StateGraph, START, END } from "@langchain/langgraph";
    import { MemorySaver } from "@langchain/langgraph";
    import { v4 as uuidv4 } from "uuid";
    import { z } from "zod";

    // Define a Zod schema to represent the state
    const State = z.object({
      url: z.string(),
      result: z.string().optional(),
    });

    const callApi = async (state: z.infer<typeof State>) => {
      // highlight-next-line
      const response = await fetch(state.url);
      const text = await response.text();
      const result = text.slice(0, 100); // Side-effect
      return {
        result,
      };
    };

    // Create a StateGraph builder and add a node for the callApi function
    const builder = new StateGraph(State)
      .addNode("callApi", callApi)
      .addEdge(START, "callApi")
      .addEdge("callApi", END);

    // Specify a checkpointer
    const checkpointer = new MemorySaver();

    // Compile the graph with the checkpointer
    const graph = builder.compile({ checkpointer });

    // Define a config with a thread ID.
    const threadId = uuidv4();
    const config = { configurable: { thread_id: threadId } };

    // Invoke the graph
    await graph.invoke({ url: "https://www.example.com" }, config);
    ```

=== "With task"

    ```typescript
    import { StateGraph, START, END } from "@langchain/langgraph";
    import { MemorySaver } from "@langchain/langgraph";
    import { task } from "@langchain/langgraph";
    import { v4 as uuidv4 } from "uuid";
    import { z } from "zod";

    // Define a Zod schema to represent the state
    const State = z.object({
      urls: z.array(z.string()),
      results: z.array(z.string()).optional(),
    });

    const makeRequest = task("makeRequest", async (url: string) => {
      // highlight-next-line
      const response = await fetch(url);
      const text = await response.text();
      return text.slice(0, 100);
    });

    const callApi = async (state: z.infer<typeof State>) => {
      // highlight-next-line
      const requests = state.urls.map((url) => makeRequest(url));
      const results = await Promise.all(requests);
      return {
        results,
      };
    };

    // Create a StateGraph builder and add a node for the callApi function
    const builder = new StateGraph(State)
      .addNode("callApi", callApi)
      .addEdge(START, "callApi")
      .addEdge("callApi", END);

    // Specify a checkpointer
    const checkpointer = new MemorySaver();

    // Compile the graph with the checkpointer
    const graph = builder.compile({ checkpointer });

    // Define a config with a thread ID.
    const threadId = uuidv4();
    const config = { configurable: { thread_id: threadId } };

    // Invoke the graph
    await graph.invoke({ urls: ["https://www.example.com"] }, config);
    ```

:::

## 워크플로우 재개

워크플로우에서 내구성 실행을 활성화하면 다음 시나리오에 대해 실행을 재개할 수 있습니다:

:::python

- **워크플로우 일시 중지 및 재개:** @[interrupt][interrupt] 함수를 사용하여 특정 지점에서 워크플로우를 일시 중지하고 @[Command] 프리미티브를 사용하여 업데이트된 상태로 재개합니다. 자세한 내용은 [**Human-in-the-Loop**](./human_in_the_loop.md)를 참조하세요.
- **장애로부터 복구:** 예외(예: LLM 제공자 중단) 후 마지막으로 성공한 체크포인트에서 워크플로우를 자동으로 재개합니다. 이는 입력 값으로 `None`을 제공하여 동일한 스레드 식별자로 워크플로우를 실행하는 것을 포함합니다(functional API의 이 [예제](../how-tos/use-functional-api.md#resuming-after-an-error) 참조).

  :::

:::js

- **워크플로우 일시 중지 및 재개:** @[interrupt][interrupt] 함수를 사용하여 특정 지점에서 워크플로우를 일시 중지하고 @[Command] 프리미티브를 사용하여 업데이트된 상태로 재개합니다. 자세한 내용은 [**Human-in-the-Loop**](./human_in_the_loop.md)를 참조하세요.
- **장애로부터 복구:** 예외(예: LLM 제공자 중단) 후 마지막으로 성공한 체크포인트에서 워크플로우를 자동으로 재개합니다. 이는 입력 값으로 `null`을 제공하여 동일한 스레드 식별자로 워크플로우를 실행하는 것을 포함합니다(functional API의 이 [예제](../how-tos/use-functional-api.md#resuming-after-an-error) 참조).

  :::

## 워크플로우 재개를 위한 시작 지점 {#starting-points-for-resuming-workflows}

:::python

- @[StateGraph (Graph API)][StateGraph]를 사용하는 경우 시작 지점은 실행이 중지된 [**node**](./low_level.md#nodes)의 시작 부분입니다.
- 노드 내부에서 서브그래프 호출을 하는 경우 시작 지점은 중단된 서브그래프를 호출한 **부모** 노드가 됩니다.
  서브그래프 내부에서는 실행이 중지된 특정 [**node**](./low_level.md#nodes)가 시작 지점이 됩니다.
- Functional API를 사용하는 경우 시작 지점은 실행이 중지된 [**entrypoint**](./functional_api.md#entrypoint)의 시작 부분입니다.

  :::

:::js

- [StateGraph (Graph API)](./low_level.md)를 사용하는 경우 시작 지점은 실행이 중지된 [**node**](./low_level.md#nodes)의 시작 부분입니다.
- 노드 내부에서 서브그래프 호출을 하는 경우 시작 지점은 중단된 서브그래프를 호출한 **부모** 노드가 됩니다.
  서브그래프 내부에서는 실행이 중지된 특정 [**node**](./low_level.md#nodes)가 시작 지점이 됩니다.
- Functional API를 사용하는 경우 시작 지점은 실행이 중지된 [**entrypoint**](./functional_api.md#entrypoint)의 시작 부분입니다.

  :::
