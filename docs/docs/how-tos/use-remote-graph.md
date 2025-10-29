# RemoteGraph를 사용하여 배포와 상호작용하는 방법

!!! info "Prerequisites"

    - [LangGraph Platform](../concepts/langgraph_platform.md)
    - [LangGraph Server](../concepts/langgraph_server.md)

`RemoteGraph`는 LangGraph Platform 배포와 마치 일반적인 로컬 정의 LangGraph 그래프(예: `CompiledGraph`)인 것처럼 상호작용할 수 있게 해주는 인터페이스입니다. 이 가이드는 `RemoteGraph`를 초기화하고 상호작용하는 방법을 보여줍니다.

## 그래프 초기화

:::python

`RemoteGraph`를 초기화할 때는 항상 다음을 지정해야 합니다:

- `name`: 상호작용할 그래프의 이름입니다. 배포의 `langgraph.json` 구성 파일에서 사용하는 것과 동일한 그래프 이름입니다.
- `api_key`: 유효한 LangSmith API 키입니다. 환경 변수(`LANGSMITH_API_KEY`)로 설정하거나 `api_key` 인수를 통해 직접 전달할 수 있습니다. `LangGraphClient` / `SyncLangGraphClient`가 `api_key` 인수로 초기화된 경우 `client` / `sync_client` 인수를 통해서도 API 키를 제공할 수 있습니다.

또한 다음 중 하나를 제공해야 합니다:

- `url`: 상호작용할 배포의 URL입니다. `url` 인수를 전달하면 제공된 URL, 헤더(제공된 경우) 및 기본 구성 값(예: 타임아웃 등)을 사용하여 동기 및 비동기 클라이언트가 모두 생성됩니다.
- `client`: 배포와 비동기적으로 상호작용하기 위한 `LangGraphClient` 인스턴스입니다(예: `.astream()`, `.ainvoke()`, `.aget_state()`, `.aupdate_state()` 등 사용).
- `sync_client`: 배포와 동기적으로 상호작용하기 위한 `SyncLangGraphClient` 인스턴스입니다(예: `.stream()`, `.invoke()`, `.get_state()`, `.update_state()` 등 사용).

!!! Note

    `client` 또는 `sync_client`와 `url` 인수를 모두 전달하면 `url` 인수보다 우선됩니다. `client` / `sync_client` / `url` 인수 중 어느 것도 제공되지 않으면 `RemoteGraph`는 런타임에 `ValueError`를 발생시킵니다.

:::

:::js

`RemoteGraph`를 초기화할 때는 항상 다음을 지정해야 합니다:

- `name`: 상호작용할 그래프의 이름입니다. 배포의 `langgraph.json` 구성 파일에서 사용하는 것과 동일한 그래프 이름입니다.
- `apiKey`: 유효한 LangSmith API 키입니다. 환경 변수(`LANGSMITH_API_KEY`)로 설정하거나 `apiKey` 인수를 통해 직접 전달할 수 있습니다. `LangGraphClient`가 `apiKey` 인수로 초기화된 경우 `client`를 통해서도 API 키를 제공할 수 있습니다.

또한 다음 중 하나를 제공해야 합니다:

- `url`: 상호작용할 배포의 URL입니다. `url` 인수를 전달하면 제공된 URL, 헤더(제공된 경우) 및 기본 구성 값(예: 타임아웃 등)을 사용하여 동기 및 비동기 클라이언트가 모두 생성됩니다.
- `client`: 배포와 비동기적으로 상호작용하기 위한 `LangGraphClient` 인스턴스입니다.

:::

### URL 사용

:::python

```python
from langgraph.pregel.remote import RemoteGraph

url = <DEPLOYMENT_URL>
graph_name = "agent"
remote_graph = RemoteGraph(graph_name, url=url)
```

:::

:::js

```ts
import { RemoteGraph } from "@langchain/langgraph/remote";

const url = `<DEPLOYMENT_URL>`;
const graphName = "agent";
const remoteGraph = new RemoteGraph({ graphId: graphName, url });
```

:::

### 클라이언트 사용

:::python

```python
from langgraph_sdk import get_client, get_sync_client
from langgraph.pregel.remote import RemoteGraph

url = <DEPLOYMENT_URL>
graph_name = "agent"
client = get_client(url=url)
sync_client = get_sync_client(url=url)
remote_graph = RemoteGraph(graph_name, client=client, sync_client=sync_client)
```

:::

:::js

```ts
import { Client } from "@langchain/langgraph-sdk";
import { RemoteGraph } from "@langchain/langgraph/remote";

const client = new Client({ apiUrl: `<DEPLOYMENT_URL>` });
const graphName = "agent";
const remoteGraph = new RemoteGraph({ graphId: graphName, client });
```

:::

## 그래프 호출

:::python
`RemoteGraph`는 `CompiledGraph`와 동일한 메서드를 구현하는 `Runnable`이므로, 컴파일된 그래프와 일반적으로 상호작용하는 것과 동일한 방식으로 상호작용할 수 있습니다. 즉, `.invoke()`, `.stream()`, `.get_state()`, `.update_state()` 등(및 비동기 버전)을 호출할 수 있습니다.

### 비동기적으로

!!! Note

    그래프를 비동기적으로 사용하려면 `RemoteGraph`를 초기화할 때 `url` 또는 `client`를 제공해야 합니다.

```python
# invoke the graph
result = await remote_graph.ainvoke({
    "messages": [{"role": "user", "content": "what's the weather in sf"}]
})

# stream outputs from the graph
async for chunk in remote_graph.astream({
    "messages": [{"role": "user", "content": "what's the weather in la"}]
}):
    print(chunk)
```

### 동기적으로

!!! Note

    그래프를 동기적으로 사용하려면 `RemoteGraph`를 초기화할 때 `url` 또는 `sync_client`를 제공해야 합니다.

```python
# invoke the graph
result = remote_graph.invoke({
    "messages": [{"role": "user", "content": "what's the weather in sf"}]
})

# stream outputs from the graph
for chunk in remote_graph.stream({
    "messages": [{"role": "user", "content": "what's the weather in la"}]
}):
    print(chunk)
```

:::

:::js
`RemoteGraph`는 `CompiledGraph`와 동일한 메서드를 구현하는 `Runnable`이므로, 컴파일된 그래프와 일반적으로 상호작용하는 것과 동일한 방식으로 상호작용할 수 있습니다. 즉, `.invoke()`, `.stream()`, `.getState()`, `.updateState()` 등을 호출할 수 있습니다.

```ts
// invoke the graph
const result = await remoteGraph.invoke({
    messages: [{role: "user", content: "what's the weather in sf"}]
})

// stream outputs from the graph
for await (const chunk of await remoteGraph.stream({
    messages: [{role: "user", content: "what's the weather in la"}]
})):
    console.log(chunk)
```

:::

## Thread 레벨 지속성

기본적으로 그래프 실행(즉, `.invoke()` 또는 `.stream()` 호출)은 상태 비저장입니다 - 체크포인트와 그래프의 최종 상태가 지속되지 않습니다. 그래프 실행의 출력을 지속하려면(예: human-in-the-loop 기능을 활성화하기 위해), thread를 생성하고 `config` 인수를 통해 thread ID를 제공할 수 있습니다. 일반 컴파일된 그래프와 동일합니다:

:::python

```python
from langgraph_sdk import get_sync_client
url = <DEPLOYMENT_URL>
graph_name = "agent"
sync_client = get_sync_client(url=url)
remote_graph = RemoteGraph(graph_name, url=url)

# create a thread (or use an existing thread instead)
thread = sync_client.threads.create()

# invoke the graph with the thread config
config = {"configurable": {"thread_id": thread["thread_id"]}}
result = remote_graph.invoke({
    "messages": [{"role": "user", "content": "what's the weather in sf"}]
}, config=config)

# verify that the state was persisted to the thread
thread_state = remote_graph.get_state(config)
print(thread_state)
```

:::

:::js

```ts
import { Client } from "@langchain/langgraph-sdk";
import { RemoteGraph } from "@langchain/langgraph/remote";

const url = `<DEPLOYMENT_URL>`;
const graphName = "agent";
const client = new Client({ apiUrl: url });
const remoteGraph = new RemoteGraph({ graphId: graphName, url });

// create a thread (or use an existing thread instead)
const thread = await client.threads.create();

// invoke the graph with the thread config
const config = { configurable: { thread_id: thread.thread_id } };
const result = await remoteGraph.invoke(
  {
    messages: [{ role: "user", content: "what's the weather in sf" }],
  },
  config
);

// verify that the state was persisted to the thread
const threadState = await remoteGraph.getState(config);
console.log(threadState);
```

:::

## 서브그래프로 사용

!!! Note

    `checkpointer`를 `RemoteGraph` 서브그래프 노드가 있는 그래프와 함께 사용해야 하는 경우, thread ID로 UUID를 사용하세요.

`RemoteGraph`는 일반 `CompiledGraph`와 동일하게 동작하므로 다른 그래프에서 서브그래프로도 사용할 수 있습니다. 예를 들어:

:::python

```python
from langgraph_sdk import get_sync_client
from langgraph.graph import StateGraph, MessagesState, START
from typing import TypedDict

url = <DEPLOYMENT_URL>
graph_name = "agent"
remote_graph = RemoteGraph(graph_name, url=url)

# define parent graph
builder = StateGraph(MessagesState)
# add remote graph directly as a node
builder.add_node("child", remote_graph)
builder.add_edge(START, "child")
graph = builder.compile()

# invoke the parent graph
result = graph.invoke({
    "messages": [{"role": "user", "content": "what's the weather in sf"}]
})
print(result)

# stream outputs from both the parent graph and subgraph
for chunk in graph.stream({
    "messages": [{"role": "user", "content": "what's the weather in sf"}]
}, subgraphs=True):
    print(chunk)
```

:::

:::js

```ts
import { MessagesAnnotation, StateGraph, START } from "@langchain/langgraph";
import { RemoteGraph } from "@langchain/langgraph/remote";

const url = `<DEPLOYMENT_URL>`;
const graphName = "agent";
const remoteGraph = new RemoteGraph({ graphId: graphName, url });

// define parent graph and add remote graph directly as a node
const graph = new StateGraph(MessagesAnnotation)
  .addNode("child", remoteGraph)
  .addEdge(START, "child")
  .compile();

// invoke the parent graph
const result = await graph.invoke({
  messages: [{ role: "user", content: "what's the weather in sf" }],
});
console.log(result);

// stream outputs from both the parent graph and subgraph
for await (const chunk of await graph.stream(
  {
    messages: [{ role: "user", content: "what's the weather in la" }],
  },
  { subgraphs: true }
)) {
  console.log(chunk);
}
```

:::
