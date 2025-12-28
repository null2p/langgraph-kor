# 서브그래프

서브그래프는 다른 그래프에서 [노드](./low_level.md#nodes)로 사용되는 [그래프](./low_level.md#graphs)입니다 — 이는 LangGraph에 적용된 캡슐화 개념입니다. 서브그래프를 사용하면 자체적으로 그래프인 여러 컴포넌트가 있는 복잡한 시스템을 구축할 수 있습니다.

![Subgraph](./img/subgraph.png)

서브그래프를 사용하는 이유는 다음과 같습니다:

- [다중 에이전트 시스템](./multi_agent.md) 구축
- 여러 그래프에서 노드 세트를 재사용하려는 경우
- 서로 다른 팀이 그래프의 서로 다른 부분을 독립적으로 작업하도록 하려는 경우, 각 부분을 서브그래프로 정의할 수 있으며, 서브그래프 인터페이스(입력 및 출력 스키마)가 존중되는 한 부모 그래프는 서브그래프의 세부 정보를 알지 못해도 구축할 수 있습니다

서브그래프를 추가할 때 주요 질문은 부모 그래프와 서브그래프가 어떻게 통신하는가, 즉 그래프 실행 중에 서로 간에 [상태](./low_level.md#state)를 어떻게 전달하는가입니다. 두 가지 시나리오가 있습니다:

- 부모 그래프와 서브그래프가 상태 [스키마](./low_level.md#state)에 **공유 상태 키**를 가지는 경우. 이 경우 [서브그래프를 부모 그래프의 노드로 포함](../how-tos/subgraph.md#shared-state-schemas)할 수 있습니다

  :::python

  ```python
  from langgraph.graph import StateGraph, MessagesState, START

  # Subgraph

  def call_model(state: MessagesState):
      response = model.invoke(state["messages"])
      return {"messages": response}

  subgraph_builder = StateGraph(State)
  subgraph_builder.add_node(call_model)
  ...
  # highlight-next-line
  subgraph = subgraph_builder.compile()

  # Parent graph

  builder = StateGraph(State)
  # highlight-next-line
  builder.add_node("subgraph_node", subgraph)
  builder.add_edge(START, "subgraph_node")
  graph = builder.compile()
  ...
  graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
  ```

  :::

  :::js

  ```typescript
  import { StateGraph, MessagesZodState, START } from "@langchain/langgraph";

  // Subgraph

  const subgraphBuilder = new StateGraph(MessagesZodState).addNode(
    "callModel",
    async (state) => {
      const response = await model.invoke(state.messages);
      return { messages: response };
    }
  );
  // ... other nodes and edges
  // highlight-next-line
  const subgraph = subgraphBuilder.compile();

  // Parent graph

  const builder = new StateGraph(MessagesZodState)
    // highlight-next-line
    .addNode("subgraphNode", subgraph)
    .addEdge(START, "subgraphNode");
  const graph = builder.compile();
  // ...
  await graph.invoke({ messages: [{ role: "user", content: "hi!" }] });
  ```

  :::

- 부모 그래프와 서브그래프가 **다른 스키마**를 가지는 경우 (상태 [스키마](./low_level.md#state)에 공유 상태 키가 없음). 이 경우 [부모 그래프의 노드 내부에서 서브그래프를 호출](../how-tos/subgraph.md#different-state-schemas)해야 합니다: 이는 부모 그래프와 서브그래프가 서로 다른 상태 스키마를 가지고 있고 서브그래프를 호출하기 전이나 후에 상태를 변환해야 할 때 유용합니다

  :::python

  ```python
  from typing_extensions import TypedDict, Annotated
  from langchain_core.messages import AnyMessage
  from langgraph.graph import StateGraph, MessagesState, START
  from langgraph.graph.message import add_messages

  class SubgraphMessagesState(TypedDict):
      # highlight-next-line
      subgraph_messages: Annotated[list[AnyMessage], add_messages]

  # Subgraph

  # highlight-next-line
  def call_model(state: SubgraphMessagesState):
      response = model.invoke(state["subgraph_messages"])
      return {"subgraph_messages": response}

  subgraph_builder = StateGraph(SubgraphMessagesState)
  subgraph_builder.add_node("call_model_from_subgraph", call_model)
  subgraph_builder.add_edge(START, "call_model_from_subgraph")
  ...
  # highlight-next-line
  subgraph = subgraph_builder.compile()

  # Parent graph

  def call_subgraph(state: MessagesState):
      response = subgraph.invoke({"subgraph_messages": state["messages"]})
      return {"messages": response["subgraph_messages"]}

  builder = StateGraph(State)
  # highlight-next-line
  builder.add_node("subgraph_node", call_subgraph)
  builder.add_edge(START, "subgraph_node")
  graph = builder.compile()
  ...
  graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
  ```

  :::

  :::js

  ```typescript
  import { StateGraph, MessagesZodState, START } from "@langchain/langgraph";
  import { z } from "zod";

  const SubgraphState = z.object({
    // highlight-next-line
    subgraphMessages: MessagesZodState.shape.messages,
  });

  // Subgraph

  const subgraphBuilder = new StateGraph(SubgraphState)
    // highlight-next-line
    .addNode("callModelFromSubgraph", async (state) => {
      const response = await model.invoke(state.subgraphMessages);
      return { subgraphMessages: response };
    })
    .addEdge(START, "callModelFromSubgraph");
  // ...
  // highlight-next-line
  const subgraph = subgraphBuilder.compile();

  // Parent graph

  const builder = new StateGraph(MessagesZodState)
    // highlight-next-line
    .addNode("subgraphNode", async (state) => {
      const response = await subgraph.invoke({
        subgraphMessages: state.messages,
      });
      return { messages: response.subgraphMessages };
    })
    .addEdge(START, "subgraphNode");
  const graph = builder.compile();
  // ...
  await graph.invoke({ messages: [{ role: "user", content: "hi!" }] });
  ```

  :::
