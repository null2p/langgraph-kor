# 스트리밍 API

[LangGraph SDK](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/)를 사용하면 LangGraph API 서버에서 [출력을 스트리밍](../../concepts/streaming.md)할 수 있습니다.

!!! note

    LangGraph SDK와 LangGraph Server는 [LangGraph Platform](../../concepts/langgraph_platform.md)의 일부입니다.

## 기본 사용법

기본 사용 예제:

=== "Python"

    ```python
    from langgraph_sdk import get_client
    client = get_client(url=<DEPLOYMENT_URL>, api_key=<API_KEY>)

    # Using the graph deployed with the name "agent"
    assistant_id = "agent"

    # create a thread
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # create a streaming run
    # highlight-next-line
    async for chunk in client.runs.stream(
        thread_id,
        assistant_id,
        input=inputs,
        stream_mode="updates"
    ):
        print(chunk.data)
    ```

=== "JavaScript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";
    const client = new Client({ apiUrl: <DEPLOYMENT_URL>, apiKey: <API_KEY> });

    // Using the graph deployed with the name "agent"
    const assistantID = "agent";

    // create a thread
    const thread = await client.threads.create();
    const threadID = thread["thread_id"];

    // create a streaming run
    // highlight-next-line
    const streamResponse = client.runs.stream(
      threadID,
      assistantID,
      {
        input,
        streamMode: "updates"
      }
    );
    for await (const chunk of streamResponse) {
      console.log(chunk.data);
    }
    ```

=== "cURL"

    스레드 생성:

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads \
    --header 'Content-Type: application/json' \
    --data '{}'
    ```

    스트리밍 실행 생성:

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
    --header 'Content-Type: application/json' \
    --header 'x-api-key: <API_KEY>'
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": <inputs>,
      \"stream_mode\": \"updates\"
    }"
    ```

??? example "확장 예제: 업데이트 스트리밍"

    이것은 LangGraph API 서버에서 실행할 수 있는 예제 그래프입니다.
    자세한 내용은 [LangGraph Platform 빠른 시작](../quick_start.md)을 참조하세요.

    ```python
    # graph.py
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

    실행 중인 LangGraph API 서버가 있으면, [LangGraph SDK](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/)를 사용하여 상호작용할 수 있습니다

    === "Python"

        ```python
        from langgraph_sdk import get_client
        client = get_client(url=<DEPLOYMENT_URL>)

        # Using the graph deployed with the name "agent"
        assistant_id = "agent"

        # create a thread
        thread = await client.threads.create()
        thread_id = thread["thread_id"]

        # create a streaming run
        # highlight-next-line
        async for chunk in client.runs.stream(  # (1)!
            thread_id,
            assistant_id,
            input={"topic": "ice cream"},
            # highlight-next-line
            stream_mode="updates"  # (2)!
        ):
            print(chunk.data)
        ```

        1. `client.runs.stream()` 메서드는 스트리밍 출력을 생성하는 이터레이터를 반환합니다.
        2. `stream_mode="updates"`를 설정하여 각 노드 이후 그래프 상태의 업데이트만 스트리밍합니다. 다른 스트림 모드도 사용할 수 있습니다. 자세한 내용은 [지원되는 스트림 모드](#supported-stream-modes)를 참조하세요.

    === "JavaScript"

        ```js
        import { Client } from "@langchain/langgraph-sdk";
        const client = new Client({ apiUrl: <DEPLOYMENT_URL> });

        // Using the graph deployed with the name "agent"
        const assistantID = "agent";

        // create a thread
        const thread = await client.threads.create();
        const threadID = thread["thread_id"];

        // create a streaming run
        // highlight-next-line
        const streamResponse = client.runs.stream(  // (1)!
          threadID,
          assistantID,
          {
            input: { topic: "ice cream" },
            // highlight-next-line
            streamMode: "updates"  // (2)!
          }
        );
        for await (const chunk of streamResponse) {
          console.log(chunk.data);
        }
        ```

        1. `client.runs.stream()` 메서드는 스트리밍 출력을 생성하는 이터레이터를 반환합니다.
        2. `streamMode: "updates"`를 설정하여 각 노드 이후 그래프 상태의 업데이트만 스트리밍합니다. 다른 스트림 모드도 사용할 수 있습니다. 자세한 내용은 [지원되는 스트림 모드](#supported-stream-modes)를 참조하세요.

    === "cURL"

        스레드 생성:

        ```bash
        curl --request POST \
        --url <DEPLOYMENT_URL>/threads \
        --header 'Content-Type: application/json' \
        --data '{}'
        ```

        스트리밍 실행 생성:

        ```bash
        curl --request POST \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
        --header 'Content-Type: application/json' \
        --data "{
          \"assistant_id\": \"agent\",
          \"input\": {\"topic\": \"ice cream\"},
          \"stream_mode\": \"updates\"
        }"
        ```

    ```output
    {'run_id': '1f02c2b3-3cef-68de-b720-eec2a4a8e920', 'attempt': 1}
    {'refine_topic': {'topic': 'ice cream and cats'}}
    {'generate_joke': {'joke': 'This is a joke about ice cream and cats'}}
    ```


### 지원되는 스트림 모드

| Mode                             | Description                                                                                                                                                                         | LangGraph Library Method                                                                                 |
|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| [`values`](#stream-graph-state)  | Stream the full graph state after each [super-step](../../concepts/low_level.md#graphs).                                                                                            | `.stream()` / `.astream()` with [`stream_mode="values"`](../../how-tos/streaming.md#stream-graph-state)  |
| [`updates`](#stream-graph-state) | Streams the updates to the state after each step of the graph. If multiple updates are made in the same step (e.g., multiple nodes are run), those updates are streamed separately. | `.stream()` / `.astream()` with [`stream_mode="updates"`](../../how-tos/streaming.md#stream-graph-state) |
| [`messages-tuple`](#messages)    | Streams LLM tokens and metadata for the graph node where the LLM is invoked (useful for chat apps).                                                                                 | `.stream()` / `.astream()` with [`stream_mode="messages"`](../../how-tos/streaming.md#messages)          |
| [`debug`](#debug)                | Streams as much information as possible throughout the execution of the graph.                                                                                                      | `.stream()` / `.astream()` with [`stream_mode="debug"`](../../how-tos/streaming.md#stream-graph-state)   |
| [`custom`](#stream-custom-data)  | Streams custom data from inside your graph                                                                                                                                          | `.stream()` / `.astream()` with [`stream_mode="custom"`](../../how-tos/streaming.md#stream-custom-data)  |
| [`events`](#stream-events)       | Stream all events (including the state of the graph); mainly useful when migrating large LCEL apps.                                                                                 | `.astream_events()`                                                                                      |

### 여러 모드 스트리밍

You can pass a list as the `stream_mode` parameter to stream multiple modes at once.

The streamed outputs will be tuples of `(mode, chunk)` where `mode` is the name of the stream mode and `chunk` is the data streamed by that mode.

=== "Python"

    ```python
    async for chunk in client.runs.stream(
        thread_id,
        assistant_id,
        input=inputs,
        stream_mode=["updates", "custom"]
    ):
        print(chunk)
    ```

=== "JavaScript"

    ```js
    const streamResponse = client.runs.stream(
      threadID,
      assistantID,
      {
        input,
        streamMode: ["updates", "custom"]
      }
    );
    for await (const chunk of streamResponse) {
      console.log(chunk);
    }
    ```

=== "cURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": <inputs>,
       \"stream_mode\": [
         \"updates\"
         \"custom\"
       ]
     }"
    ```

## 그래프 상태 스트리밍

Use the stream modes `updates` and `values` to stream the state of the graph as it executes.

* `updates` streams the **updates** to the state after each step of the graph.
* `values` streams the **full value** of the state after each step of the graph.

??? example "예제 그래프"

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

!!! note "상태 저장 실행"

    아래 예제는 스트리밍 실행의 **출력을 지속**하려고 [체크포인터](../../concepts/persistence.md) DB에 저장하고 스레드를 생성했다고 가정합니다. 스레드를 생성하려면:

    === "Python"

        ```python
        from langgraph_sdk import get_client
        client = get_client(url=<DEPLOYMENT_URL>)

        # Using the graph deployed with the name "agent"
        assistant_id = "agent"
        # create a thread
        thread = await client.threads.create()
        thread_id = thread["thread_id"]
        ```

    === "JavaScript"

        ```js
        import { Client } from "@langchain/langgraph-sdk";
        const client = new Client({ apiUrl: <DEPLOYMENT_URL> });

        // Using the graph deployed with the name "agent"
        const assistantID = "agent";
        // create a thread
        const thread = await client.threads.create();
        const threadID = thread["thread_id"]
        ```

    === "cURL"

        ```bash
        curl --request POST \
        --url <DEPLOYMENT_URL>/threads \
        --header 'Content-Type: application/json' \
        --data '{}'
        ```

    실행의 출력을 지속할 필요가 없는 경우, 스트리밍할 때 `thread_id` 대신 `None`을 전달할 수 있습니다.

=== "updates"

    각 단계 이후 노드에서 반환된 **상태 업데이트**만 스트리밍하려면 이것을 사용하세요. 스트리밍된 출력에는 노드 이름과 업데이트가 포함됩니다.

    === "Python"

        ```python
        async for chunk in client.runs.stream(
            thread_id,
            assistant_id,
            input={"topic": "ice cream"},
            # highlight-next-line
            stream_mode="updates"
        ):
            print(chunk.data)
        ```

    === "JavaScript"

        ```js
        const streamResponse = client.runs.stream(
          threadID,
          assistantID,
          {
            input: { topic: "ice cream" },
            // highlight-next-line
            streamMode: "updates"
          }
        );
        for await (const chunk of streamResponse) {
          console.log(chunk.data);
        }
        ```

    === "cURL"

        ```bash
        curl --request POST \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
        --header 'Content-Type: application/json' \
        --data "{
          \"assistant_id\": \"agent\",
          \"input\": {\"topic\": \"ice cream\"},
          \"stream_mode\": \"updates\"
        }"
        ```

===  "values"

    각 단계 이후 그래프의 **전체 상태**를 스트리밍하려면 이것을 사용하세요.

    === "Python"

        ```python
        async for chunk in client.runs.stream(
            thread_id,
            assistant_id,
            input={"topic": "ice cream"},
            # highlight-next-line
            stream_mode="values"
        ):
            print(chunk.data)
        ```

    === "JavaScript"

        ```js
        const streamResponse = client.runs.stream(
          threadID,
          assistantID,
          {
            input: { topic: "ice cream" },
            // highlight-next-line
            streamMode: "values"
          }
        );
        for await (const chunk of streamResponse) {
          console.log(chunk.data);
        }
        ```

    === "cURL"

        ```bash
        curl --request POST \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
        --header 'Content-Type: application/json' \
        --data "{
          \"assistant_id\": \"agent\",
          \"input\": {\"topic\": \"ice cream\"},
          \"stream_mode\": \"values\"
        }"
        ```


## 서브그래프

To include outputs from [subgraphs](../../concepts/subgraphs.md) in the streamed outputs, you can set `subgraphs=True` in the `.stream()` method of the parent graph. This will stream outputs from both the parent graph and any subgraphs.

```python
for chunk in client.runs.stream(
    thread_id,
    assistant_id,
    input={"foo": "foo"},
    # highlight-next-line
    stream_subgraphs=True, # (1)!
    stream_mode="updates",
):
    print(chunk)
```

1. `stream_subgraphs=True`를 설정하여 서브그래프에서 출력을 스트리밍합니다.

??? example "확장 예제: 서브그래프에서 스트리밍"

    이것은 LangGraph API 서버에서 실행할 수 있는 예제 그래프입니다.
    자세한 내용은 [LangGraph Platform 빠른 시작](../quick_start.md)을 참조하세요.

    ```python
    # graph.py
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
    ```

    실행 중인 LangGraph API 서버가 있으면, [LangGraph SDK](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/)를 사용하여 상호작용할 수 있습니다

    === "Python"

        ```python
        from langgraph_sdk import get_client
        client = get_client(url=<DEPLOYMENT_URL>)

        # Using the graph deployed with the name "agent"
        assistant_id = "agent"

        # create a thread
        thread = await client.threads.create()
        thread_id = thread["thread_id"]
    
        async for chunk in client.runs.stream(
            thread_id,
            assistant_id,
            input={"foo": "foo"},
            # highlight-next-line
            stream_subgraphs=True, # (1)!
            stream_mode="updates",
        ):
            print(chunk)
        ```

        1. `stream_subgraphs=True`를 설정하여 서브그래프에서 출력을 스트리밍합니다.

    === "JavaScript"

        ```js
        import { Client } from "@langchain/langgraph-sdk";
        const client = new Client({ apiUrl: <DEPLOYMENT_URL> });

        // Using the graph deployed with the name "agent"
        const assistantID = "agent";

        // create a thread
        const thread = await client.threads.create();
        const threadID = thread["thread_id"];

        // create a streaming run
        const streamResponse = client.runs.stream(
          threadID,
          assistantID,
          {
            input: { foo: "foo" },
            // highlight-next-line
            streamSubgraphs: true,  // (1)!
            streamMode: "updates"
          }
        );
        for await (const chunk of streamResponse) {
          console.log(chunk);
        }
        ```

        1. `streamSubgraphs: true`를 설정하여 서브그래프에서 출력을 스트리밍합니다.

    === "cURL"

        스레드 생성:

        ```bash
        curl --request POST \
        --url <DEPLOYMENT_URL>/threads \
        --header 'Content-Type: application/json' \
        --data '{}'
        ```

        스트리밍 실행 생성:

        ```bash
        curl --request POST \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
        --header 'Content-Type: application/json' \
        --data "{
          \"assistant_id\": \"agent\",
          \"input\": {\"foo\": \"foo\"},
          \"stream_subgraphs\": true,
          \"stream_mode\": [
            \"updates\"
          ]
        }"
        ```

    **참고**: 노드 업데이트뿐만 아니라 어떤 그래프(또는 서브그래프)에서 스트리밍하는지 알려주는 네임스페이스도 수신하고 있습니다.

## 디버깅 {#debug}

Use the `debug` streaming mode to stream as much information as possible throughout the execution of the graph. The streamed outputs include the name of the node as well as the full state.

=== "Python"

    ```python
    async for chunk in client.runs.stream(
        thread_id,
        assistant_id,
        input={"topic": "ice cream"},
        # highlight-next-line
        stream_mode="debug"
    ):
        print(chunk.data)
    ```

=== "JavaScript"

    ```js
    const streamResponse = client.runs.stream(
      threadID,
      assistantID,
      {
        input: { topic: "ice cream" },
        // highlight-next-line
        streamMode: "debug"
      }
    );
    for await (const chunk of streamResponse) {
      console.log(chunk.data);
    }
    ```

=== "cURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": {\"topic\": \"ice cream\"},
      \"stream_mode\": \"debug\"
    }"
    ```

## LLM 토큰 {#messages}

Use the `messages-tuple` streaming mode to stream Large Language Model (LLM) outputs **token by token** from any part of your graph, including nodes, tools, subgraphs, or tasks.

The streamed output from [`messages-tuple` mode](#supported-stream-modes) is a tuple `(message_chunk, metadata)` where:

- `message_chunk`: the token or message segment from the LLM.
- `metadata`: a dictionary containing details about the graph node and LLM invocation.
 
??? example "예제 그래프"

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
    ```

    1. LLM이 `.stream`이 아닌 `.invoke`를 사용하여 실행되는 경우에도 메시지 이벤트가 발생됩니다.

=== "Python"

    ```python
    async for chunk in client.runs.stream(
        thread_id,
        assistant_id,
        input={"topic": "ice cream"},
        # highlight-next-line
        stream_mode="messages-tuple",
    ):
        if chunk.event != "messages":
            continue

        message_chunk, metadata = chunk.data  # (1)!
        if message_chunk["content"]:
            print(message_chunk["content"], end="|", flush=True)
    ```

    1. "messages-tuple" 스트림 모드는 튜플 `(message_chunk, metadata)`의 이터레이터를 반환합니다. 여기서 `message_chunk`는 LLM에 의해 스트리밍된 토큰이고 `metadata`는 LLM이 호출된 그래프 노드 및 기타 정보에 대한 정보가 포함된 딕셔너리입니다.

=== "JavaScript"

    ```js
    const streamResponse = client.runs.stream(
      threadID,
      assistantID,
      {
        input: { topic: "ice cream" },
        // highlight-next-line
        streamMode: "messages-tuple"
      }
    );
    for await (const chunk of streamResponse) {
      if (chunk.event !== "messages") {
        continue;
      }
      console.log(chunk.data[0]["content"]);  // (1)!
    }
    ```

    1. "messages-tuple" 스트림 모드는 튜플 `(message_chunk, metadata)`의 이터레이터를 반환합니다. 여기서 `message_chunk`는 LLM에 의해 스트리밍된 토큰이고 `metadata`는 LLM이 호출된 그래프 노드 및 기타 정보에 대한 정보가 포함된 딕셔너리입니다.

=== "cURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": {\"topic\": \"ice cream\"},
      \"stream_mode\": \"messages-tuple\"
    }"
    ```

### LLM 토큰 필터링

* LLM 호출별로 스트리밍된 토큰을 필터링하려면 [LLM 호출에 `tags`를 연결](../../how-tos/streaming.md#filter-by-llm-invocation)할 수 있습니다.
* 특정 노드에서만 토큰을 스트리밍하려면 `stream_mode="messages"`를 사용하고 스트리밍된 메타데이터의 [`langgraph_node` 필드로 출력을 필터링](../../how-tos/streaming.md#filter-by-node)합니다.

## 커스텀 데이터 스트리밍

To send **custom user-defined data**:

=== "Python"

    ```python
    async for chunk in client.runs.stream(
        thread_id,
        assistant_id,
        input={"query": "example"},
        # highlight-next-line
        stream_mode="custom"
    ):
        print(chunk.data)
    ```

=== "JavaScript"

    ```js
    const streamResponse = client.runs.stream(
      threadID,
      assistantID,
      {
        input: { query: "example" },
        // highlight-next-line
        streamMode: "custom"
      }
    );
    for await (const chunk of streamResponse) {
      console.log(chunk.data);
    }
    ```

=== "cURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": {\"query\": \"example\"},
      \"stream_mode\": \"custom\"
    }"
    ```

## 이벤트 스트리밍

To stream all events, including the state of the graph:

=== "Python"

    ```python
    async for chunk in client.runs.stream(
        thread_id,
        assistant_id,
        input={"topic": "ice cream"},
        # highlight-next-line
        stream_mode="events"
    ):
        print(chunk.data)
    ```

=== "JavaScript"

    ```js
    const streamResponse = client.runs.stream(
      threadID,
      assistantID,
      {
        input: { topic: "ice cream" },
        // highlight-next-line
        streamMode: "events"
      }
    );
    for await (const chunk of streamResponse) {
      console.log(chunk.data);
    }
    ```

=== "cURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": {\"topic\": \"ice cream\"},
      \"stream_mode\": \"events\"
    }"
    ```

## 무상태 실행

If you don't want to **persist the outputs** of a streaming run in the [checkpointer](../../concepts/persistence.md) DB, you can create a stateless run without creating a thread:

=== "Python"

    ```python
    from langgraph_sdk import get_client
    client = get_client(url=<DEPLOYMENT_URL>, api_key=<API_KEY>)

    async for chunk in client.runs.stream(
        # highlight-next-line
        None,  # (1)!
        assistant_id,
        input=inputs,
        stream_mode="updates"
    ):
        print(chunk.data)
    ```

    1. `thread_id` UUID 대신 `None`을 전달하고 있습니다.

=== "JavaScript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";
    const client = new Client({ apiUrl: <DEPLOYMENT_URL>, apiKey: <API_KEY> });

    // create a streaming run
    // highlight-next-line
    const streamResponse = client.runs.stream(
      // highlight-next-line
      null,  // (1)!
      assistantID,
      {
        input,
        streamMode: "updates"
      }
    );
    for await (const chunk of streamResponse) {
      console.log(chunk.data);
    }
    ```

    1. `thread_id` UUID 대신 `None`을 전달하고 있습니다.

=== "cURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/runs/stream \
    --header 'Content-Type: application/json' \
    --header 'x-api-key: <API_KEY>'
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": <inputs>,
      \"stream_mode\": \"updates\"
    }"
    ```

## 조인 및 스트리밍

LangGraph Platform allows you to join an active [background run](../how-tos/background_run.md) and stream outputs from it. To do so, you can use [LangGraph SDK's](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/) `client.runs.join_stream` method:

=== "Python"

    ```python
    from langgraph_sdk import get_client
    client = get_client(url=<DEPLOYMENT_URL>, api_key=<API_KEY>)

    # highlight-next-line
    async for chunk in client.runs.join_stream(
        thread_id,
        # highlight-next-line
        run_id,  # (1)!
    ):
        print(chunk)
    ```

    1. 이것은 조인하려는 기존 실행의 `run_id`입니다.


=== "JavaScript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";
    const client = new Client({ apiUrl: <DEPLOYMENT_URL>, apiKey: <API_KEY> });

    // highlight-next-line
    const streamResponse = client.runs.joinStream(
      threadID,
      // highlight-next-line
      runId  // (1)!
    );
    for await (const chunk of streamResponse) {
      console.log(chunk);
    }
    ```

    1. 이것은 조인하려는 기존 실행의 `run_id`입니다.

=== "cURL"

    ```bash
    curl --request GET \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/<RUN_ID>/stream \
    --header 'Content-Type: application/json' \
    --header 'x-api-key: <API_KEY>'
    ```

!!! warning "출력이 버퍼링되지 않음"

    `.join_stream`을 사용할 때 출력이 버퍼링되지 않으므로, 조인하기 전에 생성된 출력은 수신되지 않습니다.

## API 레퍼런스

API 사용법 및 구현에 대해서는 [API 레퍼런스](../reference/api/api_ref.html#tag/thread-runs/POST/threads/{thread_id}/runs/stream)를 참조하세요. 
