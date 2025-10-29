# Server API를 사용한 타임 트래블

LangGraph는 이전 체크포인트에서 실행을 재개하는 [**타임 트래블**](../../concepts/time-travel.md) 기능을 제공하며, 동일한 상태를 재생하거나 수정하여 대안을 탐색할 수 있습니다. 모든 경우에 과거 실행을 재개하면 히스토리에 새 분기가 생성됩니다.

LangGraph Server API(LangGraph SDK를 통해)를 사용하여 타임 트래블하려면:

1. **그래프 실행**: [LangGraph SDK](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/)의 @[`client.runs.wait`][client.runs.wait] 또는 @[`client.runs.stream`][client.runs.stream] API를 사용하여 초기 입력으로 그래프를 실행합니다.
2. **기존 스레드에서 체크포인트 식별**: @[`client.threads.get_history`][client.threads.get_history] 메서드를 사용하여 특정 `thread_id`에 대한 실행 히스토리를 검색하고 원하는 `checkpoint_id`를 찾습니다.
   또는 실행을 일시 중지하려는 노드 앞에 [브레이크포인트](./human_in_the_loop_breakpoint.md)를 설정합니다. 그런 다음 해당 브레이크포인트까지 기록된 가장 최근 체크포인트를 찾을 수 있습니다.
3. **(선택 사항) 그래프 상태 수정**: @[`client.threads.update_state`][client.threads.update_state] 메서드를 사용하여 체크포인트에서 그래프의 상태를 수정하고 대체 상태에서 실행을 재개합니다.
4. **체크포인트에서 실행 재개**: `None` 입력과 적절한 `thread_id` 및 `checkpoint_id`와 함께 @[`client.runs.wait`][client.runs.wait] 또는 @[`client.runs.stream`][client.runs.stream] API를 사용합니다.

## 워크플로에서 타임 트래블 사용

??? example "Example graph"

    ```python
    from typing_extensions import TypedDict, NotRequired
    from langgraph.graph import StateGraph, START, END
    from langchain.chat_models import init_chat_model
    from langgraph.checkpoint.memory import InMemorySaver

    class State(TypedDict):
        topic: NotRequired[str]
        joke: NotRequired[str]

    llm = init_chat_model(
        "anthropic:claude-3-7-sonnet-latest",
        temperature=0,
    )

    def generate_topic(state: State):
        """LLM call to generate a topic for the joke"""
        msg = llm.invoke("Give me a funny topic for a joke")
        return {"topic": msg.content}

    def write_joke(state: State):
        """LLM call to write a joke based on the topic"""
        msg = llm.invoke(f"Write a short joke about {state['topic']}")
        return {"joke": msg.content}

    # Build workflow
    builder = StateGraph(State)

    # Add nodes
    builder.add_node("generate_topic", generate_topic)
    builder.add_node("write_joke", write_joke)

    # Add edges to connect nodes
    builder.add_edge(START, "generate_topic")
    builder.add_edge("generate_topic", "write_joke")

    # Compile
    graph = builder.compile()
    ```

### 1. 그래프 실행

=== "Python"

    ```python
    from langgraph_sdk import get_client
    client = get_client(url=<DEPLOYMENT_URL>)

    # Using the graph deployed with the name "agent"
    assistant_id = "agent"

    # create a thread
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # Run the graph
    result = await client.runs.wait(
        thread_id,
        assistant_id,
        input={}
    )
    ```

=== "JavaScript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";
    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });

    // Using the graph deployed with the name "agent"
    const assistantID = "agent";

    // create a thread
    const thread = await client.threads.create();
    const threadID = thread["thread_id"];

    // Run the graph
    const result = await client.runs.wait(
      threadID,
      assistantID,
      { input: {}}
    );
    ```

=== "cURL"

    스레드 생성:

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads \
    --header 'Content-Type: application/json' \
    --data '{}'
    ```

    그래프 실행:

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/wait \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": {}
    }"
    ```

### 2. 체크포인트 식별

=== "Python"

    ```python
    # The states are returned in reverse chronological order.
    states = await client.threads.get_history(thread_id)
    selected_state = states[1]
    print(selected_state)
    ```

=== "JavaScript"

    ```js
    // The states are returned in reverse chronological order.
    const states = await client.threads.getHistory(threadID);
    const selectedState = states[1];
    console.log(selectedState);
    ```

=== "cURL"

    ```bash
    curl --request GET \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/history \
    --header 'Content-Type: application/json'
    ```

### 3. 상태 업데이트 (선택 사항)

`update_state`는 새 체크포인트를 생성합니다. 새 체크포인트는 동일한 스레드와 연결되지만 새 체크포인트 ID를 가집니다.

=== "Python"

    ```python
    new_config = await client.threads.update_state(
        thread_id,
        {"topic": "chickens"},
        # highlight-next-line
        checkpoint_id=selected_state["checkpoint_id"]
    )
    print(new_config)
    ```

=== "JavaScript"

    ```js
    const newConfig = await client.threads.updateState(
      threadID,
      {
        values: { "topic": "chickens" },
        checkpointId: selectedState["checkpoint_id"]
      }
    );
    console.log(newConfig);
    ```

=== "cURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"checkpoint_id\": <CHECKPOINT_ID>,
      \"values\": {\"topic\": \"chickens\"}
    }"
    ```

### 4. 체크포인트에서 실행 재개

=== "Python"

    ```python
    await client.runs.wait(
        thread_id,
        assistant_id,
        # highlight-next-line
        input=None,
        # highlight-next-line
        checkpoint_id=new_config["checkpoint_id"]
    )
    ```

=== "JavaScript"

    ```js
    await client.runs.wait(
      threadID,
      assistantID,
      {
        // highlight-next-line
        input: null,
        // highlight-next-line
        checkpointId: newConfig["checkpoint_id"]
      }
    );
    ```

=== "cURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/wait \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"checkpoint_id\": <CHECKPOINT_ID>
    }"
    ```

## 더 알아보기

- [**LangGraph 타임 트래블 가이드**](../../how-tos/human_in_the_loop/time-travel.md): LangGraph에서 타임 트래블을 사용하는 방법에 대해 자세히 알아보세요.