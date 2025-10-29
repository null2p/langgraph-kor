# Server API를 사용한 Human-in-the-loop

에이전트 또는 워크플로에서 도구 호출을 검토, 편집 및 승인하려면 LangGraph의 [human-in-the-loop](../../concepts/human_in_the_loop.md) 기능을 사용하세요.

## 동적 인터럽트

=== "Python"

    ```python
    from langgraph_sdk import get_client
    # highlight-next-line
    from langgraph_sdk.schema import Command
    client = get_client(url=<DEPLOYMENT_URL>)

    # Using the graph deployed with the name "agent"
    assistant_id = "agent"

    # create a thread
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # Run the graph until the interrupt is hit.
    result = await client.runs.wait(
        thread_id,
        assistant_id,
        input={"some_text": "original text"}   # (1)!
    )

    print(result['__interrupt__']) # (2)!
    # > [
    # >     {
    # >         'value': {'text_to_revise': 'original text'},
    # >         'id': '...',
    # >     }
    # > ]


    # Resume the graph
    print(await client.runs.wait(
        thread_id,
        assistant_id,
        # highlight-next-line
        command=Command(resume="Edited text")   # (3)!
    ))
    # > {'some_text': 'Edited text'}
    ```

    1. 그래프가 초기 상태와 함께 호출됩니다.
    2. 그래프가 인터럽트에 도달하면 페이로드 및 메타데이터와 함께 인터럽트 객체를 반환합니다.
    3. `Command(resume=...)`로 그래프를 재개하여 사용자의 입력을 주입하고 실행을 계속합니다.

=== "JavaScript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";
    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });

    // Using the graph deployed with the name "agent"
    const assistantID = "agent";

    // create a thread
    const thread = await client.threads.create();
    const threadID = thread["thread_id"];

    // Run the graph until the interrupt is hit.
    const result = await client.runs.wait(
      threadID,
      assistantID,
      { input: { "some_text": "original text" } }   // (1)!
    );

    console.log(result['__interrupt__']); // (2)!
    // > [
    // >     {
    // >         'value': {'text_to_revise': 'original text'},
    // >         'resumable': True,
    // >         'ns': ['human_node:fc722478-2f21-0578-c572-d9fc4dd07c3b'],
    // >         'when': 'during'
    // >     }
    // > ]

    // Resume the graph
    console.log(await client.runs.wait(
        threadID,
        assistantID,
        // highlight-next-line
        { command: { resume: "Edited text" }}   // (3)!
    ));
    // > {'some_text': 'Edited text'}
    ```

    1. 그래프가 초기 상태와 함께 호출됩니다.
    2. 그래프가 인터럽트에 도달하면 페이로드 및 메타데이터와 함께 인터럽트 객체를 반환합니다.
    3. `{ resume: ... }` 명령 객체로 그래프를 재개하여 사용자의 입력을 주입하고 실행을 계속합니다.

=== "cURL"

    스레드 생성:

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads \
    --header 'Content-Type: application/json' \
    --data '{}'
    ```

    인터럽트가 발생할 때까지 그래프 실행:

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/wait \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": {\"some_text\": \"original text\"}
    }"
    ```

    그래프 재개:

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/wait \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"command\": {
         \"resume\": \"Edited text\"
       }
     }"
    ```

??? example "`interrupt` 사용 확장 예제"

    다음은 LangGraph API 서버에서 실행할 수 있는 예제 그래프입니다.
    자세한 내용은 [LangGraph Platform 빠른 시작](../quick_start.md)을 참조하세요.

    ```python
    from typing import TypedDict
    import uuid

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.constants import START
    from langgraph.graph import StateGraph
    # highlight-next-line
    from langgraph.types import interrupt, Command

    class State(TypedDict):
        some_text: str

    def human_node(state: State):
        # highlight-next-line
        value = interrupt( # (1)!
            {
                "text_to_revise": state["some_text"] # (2)!
            }
        )
        return {
            "some_text": value # (3)!
        }


    # Build the graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("human_node", human_node)
    graph_builder.add_edge(START, "human_node")

    graph = graph_builder.compile()
    ```

    1. `interrupt(...)`는 `human_node`에서 실행을 일시 중지하고 주어진 페이로드를 사용자에게 노출합니다.
    2. JSON 직렬화 가능한 모든 값을 `interrupt` 함수에 전달할 수 있습니다. 여기서는 수정할 텍스트를 포함하는 딕셔너리입니다.
    3. 재개되면 `interrupt(...)`의 반환 값은 사용자가 제공한 입력이며, 이는 상태를 업데이트하는 데 사용됩니다.

    LangGraph API 서버가 실행 중이면 [LangGraph SDK](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/)를 사용하여 상호 작용할 수 있습니다.

    === "Python"

        ```python
        from langgraph_sdk import get_client
        # highlight-next-line
        from langgraph_sdk.schema import Command
        client = get_client(url=<DEPLOYMENT_URL>)

        # Using the graph deployed with the name "agent"
        assistant_id = "agent"

        # create a thread
        thread = await client.threads.create()
        thread_id = thread["thread_id"]

        # Run the graph until the interrupt is hit.
        result = await client.runs.wait(
            thread_id,
            assistant_id,
            input={"some_text": "original text"}   # (1)!
        )

        print(result['__interrupt__']) # (2)!
        # > [
        # >     {
        # >         'value': {'text_to_revise': 'original text'},
        # >         'id': '...',
        # >     }
        # > ]


        # Resume the graph
        print(await client.runs.wait(
            thread_id,
            assistant_id,
            # highlight-next-line
            command=Command(resume="Edited text")   # (3)!
        ))
        # > {'some_text': 'Edited text'}
        ```

        1. 그래프가 초기 상태와 함께 호출됩니다.
        2. 그래프가 인터럽트에 도달하면 페이로드 및 메타데이터와 함께 인터럽트 객체를 반환합니다.
        3. `Command(resume=...)`로 그래프를 재개하여 사용자의 입력을 주입하고 실행을 계속합니다.

    === "JavaScript"

        ```js
        import { Client } from "@langchain/langgraph-sdk";
        const client = new Client({ apiUrl: <DEPLOYMENT_URL> });

        // Using the graph deployed with the name "agent"
        const assistantID = "agent";

        // create a thread
        const thread = await client.threads.create();
        const threadID = thread["thread_id"];

        // Run the graph until the interrupt is hit.
        const result = await client.runs.wait(
          threadID,
          assistantID,
          { input: { "some_text": "original text" } }   // (1)!
        );

        console.log(result['__interrupt__']); // (2)!
        // > [
        // >     {
        // >         'value': {'text_to_revise': 'original text'},
        // >         'resumable': True,
        // >         'ns': ['human_node:fc722478-2f21-0578-c572-d9fc4dd07c3b'],
        // >         'when': 'during'
        // >     }
        // > ]

        // Resume the graph
        console.log(await client.runs.wait(
            threadID,
            assistantID,
            // highlight-next-line
            { command: { resume: "Edited text" }}   // (3)!
        ));
        // > {'some_text': 'Edited text'}
        ```

        1. 그래프가 초기 상태와 함께 호출됩니다.
        2. 그래프가 인터럽트에 도달하면 페이로드 및 메타데이터와 함께 인터럽트 객체를 반환합니다.
        3. `{ resume: ... }` 명령 객체로 그래프를 재개하여 사용자의 입력을 주입하고 실행을 계속합니다.

    === "cURL"

        스레드 생성:

        ```bash
        curl --request POST \
        --url <DEPLOYMENT_URL>/threads \
        --header 'Content-Type: application/json' \
        --data '{}'
        ```

        인터럽트가 발생할 때까지 그래프 실행:

        ```bash
        curl --request POST \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/wait \
        --header 'Content-Type: application/json' \
        --data "{
          \"assistant_id\": \"agent\",
          \"input\": {\"some_text\": \"original text\"}
        }"
        ```

        그래프 재개:

        ```bash
        curl --request POST \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/wait \
        --header 'Content-Type: application/json' \
        --data "{
          \"assistant_id\": \"agent\",
          \"command\": {
            \"resume\": \"Edited text\"
          }
        }"
        ```

## 정적 인터럽트

정적 인터럽트(정적 브레이크포인트라고도 함)는 노드 실행 전후에 트리거됩니다.

!!! warning

    정적 인터럽트는 human-in-the-loop 워크플로우에는 **권장되지 않습니다**. 디버깅 및 테스트에 가장 적합합니다.

컴파일 시 `interrupt_before` 및 `interrupt_after`를 지정하여 정적 인터럽트를 설정할 수 있습니다:

```python
# highlight-next-line
graph = graph_builder.compile( # (1)!
    # highlight-next-line
    interrupt_before=["node_a"], # (2)!
    # highlight-next-line
    interrupt_after=["node_b", "node_c"], # (3)!
)
```

1. 브레이크포인트는 `compile` 시점에 설정됩니다.
2. `interrupt_before`는 노드가 실행되기 전에 실행을 일시 중지해야 하는 노드를 지정합니다.
3. `interrupt_after`는 노드가 실행된 후에 실행을 일시 중지해야 하는 노드를 지정합니다.

또는 실행 시점에 정적 인터럽트를 설정할 수도 있습니다:

=== "Python"

    ```python
    # highlight-next-line
    await client.runs.wait( # (1)!
        thread_id,
        assistant_id,
        inputs=inputs,
        # highlight-next-line
        interrupt_before=["node_a"], # (2)!
        # highlight-next-line
        interrupt_after=["node_b", "node_c"] # (3)!
    )
    ```

    1. `client.runs.wait`는 `interrupt_before` 및 `interrupt_after` 파라미터와 함께 호출됩니다. 이는 실행 시점 구성이며 매 호출마다 변경할 수 있습니다.
    2. `interrupt_before`는 노드가 실행되기 전에 실행을 일시 중지해야 하는 노드를 지정합니다.
    3. `interrupt_after`는 노드가 실행된 후에 실행을 일시 중지해야 하는 노드를 지정합니다.

=== "JavaScript"

    ```js
    // highlight-next-line
    await client.runs.wait( // (1)!
        threadID,
        assistantID,
        {
        input: input,
        // highlight-next-line
        interruptBefore: ["node_a"], // (2)!
        // highlight-next-line
        interruptAfter: ["node_b", "node_c"] // (3)!
        }
    )
    ```

    1. `client.runs.wait`는 `interruptBefore` 및 `interruptAfter` 파라미터와 함께 호출됩니다. 이는 실행 시점 구성이며 매 호출마다 변경할 수 있습니다.
    2. `interruptBefore`는 노드가 실행되기 전에 실행을 일시 중지해야 하는 노드를 지정합니다.
    3. `interruptAfter`는 노드가 실행된 후에 실행을 일시 중지해야 하는 노드를 지정합니다.

=== "cURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/wait \
    --header 'Content-Type: application/json' \
    --data "{
        \"assistant_id\": \"agent\",
        \"interrupt_before\": [\"node_a\"],
        \"interrupt_after\": [\"node_b\", \"node_c\"],
        \"input\": <INPUT>
    }"
    ```

다음 예제는 정적 인터럽트를 추가하는 방법을 보여줍니다:

=== "Python"

    ```python
    from langgraph_sdk import get_client
    client = get_client(url=<DEPLOYMENT_URL>)

    # Using the graph deployed with the name "agent"
    assistant_id = "agent"

    # create a thread
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # Run the graph until the breakpoint
    result = await client.runs.wait(
        thread_id,
        assistant_id,
        input=inputs   # (1)!
    )

    # Resume the graph
    await client.runs.wait(
        thread_id,
        assistant_id,
        input=None   # (2)!
    )
    ```

    1. 그래프는 첫 번째 브레이크포인트에 도달할 때까지 실행됩니다.
    2. 입력으로 `None`을 전달하여 그래프를 재개합니다. 이렇게 하면 다음 브레이크포인트에 도달할 때까지 그래프가 실행됩니다.

=== "JavaScript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";
    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });

    // Using the graph deployed with the name "agent"
    const assistantID = "agent";

    // create a thread
    const thread = await client.threads.create();
    const threadID = thread["thread_id"];

    // Run the graph until the breakpoint
    const result = await client.runs.wait(
      threadID,
      assistantID,
      { input: input }   // (1)!
    );

    // Resume the graph
    await client.runs.wait(
      threadID,
      assistantID,
      { input: null }   // (2)!
    );
    ```

    1. 그래프는 첫 번째 브레이크포인트에 도달할 때까지 실행됩니다.
    2. 입력으로 `null`을 전달하여 그래프를 재개합니다. 이렇게 하면 다음 브레이크포인트에 도달할 때까지 그래프가 실행됩니다.

=== "cURL"

    스레드 생성:

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads \
    --header 'Content-Type: application/json' \
    --data '{}'
    ```

    브레이크포인트까지 그래프 실행:

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/wait \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": <INPUT>
    }"
    ```

    그래프 재개:

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/wait \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\"
    }"
    ```


## 더 알아보기

- [Human-in-the-loop 개념 가이드](../../concepts/human_in_the_loop.md): LangGraph의 human-in-the-loop 기능에 대해 자세히 알아보세요.
- [일반적인 패턴](../../how-tos/human_in_the_loop/add-human-in-the-loop.md#common-patterns): 작업 승인/거부, 사용자 입력 요청, 도구 호출 검토, 사용자 입력 검증과 같은 패턴을 구현하는 방법을 알아보세요.