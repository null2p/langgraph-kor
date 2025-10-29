# 스레드 사용

이 가이드에서는 [스레드](../../concepts/persistence.md#threads)를 생성, 보기 및 검사하는 방법을 보여줍니다.

## 스레드 생성

그래프를 실행하고 상태를 지속하려면 먼저 스레드를 생성해야 합니다.

### 빈 스레드

새 스레드를 생성하려면 [LangGraph SDK](../../concepts/sdk.md) `create` 메서드를 사용하세요. 자세한 내용은 [Python](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.ThreadsClient.create) 및 [JS](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/js_ts_sdk_ref/#create_3) SDK 레퍼런스 문서를 참조하세요.

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    thread = await client.threads.create()

    print(thread)
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    const thread = await client.threads.create();

    console.log(thread);
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads \
        --header 'Content-Type: application/json' \
        --data '{}'
    ```

Output:

    {
      "thread_id": "123e4567-e89b-12d3-a456-426614174000",
      "created_at": "2025-05-12T14:04:08.268Z",
      "updated_at": "2025-05-12T14:04:08.268Z",
      "metadata": {},
      "status": "idle",
      "values": {}
    }

### 스레드 복사

또는 애플리케이션에 상태를 복사하려는 스레드가 이미 있는 경우 `copy` 메서드를 사용할 수 있습니다. 이렇게 하면 작업 시점에 원본 스레드와 동일한 히스토리를 가진 독립적인 스레드가 생성됩니다. 자세한 내용은 [Python](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.ThreadsClient.copy) 및 [JS](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/js_ts_sdk_ref/#copy) SDK 레퍼런스 문서를 참조하세요.

=== "Python"

    ```python
    copied_thread = await client.threads.copy(<THREAD_ID>)
    ```

=== "Javascript"

    ```js
    const copiedThread = await client.threads.copy(<THREAD_ID>);
    ```

=== "CURL"

    ```bash
    curl --request POST --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/copy \
    --header 'Content-Type: application/json'
    ```

### 미리 채워진 상태

마지막으로 `create` 메서드에 `supersteps` 목록을 제공하여 임의의 사전 정의된 상태로 스레드를 생성할 수 있습니다. `supersteps`는 상태 업데이트 시퀀스의 목록을 설명합니다. 예를 들어:

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    thread = await client.threads.create(
      graph_id="agent",
      supersteps=[
        {
          updates: [
            {
              values: {},
              as_node: '__input__',
            },
          ],
        },
        {
          updates: [
            {
              values: {
                messages: [
                  {
                    type: 'human',
                    content: 'hello',
                  },
                ],
              },
              as_node: '__start__',
            },
          ],
        },
        {
          updates: [
            {
              values: {
                messages: [
                  {
                    content: 'Hello! How can I assist you today?',
                    type: 'ai',
                  },
                ],
              },
              as_node: 'call_model',
            },
          ],
        },
      ])

    print(thread)
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    const thread = await client.threads.create({
        graphId: 'agent',
        supersteps: [
        {
          updates: [
            {
              values: {},
              asNode: '__input__',
            },
          ],
        },
        {
          updates: [
            {
              values: {
                messages: [
                  {
                    type: 'human',
                    content: 'hello',
                  },
                ],
              },
              asNode: '__start__',
            },
          ],
        },
        {
          updates: [
            {
              values: {
                messages: [
                  {
                    content: 'Hello! How can I assist you today?',
                    type: 'ai',
                  },
                ],
              },
              asNode: 'call_model',
            },
          ],
        },
      ],
    });

    console.log(thread);
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads \
        --header 'Content-Type: application/json' \
        --data '{"metadata":{"graph_id":"agent"},"supersteps":[{"updates":[{"values":{},"as_node":"__input__"}]},{"updates":[{"values":{"messages":[{"type":"human","content":"hello"}]},"as_node":"__start__"}]},{"updates":[{"values":{"messages":[{"content":"Hello\u0021 How can I assist you today?","type":"ai"}]},"as_node":"call_model"}]}]}'
    ```

Output:

    {
        "thread_id": "f15d70a1-27d4-4793-a897-de5609920b7d",
        "created_at": "2025-05-12T15:37:08.935038+00:00",
        "updated_at": "2025-05-12T15:37:08.935046+00:00",
        "metadata": {"graph_id": "agent"},
        "status": "idle",
        "config": {},
        "values": {
            "messages": [
                {
                    "content": "hello",
                    "additional_kwargs": {},
                    "response_metadata": {},
                    "type": "human",
                    "name": null,
                    "id": "8701f3be-959c-4b7c-852f-c2160699b4ab",
                    "example": false
                },
                {
                    "content": "Hello! How can I assist you today?",
                    "additional_kwargs": {},
                    "response_metadata": {},
                    "type": "ai",
                    "name": null,
                    "id": "4d8ea561-7ca1-409a-99f7-6b67af3e1aa3",
                    "example": false,
                    "tool_calls": [],
                    "invalid_tool_calls": [],
                    "usage_metadata": null
                }
            ]
        }
    }

## List threads

### LangGraph SDK

To list threads, use the [LangGraph SDK](../../concepts/sdk.md) `search` method. This will list the threads in the application that match the provided filters. See the [Python](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.ThreadsClient.search) and [JS](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/js_ts_sdk_ref/#search_2) SDK reference docs for more information.

#### 스레드 상태로 필터링

`status` 필드를 사용하여 상태에 따라 스레드를 필터링합니다. 지원되는 값은 `idle`, `busy`, `interrupted` 및 `error`입니다. 각 상태에 대한 정보는 [여기](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=thread+status#langgraph_sdk.auth.types.ThreadStatus)를 참조하세요. 예를 들어 `idle` 스레드를 보려면:

=== "Python"

    ```python
    print(await client.threads.search(status="idle",limit=1))
    ```

=== "Javascript"

    ```js
    console.log(await client.threads.search({ status: "idle", limit: 1 }));
    ```

=== "CURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/search \
    --header 'Content-Type: application/json' \
    --data '{"status": "idle", "limit": 1}'
    ```

Output:

    [
      {
        'thread_id': 'cacf79bb-4248-4d01-aabc-938dbd60ed2c',
        'created_at': '2024-08-14T17:36:38.921660+00:00',
        'updated_at': '2024-08-14T17:36:38.921660+00:00',
        'metadata': {'graph_id': 'agent'},
        'status': 'idle',
        'config': {'configurable': {}}
      }
    ]

#### 메타데이터로 필터링

`search` 메서드를 사용하면 메타데이터를 기준으로 필터링할 수 있습니다:

=== "Python"

    ```python
    print((await client.threads.search(metadata={"graph_id":"agent"},limit=1)))
    ```

=== "Javascript"

    ```js
    console.log((await client.threads.search({ metadata: { "graph_id": "agent" }, limit: 1 })));
    ```

=== "CURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/search \
    --header 'Content-Type: application/json' \
    --data '{"metadata": {"graph_id":"agent"}, "limit": 1}'
    ```

Output:

    [
      {
        'thread_id': 'cacf79bb-4248-4d01-aabc-938dbd60ed2c',
        'created_at': '2024-08-14T17:36:38.921660+00:00',
        'updated_at': '2024-08-14T17:36:38.921660+00:00',
        'metadata': {'graph_id': 'agent'},
        'status': 'idle',
        'config': {'configurable': {}}
      }
    ]

#### 정렬

SDK는 `sort_by` 및 `sort_order` 파라미터를 사용하여 `thread_id`, `status`, `created_at` 및 `updated_at`로 스레드를 정렬하는 것도 지원합니다.

### LangGraph Platform UI

LangGraph Platform UI를 통해 배포의 스레드를 볼 수도 있습니다.

배포 내에서 "Threads" 탭을 선택하세요. 배포의 모든 스레드 테이블이 로드됩니다.

스레드 상태로 필터링하려면 상단 바에서 상태를 선택하세요. 지원되는 속성으로 정렬하려면 원하는 열의 화살표 아이콘을 클릭하세요.

## Inspect threads

### LangGraph SDK

#### 스레드 가져오기

`thread_id`가 주어진 특정 스레드를 보려면 `get` 메서드를 사용하세요:

=== "Python"

    ```python
    print((await client.threads.get(<THREAD_ID>)))
    ```

=== "Javascript"

    ```js
    console.log((await client.threads.get(<THREAD_ID>)));
    ```

=== "CURL"

    ```bash
    curl --request GET \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID> \
    --header 'Content-Type: application/json'
    ```

Output:

    {
      'thread_id': 'cacf79bb-4248-4d01-aabc-938dbd60ed2c',
      'created_at': '2024-08-14T17:36:38.921660+00:00',
      'updated_at': '2024-08-14T17:36:38.921660+00:00',
      'metadata': {'graph_id': 'agent'},
      'status': 'idle',
      'config': {'configurable': {}}
    }

#### 스레드 상태 검사

주어진 스레드의 현재 상태를 보려면 `get_state` 메서드를 사용하세요:

=== "Python"

    ```python
    print((await client.threads.get_state(<THREAD_ID>)))
    ```

=== "Javascript"

    ```js
    console.log((await client.threads.getState(<THREAD_ID>)));
    ```

=== "CURL"

    ```bash
    curl --request GET \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state \
    --header 'Content-Type: application/json'
    ```

Output:

    {
        "values": {
            "messages": [
                {
                    "content": "hello",
                    "additional_kwargs": {},
                    "response_metadata": {},
                    "type": "human",
                    "name": null,
                    "id": "8701f3be-959c-4b7c-852f-c2160699b4ab",
                    "example": false
                },
                {
                    "content": "Hello! How can I assist you today?",
                    "additional_kwargs": {},
                    "response_metadata": {},
                    "type": "ai",
                    "name": null,
                    "id": "4d8ea561-7ca1-409a-99f7-6b67af3e1aa3",
                    "example": false,
                    "tool_calls": [],
                    "invalid_tool_calls": [],
                    "usage_metadata": null
                }
            ]
        },
        "next": [],
        "tasks": [],
        "metadata": {
            "thread_id": "f15d70a1-27d4-4793-a897-de5609920b7d",
            "checkpoint_id": "1f02f46f-7308-616c-8000-1b158a9a6955",
            "graph_id": "agent_with_quite_a_long_name",
            "source": "update",
            "step": 1,
            "writes": {
                "call_model": {
                    "messages": [
                        {
                            "content": "Hello! How can I assist you today?",
                            "type": "ai"
                        }
                    ]
                }
            },
            "parents": {}
        },
        "created_at": "2025-05-12T15:37:09.008055+00:00",
        "checkpoint": {
            "checkpoint_id": "1f02f46f-733f-6b58-8001-ea90dcabb1bd",
            "thread_id": "f15d70a1-27d4-4793-a897-de5609920b7d",
            "checkpoint_ns": ""
        },
        "parent_checkpoint": {
            "checkpoint_id": "1f02f46f-7308-616c-8000-1b158a9a6955",
            "thread_id": "f15d70a1-27d4-4793-a897-de5609920b7d",
            "checkpoint_ns": ""
        },
        "checkpoint_id": "1f02f46f-733f-6b58-8001-ea90dcabb1bd",
        "parent_checkpoint_id": "1f02f46f-7308-616c-8000-1b158a9a6955"
    }

선택적으로 주어진 체크포인트에서 스레드의 상태를 보려면 체크포인트 ID(또는 전체 체크포인트 객체)를 전달하면 됩니다:

=== "Python"

    ```python
    thread_state = await client.threads.get_state(
      thread_id=<THREAD_ID>
      checkpoint_id=<CHECKPOINT_ID>
    )
    ```

=== "Javascript"

    ```js
    const threadState = await client.threads.getState(<THREAD_ID>, <CHECKPOINT_ID>);
    ```

=== "CURL"

    ```bash
    curl --request GET \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state/<CHECKPOINT_ID> \
    --header 'Content-Type: application/json'
    ```

#### 전체 Thread 히스토리 검사

thread의 히스토리를 보려면 `get_history` 메서드를 사용하세요. thread가 경험한 모든 상태의 목록을 반환합니다. 자세한 내용은 [Python](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=thread+status#langgraph_sdk.client.ThreadsClient.get_history) 및 [JS](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/js_ts_sdk_ref/#gethistory) 레퍼런스 문서를 참조하세요.

### LangGraph Platform UI

LangGraph Platform UI를 통해 배포의 thread를 볼 수도 있습니다.

배포 내에서 "Threads" 탭을 선택하세요. 배포의 모든 thread 테이블이 로드됩니다.

thread를 선택하여 현재 상태를 검사하세요. 전체 히스토리를 보고 추가 디버깅을 하려면 [LangGraph Studio](../../concepts//langgraph_studio.md)에서 thread를 여세요.
