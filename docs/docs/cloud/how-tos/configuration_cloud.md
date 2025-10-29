# 어시스턴트 관리

이 가이드에서는 [어시스턴트](../../concepts/assistants.md)를 생성, 구성 및 관리하는 방법을 보여줍니다.

먼저 런타임 컨텍스트 개념에 대한 간단한 복습으로 다음의 간단한 `call_model` 노드와 컨텍스트 스키마를 고려하세요. 이 노드는 `Runtime` 객체의 `context` 속성으로 정의된 `model_provider`를 읽고 사용하려고 시도하는 것을 확인하세요.

=== "Python"

    ```python
    @dataclass
    class ContextSchema:
        llm_provider: str = "anthropic"

    builder = StateGraph(AgentState, context_schema=ContextSchema)

    def call_model(state, runtime: Runtime[ContextSchema]):
        messages = state["messages"]
        model = _get_model(runtime.context.llm_provider)
        response = model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}
    ```

=== "Javascript"

    ```js
    import { Annotation } from "@langchain/langgraph";

    const ConfigSchema = Annotation.Root({
        model_name: Annotation<string>,
        system_prompt:
    });

    const builder = new StateGraph(AgentState, ConfigSchema)

    function callModel(state: State, config: RunnableConfig) {
      const messages = state.messages;
      const modelName = config.configurable?.model_name ?? "anthropic";
      const model = _getModel(modelName);
      const response = model.invoke(messages);
      // We return a list, because this will get added to the existing list
      return { messages: [response] };
    }
    ```

:::python
런타임 컨텍스트에 대한 자세한 내용은 [여기](../../concepts/low_level.md#runtime-context)를 참조하세요.
:::

## 어시스턴트 생성

### LangGraph SDK

어시스턴트를 생성하려면 [LangGraph SDK](../../concepts/sdk.md)의 `create` 메서드를 사용하세요. 자세한 내용은 [Python](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.AssistantsClient.create) 및 [JS](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/js_ts_sdk_ref/#create) SDK 레퍼런스 문서를 참조하세요.

이 예제는 위와 동일한 구성 스키마를 사용하며, `model_name`을 `openai`로 설정한 어시스턴트를 생성합니다.

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    openai_assistant = await client.assistants.create(
        # "agent" is the name of a graph we deployed
        "agent", config={"configurable": {"model_name": "openai"}}, name="Open AI Assistant"
    )

    print(openai_assistant)
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    const openAIAssistant = await client.assistants.create({
        graphId: 'agent',
        name: "Open AI Assistant",
        config: { "configurable": { "model_name": "openai" } },
    });

    console.log(openAIAssistant);
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/assistants \
        --header 'Content-Type: application/json' \
        --data '{"graph_id":"agent", "config":{"configurable":{"model_name":"openai"}}, "name": "Open AI Assistant"}'
    ```

출력:

    {
        "assistant_id": "62e209ca-9154-432a-b9e9-2d75c7a9219b",
        "graph_id": "agent",
        "name": "Open AI Assistant"
        "config": {
            "configurable": {
                "model_name": "openai"
            }
        },
        "metadata": {}
        "created_at": "2024-08-31T03:09:10.230718+00:00",
        "updated_at": "2024-08-31T03:09:10.230718+00:00",
    }

### LangGraph Platform UI

LangGraph Platform UI에서도 어시스턴트를 생성할 수 있습니다.

배포 내에서 "Assistants" 탭을 선택하세요. 모든 그래프에 걸쳐 배포의 모든 어시스턴트 테이블이 로드됩니다.

새 어시스턴트를 생성하려면 "+ New assistant" 버튼을 선택하세요. 이렇게 하면 이 어시스턴트가 사용할 그래프를 지정하고, 이름, 설명 및 해당 그래프의 구성 스키마를 기반으로 어시스턴트에 대한 원하는 구성을 제공할 수 있는 양식이 열립니다.

확인하려면 "Create assistant"를 클릭하세요. 그러면 어시스턴트를 테스트할 수 있는 [LangGraph Studio](../../concepts/langgraph_studio.md)로 이동합니다. 배포의 "Assistants" 탭으로 돌아가면 테이블에 새로 생성된 어시스턴트가 표시됩니다.

## 어시스턴트 사용

### LangGraph SDK

이제 `model_name`이 `openai`로 정의된 "Open AI Assistant"라는 어시스턴트를 생성했습니다. 이제 이 구성으로 이 어시스턴트를 사용할 수 있습니다:

=== "Python"

    ```python
    thread = await client.threads.create()
    input = {"messages": [{"role": "user", "content": "who made you?"}]}
    async for event in client.runs.stream(
        thread["thread_id"],
        # this is where we specify the assistant id to use
        openai_assistant["assistant_id"],
        input=input,
        stream_mode="updates",
    ):
        print(f"Receiving event of type: {event.event}")
        print(event.data)
        print("\n\n")
    ```

=== "Javascript"

    ```js
    const thread = await client.threads.create();
    const input = { "messages": [{ "role": "user", "content": "who made you?" }] };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      // this is where we specify the assistant id to use
      openAIAssistant["assistant_id"],
      {
        input,
        streamMode: "updates"
      }
    );

    for await (const event of streamResponse) {
      console.log(`Receiving event of type: ${event.event}`);
      console.log(event.data);
      console.log("\n\n");
    }
    ```

=== "CURL"

    ```bash
    thread_id=$(curl --request POST \
        --url <DEPLOYMENT_URL>/threads \
        --header 'Content-Type: application/json' \
        --data '{}' | jq -r '.thread_id') && \
    curl --request POST \
        --url "<DEPLOYMENT_URL>/threads/${thread_id}/runs/stream" \
        --header 'Content-Type: application/json' \
        --data '{
            "assistant_id": <OPENAI_ASSISTANT_ID>,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "who made you?"
                    }
                ]
            },
            "stream_mode": [
                "updates"
            ]
        }' | \
        sed 's/\r$//' | \
        awk '
        /^event:/ {
            if (data_content != "") {
                print data_content "\n"
            }
            sub(/^event: /, "Receiving event of type: ", $0)
            printf "%s...\n", $0
            data_content = ""
        }
        /^data:/ {
            sub(/^data: /, "", $0)
            data_content = $0
        }
        END {
            if (data_content != "") {
                print data_content "\n\n"
            }
        }
    '
    ```

출력:

    ```
    Receiving event of type: metadata
    {'run_id': '1ef6746e-5893-67b1-978a-0f1cd4060e16'}



    Receiving event of type: updates
    {'agent': {'messages': [{'content': 'I was created by OpenAI, a research organization focused on developing and advancing artificial intelligence technology.', 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5'}, 'type': 'ai', 'name': None, 'id': 'run-e1a6b25c-8416-41f2-9981-f9cfe043f414', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}
    ```

### LangGraph Platform UI

배포 내에서 "Assistants" 탭을 선택하세요. 사용하려는 어시스턴트의 "Studio" 버튼을 클릭하세요. 그러면 선택한 어시스턴트와 함께 LangGraph Studio가 열립니다. (Graph 또는 Chat 모드에서) 입력을 제출하면 선택한 어시스턴트와 해당 구성이 사용됩니다.

## 어시스턴트의 새 버전 생성

### LangGraph SDK

어시스턴트를 편집하려면 `update` 메서드를 사용하세요. 제공된 편집 내용으로 어시스턴트의 새 버전이 생성됩니다. 자세한 내용은 [Python](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/#langgraph_sdk.client.AssistantsClient.update) 및 [JS](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/js_ts_sdk_ref/#update) SDK 레퍼런스 문서를 참조하세요.

!!! note "참고"

    전체 config(및 사용 중인 경우 metadata)를 전달해야 합니다. update 엔드포인트는 이전 버전에 의존하지 않고 처음부터 완전히 새로운 버전을 생성합니다.

예를 들어, 어시스턴트의 시스템 프롬프트를 업데이트하려면:

=== "Python"

    ```python
    openai_assistant_v2 = await client.assistants.update(
        openai_assistant["assistant_id"],
        config={
            "configurable": {
                "model_name": "openai",
                "system_prompt": "You are an unhelpful assistant!",
            }
        },
    )
    ```

=== "Javascript"

    ```js
    const openaiAssistantV2 = await client.assistants.update(
        openai_assistant["assistant_id"],
        {
            config: {
                configurable: {
                    model_name: 'openai',
                    system_prompt: 'You are an unhelpful assistant!',
                },
        },
    });
    ```

=== "CURL"

    ```bash
    curl --request PATCH \
    --url <DEPOLYMENT_URL>/assistants/<ASSISTANT_ID> \
    --header 'Content-Type: application/json' \
    --data '{
    "config": {"model_name": "openai", "system_prompt": "You are an unhelpful assistant!"}
    }'
    ```

이렇게 하면 업데이트된 파라미터로 어시스턴트의 새 버전이 생성되고 이것이 어시스턴트의 활성 버전으로 설정됩니다. 이제 그래프를 실행하고 이 어시스턴트 id를 전달하면 이 최신 버전을 사용합니다.

### LangGraph Platform UI

LangGraph Platform UI에서도 어시스턴트를 편집할 수 있습니다.

배포 내에서 "Assistants" 탭을 선택하세요. 모든 그래프에 걸쳐 배포의 모든 어시스턴트 테이블이 로드됩니다.

기존 어시스턴트를 편집하려면 지정된 어시스턴트의 "Edit" 버튼을 선택하세요. 그러면 어시스턴트의 이름, 설명 및 구성을 편집할 수 있는 양식이 열립니다.

또한 LangGraph Studio를 사용하는 경우 "Manage Assistants" 버튼을 통해 어시스턴트를 편집하고 새 버전을 만들 수 있습니다.

## 이전 어시스턴트 버전 사용

### LangGraph SDK

어시스턴트의 활성 버전을 변경할 수도 있습니다. 그렇게 하려면 `setLatest` 메서드를 사용하세요.

위의 예제에서 어시스턴트의 첫 번째 버전으로 롤백하려면:

=== "Python"

    ```python
    await client.assistants.set_latest(openai_assistant['assistant_id'], 1)
    ```

=== "Javascript"

    ```js
    await client.assistants.setLatest(openaiAssistant['assistant_id'], 1);
    ```

=== "CURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/assistants/<ASSISTANT_ID>/latest \
    --header 'Content-Type: application/json' \
    --data '{
    "version": 1
    }'
    ```

이제 그래프를 실행하고 이 어시스턴트 id를 전달하면 어시스턴트의 첫 번째 버전을 사용합니다.

### LangGraph Platform UI

LangGraph Studio를 사용하는 경우 어시스턴트의 활성 버전을 설정하려면 "Manage Assistants" 버튼을 클릭하고 사용하려는 어시스턴트를 찾으세요. 어시스턴트와 버전을 선택한 다음 "Active" 토글을 클릭하세요. 그러면 선택한 버전이 활성화되도록 어시스턴트가 업데이트됩니다.

!!! warning "어시스턴트 삭제"
어시스턴트를 삭제하면 모든 버전이 삭제됩니다. 현재 단일 버전을 삭제할 수 있는 방법은 없지만, 어시스턴트를 올바른 버전으로 지정하면 사용하지 않으려는 버전을 건너뛸 수 있습니다.
