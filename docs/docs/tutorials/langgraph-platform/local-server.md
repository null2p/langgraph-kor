# 로컬 서버 실행하기

이 가이드는 LangGraph 애플리케이션을 로컬에서 실행하는 방법을 보여줍니다.

## 사전 요구 사항

시작하기 전에 다음 사항을 준비하세요:

- [LangSmith](https://smith.langchain.com/settings) API 키 - 무료로 가입 가능

## 1. LangGraph CLI 설치

:::python

```shell
# Python >= 3.11이 필요합니다.

pip install --upgrade "langgraph-cli[inmem]"
```

:::

:::js

```shell
npx @langchain/langgraph-cli
```

:::

## 2. LangGraph 앱 만들기 🌱

:::python
[`new-langgraph-project-python` 템플릿](https://github.com/langchain-ai/new-langgraph-project)에서 새 앱을 생성하세요. 이 템플릿은 자체 로직으로 확장할 수 있는 단일 노드 애플리케이션을 보여줍니다.

```shell
langgraph new path/to/your/app --template new-langgraph-project-python
```

!!! tip "추가 템플릿"

    템플릿을 지정하지 않고 `langgraph new`를 사용하면, 사용 가능한 템플릿 목록에서 선택할 수 있는 대화형 메뉴가 표시됩니다.

:::

:::js
[`new-langgraph-project-js` 템플릿](https://github.com/langchain-ai/new-langgraphjs-project)에서 새 앱을 생성하세요. 이 템플릿은 자체 로직으로 확장할 수 있는 단일 노드 애플리케이션을 보여줍니다.

```shell
npm create langgraph
```

:::

## 3. 종속성 설치

새 LangGraph 앱의 루트에서, 로컬 변경 사항이 서버에서 사용되도록 `edit` 모드로 종속성을 설치하세요:

:::python

```shell
cd path/to/your/app
pip install -e .
```

:::

:::js

```shell
cd path/to/your/app
npm install
```

:::

## 4. `.env` 파일 생성

새 LangGraph 앱의 루트에 `.env.example` 파일이 있습니다. 새 LangGraph 앱의 루트에 `.env` 파일을 생성하고 `.env.example` 파일의 내용을 복사한 후, 필요한 API 키를 입력하세요:

```bash
LANGSMITH_API_KEY=lsv2...
```

## 5. LangGraph Server 시작 🚀

LangGraph API 서버를 로컬에서 시작하세요:

:::python

```shell
langgraph dev
```

:::

:::js

```shell
npx @langchain/langgraph-cli dev
```

:::

샘플 출력:

```
>    Ready!
>
>    - API: [http://localhost:2024](http://localhost:2024/)
>
>    - Docs: http://localhost:2024/docs
>
>    - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

`langgraph dev` 명령은 LangGraph Server를 인메모리 모드로 시작합니다. 이 모드는 개발 및 테스트 목적에 적합합니다. 프로덕션 사용의 경우, 영구 스토리지 백엔드에 액세스할 수 있는 LangGraph Server를 배포하세요. 자세한 내용은 [배포 옵션](../../concepts/deployment_options.md)을 참조하세요.

## 6. LangGraph Studio에서 애플리케이션 테스트

[LangGraph Studio](../../concepts/langgraph_studio.md)는 LangGraph API 서버에 연결하여 애플리케이션을 로컬에서 시각화하고, 상호작용하고, 디버깅할 수 있는 특수한 UI입니다. `langgraph dev` 명령의 출력에서 제공된 URL을 방문하여 LangGraph Studio에서 그래프를 테스트하세요:

```
>    - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

사용자 지정 호스트/포트에서 실행되는 LangGraph Server의 경우, baseURL 파라미터를 업데이트하세요.

??? info "Safari 호환성"

    Safari에는 localhost 서버에 연결할 때 제한 사항이 있으므로 명령에 `--tunnel` 플래그를 사용하여 안전한 터널을 생성하세요:

    ```shell
    langgraph dev --tunnel
    ```

## 7. API 테스트

:::python
=== "Python SDK (async)"

    1. LangGraph Python SDK를 설치하세요:

        ```shell
        pip install langgraph-sdk
        ```

    1. 어시스턴트에게 메시지를 보내세요 (threadless run):

        ```python
        from langgraph_sdk import get_client
        import asyncio

        client = get_client(url="http://localhost:2024")

        async def main():
            async for chunk in client.runs.stream(
                None,  # Threadless run
                "agent", # 어시스턴트 이름. langgraph.json에 정의되어 있습니다.
                input={
                "messages": [{
                    "role": "human",
                    "content": "What is LangGraph?",
                    }],
                },
            ):
                print(f"Receiving new event of type: {chunk.event}...")
                print(chunk.data)
                print("\n\n")

        asyncio.run(main())
        ```

=== "Python SDK (sync)"

    1. LangGraph Python SDK를 설치하세요:

        ```shell
        pip install langgraph-sdk
        ```

    1. 어시스턴트에게 메시지를 보내세요 (threadless run):

        ```python
        from langgraph_sdk import get_sync_client

        client = get_sync_client(url="http://localhost:2024")

        for chunk in client.runs.stream(
            None,  # Threadless run
            "agent", # 어시스턴트 이름. langgraph.json에 정의되어 있습니다.
            input={
                "messages": [{
                    "role": "human",
                    "content": "What is LangGraph?",
                }],
            },
            stream_mode="messages-tuple",
        ):
            print(f"Receiving new event of type: {chunk.event}...")
            print(chunk.data)
            print("\n\n")
        ```

=== "Rest API"

    ```bash
    curl -s --request POST \
        --url "http://localhost:2024/runs/stream" \
        --header 'Content-Type: application/json' \
        --data "{
            \"assistant_id\": \"agent\",
            \"input\": {
                \"messages\": [
                    {
                        \"role\": \"human\",
                        \"content\": \"What is LangGraph?\"
                    }
                ]
            },
            \"stream_mode\": \"messages-tuple\"
        }"
    ```

:::

:::js
=== "Javascript SDK"

    1. LangGraph JS SDK를 설치하세요:

        ```shell
        npm install @langchain/langgraph-sdk
        ```

    1. 어시스턴트에게 메시지를 보내세요 (threadless run):

        ```js
        const { Client } = await import("@langchain/langgraph-sdk");

        // langgraph dev 호출 시 기본 포트를 변경한 경우에만 apiUrl을 설정하세요
        const client = new Client({ apiUrl: "http://localhost:2024"});

        const streamResponse = client.runs.stream(
            null, // Threadless run
            "agent", // 어시스턴트 ID
            {
                input: {
                    "messages": [
                        { "role": "user", "content": "What is LangGraph?"}
                    ]
                },
                streamMode: "messages-tuple",
            }
        );

        for await (const chunk of streamResponse) {
            console.log(`Receiving new event of type: ${chunk.event}...`);
            console.log(JSON.stringify(chunk.data));
            console.log("\n\n");
        }
        ```

=== "Rest API"

    ```bash
    curl -s --request POST \
        --url "http://localhost:2024/runs/stream" \
        --header 'Content-Type: application/json' \
        --data "{
            \"assistant_id\": \"agent\",
            \"input\": {
                \"messages\": [
                    {
                        \"role\": \"human\",
                        \"content\": \"What is LangGraph?\"
                    }
                ]
            },
            \"stream_mode\": \"messages-tuple\"
        }"
    ```

:::

## 다음 단계

이제 LangGraph 앱이 로컬에서 실행되고 있으니, 배포 및 고급 기능을 탐색하여 여정을 더 나아가세요:

- [배포 빠른 시작](../../cloud/quick_start.md): LangGraph Platform을 사용하여 LangGraph 앱을 배포하세요.
- [LangGraph Platform 개요](../../concepts/langgraph_platform.md): LangGraph Platform의 기본 개념을 알아보세요.
- [LangGraph Server API 참조](../../cloud/reference/api/api_ref.html): LangGraph Server API 문서를 살펴보세요.

:::python

- [Python SDK 참조](../../cloud/reference/sdk/python_sdk_ref.md): Python SDK API 참조를 살펴보세요.
  :::

:::js

- [JS/TS SDK 참조](../../cloud/reference/sdk/js_ts_sdk_ref.md): JS/TS SDK API 참조를 살펴보세요.
  :::
