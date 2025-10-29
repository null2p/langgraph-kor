# 배포 빠른 시작

이 가이드는 클라우드 배포를 위해 LangGraph Platform을 설정하고 사용하는 방법을 보여줍니다.

## 사전 요구 사항

시작하기 전에 다음이 있는지 확인하세요:

- [GitHub 계정](https://github.com/)
- [LangSmith 계정](https://smith.langchain.com/) – 무료로 가입 가능

## 1. GitHub에서 리포지토리 생성

**LangGraph Platform**에 애플리케이션을 배포하려면 애플리케이션 코드가 GitHub 리포지토리에 있어야 합니다. 공개 및 비공개 리포지토리가 모두 지원됩니다. 이 빠른 시작에서는 애플리케이션에 [`new-langgraph-project` 템플릿](https://github.com/langchain-ai/react-agent)을 사용하세요:

1. [`new-langgraph-project` 리포지토리](https://github.com/langchain-ai/new-langgraph-project) 또는 [`new-langgraphjs-project` 템플릿](https://github.com/langchain-ai/new-langgraphjs-project)으로 이동합니다.
1. 오른쪽 상단 모서리의 `Fork` 버튼을 클릭하여 리포지토리를 GitHub 계정으로 포크합니다.
1. **Create fork**를 클릭합니다.

## 2. LangGraph Platform에 배포

1. [LangSmith](https://smith.langchain.com/)에 로그인합니다.
1. 왼쪽 사이드바에서 **Deployments**를 선택합니다.
1. **+ New Deployment** 버튼을 클릭합니다. 필수 필드를 입력할 수 있는 패널이 열립니다.
1. 처음 사용하는 사용자이거나 이전에 연결되지 않은 비공개 리포지토리를 추가하는 경우 **Import from GitHub** 버튼을 클릭하고 지침에 따라 GitHub 계정을 연결합니다.
1. New LangGraph Project 리포지토리를 선택합니다.
1. **Submit**을 클릭하여 배포합니다.

    완료하는 데 약 15분이 걸릴 수 있습니다. **Deployment details** 보기에서 상태를 확인할 수 있습니다.

## 3. LangGraph Studio에서 애플리케이션 테스트

애플리케이션이 배포되면:

1. 방금 생성한 배포를 선택하여 자세한 내용을 확인합니다.
1. 오른쪽 상단 모서리의 **LangGraph Studio** 버튼을 클릭합니다.

    LangGraph Studio가 열려 그래프가 표시됩니다.

    <figure markdown="1">
    [![image](deployment/img/langgraph_studio.png){: style="max-height:400px"}](deployment/img/langgraph_studio.png)
    <figcaption>
        LangGraph Studio에서 실행되는 샘플 그래프.
    </figcaption>
    </figure>

## 4. 배포의 API URL 가져오기

1. LangGraph의 **Deployment details** 보기에서 **API URL**을 클릭하여 클립보드에 복사합니다.
1. `URL`을 클릭하여 클립보드에 복사합니다.

## 5. API 테스트

이제 API를 테스트할 수 있습니다:

=== "Python SDK (Async)"

    1. LangGraph Python SDK를 설치하세요:

        ```shell
        pip install langgraph-sdk
        ```

    1. 어시스턴트에게 메시지를 보내세요 (스레드 없는 실행):

        ```python
        from langgraph_sdk import get_client

        client = get_client(url="your-deployment-url", api_key="your-langsmith-api-key")

        async for chunk in client.runs.stream(
            None,  # Threadless run
            "agent", # Name of assistant. Defined in langgraph.json.
            input={
                "messages": [{
                    "role": "human",
                    "content": "What is LangGraph?",
                }],
            },
            stream_mode="updates",
        ):
            print(f"Receiving new event of type: {chunk.event}...")
            print(chunk.data)
            print("\n\n")
        ```

=== "Python SDK (Sync)"

    1. LangGraph Python SDK를 설치하세요:

        ```shell
        pip install langgraph-sdk
        ```

    1. 어시스턴트에게 메시지를 보내세요 (스레드 없는 실행):

        ```python
        from langgraph_sdk import get_sync_client

        client = get_sync_client(url="your-deployment-url", api_key="your-langsmith-api-key")

        for chunk in client.runs.stream(
            None,  # Threadless run
            "agent", # Name of assistant. Defined in langgraph.json.
            input={
                "messages": [{
                    "role": "human",
                    "content": "What is LangGraph?",
                }],
            },
            stream_mode="updates",
        ):
            print(f"Receiving new event of type: {chunk.event}...")
            print(chunk.data)
            print("\n\n")
        ```

=== "JavaScript SDK"

    1. LangGraph JS SDK를 설치하세요

        ```shell
        npm install @langchain/langgraph-sdk
        ```

    1. 어시스턴트에게 메시지를 보내세요 (스레드 없는 실행):

        ```js
        const { Client } = await import("@langchain/langgraph-sdk");

        const client = new Client({ apiUrl: "your-deployment-url", apiKey: "your-langsmith-api-key" });

        const streamResponse = client.runs.stream(
            null, // Threadless run
            "agent", // Assistant ID
            {
                input: {
                    "messages": [
                        { "role": "user", "content": "What is LangGraph?"}
                    ]
                },
                streamMode: "messages",
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
        --url <DEPLOYMENT_URL>/runs/stream \
        --header 'Content-Type: application/json' \
        --header "X-Api-Key: <LANGSMITH API KEY> \
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
            \"stream_mode\": \"updates\"
        }" 
    ```


## 다음 단계

축하합니다! LangGraph Platform을 사용하여 애플리케이션을 배포했습니다.

확인해볼 다른 리소스는 다음과 같습니다:

- [LangGraph Platform overview](../concepts/langgraph_platform.md)
- [Deployment options](../concepts/deployment_options.md)


