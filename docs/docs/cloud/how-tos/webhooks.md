# 웹훅 사용

LangGraph Platform을 사용할 때 API 호출이 완료된 후 업데이트를 받기 위해 웹훅을 사용하고 싶을 수 있습니다. 웹훅은 실행이 처리를 완료한 후 서비스에서 작업을 트리거하는 데 유용합니다. 이를 구현하려면 `POST` 요청을 수락할 수 있는 엔드포인트를 노출하고 이 엔드포인트를 API 요청에서 `webhook` 파라미터로 전달해야 합니다.

현재 SDK는 웹훅 엔드포인트 정의를 위한 내장 지원을 제공하지 않지만 API 요청을 사용하여 수동으로 지정할 수 있습니다.

## 지원되는 엔드포인트

다음 API 엔드포인트는 `webhook` 파라미터를 허용합니다:

| Operation            | HTTP Method | Endpoint                          |
|----------------------|-------------|-----------------------------------|
| Create Run           | `POST`      | `/thread/{thread_id}/runs`        |
| Create Thread Cron   | `POST`      | `/thread/{thread_id}/runs/crons`  |
| Stream Run           | `POST`      | `/thread/{thread_id}/runs/stream` |
| Wait Run             | `POST`      | `/thread/{thread_id}/runs/wait`   |
| Create Cron          | `POST`      | `/runs/crons`                     |
| Stream Run Stateless | `POST`      | `/runs/stream`                    |
| Wait Run Stateless   | `POST`      | `/runs/wait`                      |

이 가이드에서는 실행 스트리밍 후 웹훅을 트리거하는 방법을 보여줍니다.

## 어시스턴트 및 스레드 설정

API 호출을 하기 전에 어시스턴트와 스레드를 설정하세요.

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    assistant_id = "agent"
    thread = await client.threads.create()
    print(thread)
    ```

=== "JavaScript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    const assistantID = "agent";
    const thread = await client.threads.create();
    console.log(thread);
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/assistants/search \
        --header 'Content-Type: application/json' \
        --data '{ "limit": 10, "offset": 0 }' | jq -c 'map(select(.config == null or .config == {})) | .[0]' && \
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads \
        --header 'Content-Type: application/json' \
        --data '{}'
    ```

Example response:

```json
{
    "thread_id": "9dde5490-2b67-47c8-aa14-4bfec88af217",
    "created_at": "2024-08-30T23:07:38.242730+00:00",
    "updated_at": "2024-08-30T23:07:38.242730+00:00",
    "metadata": {},
    "status": "idle",
    "config": {},
    "values": null
}
```

## 그래프 실행에서 웹훅 사용

웹훅을 사용하려면 API 요청에서 `webhook` 파라미터를 지정하세요. 실행이 완료되면 LangGraph Platform은 지정된 웹훅 URL로 `POST` 요청을 보냅니다.

예를 들어 서버가 `https://my-server.app/my-webhook-endpoint`에서 웹훅 이벤트를 수신하는 경우, 요청에 다음을 포함하세요:

=== "Python"

    ```python
    input = { "messages": [{ "role": "user", "content": "Hello!" }] }

    async for chunk in client.runs.stream(
        thread_id=thread["thread_id"],
        assistant_id=assistant_id,
        input=input,
        stream_mode="events",
        webhook="https://my-server.app/my-webhook-endpoint"
    ):
        pass
    ```

=== "JavaScript"

    ```js
    const input = { messages: [{ role: "human", content: "Hello!" }] };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantID,
      {
        input: input,
        webhook: "https://my-server.app/my-webhook-endpoint"
      }
    );

    for await (const chunk of streamResponse) {
      // Handle stream output
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
        --header 'Content-Type: application/json' \
        --data '{
            "assistant_id": <ASSISTANT_ID>,
            "input": {"messages": [{"role": "user", "content": "Hello!"}]},
            "webhook": "https://my-server.app/my-webhook-endpoint"
        }'
    ```

## 웹훅 페이로드

LangGraph Platform은 [Run](../../concepts/assistants.md#execution) 형식으로 웹훅 알림을 보냅니다. 자세한 내용은 [API 레퍼런스](https://langchain-ai.github.io/langgraph/cloud/reference/api/api_ref.html#model/run)를 참조하세요. 요청 페이로드에는 `kwargs` 필드에 실행 입력, 구성 및 기타 메타데이터가 포함됩니다.

## 웹훅 보안

권한이 부여된 요청만 웹훅 엔드포인트에 도달하도록 하려면 쿼리 매개변수로 보안 토큰을 추가하는 것을 고려하세요:

```
https://my-server.app/my-webhook-endpoint?token=YOUR_SECRET_TOKEN
```

서버는 요청을 처리하기 전에 이 토큰을 추출하고 검증해야 합니다.

## 웹훅 비활성화

`langgraph-api>=0.2.78`부터 개발자는 `langgraph.json` 파일에서 웹훅을 비활성화할 수 있습니다:

```json
{
  "http": {
    "disable_webhooks": true
  }
}
```

이 기능은 주로 자체 호스팅 배포를 위한 것으로, 플랫폼 관리자나 개발자가 보안 태세를 단순화하기 위해 웹훅을 비활성화하는 것을 선호할 수 있습니다. 특히 방화벽 규칙이나 기타 네트워크 제어를 구성하지 않는 경우 더욱 그렇습니다. 웹훅을 비활성화하면 신뢰할 수 없는 페이로드가 내부 엔드포인트로 전송되는 것을 방지하는 데 도움이 됩니다.

전체 구성 세부 정보는 [구성 파일 레퍼런스](https://langchain-ai.github.io/langgraph/cloud/reference/cli/?h=disable_webhooks#configuration-file)를 참조하세요.

## 웹훅 테스트

다음과 같은 온라인 서비스를 사용하여 웹훅을 테스트할 수 있습니다:

- **[Beeceptor](https://beeceptor.com/)** – 테스트 엔드포인트를 빠르게 생성하고 들어오는 웹훅 페이로드를 검사합니다.
- **[Webhook.site](https://webhook.site/)** – 실시간으로 들어오는 웹훅 요청을 보고, 디버그하고, 기록합니다.

이러한 도구는 LangGraph Platform이 올바르게 웹훅을 트리거하고 서비스로 전송하는지 확인하는 데 도움이 됩니다.
