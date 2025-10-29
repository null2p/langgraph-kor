# cron 작업 사용

때때로 사용자 상호작용을 기반으로 그래프를 실행하는 것이 아니라 일정에 따라 그래프가 실행되도록 예약하고 싶을 수 있습니다 - 예를 들어 그래프가 팀의 할 일 목록을 작성하여 주간 이메일로 보내도록 하고 싶을 때입니다. LangGraph Platform을 사용하면 `Crons` 클라이언트를 사용하여 자체 스크립트를 작성할 필요 없이 이를 수행할 수 있습니다. 그래프 작업을 예약하려면 [cron 표현식](https://crontab.cronhub.io/)을 전달하여 클라이언트에게 그래프를 실행할 시기를 알려야 합니다. `Cron` 작업은 백그라운드에서 실행되며 그래프의 정상적인 호출을 방해하지 않습니다.

## 설정

먼저 SDK 클라이언트, 어시스턴트 및 스레드를 설정합니다:

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    # create thread
    thread = await client.threads.create()
    print(thread)
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantId = "agent";
    // create thread
    const thread = await client.threads.create();
    console.log(thread);
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/assistants/search \
        --header 'Content-Type: application/json' \
        --data '{
            "limit": 10,
            "offset": 0
        }' | jq -c 'map(select(.config == null or .config == {})) | .[0].graph_id' && \
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads \
        --header 'Content-Type: application/json' \
        --data '{}'
    ```

출력:

    {
        'thread_id': '9dde5490-2b67-47c8-aa14-4bfec88af217',
        'created_at': '2024-08-30T23:07:38.242730+00:00',
        'updated_at': '2024-08-30T23:07:38.242730+00:00',
        'metadata': {},
        'status': 'idle',
        'config': {},
        'values': None
    }

## 스레드에서 Cron 작업 생성

특정 스레드와 연결된 cron 작업을 생성하려면 다음과 같이 작성할 수 있습니다:


=== "Python"

    ```python
    # This schedules a job to run at 15:27 (3:27PM) every day
    cron_job = await client.crons.create_for_thread(
        thread["thread_id"],
        assistant_id,
        schedule="27 15 * * *",
        input={"messages": [{"role": "user", "content": "What time is it?"}]},
    )
    ```

=== "Javascript"

    ```js
    // This schedules a job to run at 15:27 (3:27PM) every day
    const cronJob = await client.crons.create_for_thread(
      thread["thread_id"],
      assistantId,
      {
        schedule: "27 15 * * *",
        input: { messages: [{ role: "user", content: "What time is it?" }] }
      }
    );
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/crons \
        --header 'Content-Type: application/json' \
        --data '{
            "assistant_id": <ASSISTANT_ID>,
        }'
    ```

더 이상 유용하지 않은 `Cron` 작업을 삭제하는 것은 **매우** 중요합니다. 그렇지 않으면 LLM에 대한 원치 않는 API 요금이 발생할 수 있습니다! 다음 코드를 사용하여 `Cron` 작업을 삭제할 수 있습니다:

=== "Python"

    ```python
    await client.crons.delete(cron_job["cron_id"])
    ```

=== "Javascript"

    ```js
    await client.crons.delete(cronJob["cron_id"]);
    ```

=== "CURL"

    ```bash
    curl --request DELETE \
        --url <DEPLOYMENT_URL>/runs/crons/<CRON_ID>
    ```

## 상태 비저장 Cron 작업

다음 코드를 사용하여 상태 비저장 cron 작업을 생성할 수도 있습니다:

=== "Python"

    ```python
    # This schedules a job to run at 15:27 (3:27PM) every day
    cron_job_stateless = await client.crons.create(
        assistant_id,
        schedule="27 15 * * *",
        input={"messages": [{"role": "user", "content": "What time is it?"}]},
    )
    ```

=== "Javascript"

    ```js
    // This schedules a job to run at 15:27 (3:27PM) every day
    const cronJobStateless = await client.crons.create(
      assistantId,
      {
        schedule: "27 15 * * *",
        input: { messages: [{ role: "user", content: "What time is it?" }] }
      }
    );
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/runs/crons \
        --header 'Content-Type: application/json' \
        --data '{
            "assistant_id": <ASSISTANT_ID>,
        }'
    ```

다시 한번 강조하지만, 작업을 완료한 후에는 반드시 삭제하는 것을 잊지 마세요!

=== "Python"

    ```python
    await client.crons.delete(cron_job_stateless["cron_id"])
    ```

=== "Javascript"

    ```js
    await client.crons.delete(cronJobStateless["cron_id"]);
    ```

=== "CURL"

    ```bash
    curl --request DELETE \
        --url <DEPLOYMENT_URL>/runs/crons/<CRON_ID>
    ```
