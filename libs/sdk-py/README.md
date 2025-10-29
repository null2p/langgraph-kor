# LangGraph Python SDK

이 리포지토리는 LangSmith Deployment REST API와 상호작용하기 위한 Python SDK를 포함합니다.

## 빠른 시작

Python SDK를 시작하려면 [패키지를 설치](https://pypi.org/project/langgraph-sdk/)하세요

```bash
pip install -U langgraph-sdk
```

실행 중인 LangGraph API 서버가 필요합니다. `langgraph-cli`를 사용하여 로컬에서 서버를 실행하는 경우 SDK는 자동으로 `http://localhost:8123`을 가리킵니다. 그렇지 않으면 클라이언트를 생성할 때 서버 URL을 지정해야 합니다.

```python
from langgraph_sdk import get_client

# 원격 서버를 사용하는 경우 `get_client(url=REMOTE_URL)`로 클라이언트를 초기화합니다
client = get_client()

# 모든 어시스턴트를 조회합니다
assistants = await client.assistants.search()

# config에 등록한 각 그래프에 대해 어시스턴트가 자동으로 생성됩니다.
agent = assistants[0]

# 새 스레드를 시작합니다
thread = await client.threads.create()

# 스트리밍 실행을 시작합니다
input = {"messages": [{"role": "human", "content": "what's the weather in la"}]}
async for chunk in client.runs.stream(thread['thread_id'], agent['assistant_id'], input=input):
    print(chunk)
```
