---
search:
  boost: 2
---

# LangGraph SDK

:::python
LangGraph Platform은 [LangGraph Server](./langgraph_server.md)와 상호작용하기 위한 Python SDK를 제공합니다.

!!! tip "Python SDK 레퍼런스"

    Python SDK에 대한 자세한 정보는 [Python SDK 레퍼런스 문서](../cloud/reference/sdk/python_sdk_ref.md)를 참조하세요.

## 설치

다음 명령을 사용하여 LangGraph SDK를 설치할 수 있습니다:

```bash
pip install langgraph-sdk
```

## Python sync vs. async

Python SDK는 LangGraph Server와 상호작용하기 위한 동기(`get_sync_client`) 및 비동기(`get_client`) 클라이언트를 모두 제공합니다:

=== "Sync"

    ```python
    from langgraph_sdk import get_sync_client

    client = get_sync_client(url=..., api_key=...)
    client.assistants.search()
    ```

=== "Async"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=..., api_key=...)
    await client.assistants.search()
    ```

## 더 알아보기

- [Python SDK 레퍼런스](../cloud/reference/sdk/python_sdk_ref.md)
- [LangGraph CLI API 레퍼런스](../cloud/reference/cli.md)
  :::

:::js
LangGraph Platform은 [LangGraph Server](./langgraph_server.md)와 상호작용하기 위한 JS/TS SDK를 제공합니다.

## 설치

다음 명령을 사용하여 프로젝트에 LangGraph SDK를 추가할 수 있습니다:

```bash
npm install @langchain/langgraph-sdk
```

## 더 알아보기

- [LangGraph CLI API 레퍼런스](../cloud/reference/cli.md)
  :::
