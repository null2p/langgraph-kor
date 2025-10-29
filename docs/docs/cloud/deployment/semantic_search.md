# LangGraph 배포에 semantic search 추가 방법

이 가이드는 LangGraph 배포의 cross-thread [store](../../concepts/persistence.md#memory-store)에 semantic search를 추가하는 방법을 설명하여, 에이전트가 의미론적 유사성을 기준으로 메모리 및 기타 문서를 검색할 수 있도록 합니다.

## Prerequisites

- LangGraph 배포 ([배포 방법](setup_pyproject.md) 참조)
- 임베딩 제공자를 위한 API 키 (이 경우 OpenAI)
- `langchain >= 0.3.8` (아래 문자열 형식을 사용하여 지정하는 경우)

## Steps

1. `langgraph.json` 구성 파일을 업데이트하여 store 구성을 포함합니다:

```json
{
    ...
    "store": {
        "index": {
            "embed": "openai:text-embedding-3-small",
            "dims": 1536,
            "fields": ["$"]
        }
    }
}
```

이 구성은:

- 임베딩 생성을 위해 OpenAI의 text-embedding-3-small 모델을 사용합니다
- 임베딩 차원을 1536으로 설정합니다 (모델의 출력과 일치)
- 저장된 데이터의 모든 필드를 인덱싱합니다 (`["$"]`는 모든 것을 인덱싱한다는 의미이며, `["text", "metadata.title"]`과 같이 특정 필드를 지정할 수도 있습니다)

2. 위의 문자열 임베딩 형식을 사용하려면 종속성에 `langchain >= 0.3.8`이 포함되어 있는지 확인하세요:

```toml
# In pyproject.toml
[project]
dependencies = [
    "langchain>=0.3.8"
]
```

또는 requirements.txt를 사용하는 경우:

```
langchain>=0.3.8
```

## Usage

구성이 완료되면 LangGraph 노드에서 semantic search를 사용할 수 있습니다. store는 메모리를 구성하기 위해 namespace 튜플이 필요합니다:

```python
def search_memory(state: State, *, store: BaseStore):
    # Search the store using semantic similarity
    # The namespace tuple helps organize different types of memories
    # e.g., ("user_facts", "preferences") or ("conversation", "summaries")
    results = store.search(
        namespace=("memory", "facts"),  # Organize memories by type
        query="your search query",
        limit=3  # number of results to return
    )
    return results
```

## Custom Embeddings

커스텀 임베딩을 사용하려면 커스텀 임베딩 함수의 경로를 전달할 수 있습니다:

```json
{
    ...
    "store": {
        "index": {
            "embed": "path/to/embedding_function.py:embed",
            "dims": 1536,
            "fields": ["$"]
        }
    }
}
```

배포는 지정된 경로에서 함수를 찾습니다. 함수는 비동기여야 하며 문자열 리스트를 받아야 합니다:

```python
# path/to/embedding_function.py
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def aembed_texts(texts: list[str]) -> list[list[float]]:
    """Custom embedding function that must:
    1. Be async
    2. Accept a list of strings
    3. Return a list of float arrays (embeddings)
    """
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [e.embedding for e in response.data]
```

## Querying via the API

LangGraph SDK를 사용하여 store를 쿼리할 수도 있습니다. SDK는 비동기 작업을 사용하므로:

```python
from langgraph_sdk import get_client

async def search_store():
    client = get_client()
    results = await client.store.search_items(
        ("memory", "facts"),
        query="your search query",
        limit=3  # number of results to return
    )
    return results

# Use in an async context
results = await search_store()
```
