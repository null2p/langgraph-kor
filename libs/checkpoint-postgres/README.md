# LangGraph Checkpoint Postgres

Postgres를 사용하는 LangGraph CheckpointSaver의 구현입니다.

## 의존성

기본적으로 `langgraph-checkpoint-postgres`는 추가 항목 없이 `psycopg` (Psycopg 3)를 설치합니다. 그러나 [여기](https://www.psycopg.org/psycopg3/docs/basic/install.html)에서 필요에 가장 적합한 특정 설치를 선택할 수 있습니다 (예: `psycopg[binary]`).

## 사용법

> [!IMPORTANT]
> Postgres 체크포인터를 처음 사용할 때는 필수 테이블을 생성하기 위해 `.setup()` 메서드를 반드시 호출하세요. 아래 예제를 참조하세요.

> [!IMPORTANT]
> Postgres 연결을 수동으로 생성하여 `PostgresSaver` 또는 `AsyncPostgresSaver`에 전달할 때는 `autocommit=True`와 `row_factory=dict_row` (`from psycopg.rows import dict_row`)를 반드시 포함하세요. 전체 예제는 이 [how-to 가이드](https://langchain-ai.github.io/langgraph/how-tos/persistence_postgres/)를 참조하세요.
>
> **이 매개변수들이 필요한 이유:**
> - `autocommit=True`: `.setup()` 메서드가 체크포인트 테이블을 데이터베이스에 올바르게 커밋하기 위해 필요합니다. 이것이 없으면 테이블 생성이 지속되지 않을 수 있습니다.
> - `row_factory=dict_row`: PostgresSaver 구현이 딕셔너리 스타일 구문(예: `row["column_name"]`)을 사용하여 데이터베이스 행에 액세스하기 때문에 필요합니다. 기본 `tuple_row` 팩토리는 인덱스 기반 액세스(예: `row[0]`)만 지원하는 튜플을 반환하므로, 체크포인터가 이름으로 열에 액세스하려고 할 때 `TypeError` 예외가 발생합니다.
>
> **잘못된 사용 예:**
> ```python
> # ❌ 체크포인터 작업 중 TypeError로 실패합니다
> with psycopg.connect(DB_URI) as conn:  # autocommit=True와 row_factory=dict_row가 누락됨
>     checkpointer = PostgresSaver(conn)
>     checkpointer.setup()  # 테이블이 제대로 지속되지 않을 수 있음
>     # 데이터베이스에서 읽는 모든 작업은 다음과 같이 실패합니다:
>     # TypeError: tuple indices must be integers or slices, not str
> ```

```python
from langgraph.checkpoint.postgres import PostgresSaver

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

DB_URI = "postgres://postgres:postgres@localhost:5432/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # 체크포인터를 처음 사용할 때 .setup()을 호출합니다
    checkpointer.setup()
    checkpoint = {
        "v": 4,
        "ts": "2024-07-31T20:14:19.804150+00:00",
        "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        "channel_values": {
            "my_key": "meow",
            "node": "node"
        },
        "channel_versions": {
            "__start__": 2,
            "my_key": 3,
            "start:node": 3,
            "node": 3
        },
        "versions_seen": {
            "__input__": {},
            "__start__": {
            "__start__": 1
            },
            "node": {
            "start:node": 2
            }
        },
    }

    # 체크포인트 저장
    checkpointer.put(write_config, checkpoint, {}, {})

    # 체크포인트 로드
    checkpointer.get(read_config)

    # 체크포인트 목록
    list(checkpointer.list(read_config))
```

### 비동기

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpoint = {
        "v": 4,
        "ts": "2024-07-31T20:14:19.804150+00:00",
        "id": "1ef4f797-8335-6428-8001-8a1503f9b875",
        "channel_values": {
            "my_key": "meow",
            "node": "node"
        },
        "channel_versions": {
            "__start__": 2,
            "my_key": 3,
            "start:node": 3,
            "node": 3
        },
        "versions_seen": {
            "__input__": {},
            "__start__": {
            "__start__": 1
            },
            "node": {
            "start:node": 2
            }
        },
    }

    # 체크포인트 저장
    await checkpointer.aput(write_config, checkpoint, {}, {})

    # 체크포인트 로드
    await checkpointer.aget(read_config)

    # 체크포인트 목록
    [c async for c in checkpointer.alist(read_config)]
```
