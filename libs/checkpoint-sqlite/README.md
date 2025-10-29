# LangGraph SQLite Checkpoint

SQLite DB를 사용하는 LangGraph CheckpointSaver의 구현입니다 (동기 및 비동기 모두, `aiosqlite`를 통해)

## 사용법

```python
from langgraph.checkpoint.sqlite import SqliteSaver

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

with SqliteSaver.from_conn_string(":memory:") as checkpointer:
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

    # 체크포인트 목록 조회
    list(checkpointer.list(read_config))
```

### 비동기

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
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

    # 체크포인트 목록 조회
    [c async for c in checkpointer.alist(read_config)]
```
