# LangGraph Checkpoint

이 라이브러리는 LangGraph 체크포인터의 기본 인터페이스를 정의합니다. 체크포인터는 LangGraph에 지속성 계층을 제공합니다. 체크포인터를 사용하면 그래프의 상태와 상호작용하고 관리할 수 있습니다. 체크포인터와 함께 그래프를 사용하면 체크포인터가 매 슈퍼스텝마다 그래프 상태의 _체크포인트_를 저장하여 휴먼-인-더-루프, 상호작용 간 "메모리" 등과 같은 여러 강력한 기능을 활성화합니다.

## 주요 개념

### 체크포인트

체크포인트는 특정 시점의 그래프 상태 스냅샷입니다. 체크포인트 튜플은 체크포인트와 관련된 설정, 메타데이터 및 대기 중인 쓰기 작업을 포함하는 객체를 의미합니다.

### 스레드

스레드는 여러 다른 실행의 체크포인팅을 가능하게 하며, 다중 테넌트 채팅 애플리케이션 및 별도의 상태 유지가 필요한 기타 시나리오에 필수적입니다. 스레드는 체크포인터가 저장하는 일련의 체크포인트에 할당된 고유 ID입니다. 체크포인터를 사용할 때 그래프를 실행할 때 `thread_id`와 선택적으로 `checkpoint_id`를 지정해야 합니다.

- `thread_id`는 단순히 스레드의 ID입니다. 이것은 항상 필수입니다.
- `checkpoint_id`는 선택적으로 전달할 수 있습니다. 이 식별자는 스레드 내의 특정 체크포인트를 참조합니다. 이것은 스레드 중간 지점부터 그래프 실행을 시작하는 데 사용할 수 있습니다.

그래프를 호출할 때 설정의 구성 가능한 부분으로 이것들을 전달해야 합니다. 예:

```python
{"configurable": {"thread_id": "1"}}  # 유효한 설정
{"configurable": {"thread_id": "1", "checkpoint_id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"}}  # 이것도 유효한 설정
```

### Serde

`langgraph_checkpoint`는 직렬화/역직렬화(serde) 프로토콜도 정의하며, LangChain 및 LangGraph 프리미티브, datetime, enum 등을 포함한 다양한 타입을 처리하는 기본 구현(`langgraph.checkpoint.serde.jsonplus.JsonPlusSerializer`)을 제공합니다.

### 대기 중인 쓰기

주어진 슈퍼스텝에서 그래프 노드가 실행 중 실패하면 LangGraph는 해당 슈퍼스텝에서 성공적으로 완료된 다른 노드들의 대기 중인 체크포인트 쓰기를 저장하므로, 해당 슈퍼스텝에서 그래프 실행을 재개할 때 성공한 노드들을 다시 실행하지 않습니다.

## 인터페이스

각 체크포인터는 `langgraph.checkpoint.base.BaseCheckpointSaver` 인터페이스를 따라야 하며 다음 메서드를 구현해야 합니다:

- `.put` - 체크포인트를 구성 및 메타데이터와 함께 저장합니다.
- `.put_writes` - 체크포인트에 연결된 중간 쓰기 작업(즉, 대기 중인 쓰기)을 저장합니다.
- `.get_tuple` - 주어진 구성(`thread_id` 및 `checkpoint_id`)에 대한 체크포인트 튜플을 가져옵니다.
- `.list` - 주어진 구성 및 필터 기준과 일치하는 체크포인트를 나열합니다.

체크포인터가 비동기 그래프 실행(즉, `.ainvoke`, `.astream`, `.abatch`를 통한 그래프 실행)과 함께 사용될 경우, 체크포인터는 위 메서드의 비동기 버전(`.aput`, `.aput_writes`, `.aget_tuple`, `.alist`)을 구현해야 합니다.

## 사용법

```python
from langgraph.checkpoint.memory import InMemorySaver

write_config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
read_config = {"configurable": {"thread_id": "1"}}

checkpointer = InMemorySaver()
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

# 체크포인트 나열
list(checkpointer.list(read_config))
```
