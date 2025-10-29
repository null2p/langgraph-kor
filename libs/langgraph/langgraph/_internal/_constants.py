"""Pregel 작업에 사용되는 상수입니다."""

import sys
from typing import Literal, cast

# --- 예약된 쓰기 키 ---
INPUT = sys.intern("__input__")
# 그래프에 입력으로 전달되는 값들
INTERRUPT = sys.intern("__interrupt__")
# 노드에 의해 발생한 동적 인터럽트
RESUME = sys.intern("__resume__")
# 인터럽트 후 노드를 재개하기 위해 전달되는 값들
ERROR = sys.intern("__error__")
# 노드에 의해 발생한 에러
NO_WRITES = sys.intern("__no_writes__")
# 노드가 아무것도 쓰지 않았음을 나타내는 마커
TASKS = sys.intern("__pregel_tasks")
# 노드/엣지에 의해 반환된 Send 객체들, 아래의 PUSH에 해당
RETURN = sys.intern("__return__")
# 단순히 반환 값을 기록하는 태스크의 쓰기
PREVIOUS = sys.intern("__previous__")
# 각 노드의 Control 값들을 처리하는 암묵적 분기


# --- 예약된 캐시 네임스페이스 ---
CACHE_NS_WRITES = sys.intern("__pregel_ns_writes")
# 노드 쓰기를 위한 캐시 네임스페이스

# --- 예약된 config.configurable 키 ---
CONFIG_KEY_SEND = sys.intern("__pregel_send")
# state/엣지/예약된 키에 대한 쓰기를 받는 `write` 함수를 보유
CONFIG_KEY_READ = sys.intern("__pregel_read")
# 현재 상태의 복사본을 반환하는 `read` 함수를 보유
CONFIG_KEY_CALL = sys.intern("__pregel_call")
# 노드/함수, 인자를 받아 future를 반환하는 `call` 함수를 보유
CONFIG_KEY_CHECKPOINTER = sys.intern("__pregel_checkpointer")
# 부모 그래프에서 자식 그래프로 전달된 `BaseCheckpointSaver`를 보유
CONFIG_KEY_STREAM = sys.intern("__pregel_stream")
# 부모 그래프에서 자식 그래프로 전달된 `StreamProtocol`을 보유
CONFIG_KEY_CACHE = sys.intern("__pregel_cache")
# 서브그래프에 제공되는 `BaseCache`를 보유
CONFIG_KEY_RESUMING = sys.intern("__pregel_resuming")
# 서브그래프가 이전 체크포인트에서 재개해야 하는지를 나타내는 boolean을 보유
CONFIG_KEY_TASK_ID = sys.intern("__pregel_task_id")
# 현재 태스크의 태스크 ID를 보유
CONFIG_KEY_THREAD_ID = sys.intern("thread_id")
# 현재 실행의 스레드 ID를 보유
CONFIG_KEY_CHECKPOINT_MAP = sys.intern("checkpoint_map")
# 부모 그래프를 위한 checkpoint_ns -> checkpoint_id 매핑을 보유
CONFIG_KEY_CHECKPOINT_ID = sys.intern("checkpoint_id")
# 현재 checkpoint_id를 보유 (있는 경우)
CONFIG_KEY_CHECKPOINT_NS = sys.intern("checkpoint_ns")
# 현재 checkpoint_ns를 보유, 루트 그래프의 경우 ""
CONFIG_KEY_NODE_FINISHED = sys.intern("__pregel_node_finished")
# 노드가 완료될 때 호출될 콜백을 보유
CONFIG_KEY_SCRATCHPAD = sys.intern("__pregel_scratchpad")
# 현재 태스크로 범위가 지정된 임시 저장소를 위한 가변 dict를 보유
CONFIG_KEY_RUNNER_SUBMIT = sys.intern("__pregel_runner_submit")
# runner로부터 태스크를 받아 실행하고 결과를 반환하는 함수를 보유
CONFIG_KEY_DURABILITY = sys.intern("__pregel_durability")
# "sync", "async" 또는 "exit" 중 하나인 내구성 모드를 보유
CONFIG_KEY_RUNTIME = sys.intern("__pregel_runtime")
# context, store, stream writer 등을 포함하는 `Runtime` 인스턴스를 보유
CONFIG_KEY_RESUME_MAP = sys.intern("__pregel_resume_map")
# 태스크 재개를 위한 task ns -> resume value 매핑을 보유

# --- 기타 상수 ---
PUSH = sys.intern("__pregel_push")
# push 스타일 태스크를 나타냄, 즉 Send 객체에 의해 생성된 것들
PULL = sys.intern("__pregel_pull")
# pull 스타일 태스크를 나타냄, 즉 엣지에 의해 트리거된 것들
NS_SEP = sys.intern("|")
# checkpoint_ns용, 각 레벨을 구분함 (예: graph|subgraph|subsubgraph)
NS_END = sys.intern(":")
# checkpoint_ns용, 각 레벨에서 네임스페이스와 task_id를 구분함
CONF = cast(Literal["configurable"], sys.intern("configurable"))
# RunnableConfig에서 configurable dict의 키
NULL_TASK_ID = sys.intern("00000000-0000-0000-0000-000000000000")
# 태스크와 연관되지 않은 쓰기에 사용할 task_id

# langgraph.constants와의 순환 import를 피하기 위해 재정의
_TAG_HIDDEN = sys.intern("langsmith:hidden")

RESERVED = {
    _TAG_HIDDEN,
    # 예약된 쓰기 키
    INPUT,
    INTERRUPT,
    RESUME,
    ERROR,
    NO_WRITES,
    # 예약된 config.configurable 키
    CONFIG_KEY_SEND,
    CONFIG_KEY_READ,
    CONFIG_KEY_CHECKPOINTER,
    CONFIG_KEY_STREAM,
    CONFIG_KEY_CHECKPOINT_MAP,
    CONFIG_KEY_RESUMING,
    CONFIG_KEY_TASK_ID,
    CONFIG_KEY_CHECKPOINT_MAP,
    CONFIG_KEY_CHECKPOINT_ID,
    CONFIG_KEY_CHECKPOINT_NS,
    CONFIG_KEY_RESUME_MAP,
    # 기타 상수
    PUSH,
    PULL,
    NS_SEP,
    NS_END,
    CONF,
}
