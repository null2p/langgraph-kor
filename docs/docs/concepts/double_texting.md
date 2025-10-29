---
search:
  boost: 2
---

# Double Texting

!!! info "사전 요구 사항"
    - [LangGraph Server](./langgraph_server.md)

사용자가 의도하지 않은 방식으로 그래프와 상호작용할 수 있습니다.
예를 들어, 사용자가 하나의 메시지를 보낸 후 그래프가 실행을 마치기 전에 두 번째 메시지를 보낼 수 있습니다.
더 일반적으로, 사용자는 첫 번째 실행이 완료되기 전에 그래프를 두 번째로 호출할 수 있습니다.
우리는 이것을 "double texting"이라고 부릅니다.

현재 LangGraph는 오픈 소스가 아닌 [LangGraph Platform](langgraph_platform.md)의 일부로만 이를 처리합니다.
그 이유는 이를 처리하기 위해서는 그래프가 어떻게 배포되는지 알아야 하고, LangGraph Platform이 배포를 다루므로 로직이 거기에 있어야 하기 때문입니다.
LangGraph Platform을 사용하고 싶지 않다면 아래에서 구현한 옵션을 자세히 설명합니다.

![](img/double_texting.png)

## Reject

가장 간단한 옵션으로, 후속 실행을 거부하고 double texting을 허용하지 않습니다.
reject double text 옵션을 구성하는 방법은 [how-to 가이드](../cloud/how-tos/reject_concurrent.md)를 참조하세요.

## Enqueue

비교적 간단한 옵션으로, 첫 번째 실행이 전체 실행을 완료할 때까지 계속한 다음 새 입력을 별도의 실행으로 보냅니다.
enqueue double text 옵션을 구성하는 방법은 [how-to 가이드](../cloud/how-tos/enqueue_concurrent.md)를 참조하세요.

## Interrupt

이 옵션은 현재 실행을 중단하지만 그 시점까지 수행된 모든 작업을 저장합니다.
그런 다음 사용자 입력을 삽입하고 거기서부터 계속합니다.

이 옵션을 활성화하면 그래프가 발생할 수 있는 이상한 엣지 케이스를 처리할 수 있어야 합니다.
예를 들어, 툴을 호출했지만 해당 툴을 실행한 결과를 아직 받지 못했을 수 있습니다.
dangling 툴 호출이 없도록 해당 툴 호출을 제거해야 할 수 있습니다.

interrupt double text 옵션을 구성하는 방법은 [how-to 가이드](../cloud/how-tos/interrupt_concurrent.md)를 참조하세요.

## Rollback

이 옵션은 현재 실행을 중단하고 원래 실행 입력을 포함하여 그 시점까지 수행된 모든 작업을 롤백합니다. 그런 다음 새 사용자 입력을 보내며, 기본적으로 원래 입력인 것처럼 작동합니다.

rollback double text 옵션을 구성하는 방법은 [how-to 가이드](../cloud/how-tos/rollback_concurrent.md)를 참조하세요.
