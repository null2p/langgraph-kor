---
search:
  boost: 2
tags:
  - human-in-the-loop
  - hil
  - overview
hide:
  - tags
---

# Human-in-the-loop

에이전트 또는 워크플로에서 툴 호출을 검토, 편집 및 승인하려면 [LangGraph의 Human-in-the-loop 기능을 사용](../how-tos/human_in_the_loop/add-human-in-the-loop.md)하여 워크플로의 어느 지점에서나 사람의 개입을 활성화합니다. 이는 모델 출력이 검증, 수정 또는 추가 컨텍스트가 필요할 수 있는 대규모 언어 모델(LLM) 기반 애플리케이션에서 특히 유용합니다.

<figure markdown="1">
![image](../concepts/img/human_in_the_loop/tool-call-review.png){: style="max-height:400px"}
</figure>

!!! tip

    Human-in-the-loop 사용 방법에 대한 정보는 [사람 개입 활성화](../how-tos/human_in_the_loop/add-human-in-the-loop.md)와 [Server API를 사용한 Human-in-the-loop](../cloud/how-tos/add-human-in-the-loop.md)를 참조하세요.

## 주요 기능 {#key-capabilities}

* **지속적인 실행 상태**: 중단(Interrupt)은 그래프 상태를 저장하는 LangGraph의 [지속성](./persistence.md) 레이어를 사용하여 재개할 때까지 그래프 실행을 무기한 일시 중지합니다. 이는 LangGraph가 각 단계 후 그래프 상태를 체크포인트하기 때문에 가능하며, 시스템이 실행 컨텍스트를 유지하고 나중에 중단된 지점부터 워크플로를 재개할 수 있게 합니다. 이는 시간 제약 없이 비동기 사람 검토 또는 입력을 지원합니다.

    그래프를 일시 중지하는 두 가지 방법이 있습니다:

    - [동적 중단](../how-tos/human_in_the_loop/add-human-in-the-loop.md#pause-using-interrupt): `interrupt`를 사용하여 그래프의 현재 상태를 기반으로 특정 노드 내부에서 그래프를 일시 중지합니다.
    - [정적 중단](../how-tos/human_in_the_loop/add-human-in-the-loop.md#debug-with-interrupts): `interrupt_before` 및 `interrupt_after`를 사용하여 노드 실행 전 또는 후의 미리 정의된 지점에서 그래프를 일시 중지합니다.

    <figure markdown="1">
    ![image](./img/breakpoints.png){: style="max-height:400px"}
    <figcaption>step_3 이전에 중단점이 있는 3개의 순차적 단계로 구성된 예제 그래프. </figcaption> </figure>

* **유연한 통합 지점**: Human-in-the-loop 로직은 워크플로의 어느 지점에서나 도입될 수 있습니다. 이를 통해 API 호출 승인, 출력 수정 또는 대화 안내와 같은 타겟팅된 사람의 참여가 가능합니다.

## 패턴

`interrupt` 및 `Command`를 사용하여 구현할 수 있는 네 가지 일반적인 디자인 패턴이 있습니다:

- [승인 또는 거부](../how-tos/human_in_the_loop/add-human-in-the-loop.md#approve-or-reject): API 호출과 같은 중요한 단계 전에 그래프를 일시 중지하여 작업을 검토하고 승인합니다. 작업이 거부되면 그래프가 단계를 실행하지 못하도록 하고 잠재적으로 대안적인 작업을 수행할 수 있습니다. 이 패턴은 종종 사람의 입력을 기반으로 그래프를 라우팅하는 것을 포함합니다.
- [그래프 상태 편집](../how-tos/human_in_the_loop/add-human-in-the-loop.md#review-and-edit-state): 그래프를 일시 중지하여 그래프 상태를 검토하고 편집합니다. 이는 실수를 수정하거나 추가 정보로 상태를 업데이트하는 데 유용합니다. 이 패턴은 종종 사람의 입력으로 상태를 업데이트하는 것을 포함합니다.
- [툴 호출 검토](../how-tos/human_in_the_loop/add-human-in-the-loop.md#review-tool-calls): 그래프를 일시 중지하여 툴 실행 전에 LLM이 요청한 툴 호출을 검토하고 편집합니다.
- [사람 입력 검증](../how-tos/human_in_the_loop/add-human-in-the-loop.md#validate-human-input): 다음 단계로 진행하기 전에 사람의 입력을 검증하기 위해 그래프를 일시 중지합니다.