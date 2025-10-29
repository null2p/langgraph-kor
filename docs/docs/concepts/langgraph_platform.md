---
search:
  boost: 2
---

# LangGraph Platform

**LangGraph Platform**으로 에이전트를 개발, 배포, 확장 및 관리하세요 — 장기 실행되는 에이전틱 워크플로우를 위해 특별히 구축된 플랫폼입니다.

!!! tip "LangGraph Platform 시작하기"

    LangGraph Platform을 사용하여 로컬에서 LangGraph 애플리케이션을 실행하는 방법에 대한 지침은 [LangGraph Platform 빠른 시작](../tutorials/langgraph-platform/local-server.md)을 확인하세요.

## 왜 LangGraph Platform을 사용하나요?

<div align="center"><iframe width="560" height="315" src="https://www.youtube.com/embed/pfAQxBS5z88?si=XGS6Chydn6lhSO1S" title="What is LangGraph Platform?" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe></div>

LangGraph Platform을 사용하면 LangGraph 또는 다른 프레임워크로 구축되었든 에이전트를 프로덕션에 쉽게 실행할 수 있어, 인프라가 아닌 앱 로직에 집중할 수 있습니다. 클릭 한 번으로 배포하여 라이브 엔드포인트를 얻고, 강력한 API와 내장 작업 큐를 사용하여 프로덕션 규모를 처리합니다.

- **[스트리밍 지원](../cloud/how-tos/streaming.md)**: 에이전트가 더 정교해짐에 따라 토큰 출력과 중간 상태를 모두 사용자에게 다시 스트리밍하는 것이 유리한 경우가 많습니다. 이것이 없으면 사용자는 피드백 없이 잠재적으로 긴 작업을 기다려야 합니다. LangGraph Server는 다양한 애플리케이션 요구 사항에 최적화된 여러 스트리밍 모드를 제공합니다.

- **[백그라운드 Run](../cloud/how-tos/background_run.md)**: 처리하는 데 더 오래 걸리는 에이전트(예: 시간 단위)의 경우 연결을 열어두는 것이 비실용적일 수 있습니다. LangGraph Server는 백그라운드에서 에이전트 run을 시작하는 것을 지원하고, run 상태를 효과적으로 모니터링하기 위한 폴링 엔드포인트와 웹훅을 모두 제공합니다.

- **긴 run 지원**: 일반 서버 설정은 완료하는 데 오랜 시간이 걸리는 요청을 처리할 때 종종 타임아웃이나 중단이 발생합니다. LangGraph Server의 API는 정기적인 하트비트 신호를 보내 장기간 프로세스 중 예상치 못한 연결 종료를 방지하여 이러한 작업을 강력하게 지원합니다.

- **급증 처리**: 특히 실시간 사용자 상호작용이 있는 특정 애플리케이션은 수많은 요청이 동시에 서버에 도달하는 "급증" 요청 부하를 경험할 수 있습니다. LangGraph Server에는 작업 큐가 포함되어 있어 과중한 부하에서도 손실 없이 요청을 일관되게 처리합니다.

- **[Double-texting](../cloud/how-tos/interrupt_concurrent.md)**: 사용자 주도 애플리케이션에서는 사용자가 빠르게 여러 메시지를 보내는 것이 일반적입니다. 이러한 "double texting"은 적절하게 처리되지 않으면 에이전트 플로우를 방해할 수 있습니다. LangGraph Server는 이러한 상호작용을 해결하고 관리하기 위한 내장 전략을 제공합니다.

- **[체크포인터 및 메모리 관리](persistence.md#checkpoints)**: 지속성이 필요한 에이전트(예: 대화 메모리)의 경우 강력한 스토리지 솔루션을 배포하는 것이 복잡할 수 있습니다. LangGraph Platform에는 최적화된 [체크포인터](persistence.md#checkpoints)와 [메모리 스토어](persistence.md#memory-store)가 포함되어 있어 사용자 정의 솔루션 없이도 세션 간 상태를 관리합니다.

- **[Human-in-the-loop 지원](../cloud/how-tos/human_in_the_loop_breakpoint.md)**: 많은 애플리케이션에서 사용자는 에이전트 프로세스에 개입할 수 있는 방법이 필요합니다. LangGraph Server는 human-in-the-loop 시나리오를 위한 특수 엔드포인트를 제공하여 에이전트 워크플로우에 수동 감독을 통합하는 것을 단순화합니다.

- **[LangGraph Studio](./langgraph_studio.md)**: LangGraph Server API 프로토콜을 구현하는 에이전틱 시스템의 시각화, 상호작용 및 디버깅을 가능하게 합니다. Studio는 또한 LangSmith와 통합되어 추적, 평가 및 프롬프트 엔지니어링을 가능하게 합니다.

- **[배포](./deployment_options.md)**: LangGraph Platform에 배포하는 네 가지 방법이 있습니다: [Cloud SaaS](../concepts/langgraph_cloud.md), [Self-Hosted Data Plane](../concepts/langgraph_self_hosted_data_plane.md), [Self-Hosted Control Plane](../concepts/langgraph_self_hosted_control_plane.md), 및 [Standalone Container](../concepts/langgraph_standalone_container.md).
