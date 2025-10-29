---
search:
  boost: 2
---

# LangGraph Studio

!!! info "사전 요구 사항"

    - [LangGraph Platform](./langgraph_platform.md)
    - [LangGraph Server](./langgraph_server.md)
    - [LangGraph CLI](./langgraph_cli.md)

LangGraph Studio는 LangGraph Server API 프로토콜을 구현하는 에이전트 시스템의 시각화, 상호작용 및 디버깅을 가능하게 하는 전문 에이전트 IDE입니다. Studio는 또한 LangSmith와 통합되어 추적, 평가 및 프롬프트 엔지니어링을 가능하게 합니다.

![](img/lg_studio.png)

## 기능

LangGraph Studio의 주요 기능:

- 그래프 아키텍처 시각화
- [에이전트 실행 및 상호작용](../cloud/how-tos/invoke_studio.md)
- [어시스턴트 관리](../cloud/how-tos/studio/manage_assistants.md)
- [Thread 관리](../cloud/how-tos/threads_studio.md)
- [프롬프트 반복](../cloud/how-tos/iterate_graph_studio.md)
- [데이터셋에서 실험 실행](../cloud/how-tos/studio/run_evals.md)
- [장기 메모리](memory.md) 관리
- [타임 트래블](time-travel.md)을 통한 에이전트 상태 디버그

LangGraph Studio는 [LangGraph Platform](../cloud/quick_start.md)에 배포된 그래프 또는 [LangGraph Server](../tutorials/langgraph-platform/local-server.md)를 통해 로컬에서 실행되는 그래프에서 작동합니다.

Studio는 두 가지 모드를 지원합니다:

### 그래프 모드

그래프 모드는 Studio의 전체 기능을 노출하며 순회한 노드, 중간 상태 및 LangSmith 통합(데이터셋 및 playground에 추가 등)을 포함하여 에이전트 실행에 대한 가능한 한 많은 세부 정보를 원할 때 유용합니다.

### 챗 모드

챗 모드는 챗 전용 에이전트를 반복하고 테스트하기 위한 더 간단한 UI입니다. 비즈니스 사용자와 전반적인 에이전트 동작을 테스트하려는 사람들에게 유용합니다. 챗 모드는 상태가 [`MessagesState`](https://langchain-ai.github.io/langgraph/how-tos/graph-api/#messagesstate)를 포함하거나 확장하는 그래프에서만 지원됩니다.

## 더 알아보기

- LangGraph Studio를 [시작하는 방법](../cloud/how-tos/studio/quick_start.md)에 대한 가이드를 참조하세요.
