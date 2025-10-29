# LangSmith 추적 디버그

이 가이드는 대화형 조사 및 디버깅을 위해 LangGraph Studio에서 LangSmith 추적을 여는 방법을 설명합니다.

## 배포된 thread 열기

1. LangSmith 추적을 열고 루트 run을 선택합니다.
2. "Run in Studio"를 클릭합니다.

그러면 추적의 부모 thread가 선택된 상태로 관련 LangGraph Platform 배포에 연결된 LangGraph Studio가 열립니다.

## 원격 추적으로 로컬 에이전트 테스트

이 섹션에서는 LangSmith의 원격 추적에 대해 로컬 에이전트를 테스트하는 방법을 설명합니다. 이를 통해 프로덕션 추적을 로컬 테스트 입력으로 사용할 수 있으며, 개발 환경에서 에이전트 수정 사항을 디버그하고 검증할 수 있습니다.

### 요구 사항

- LangSmith 추적 thread
- 로컬에서 실행되는 에이전트. 설정 지침은 [여기](../how-tos/studio/quick_start.md#local-development-server)를 참조하세요.

!!! info "로컬 에이전트 요구 사항"

    - langgraph>=0.3.18
    - langgraph-api>=0.0.32
    - 원격 추적에 있는 동일한 노드 집합 포함

### Thread 복제

1. LangSmith 추적을 열고 루트 run을 선택합니다.
2. "Run in Studio" 옆의 드롭다운을 클릭합니다.
3. 로컬 에이전트의 URL을 입력합니다.
4. "Clone thread locally"를 선택합니다.
5. 여러 그래프가 있는 경우 대상 그래프를 선택합니다.

원격 thread에서 추론되고 복사된 thread 히스토리로 로컬 에이전트에 새 thread가 생성되며, 로컬로 실행 중인 애플리케이션의 LangGraph Studio로 이동합니다.
