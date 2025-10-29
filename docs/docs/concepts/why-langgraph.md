# 개요

LangGraph는 강력하고 적응 가능한 AI 에이전트를 구축하려는 개발자를 위해 만들어졌습니다. 개발자들이 LangGraph를 선택하는 이유:

- **신뢰성과 제어 가능성.** 중재 검사 및 human-in-the-loop 승인으로 에이전트 동작을 조정합니다. LangGraph는 장기 실행 워크플로우를 위한 컨텍스트를 유지하여 에이전트가 올바른 방향을 유지하도록 합니다.
- **저수준이고 확장 가능.** 커스터마이징을 제한하는 엄격한 추상화 없이 완전히 설명적인 저수준 프리미티브로 맞춤형 에이전트를 구축합니다. 각 에이전트가 사용 사례에 맞춘 특정 역할을 수행하는 확장 가능한 다중 에이전트 시스템을 설계합니다.
- **일급 스트리밍 지원.** 토큰 단위 스트리밍 및 중간 단계 스트리밍을 통해 LangGraph는 사용자에게 실시간으로 전개되는 에이전트 추론과 동작에 대한 명확한 가시성을 제공합니다.

## LangGraph 기초 학습

LangGraph의 주요 개념과 기능을 익히려면 다음 LangGraph 기초 튜토리얼 시리즈를 완료하세요:

1. [기본 챗봇 구축하기](../tutorials/get-started/1-build-basic-chatbot.md)
2. [도구 추가하기](../tutorials/get-started/2-add-tools.md)
3. [메모리 추가하기](../tutorials/get-started/3-add-memory.md)
4. [Human-in-the-loop 제어 추가하기](../tutorials/get-started/4-human-in-the-loop.md)
5. [상태 커스터마이징하기](../tutorials/get-started/5-customize-state.md)
6. [타임 트래블](../tutorials/get-started/6-time-travel.md)

이 일련의 튜토리얼을 완료하면 다음 기능을 가진 LangGraph 지원 챗봇을 구축하게 됩니다:

* ✅ **일반적인 질문에 답변** - 웹 검색을 통해
* ✅ **대화 상태 유지** - 호출 간
* ✅ **복잡한 쿼리 라우팅** - 검토를 위해 사람에게
* ✅ **사용자 정의 상태 사용** - 동작 제어를 위해
* ✅ **되감기 및 탐색** - 대안 대화 경로
