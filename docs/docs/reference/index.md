---
title: Reference
description: API reference for LangGraph
search:
  boost: 0.5
---

<style>
.md-sidebar {
  display: block !important;
}
</style>

# Reference

LangGraph reference 문서에 오신 것을 환영합니다! 이 페이지들은 LangGraph로 빌드할 때 사용할 핵심 인터페이스에 대해 자세히 설명합니다. 각 섹션은 생태계의 다른 부분을 다룹니다.

!!! tip

    막 시작하는 경우, 주요 개념과 사용 패턴에 대한 소개를 위해 [LangGraph 기초](../concepts/why-langgraph.md)를 참조하세요.


## LangGraph

LangGraph 오픈소스 라이브러리의 핵심 API입니다.

- [Graphs](graphs.md): 주요 그래프 추상화 및 사용법.
- [Functional API](func.md): 그래프를 위한 함수형 프로그래밍 인터페이스.
- [Pregel](pregel.md): Pregel에서 영감을 받은 계산 모델.
- [Checkpointing](checkpoints.md): 그래프 상태 저장 및 복원.
- [Storage](store.md): 저장소 백엔드 및 옵션.
- [Caching](cache.md): 성능을 위한 캐싱 메커니즘.
- [Types](types.md): 그래프 컴포넌트의 타입 정의.
- [Config](config.md): 구성 옵션.
- [Errors](errors.md): 에러 타입 및 처리.
- [Constants](constants.md): 전역 상수.
- [Channels](channels.md): 메시지 전달 및 채널.

## Prebuilt components

일반적인 워크플로우, 에이전트 및 기타 패턴을 위한 상위 수준 추상화입니다.

- [Agents](agents.md): 내장 에이전트 패턴.
- [Supervisor](supervisor.md): 오케스트레이션 및 위임.
- [Swarm](swarm.md): 다중 에이전트 협업.
- [MCP Adapters](mcp.md): 외부 시스템과의 통합.

## LangGraph Platform

LangGraph Platform을 배포하고 연결하기 위한 도구입니다.

- [SDK (Python)](../cloud/reference/sdk/python_sdk_ref.md): LangGraph Server 인스턴스와 상호 작용하기 위한 Python SDK.
- [SDK (JS/TS)](../cloud/reference/sdk/js_ts_sdk_ref.md): LangGraph Server 인스턴스와 상호 작용하기 위한 JavaScript/TypeScript SDK.
- [RemoteGraph](remote_graph.md): LangGraph Server 인스턴스에 연결하기 위한 `Pregel` 추상화.

더 많은 reference 문서는 [LangGraph Platform reference](https://docs.langchain.com/langgraph-platform/reference-overview)를 참조하세요.
