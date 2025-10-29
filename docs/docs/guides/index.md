# 가이드

이 섹션의 페이지들은 다음 주제에 대한 개념적 개요와 how-to를 제공합니다:

## 에이전트 개발

- [개요](../agents/overview.md): 사전 구축된 컴포넌트를 사용하여 에이전트를 구축합니다.
- [에이전트 실행](../agents/run_agents.md): 입력 제공, 출력 해석, 스트리밍 활성화 및 실행 제한 제어를 통해 에이전트를 실행합니다.

## LangGraph API

- [Graph API](../concepts/low_level.md): Graph API를 사용하여 그래프 패러다임을 사용한 워크플로우를 정의합니다.
- [Functional API](../concepts/functional_api.md): Functional API를 사용하여 그래프 구조에 대해 생각하지 않고 함수형 패러다임으로 워크플로우를 구축합니다.
- [Runtime](../concepts/pregel.md): Pregel은 LangGraph의 런타임을 구현하여 LangGraph 애플리케이션의 실행을 관리합니다.

## 핵심 기능

이러한 기능은 LangGraph OSS와 LangGraph Platform 모두에서 사용할 수 있습니다.

- [스트리밍](../concepts/streaming.md): LangGraph 그래프에서 출력을 스트리밍합니다.
- [영속성](../concepts/persistence.md): LangGraph 그래프의 상태를 유지합니다.
- [지속 가능한 실행](../concepts/durable_execution.md): 그래프 실행의 주요 지점에서 진행 상황을 저장합니다.
- [메모리](../concepts/memory.md): 이전 상호작용에 대한 정보를 기억합니다.
- [컨텍스트](../agents/context.md): 그래프 실행에 대한 컨텍스트를 제공하기 위해 외부 데이터를 LangGraph 그래프에 전달합니다.
- [모델](../agents/models.md): LangGraph 애플리케이션에 다양한 LLM을 통합합니다.
- [도구](../concepts/tools.md): 외부 시스템과 직접 인터페이스합니다.
- [Human-in-the-loop](../concepts/human_in_the_loop.md): 워크플로우의 어느 시점에서든 그래프를 일시 중지하고 인간 입력을 기다립니다.
- [타임 트래블](../concepts/time-travel.md): LangGraph 그래프 실행의 특정 시점으로 되돌아갑니다.
- [서브그래프](../concepts/subgraphs.md): 모듈식 그래프를 구축합니다.
- [다중 에이전트](../concepts/multi_agent.md): 복잡한 워크플로우를 여러 에이전트로 나눕니다.
- [MCP](../concepts/mcp.md): LangGraph 그래프에서 MCP 서버를 사용합니다.
- [평가](../agents/evals.md): LangSmith를 사용하여 그래프 성능을 평가합니다.
