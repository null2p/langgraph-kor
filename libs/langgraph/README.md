<picture class="github-only">
  <source media="(prefers-color-scheme: light)" srcset="https://langchain-ai.github.io/langgraph/static/wordmark_dark.svg">
  <source media="(prefers-color-scheme: dark)" srcset="https://langchain-ai.github.io/langgraph/static/wordmark_light.svg">
  <img alt="LangGraph Logo" src="https://langchain-ai.github.io/langgraph/static/wordmark_dark.svg" width="80%">
</picture>

<div>
<br>
</div>

[![Version](https://img.shields.io/pypi/v/langgraph.svg)](https://pypi.org/project/langgraph/)
[![Downloads](https://static.pepy.tech/badge/langgraph/month)](https://pepy.tech/project/langgraph)
[![Open Issues](https://img.shields.io/github/issues-raw/langchain-ai/langgraph)](https://github.com/langchain-ai/langgraph/issues)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://langchain-ai.github.io/langgraph/)

Klarna, Replit, Elastic 등 에이전트의 미래를 만들어가는 기업들이 신뢰하는 LangGraph는 장기 실행되는 상태 유지 에이전트를 구축, 관리 및 배포하기 위한 로우레벨 오케스트레이션 프레임워크입니다.

## 시작하기

LangGraph 설치:

```
pip install -U langgraph
```

그런 다음, [사전 구축된 컴포넌트를 사용하여](https://langchain-ai.github.io/langgraph/agents/agents/) 에이전트를 생성합니다:

```python
# pip install -qU "langchain[anthropic]" 모델을 호출하기 위해

from langgraph.prebuilt import create_react_agent

def get_weather(city: str) -> str:
    """주어진 도시의 날씨를 가져옵니다."""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    prompt="You are a helpful assistant"
)

# 에이전트 실행
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

자세한 정보는 [빠른 시작](https://langchain-ai.github.io/langgraph/agents/agents/)을 참조하세요. 또는 커스터마이징 가능한 아키텍처, 장기 메모리 및 기타 복잡한 작업 처리를 갖춘 [에이전트 워크플로우](https://langchain-ai.github.io/langgraph/concepts/low_level/)를 구축하는 방법을 배우려면 [LangGraph 기본 튜토리얼](https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/)을 참조하세요.

## 핵심 이점

LangGraph는 *모든* 장기 실행되는 상태 유지 워크플로우 또는 에이전트를 위한 로우레벨 지원 인프라를 제공합니다. LangGraph는 프롬프트나 아키텍처를 추상화하지 않으며, 다음과 같은 핵심 이점을 제공합니다:

- [내구성 있는 실행](https://langchain-ai.github.io/langgraph/concepts/durable_execution/): 실패를 견디고 장기간 실행할 수 있는 에이전트를 구축하여 중단된 지점에서 정확히 자동으로 재개합니다.
- [Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/): 실행 중 언제든지 에이전트 상태를 검사하고 수정하여 인간의 감독을 원활하게 통합합니다.
- [포괄적인 메모리](https://langchain-ai.github.io/langgraph/concepts/memory/): 진행 중인 추론을 위한 단기 작업 메모리와 세션 간 장기 영구 메모리를 모두 갖춘 진정한 상태 유지 에이전트를 만듭니다.
- [LangSmith를 사용한 디버깅](http://www.langchain.com/langsmith): 실행 경로를 추적하고 상태 전환을 캡처하며 상세한 런타임 메트릭을 제공하는 시각화 도구로 복잡한 에이전트 동작에 대한 깊은 가시성을 확보합니다.
- [프로덕션 준비 배포](https://langchain-ai.github.io/langgraph/concepts/deployment_options/): 상태 유지 장기 실행 워크플로우의 고유한 과제를 처리하도록 설계된 확장 가능한 인프라로 정교한 에이전트 시스템을 자신 있게 배포합니다.

## LangGraph의 생태계

LangGraph는 독립적으로 사용할 수 있지만 모든 LangChain 제품과 원활하게 통합되어 개발자에게 에이전트 구축을 위한 완전한 도구 모음을 제공합니다. LLM 애플리케이션 개발을 개선하려면 LangGraph를 다음과 함께 사용하세요:

- [LangSmith](http://www.langchain.com/langsmith) — 에이전트 평가 및 관찰성에 유용합니다. 성능이 좋지 않은 LLM 앱 실행을 디버그하고, 에이전트 궤적을 평가하며, 프로덕션에서 가시성을 확보하고, 시간이 지남에 따라 성능을 향상시킵니다.
- [LangSmith Deployment](https://langchain-ai.github.io/langgraph/concepts/langgraph_platform/) — 장기 실행되는 상태 유지 워크플로우를 위한 전용 배포 플랫폼으로 에이전트를 손쉽게 배포하고 확장합니다. 팀 간에 에이전트를 발견, 재사용, 구성 및 공유하고 [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/)에서 시각적 프로토타이핑을 통해 빠르게 반복합니다.
- [LangChain](https://python.langchain.com/docs/introduction/) – LLM 애플리케이션 개발을 간소화하기 위한 통합 및 구성 가능한 컴포넌트를 제공합니다.

> [!NOTE]
> LangGraph의 JS 버전을 찾고 계신가요? [JS 저장소](https://github.com/langchain-ai/langgraphjs)와 [JS 문서](https://langchain-ai.github.io/langgraphjs/)를 참조하세요.

## 추가 리소스

- [가이드](https://langchain-ai.github.io/langgraph/guides/): 스트리밍, 메모리 및 지속성 추가, 디자인 패턴(예: 분기, 서브그래프 등)과 같은 주제에 대한 빠르고 실행 가능한 코드 스니펫.
- [참조](https://langchain-ai.github.io/langgraph/reference/graphs/): 핵심 클래스, 메서드, 그래프 및 체크포인팅 API 사용 방법, 상위 수준의 사전 구축된 컴포넌트에 대한 상세한 참조.
- [예제](https://langchain-ai.github.io/langgraph/examples/): LangGraph 시작하기에 대한 안내 예제.
- [LangChain 포럼](https://forum.langchain.com/): 커뮤니티와 연결하고 모든 기술적 질문, 아이디어 및 피드백을 공유하세요.
- [LangChain Academy](https://academy.langchain.com/courses/intro-to-langgraph): 무료 구조화된 과정에서 LangGraph의 기본을 배우세요.
- [템플릿](https://langchain-ai.github.io/langgraph/concepts/template_applications/): 일반적인 에이전트 워크플로우(예: ReAct 에이전트, 메모리, 검색 등)를 위한 사전 구축된 참조 앱으로 복제하고 적용할 수 있습니다.
- [사례 연구](https://www.langchain.com/built-with-langgraph): 업계 리더들이 LangGraph를 사용하여 대규모로 AI 애플리케이션을 제공하는 방법을 들어보세요.

## 감사의 말

LangGraph는 [Pregel](https://research.google/pubs/pub37252/)과 [Apache Beam](https://beam.apache.org/)에서 영감을 받았습니다. 공개 인터페이스는 [NetworkX](https://networkx.org/documentation/latest/)에서 영감을 얻었습니다. LangGraph는 LangChain의 제작자인 LangChain Inc에서 구축했지만 LangChain 없이도 사용할 수 있습니다.
