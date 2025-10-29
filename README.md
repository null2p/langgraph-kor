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

Klarna, Replit, Elastic 등 에이전트의 미래를 만들어가는 기업들이 신뢰하는 LangGraph는 장기 실행되는 상태 저장형 에이전트를 구축, 관리 및 배포하기 위한 저수준 오케스트레이션 프레임워크입니다.

## 시작하기

LangGraph 설치:

```
pip install -U langgraph
```

그런 다음 [사전 구축된 컴포넌트를 사용하여](https://langchain-ai.github.io/langgraph/agents/agents/) 에이전트를 생성합니다:

```python
# 모델을 호출하려면 pip install -qU "langchain[anthropic]" 실행

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

자세한 내용은 [빠른 시작](https://langchain-ai.github.io/langgraph/agents/agents/)을 참조하세요. 또는 커스터마이징 가능한 아키텍처, 장기 메모리 및 기타 복잡한 작업 처리를 갖춘 [에이전트 워크플로우](https://langchain-ai.github.io/langgraph/concepts/low_level/)를 구축하는 방법을 배우려면 [LangGraph 기초 튜토리얼](https://langchain-ai.github.io/langgraph/tutorials/get-started/1-build-basic-chatbot/)을 참조하세요.

## 핵심 이점

LangGraph는 *모든* 장기 실행되는 상태 저장형 워크플로우나 에이전트를 위한 저수준 지원 인프라를 제공합니다. LangGraph는 프롬프트나 아키텍처를 추상화하지 않으며, 다음과 같은 핵심 이점을 제공합니다:

- [지속 가능한 실행](https://langchain-ai.github.io/langgraph/concepts/durable_execution/): 장애가 발생해도 유지되고 장기간 실행될 수 있으며, 중단된 지점에서 자동으로 재개되는 에이전트를 구축합니다.
- [Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/): 실행 중 언제든지 에이전트 상태를 검사하고 수정하여 사람의 감독을 원활하게 통합합니다.
- [포괄적인 메모리](https://langchain-ai.github.io/langgraph/concepts/memory/): 진행 중인 추론을 위한 단기 작업 메모리와 세션 간 장기 영구 메모리를 모두 갖춘 진정한 상태 저장형 에이전트를 생성합니다.
- [LangSmith를 통한 디버깅](http://www.langchain.com/langsmith): 실행 경로를 추적하고 상태 전환을 캡처하며 세부 런타임 메트릭을 제공하는 시각화 도구를 통해 복잡한 에이전트 동작에 대한 깊은 가시성을 확보합니다.
- [프로덕션 준비 배포](https://langchain-ai.github.io/langgraph/concepts/deployment_options/): 상태 저장형 장기 실행 워크플로우의 고유한 과제를 처리하도록 설계된 확장 가능한 인프라로 정교한 에이전트 시스템을 안심하고 배포합니다.

## LangGraph 생태계

LangGraph는 독립적으로 사용할 수 있지만, 모든 LangChain 제품과도 완벽하게 통합되어 개발자에게 에이전트 구축을 위한 전체 도구 모음을 제공합니다. LLM 애플리케이션 개발을 개선하려면 LangGraph를 다음과 함께 사용하세요:

- [LangSmith](http://www.langchain.com/langsmith) — 에이전트 평가 및 관찰성에 유용합니다. 성능이 낮은 LLM 앱 실행을 디버그하고, 에이전트 궤적을 평가하며, 프로덕션 환경에서 가시성을 확보하고, 시간이 지남에 따라 성능을 개선합니다.
- [LangSmith Deployment](https://langchain-ai.github.io/langgraph/concepts/langgraph_platform/) — 장기 실행되는 상태 저장형 워크플로우를 위해 특별히 구축된 배포 플랫폼으로 에이전트를 손쉽게 배포하고 확장합니다. 팀 전체에서 에이전트를 발견, 재사용, 구성 및 공유하고, [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/)에서 시각적 프로토타이핑을 통해 빠르게 반복합니다.
- [LangChain](https://python.langchain.com/docs/introduction/) – LLM 애플리케이션 개발을 간소화하는 통합 및 구성 가능한 컴포넌트를 제공합니다.

> [!NOTE]
> LangGraph의 JS 버전을 찾고 계신가요? [JS 저장소](https://github.com/langchain-ai/langgraphjs)와 [JS 문서](https://langchain-ai.github.io/langgraphjs/)를 참조하세요.

## 추가 리소스

- [가이드](https://langchain-ai.github.io/langgraph/guides/): 스트리밍, 메모리 및 영속성 추가, 디자인 패턴(예: 분기, 서브그래프 등)과 같은 주제에 대한 빠르고 실행 가능한 코드 스니펫입니다.
- [레퍼런스](https://langchain-ai.github.io/langgraph/reference/graphs/): 핵심 클래스, 메서드, 그래프 및 체크포인팅 API 사용 방법, 상위 수준의 사전 구축된 컴포넌트에 대한 자세한 레퍼런스입니다.
- [예제](https://langchain-ai.github.io/langgraph/examples/): LangGraph 시작하기에 대한 안내된 예제입니다.
- [LangChain 포럼](https://forum.langchain.com/): 커뮤니티와 연결하고 모든 기술 질문, 아이디어 및 피드백을 공유하세요.
- [LangChain Academy](https://academy.langchain.com/courses/intro-to-langgraph): 무료 구조화된 강좌에서 LangGraph의 기초를 배우세요.
- [템플릿](https://langchain-ai.github.io/langgraph/concepts/template_applications/): 일반적인 에이전틱 워크플로우(예: ReAct 에이전트, 메모리, 검색 등)를 위한 사전 구축된 참조 앱으로, 복제하여 적응시킬 수 있습니다.
- [사례 연구](https://www.langchain.com/built-with-langgraph): 업계 리더들이 LangGraph를 사용하여 대규모로 AI 애플리케이션을 출시하는 방법을 들어보세요.

## 감사의 말

LangGraph는 [Pregel](https://research.google/pubs/pub37252/)과 [Apache Beam](https://beam.apache.org/)에서 영감을 받았습니다. 공개 인터페이스는 [NetworkX](https://networkx.org/documentation/latest/)에서 영감을 받았습니다. LangGraph는 LangChain의 제작자인 LangChain Inc에서 구축했지만, LangChain 없이도 사용할 수 있습니다.