---
search:
  boost: 2
tags:
  - agent
hide:
  - tags
---

# UI

[Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui)를 통해 모든 LangGraph 에이전트와 상호작용할 수 있는 미리 빌드된 챗 UI를 사용할 수 있습니다. [배포된 버전](https://agentchat.vercel.app)을 사용하는 것이 가장 빠르게 시작하는 방법이며, 로컬 그래프와 배포된 그래프 모두와 상호작용할 수 있습니다.

## UI에서 에이전트 실행

먼저 LangGraph API 서버를 [로컬에서](../tutorials/langgraph-platform/local-server.md) 설정하거나 [LangGraph Platform](https://langchain-ai.github.io/langgraph/cloud/quick_start/)에 에이전트를 배포합니다.

그런 다음 [Agent Chat UI](https://agentchat.vercel.app)로 이동하거나, 저장소를 클론하여 [개발 서버를 로컬에서 실행](https://github.com/langchain-ai/agent-chat-ui?tab=readme-ov-file#setup)합니다:

<video controls src="../assets/base-chat-ui.mp4" type="video/mp4"></video>

!!! Tip

    UI는 툴 호출과 툴 결과 메시지를 렌더링하는 기본 지원이 있습니다. 표시되는 메시지를 사용자 정의하려면 Agent Chat UI 문서의 [챗에서 메시지 숨기기](https://github.com/langchain-ai/agent-chat-ui?tab=readme-ov-file#hiding-messages-in-the-chat) 섹션을 참조하세요.

## Human-in-the-loop 추가

Agent Chat UI는 [Human-in-the-loop](../concepts/human_in_the_loop.md) 워크플로를 완전히 지원합니다. 테스트하려면 ([배포](../tutorials/langgraph-platform/local-server.md) 가이드의) `src/agent/graph.py`에 있는 에이전트 코드를 이 [에이전트 구현](../how-tos/human_in_the_loop/add-human-in-the-loop.md#add-interrupts-to-any-tool)으로 교체하세요:

<video controls src="../assets/interrupt-chat-ui.mp4" type="video/mp4"></video>

!!! Important

    Agent Chat UI는 LangGraph 에이전트가 @[`HumanInterrupt` 스키마][HumanInterrupt]를 사용하여 중단할 때 가장 잘 작동합니다. 해당 스키마를 사용하지 않으면 Agent Chat UI는 `interrupt` 함수에 전달된 입력을 렌더링할 수 있지만, 그래프를 재개하는 것에 대한 완전한 지원은 제공하지 않습니다.

## Generative UI

Agent Chat UI에서 Generative UI를 사용할 수도 있습니다.

Generative UI를 사용하면 [React](https://react.dev/) 컴포넌트를 정의하고, LangGraph 서버에서 UI로 푸시할 수 있습니다. Generative UI LangGraph 에이전트 빌드에 대한 자세한 문서는 [이 문서](https://langchain-ai.github.io/langgraph/cloud/how-tos/generative_ui_react/)를 읽어보세요.
