# llms.txt

아래에서 [`llms.txt`](https://llmstxt.org/) 형식의 문서 파일 목록을 확인할 수 있습니다. 특히 `llms.txt`와 `llms-full.txt`가 있습니다. 이 파일들을 통해 대규모 언어 모델(LLM)과 에이전트가 프로그래밍 문서 및 API에 액세스할 수 있으며, 특히 통합 개발 환경(IDE) 내에서 유용합니다.

| Language Version | llms.txt                                                                                                   | llms-full.txt                                                                                                        |
|------------------|------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| LangGraph Python | [https://langchain-ai.github.io/langgraph/llms.txt](https://langchain-ai.github.io/langgraph/llms.txt)     | [https://langchain-ai.github.io/langgraph/llms-full.txt](https://langchain-ai.github.io/langgraph/llms-full.txt)     |
| LangGraph JS     | [https://langchain-ai.github.io/langgraphjs/llms.txt](https://langchain-ai.github.io/langgraphjs/llms.txt) | [https://langchain-ai.github.io/langgraphjs/llms-full.txt](https://langchain-ai.github.io/langgraphjs/llms-full.txt) |
| LangChain Python | [https://python.langchain.com/llms.txt](https://python.langchain.com/llms.txt)                             | N/A                                                                                                                  |
| LangChain JS     | [https://js.langchain.com/llms.txt](https://js.langchain.com/llms.txt)                                     | N/A                                                                                                                  |

!!! info "출력 검토"

    최신 문서에 액세스할 수 있더라도 현재 최신 모델이 항상 올바른 코드를 생성하는 것은 아닙니다. 생성된 코드를 시작점으로 취급하고 프로덕션에 코드를 배포하기 전에 항상 검토하세요.

## `llms.txt`와 `llms-full.txt`의 차이점

- **`llms.txt`**는 콘텐츠에 대한 간략한 설명과 함께 링크를 포함하는 인덱스 파일입니다. LLM이나 에이전트는 상세한 정보에 액세스하기 위해 이러한 링크를 따라가야 합니다.

- **`llms-full.txt`**는 모든 상세한 콘텐츠를 단일 파일에 직접 포함하여 추가 탐색이 필요하지 않습니다.

`llms-full.txt`를 사용할 때 고려해야 할 핵심 사항은 파일 크기입니다. 방대한 문서의 경우 이 파일이 너무 커져서 LLM의 컨텍스트 윈도우에 맞지 않을 수 있습니다.

## MCP 서버를 통한 `llms.txt` 사용

2025년 3월 9일 현재 IDE는 [아직 `llms.txt`에 대한 강력한 네이티브 지원이 없습니다](https://x.com/jeremyphoward/status/1902109312216129905?t=1eHFv2vdNdAckajnug0_Vw&s=19). 그러나 MCP 서버를 통해 `llms.txt`를 효과적으로 사용할 수 있습니다.

### 🚀 `mcpdoc` 서버 사용

LLM과 IDE를 위한 문서를 제공하도록 설계된 **MCP 서버**를 제공합니다:

👉 **[langchain-ai/mcpdoc GitHub 저장소](https://github.com/langchain-ai/mcpdoc)**

이 MCP 서버를 사용하면 **Cursor**, **Windsurf**, **Claude**, **Claude Code**와 같은 도구에 `llms.txt`를 통합할 수 있습니다.

📘 **설정 지침 및 사용 예제**는 저장소에서 확인할 수 있습니다.

## `llms-full.txt` 사용

LangGraph `llms-full.txt` 파일은 일반적으로 수십만 개의 토큰을 포함하여 대부분의 LLM의 컨텍스트 윈도우 제한을 초과합니다. 이 파일을 효과적으로 사용하려면:

1. **IDE 사용 시 (예: Cursor, Windsurf)**:
    - `llms-full.txt`를 커스텀 문서로 추가합니다. IDE가 자동으로 콘텐츠를 청크하고 인덱싱하여 Retrieval-Augmented Generation(RAG)을 구현합니다.

2. **IDE 지원 없이**:
    - 큰 컨텍스트 윈도우를 가진 챗 모델을 사용합니다.
    - 문서를 효율적으로 관리하고 쿼리하기 위해 RAG 전략을 구현합니다.
