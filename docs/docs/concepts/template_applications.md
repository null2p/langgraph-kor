---
search:
  boost: 2
---

# 템플릿 애플리케이션

템플릿은 LangGraph로 빌드할 때 빠르게 시작할 수 있도록 설계된 오픈 소스 참조 애플리케이션입니다. 필요에 맞게 커스터마이징할 수 있는 일반적인 에이전트 워크플로의 작동 예제를 제공합니다.

LangGraph CLI를 사용하여 템플릿에서 애플리케이션을 만들 수 있습니다.

:::python
!!! info "요구 사항"

    - Python >= 3.11
    - [LangGraph CLI](https://langchain-ai.github.io/langgraph/cloud/reference/cli/): langchain-cli[inmem] >= 0.1.58 필요

## LangGraph CLI 설치

```bash
pip install "langgraph-cli[inmem]" --upgrade
```

또는 [`uv`](https://docs.astral.sh/uv/getting-started/installation/) 사용 (권장):

```bash
uvx --from "langgraph-cli[inmem]" langgraph dev --help
```

:::

:::js

```bash
npx @langchain/langgraph-cli --help
```

:::

## 사용 가능한 템플릿

:::python
| 템플릿 | 설명 | 링크 |
| -------- | ----------- | ------ |
| **New LangGraph Project** | 메모리가 있는 간단하고 최소한의 챗봇입니다. | [Repo](https://github.com/langchain-ai/new-langgraph-project) |
| **ReAct Agent** | 많은 도구로 유연하게 확장할 수 있는 간단한 에이전트입니다. | [Repo](https://github.com/langchain-ai/react-agent) |
| **Memory Agent** | thread 간에 사용할 메모리를 저장하는 추가 도구가 있는 ReAct 스타일 에이전트입니다. | [Repo](https://github.com/langchain-ai/memory-agent) |
| **Retrieval Agent** | 검색 기반 질문 답변 시스템이 포함된 에이전트입니다. | [Repo](https://github.com/langchain-ai/retrieval-agent-template) |
| **Data-Enrichment Agent** | 웹 검색을 수행하고 발견 사항을 구조화된 형식으로 정리하는 에이전트입니다. | [Repo](https://github.com/langchain-ai/data-enrichment) |

:::

:::js
| 템플릿 | 설명 | 링크 |
| -------- | ----------- | ------ |
| **New LangGraph Project** | 메모리가 있는 간단하고 최소한의 챗봇입니다. | [Repo](https://github.com/langchain-ai/new-langgraphjs-project) |
| **ReAct Agent** | 많은 도구로 유연하게 확장할 수 있는 간단한 에이전트입니다. | [Repo](https://github.com/langchain-ai/react-agent-js) |
| **Memory Agent** | thread 간에 사용할 메모리를 저장하는 추가 도구가 있는 ReAct 스타일 에이전트입니다. | [Repo](https://github.com/langchain-ai/memory-agent-js) |
| **Retrieval Agent** | 검색 기반 질문 답변 시스템이 포함된 에이전트입니다. | [Repo](https://github.com/langchain-ai/retrieval-agent-template-js) |
| **Data-Enrichment Agent** | 웹 검색을 수행하고 발견 사항을 구조화된 형식으로 정리하는 에이전트입니다. | [Repo](https://github.com/langchain-ai/data-enrichment-js) |
:::

## 🌱 LangGraph 앱 만들기

템플릿에서 새 앱을 만들려면 `langgraph new` 명령을 사용하세요.

:::python

```bash
langgraph new
```

또는 [`uv`](https://docs.astral.sh/uv/getting-started/installation/) 사용 (권장):

```bash
uvx --from "langgraph-cli[inmem]" langgraph new
```

:::

:::js

```bash
npm create langgraph
```

:::

## 다음 단계

템플릿 및 커스터마이징 방법에 대한 자세한 내용은 새 LangGraph 앱의 루트에 있는 `README.md` 파일을 검토하세요.

앱을 올바르게 구성하고 API 키를 추가한 후 LangGraph CLI를 사용하여 앱을 시작할 수 있습니다:

:::python

```bash
langgraph dev
```

또는 [`uv`](https://docs.astral.sh/uv/getting-started/installation/) 사용 (권장):

```bash
uvx --from "langgraph-cli[inmem]" --with-editable . langgraph dev
```

!!! info "로컬 패키지 누락?"

    `uv`를 사용하지 않고 로컬 패키지를 설치(`pip install -e .`)한 후에도 "`ModuleNotFoundError`" 또는 "`ImportError`"가 발생하는 경우, CLI가 로컬 패키지를 "인식"할 수 있도록 CLI를 로컬 가상 환경에 설치해야 할 가능성이 높습니다. `python -m pip install "langgraph-cli[inmem]"`을 실행하고 `langgraph dev`를 실행하기 전에 가상 환경을 다시 활성화하면 됩니다.

:::

:::js

```bash
npx @langchain/langgraph-cli dev
```

:::

앱 배포 방법에 대한 자세한 내용은 다음 가이드를 참조하세요:

- **[로컬 LangGraph Server 시작](../tutorials/langgraph-platform/local-server.md)**: 이 빠른 시작 가이드는 **ReAct Agent** 템플릿에 대한 LangGraph Server를 로컬에서 시작하는 방법을 보여줍니다. 다른 템플릿에 대한 단계도 유사합니다.
- **[LangGraph Platform에 배포](../cloud/quick_start.md)**: LangGraph Platform을 사용하여 LangGraph 앱을 배포하세요.
