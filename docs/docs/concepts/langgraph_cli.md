---
search:
  boost: 2
---

# LangGraph CLI

**LangGraph CLI**는 [LangGraph API 서버](./langgraph_server.md)를 로컬에서 빌드하고 실행하기 위한 멀티 플랫폼 명령줄 도구입니다. 결과 서버에는 그래프의 run, thread, 어시스턴트 등을 위한 모든 API 엔드포인트와 체크포인팅 및 스토리지를 위한 관리형 데이터베이스를 포함하여 에이전트를 실행하는 데 필요한 기타 서비스가 포함됩니다.

:::python

## 설치

LangGraph CLI는 pip 또는 [Homebrew](https://brew.sh/)를 통해 설치할 수 있습니다:

=== "pip"

    ```bash
    pip install langgraph-cli
    ```

=== "Homebrew"

    ```bash
    brew install langgraph-cli
    ```
:::

:::js

## 설치

LangGraph.js CLI는 NPM 레지스트리에서 설치할 수 있습니다:

=== "npx"
    ```bash
    npx @langchain/langgraph-cli
    ```

=== "npm"
    ```bash
    npm install @langchain/langgraph-cli
    ```

=== "yarn"
    ```bash
    yarn add @langchain/langgraph-cli
    ```

=== "pnpm"
    ```bash
    pnpm add @langchain/langgraph-cli
    ```

=== "bun"
    ```bash
    bun add @langchain/langgraph-cli
    ```
:::

## 명령

LangGraph CLI는 다음과 같은 핵심 기능을 제공합니다:

| 명령                                                        | 설명                                                                                                                                                                                                                                                                            |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`langgraph build`](../cloud/reference/cli.md#build)           | 직접 배포할 수 있는 [LangGraph API 서버](./langgraph_server.md)를 위한 Docker 이미지를 빌드합니다.                                                                                                                                                                             |
| [`langgraph dev`](../cloud/reference/cli.md#dev)               | Docker 설치가 필요하지 않은 경량 개발 서버를 시작합니다. 이 서버는 빠른 개발 및 테스트에 이상적입니다.                                                                                                                                                                                  |
| [`langgraph dockerfile`](../cloud/reference/cli.md#dockerfile) | [LangGraph API 서버](./langgraph_server.md)의 인스턴스를 위한 이미지를 빌드하고 배포하는 데 사용할 수 있는 [Dockerfile](https://docs.docker.com/reference/dockerfile/)을 생성합니다. Dockerfile을 추가로 사용자 정의하거나 더 맞춤화된 방식으로 배포하려는 경우 유용합니다. |
| [`langgraph up`](../cloud/reference/cli.md#up)                 | Docker 컨테이너에서 로컬로 [LangGraph API 서버](./langgraph_server.md)의 인스턴스를 시작합니다. 이를 위해서는 Docker 서버가 로컬에서 실행되고 있어야 합니다. 또한 로컬 개발을 위한 LangSmith API 키 또는 프로덕션 사용을 위한 라이선스 키가 필요합니다.                          |

자세한 내용은 [LangGraph CLI 레퍼런스](../cloud/reference/cli.md)를 참조하세요.
