---
search:
  boost: 2
---

# 애플리케이션 구조

## 개요

LangGraph 애플리케이션은 하나 이상의 그래프, 구성 파일(`langgraph.json`), 의존성을 지정하는 파일, 그리고 환경 변수를 지정하는 선택적 `.env` 파일로 구성됩니다.

이 가이드는 애플리케이션의 일반적인 구조를 보여주고 LangGraph Platform을 사용하여 애플리케이션을 배포하는 데 필요한 정보를 지정하는 방법을 보여줍니다.

## 핵심 개념

LangGraph Platform을 사용하여 배포하려면 다음 정보를 제공해야 합니다:

1. 애플리케이션에 사용할 의존성, 그래프 및 환경 변수를 지정하는 [LangGraph 구성 파일](#configuration-file-concepts)(`langgraph.json`)
2. 애플리케이션의 로직을 구현하는 [그래프](#graphs)
3. 애플리케이션 실행에 필요한 [의존성](#dependencies)을 지정하는 파일
4. 애플리케이션 실행에 필요한 [환경 변수](#environment-variables)

## 파일 구조

다음은 애플리케이션의 디렉토리 구조 예시입니다:

:::python
=== "Python (requirements.txt)"

    ```plaintext
    my-app/
    ├── my_agent # 모든 프로젝트 코드가 여기에 있습니다
    │   ├── utils # 그래프를 위한 유틸리티
    │   │   ├── __init__.py
    │   │   ├── tools.py # 그래프를 위한 도구
    │   │   ├── nodes.py # 그래프를 위한 노드 함수
    │   │   └── state.py # 그래프의 상태 정의
    │   ├── __init__.py
    │   └── agent.py # 그래프를 구성하는 코드
    ├── .env # 환경 변수
    ├── requirements.txt # 패키지 의존성
    └── langgraph.json # LangGraph 구성 파일
    ```

=== "Python (pyproject.toml)"

    ```plaintext
    my-app/
    ├── my_agent # 모든 프로젝트 코드가 여기에 있습니다
    │   ├── utils # 그래프를 위한 유틸리티
    │   │   ├── __init__.py
    │   │   ├── tools.py # 그래프를 위한 도구
    │   │   ├── nodes.py # 그래프를 위한 노드 함수
    │   │   └── state.py # 그래프의 상태 정의
    │   ├── __init__.py
    │   └── agent.py # 그래프를 구성하는 코드
    ├── .env # 환경 변수
    ├── langgraph.json  # LangGraph 구성 파일
    └── pyproject.toml # 프로젝트 의존성
    ```

:::

:::js

```plaintext
my-app/
├── src # 모든 프로젝트 코드가 여기에 있습니다
│   ├── utils # 그래프를 위한 선택적 유틸리티
│   │   ├── tools.ts # 그래프를 위한 도구
│   │   ├── nodes.ts # 그래프를 위한 노드 함수
│   │   └── state.ts # 그래프의 상태 정의
│   └── agent.ts # 그래프를 구성하는 코드
├── package.json # 패키지 의존성
├── .env # 환경 변수
└── langgraph.json # LangGraph 구성 파일
```

:::

!!! note

    LangGraph 애플리케이션의 디렉토리 구조는 사용하는 프로그래밍 언어와 패키지 관리자에 따라 다를 수 있습니다.

## 구성 파일 {#configuration-file-concepts}

`langgraph.json` 파일은 LangGraph 애플리케이션을 배포하는 데 필요한 의존성, 그래프, 환경 변수 및 기타 설정을 지정하는 JSON 파일입니다.

JSON 파일에서 지원되는 모든 키에 대한 자세한 내용은 [LangGraph 구성 파일 참조](../cloud/reference/cli.md#configuration-file)를 참조하세요.

!!! tip

    [LangGraph CLI](./langgraph_cli.md)는 기본적으로 현재 디렉토리의 구성 파일 `langgraph.json`을 사용합니다.

### 예시

:::python

- 의존성에는 사용자 정의 로컬 패키지와 `langchain_openai` 패키지가 포함됩니다.
- 단일 그래프는 `variable` 변수를 사용하여 `./your_package/your_file.py` 파일에서 로드됩니다.
- 환경 변수는 `.env` 파일에서 로드됩니다.

```json
{
  "dependencies": ["langchain_openai", "./your_package"],
  "graphs": {
    "my_agent": "./your_package/your_file.py:agent"
  },
  "env": "./.env"
}
```

:::

:::js

- 의존성은 로컬 디렉토리의 의존성 파일(예: `package.json`)에서 로드됩니다.
- 단일 그래프는 `agent` 함수를 사용하여 `./your_package/your_file.js` 파일에서 로드됩니다.
- 환경 변수 `OPENAI_API_KEY`는 인라인으로 설정됩니다.

```json
{
  "dependencies": ["."],
  "graphs": {
    "my_agent": "./your_package/your_file.js:agent"
  },
  "env": {
    "OPENAI_API_KEY": "secret-key"
  }
}
```

:::

## 의존성 {#dependencies}

:::python
LangGraph 애플리케이션은 다른 Python 패키지에 의존할 수 있습니다.
:::

:::js
LangGraph 애플리케이션은 다른 TypeScript/JavaScript 라이브러리에 의존할 수 있습니다.
:::

의존성이 올바르게 설정되려면 일반적으로 다음 정보를 지정해야 합니다:

:::python

1. 의존성을 지정하는 디렉토리 내의 파일 (예: `requirements.txt`, `pyproject.toml` 또는 `package.json`)
   :::

:::js

1. 의존성을 지정하는 디렉토리 내의 파일 (예: `package.json`)
   :::

2. LangGraph 애플리케이션 실행에 필요한 의존성을 지정하는 [LangGraph 구성 파일](#configuration-file-concepts)의 `dependencies` 키
3. 추가 바이너리 또는 시스템 라이브러리는 [LangGraph 구성 파일](#configuration-file-concepts)의 `dockerfile_lines` 키를 사용하여 지정할 수 있습니다.

## 그래프 {#graphs}

배포된 LangGraph 애플리케이션에서 사용할 수 있는 그래프를 지정하려면 [LangGraph 구성 파일](#configuration-file-concepts)의 `graphs` 키를 사용합니다.

구성 파일에서 하나 이상의 그래프를 지정할 수 있습니다. 각 그래프는 이름(고유해야 함)과 다음 중 하나에 대한 경로로 식별됩니다: (1) 컴파일된 그래프 또는 (2) 그래프를 생성하는 함수가 정의된 경로

## 환경 변수 {#environment-variables}

로컬에서 배포된 LangGraph 애플리케이션으로 작업하는 경우 [LangGraph 구성 파일](#configuration-file-concepts)의 `env` 키에서 환경 변수를 구성할 수 있습니다.

프로덕션 배포의 경우 일반적으로 배포 환경에서 환경 변수를 구성하려고 할 것입니다.
