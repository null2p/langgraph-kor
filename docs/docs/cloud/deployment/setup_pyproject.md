# pyproject.toml로 LangGraph 애플리케이션 설정 방법

LangGraph 애플리케이션을 LangGraph Platform에 배포하거나 셀프 호스팅하려면 [LangGraph 구성 파일](../reference/cli.md#configuration-file)로 구성해야 합니다. 이 가이드는 `pyproject.toml`을 사용하여 패키지의 의존성을 정의하고 배포를 위한 LangGraph 애플리케이션을 설정하는 기본 단계를 다룹니다.

이 워크스루는 [이 리포지토리](https://github.com/langchain-ai/langgraph-example-pyproject)를 기반으로 하며, LangGraph 애플리케이션을 배포용으로 설정하는 방법을 자세히 알아보기 위해 사용해볼 수 있습니다.

!!! tip "requirements.txt로 설정"
    의존성 관리에 `requirements.txt` 사용을 선호하는 경우, [이 가이드](./setup.md)를 참조하세요.

!!! tip "모노레포로 설정"
    모노레포 내에 위치한 그래프를 배포하는 데 관심이 있다면, 방법의 예시로 [이](https://github.com/langchain-ai/langgraph-example-monorepo) 리포지토리를 살펴보세요.

The final repository structure will look something like this:

```bash
my-app/
├── my_agent # all project code lies within here
│   ├── utils # utilities for your graph
│   │   ├── __init__.py
│   │   ├── tools.py # tools for your graph
│   │   ├── nodes.py # node functions for you graph
│   │   └── state.py # state definition of your graph
│   ├── __init__.py
│   └── agent.py # code for constructing your graph
├── .env # environment variables
├── langgraph.json  # configuration file for LangGraph
└── pyproject.toml # dependencies for your project
```

각 단계 후에는 코드를 어떻게 구성할 수 있는지 보여주는 예제 파일 디렉토리가 제공됩니다.

## 의존성 지정

의존성은 선택적으로 다음 파일 중 하나에 지정할 수 있습니다: `pyproject.toml`, `setup.py` 또는 `requirements.txt`. 이러한 파일이 생성되지 않은 경우 나중에 [LangGraph 구성 파일](#create-langgraph-configuration-file)에서 의존성을 지정할 수 있습니다.

아래 의존성은 이미지에 포함되며, 호환 가능한 버전 범위로 코드에서도 사용할 수 있습니다:

```
langgraph>=0.3.27
langgraph-sdk>=0.1.66
langgraph-checkpoint>=2.0.23
langchain-core>=0.2.38
langsmith>=0.1.63
orjson>=3.9.7,<3.10.17
httpx>=0.25.0
tenacity>=8.0.0
uvicorn>=0.26.0
sse-starlette>=2.1.0,<2.2.0
uvloop>=0.18.0
httptools>=0.5.0
jsonschema-rs>=0.20.0
structlog>=24.1.0
cloudpickle>=3.0.0
```

Example `pyproject.toml` file:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-agent"
version = "0.0.1"
description = "An excellent agent build for LangGraph Platform."
authors = [
    {name = "Polly the parrot", email = "1223+polly@users.noreply.github.com"}
]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "langgraph>=0.2.0",
    "langchain-fireworks>=0.1.3"
]

[tool.hatch.build.targets.wheel]
packages = ["my_agent"]
```

Example file directory:

```bash
my-app/
└── pyproject.toml   # Python packages required for your graph
```

## 환경 변수 지정

환경 변수는 선택적으로 파일(예: `.env`)에 지정할 수 있습니다. 배포를 위한 추가 변수 구성은 [환경 변수 레퍼런스](../reference/env_var.md)를 참조하세요.

Example `.env` file:

```
MY_ENV_VAR_1=foo
MY_ENV_VAR_2=bar
FIREWORKS_API_KEY=key
```

Example file directory:

```bash
my-app/
├── .env # file with environment variables
└── pyproject.toml
```

## 그래프 정의

그래프를 구현하세요! 그래프는 단일 파일 또는 여러 파일로 정의할 수 있습니다. LangGraph 애플리케이션에 포함될 각 @[CompiledStateGraph][CompiledStateGraph]의 변수 이름을 기록해두세요. 변수 이름은 나중에 [LangGraph 구성 파일](../reference/cli.md#configuration-file)을 생성할 때 사용됩니다.

정의한 다른 모듈에서 가져오는 방법을 보여주는 예제 `agent.py` 파일입니다(모듈에 대한 코드는 여기에 표시되지 않으며, 구현을 보려면 [이 리포지토리](https://github.com/langchain-ai/langgraph-example-pyproject)를 참조하세요):

```python
# my_agent/agent.py
from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END, START
from my_agent.utils.nodes import call_model, should_continue, tool_node # import nodes
from my_agent.utils.state import AgentState # import state

# Define the runtime context
class GraphContext(TypedDict):
    model_name: Literal["anthropic", "openai"]

workflow = StateGraph(AgentState, context_schema=GraphContext)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)
workflow.add_edge("action", "agent")

graph = workflow.compile()
```

Example file directory:

```bash
my-app/
├── my_agent # all project code lies within here
│   ├── utils # utilities for your graph
│   │   ├── __init__.py
│   │   ├── tools.py # tools for your graph
│   │   ├── nodes.py # node functions for you graph
│   │   └── state.py # state definition of your graph
│   ├── __init__.py
│   └── agent.py # code for constructing your graph
├── .env
└── pyproject.toml
```

## LangGraph 구성 파일 생성

`langgraph.json`이라는 [LangGraph 구성 파일](../reference/cli.md#configuration-file)을 생성합니다. 구성 파일의 JSON 객체에서 각 키에 대한 자세한 설명은 [LangGraph 구성 파일 레퍼런스](../reference/cli.md#configuration-file)를 참조하세요.

Example `langgraph.json` file:

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./my_agent/agent.py:graph"
  },
  "env": ".env"
}
```

`CompiledGraph`의 변수 이름이 최상위 `graphs` 키의 각 하위 키 값 끝에 나타납니다(즉, `:<variable_name>`).

!!! warning "구성 파일 위치"
    LangGraph 구성 파일은 컴파일된 그래프와 관련 의존성을 포함하는 Python 파일과 같은 레벨 또는 더 높은 레벨의 디렉토리에 배치되어야 합니다.

Example file directory:

```bash
my-app/
├── my_agent # all project code lies within here
│   ├── utils # utilities for your graph
│   │   ├── __init__.py
│   │   ├── tools.py # tools for your graph
│   │   ├── nodes.py # node functions for you graph
│   │   └── state.py # state definition of your graph
│   ├── __init__.py
│   └── agent.py # code for constructing your graph
├── .env # environment variables
├── langgraph.json  # configuration file for LangGraph
└── pyproject.toml # dependencies for your project
```

## 다음 단계

프로젝트를 설정하고 GitHub 리포지토리에 배치한 후에는 [앱을 배포](./cloud.md)할 차례입니다.
