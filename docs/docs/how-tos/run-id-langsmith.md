# LangSmith에서 그래프 run에 대한 커스텀 run ID 전달 또는 태그 및 메타데이터 설정 방법

!!! tip "사전 요구사항"
    이 가이드는 다음 내용에 익숙하다고 가정합니다:

    - [LangSmith 문서](https://docs.smith.langchain.com)
    - [LangSmith Platform](https://smith.langchain.com)
    - [RunnableConfig](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.config.RunnableConfig.html#langchain_core.runnables.config.RunnableConfig)
    - [추적에 메타데이터 및 태그 추가](https://docs.smith.langchain.com/how_to_guides/tracing/trace_with_langchain#add-metadata-and-tags-to-traces)
    - [run 이름 커스터마이징](https://docs.smith.langchain.com/how_to_guides/tracing/trace_with_langchain#customize-run-name)

그래프 run을 디버깅하는 것은 때때로 IDE나 터미널에서 어려울 수 있습니다. [LangSmith](https://docs.smith.langchain.com)를 사용하면 추적 데이터를 사용하여 LangGraph로 구축된 LLM 앱을 디버그, 테스트 및 모니터링할 수 있습니다 — 시작 방법에 대한 자세한 내용은 [LangSmith 문서](https://docs.smith.langchain.com)를 참조하세요.

그래프 호출 중에 생성된 추적을 더 쉽게 식별하고 분석하기 위해 런타임에 추가 구성을 설정할 수 있습니다([RunnableConfig](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.config.RunnableConfig.html#langchain_core.runnables.config.RunnableConfig) 참조):

| **Field**   | **Type**            | **Description**                                                                                                    |
|-------------|---------------------|--------------------------------------------------------------------------------------------------------------------|
| run_name    | `str`               | 이 호출에 대한 tracer run의 이름. 기본값은 클래스의 이름입니다.                                          |
| run_id      | `UUID`              | 이 호출에 대한 tracer run의 고유 식별자. 제공되지 않으면 새 UUID가 생성됩니다.                 |
| tags        | `List[str]`         | 이 호출 및 모든 하위 호출(예: LLM을 호출하는 Chain)에 대한 태그. 이를 사용하여 호출을 필터링할 수 있습니다.            |
| metadata    | `Dict[str, Any]`    | 이 호출 및 모든 하위 호출(예: LLM을 호출하는 Chain)에 대한 메타데이터. 키는 문자열이어야 하고 값은 JSON 직렬화 가능해야 합니다. |

LangGraph 그래프는 [LangChain Runnable Interface](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html)를 구현하며 `invoke`, `ainvoke`, `stream` 등의 메서드에서 두 번째 인수(`RunnableConfig`)를 허용합니다.

LangSmith 플랫폼을 사용하면 `run_name`, `run_id`, `tags` 및 `metadata`를 기반으로 추적을 검색하고 필터링할 수 있습니다.

## TLDR

```python
import uuid
# 랜덤 UUID 생성 -- UUID여야 합니다
config = {"run_id": uuid.uuid4()}, "tags": ["my_tag1"], "metadata": {"a": 5}}
# invoke, batch, ainvoke, astream_events 등과 같은
# 모든 표준 Runnable 메서드에서 작동합니다
graph.stream(inputs, config, stream_mode="values")
```

나머지 how-to 가이드에서는 전체 에이전트를 보여줍니다.

## 설정

먼저 필요한 패키지를 설치하고 API 키를 설정합니다

```python
%%capture --no-stderr
%pip install --quiet -U langgraph langchain_openai
```

```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
_set_env("LANGSMITH_API_KEY")
```

!!! tip
    LangSmith에 가입하여 LangGraph 프로젝트의 문제를 빠르게 발견하고 성능을 개선하세요. [LangSmith](https://docs.smith.langchain.com)를 사용하면 추적 데이터를 사용하여 LangGraph로 구축된 LLM 앱을 디버그, 테스트 및 모니터링할 수 있습니다 — 시작 방법에 대한 자세한 내용은 [여기](https://docs.smith.langchain.com)를 참조하세요.

## 그래프 정의

이 예제에서는 [prebuilt ReAct agent](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/)를 사용합니다.

```python
from langchain_openai import ChatOpenAI
from typing import Literal
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# 먼저 사용할 모델을 초기화합니다.
model = ChatOpenAI(model="gpt-4o", temperature=0)


# 이 튜토리얼에서는 두 도시(NYC & SF)의 날씨에 대해 미리 정의된 값을 반환하는 커스텀 도구를 사용합니다
@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]


# 그래프 정의
graph = create_react_agent(model, tools=tools)
```

## 그래프 실행

이제 그래프를 정의했으니 한 번 실행하고 LangSmith에서 추적을 확인해봅시다. LangSmith에서 추적에 쉽게 액세스할 수 있도록 config에 커스텀 `run_id`를 전달합니다.

이는 `LANGSMITH_API_KEY` 환경 변수를 설정했다고 가정합니다.

`LANGCHAIN_PROJECT` 환경 변수를 설정하여 추적할 프로젝트를 구성할 수도 있으며, 기본적으로 run은 `default` 프로젝트에 추적됩니다.

```python
import uuid


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages": [("user", "what is the weather in sf")]}

config = {"run_name": "agent_007", "tags": ["cats are awesome"]}

print_stream(graph.stream(inputs, config, stream_mode="values"))
```

**출력:**
```
================================ Human Message ==================================

what is the weather in sf
================================== Ai Message ===================================
Tool Calls:
  get_weather (call_9ZudXyMAdlUjptq9oMGtQo8o)
 Call ID: call_9ZudXyMAdlUjptq9oMGtQo8o
  Args:
    city: sf
================================= Tool Message ==================================
Name: get_weather

It's always sunny in sf
================================== Ai Message ===================================

The weather in San Francisco is currently sunny.
```

## LangSmith에서 추적 보기

이제 그래프를 실행했으니 LangSmith로 이동하여 추적을 확인해봅시다. 먼저 추적한 프로젝트(우리의 경우 default 프로젝트)를 클릭합니다. 커스텀 run 이름 "agent_007"을 가진 run을 볼 수 있습니다.

![LangSmith Trace View](assets/d38d1f2b-0f4c-4707-b531-a3c749de987f.png)

또한 제공된 태그나 메타데이터를 사용하여 나중에 추적을 필터링할 수 있습니다. 예를 들어,

![LangSmith Filter View](assets/410e0089-2ab8-46bb-a61a-827187fd46b3.png)
