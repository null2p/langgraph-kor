# 런타임에 그래프 재빌드

새 실행을 위해 다른 구성으로 그래프를 재빌드해야 할 수 있습니다. 예를 들어, config에 따라 다른 그래프 state 또는 그래프 구조를 사용해야 할 수 있습니다. 이 가이드는 이를 수행하는 방법을 보여줍니다.

!!! note "Note"
    대부분의 경우 config를 기반으로 동작을 커스터마이징하는 것은 각 노드가 config를 읽고 이를 기반으로 동작을 변경할 수 있는 단일 그래프에서 처리해야 합니다

## Prerequisites

먼저 배포를 위한 앱 설정에 대한 [이 how-to 가이드](./setup.md)를 확인하세요.

## Define graphs

LLM을 호출하고 사용자에게 응답을 반환하는 간단한 그래프가 있는 앱이 있다고 가정해 봅시다. 앱 파일 디렉토리는 다음과 같습니다:

```
my-app/
|-- requirements.txt
|-- .env
|-- openai_agent.py     # code for your graph
```

여기서 그래프는 `openai_agent.py`에 정의되어 있습니다.

### No rebuild

표준 LangGraph API 구성에서 서버는 `openai_agent.py`의 최상위 레벨에 정의된 컴파일된 그래프 인스턴스를 사용하며, 다음과 같습니다:

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessageGraph

model = ChatOpenAI(temperature=0)

graph_workflow = MessageGraph()

graph_workflow.add_node("agent", model)
graph_workflow.add_edge("agent", END)
graph_workflow.add_edge(START, "agent")

agent = graph_workflow.compile()
```

서버가 그래프를 인식하도록 하려면 LangGraph API 구성(`langgraph.json`)에서 `CompiledStateGraph` 인스턴스를 포함하는 변수의 경로를 지정해야 합니다. 예:

```
{
    "dependencies": ["."],
    "graphs": {
        "openai_agent": "./openai_agent.py:agent",
    },
    "env": "./.env"
}
```

### Rebuild

커스텀 구성으로 새 실행마다 그래프를 재빌드하려면 `openai_agent.py`를 다시 작성하여 config를 받아 그래프(또는 컴파일된 그래프) 인스턴스를 반환하는 _함수_를 제공해야 합니다. 사용자 ID '1'에 대해서는 기존 그래프를 반환하고 다른 사용자에 대해서는 도구 호출 에이전트를 반환하려고 한다고 가정해 봅시다. 다음과 같이 `openai_agent.py`를 수정할 수 있습니다:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessageGraph
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


model = ChatOpenAI(temperature=0)

def make_default_graph():
    """Make a simple LLM agent"""
    graph_workflow = StateGraph(State)
    def call_model(state):
        return {"messages": [model.invoke(state["messages"])]}

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge("agent", END)
    graph_workflow.add_edge(START, "agent")

    agent = graph_workflow.compile()
    return agent


def make_alternative_graph():
    """Make a tool-calling agent"""

    @tool
    def add(a: float, b: float):
        """Adds two numbers."""
        return a + b

    tool_node = ToolNode([add])
    model_with_tools = model.bind_tools([add])
    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def should_continue(state: State):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END

    graph_workflow = StateGraph(State)

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)
    graph_workflow.add_edge("tools", "agent")
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_conditional_edges("agent", should_continue)

    agent = graph_workflow.compile()
    return agent


# this is the graph making function that will decide which graph to
# build based on the provided config
def make_graph(config: RunnableConfig):
    user_id = config.get("configurable", {}).get("user_id")
    # route to different graph state / structure based on the user ID
    if user_id == "1":
        return make_default_graph()
    else:
        return make_alternative_graph()
```

마지막으로 `langgraph.json`에서 그래프 생성 함수(`make_graph`)의 경로를 지정해야 합니다:

```
{
    "dependencies": ["."],
    "graphs": {
        "openai_agent": "./openai_agent.py:make_graph",
    },
    "env": "./.env"
}
```

LangGraph API 구성 파일에 대한 자세한 정보는 [여기](../reference/cli.md#configuration-file)를 참조하세요.
