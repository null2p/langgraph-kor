# LangGraph Prebuilt

이 라이브러리는 LangGraph 에이전트와 도구를 생성하고 실행하기 위한 상위 수준 API를 정의합니다.

> [!IMPORTANT]
> 이 라이브러리는 `langgraph`와 함께 번들로 제공되므로 직접 설치하지 마세요

## Agents

`langgraph-prebuilt`는 도구 호출 [ReAct 스타일](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#react-implementation) 에이전트의 [구현](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent)을 제공합니다 - `create_react_agent`:

```bash
pip install langchain-anthropic
```

```python
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

# 에이전트가 사용할 도구를 정의합니다
def search(query: str):
    """웹을 검색하기 위한 호출입니다."""
    # 이것은 플레이스홀더입니다만, LLM에게는 말하지 마세요...
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

tools = [search]
model = ChatAnthropic(model="claude-3-7-sonnet-latest")

app = create_react_agent(model, tools)
# 에이전트를 실행합니다
app.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
)
```

## Tools

### ToolNode

`langgraph-prebuilt`는 도구 호출을 실행하는 노드의 [구현](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.tool_node.ToolNode)을 제공합니다 - `ToolNode`:

```python
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

def search(query: str):
    """웹을 검색하기 위한 호출입니다."""
    # 이것은 플레이스홀더입니다만, LLM에게는 말하지 마세요...
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

tool_node = ToolNode([search])
tool_calls = [{"name": "search", "args": {"query": "what is the weather in sf"}, "id": "1"}]
ai_message = AIMessage(content="", tool_calls=tool_calls)
# 도구 호출을 실행합니다
tool_node.invoke({"messages": [ai_message]})
```

### ValidationNode

`langgraph-prebuilt`는 pydantic 스키마에 대해 도구 호출을 검증하는 노드의 [구현](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.tool_validator.ValidationNode)을 제공합니다 - `ValidationNode`:

```python
from pydantic import BaseModel, field_validator
from langgraph.prebuilt import ValidationNode
from langchain_core.messages import AIMessage


class SelectNumber(BaseModel):
    a: int

    @field_validator("a")
    def a_must_be_meaningful(cls, v):
        if v != 37:
            raise ValueError("Only 37 is allowed")
        return v

validation_node = ValidationNode([SelectNumber])
validation_node.invoke({
    "messages": [AIMessage("", tool_calls=[{"name": "SelectNumber", "args": {"a": 42}, "id": "1"}])]
})
```

## Agent Inbox

이 라이브러리는 LangGraph 에이전트와 함께 [Agent Inbox](https://github.com/langchain-ai/agent-inbox)를 사용하기 위한 스키마를 포함합니다. Agent Inbox 사용 방법에 대한 자세한 내용은 [여기](https://github.com/langchain-ai/agent-inbox#interrupts)에서 확인하세요.

```python
from langgraph.types import interrupt
from langgraph.prebuilt.interrupt import HumanInterrupt, HumanResponse

def my_graph_function():
    # 상태의 `messages` 필드에서 마지막 도구 호출을 추출합니다
    tool_call = state["messages"][-1].tool_calls[0]
    # interrupt를 생성합니다
    request: HumanInterrupt = {
        "action_request": {
            "action": tool_call['name'],
            "args": tool_call['args']
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": False,
            "allow_accept": False
        },
        "description": _generate_email_markdown(state) # 상세한 마크다운 설명을 생성합니다.
    }
    # interrupt 요청을 리스트 안에 넣어 전송하고, 첫 번째 응답을 추출합니다
    response = interrupt([request])[0]
    if response['type'] == "response":
        # 응답으로 무언가를 수행합니다
    ...
```