# LangGraph를 AutoGen, CrewAI 및 기타 프레임워크와 통합하는 방법

이 가이드는 AutoGen 에이전트를 LangGraph와 통합하여 지속성, 스트리밍 및 메모리와 같은 기능을 활용하고, 통합된 솔루션을 LangGraph Platform에 배포하여 확장 가능한 프로덕션 사용을 하는 방법을 보여줍니다. 이 가이드에서는 AutoGen과 통합되는 LangGraph 챗봇을 구축하는 방법을 보여주지만, 다른 프레임워크와도 동일한 접근 방식을 따를 수 있습니다.

AutoGen을 LangGraph와 통합하면 여러 이점이 있습니다:

- 향상된 기능: AutoGen 에이전트에 [지속성](../concepts/persistence.md), [스트리밍](../concepts/streaming.md), [단기 및 장기 메모리](../concepts/memory.md) 등을 추가합니다.
- 다중 에이전트 시스템: 개별 에이전트가 서로 다른 프레임워크로 구축된 [다중 에이전트 시스템](../concepts/multi_agent.md)을 구축합니다.
- 프로덕션 배포: 통합 솔루션을 [LangGraph Platform](../concepts/langgraph_platform.md)에 배포하여 확장 가능한 프로덕션 사용을 합니다.

## 사전 요구 사항

- Python 3.9+
- Autogen: `pip install autogen`
- LangGraph: `pip install langgraph`
- OpenAI API 키

## 설정

환경을 설정합니다:

```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
```

## 1. AutoGen 에이전트 정의

코드를 실행할 수 있는 AutoGen 에이전트를 생성합니다. 이 예제는 AutoGen의 [공식 튜토리얼](https://github.com/microsoft/autogen/blob/0.2/notebook/agentchat_web_info.ipynb)에서 적용되었습니다:

```python
import autogen
import os

config_list = [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]

llm_config = {
    "timeout": 600,
    "cache_seed": 42,
    "config_list": config_list,
    "temperature": 0,
}

autogen_agent = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "web",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    llm_config=llm_config,
    system_message="Reply TERMINATE if the task has been solved at full satisfaction. Otherwise, reply CONTINUE, or the reason why the task is not solved yet.",
)
```

## 2. 그래프 생성

이제 AutoGen 에이전트를 호출하는 LangGraph 챗봇 그래프를 생성합니다.

```python
from langchain_core.messages import convert_to_openai_messages
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import InMemorySaver

def call_autogen_agent(state: MessagesState):
    # Convert LangGraph messages to OpenAI format for AutoGen
    messages = convert_to_openai_messages(state["messages"])
    
    # Get the last user message
    last_message = messages[-1]
    
    # Pass previous message history as context (excluding the last message)
    carryover = messages[:-1] if len(messages) > 1 else []
    
    # Initiate chat with AutoGen
    response = user_proxy.initiate_chat(
        autogen_agent,
        message=last_message,
        carryover=carryover
    )
    
    # Extract the final response from the agent
    final_content = response.chat_history[-1]["content"]
    
    # Return the response in LangGraph format
    return {"messages": {"role": "assistant", "content": final_content}}

# Create the graph with memory for persistence
checkpointer = InMemorySaver()

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("autogen", call_autogen_agent)
builder.add_edge(START, "autogen")

# Compile with checkpointer for persistence
graph = builder.compile(checkpointer=checkpointer)
```

```python
from IPython.display import display, Image

display(Image(graph.get_graph().draw_mermaid_png()))
```

![Graph](./assets/autogen-output.png)

## 3. 로컬에서 그래프 테스트

LangGraph Platform에 배포하기 전에 로컬에서 그래프를 테스트할 수 있습니다:

```python
# pass the thread ID to persist agent outputs for future interactions
# highlight-next-line
config = {"configurable": {"thread_id": "1"}}

for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "Find numbers between 10 and 30 in fibonacci sequence",
            }
        ]
    },
    # highlight-next-line
    config,
):
    print(chunk)
```

**Output:**
```
user_proxy (to assistant):

Find numbers between 10 and 30 in fibonacci sequence

--------------------------------------------------------------------------------
assistant (to user_proxy):

To find numbers between 10 and 30 in the Fibonacci sequence, we can generate the Fibonacci sequence and check which numbers fall within this range. Here's a plan:

1. Generate Fibonacci numbers starting from 0.
2. Continue generating until the numbers exceed 30.
3. Collect and print the numbers that are between 10 and 30.

...
```

LangGraph의 [지속성](https://langchain-ai.github.io/langgraph/concepts/persistence/) 기능을 활용하고 있으므로 이제 동일한 thread ID를 사용하여 대화를 계속할 수 있습니다. LangGraph는 이전 히스토리를 자동으로 AutoGen 에이전트에 전달합니다:

```python
for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "Multiply the last number by 3",
            }
        ]
    },
    # highlight-next-line
    config,
):
    print(chunk)
```

**Output:**
```
user_proxy (to assistant):

Multiply the last number by 3
Context: 
Find numbers between 10 and 30 in fibonacci sequence
The Fibonacci numbers between 10 and 30 are 13 and 21. 

These numbers are part of the Fibonacci sequence, which is generated by adding the two preceding numbers to get the next number, starting from 0 and 1. 

The sequence goes: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...

As you can see, 13 and 21 are the only numbers in this sequence that fall between 10 and 30.

TERMINATE

--------------------------------------------------------------------------------
assistant (to user_proxy):

The last number in the Fibonacci sequence between 10 and 30 is 21. Multiplying 21 by 3 gives:

21 * 3 = 63

TERMINATE

--------------------------------------------------------------------------------
{'call_autogen_agent': {'messages': {'role': 'assistant', 'content': 'The last number in the Fibonacci sequence between 10 and 30 is 21. Multiplying 21 by 3 gives:\n\n21 * 3 = 63\n\nTERMINATE'}}}
```

## 4. 배포 준비

LangGraph Platform에 배포하려면 다음과 같은 파일 구조를 생성합니다:

```
my-autogen-agent/
├── agent.py          # 메인 에이전트 코드
├── requirements.txt  # Python 의존성
└── langgraph.json   # LangGraph 구성
```

=== "agent.py"

    ```python
    import os
    import autogen
    from langchain_core.messages import convert_to_openai_messages
    from langgraph.graph import StateGraph, MessagesState, START
    from langgraph.checkpoint.memory import InMemorySaver

    # AutoGen configuration
    config_list = [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]

    llm_config = {
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
        "temperature": 0,
    }

    # Create AutoGen agents
    autogen_agent = autogen.AssistantAgent(
        name="assistant",
        llm_config=llm_config,
    )

    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={
            "work_dir": "/tmp/autogen_work",
            "use_docker": False,
        },
        llm_config=llm_config,
        system_message="Reply TERMINATE if the task has been solved at full satisfaction.",
    )

    def call_autogen_agent(state: MessagesState):
        """AutoGen 에이전트를 호출하는 노드 함수"""
        messages = convert_to_openai_messages(state["messages"])
        last_message = messages[-1]
        carryover = messages[:-1] if len(messages) > 1 else []
        
        response = user_proxy.initiate_chat(
            autogen_agent,
            message=last_message,
            carryover=carryover
        )
        
        final_content = response.chat_history[-1]["content"]
        return {"messages": {"role": "assistant", "content": final_content}}

    # 그래프 생성 및 컴파일
    def create_graph():
        checkpointer = InMemorySaver()
        builder = StateGraph(MessagesState)
        builder.add_node("autogen", call_autogen_agent)
        builder.add_edge(START, "autogen")
        return builder.compile(checkpointer=checkpointer)

    # LangGraph Platform을 위한 그래프 내보내기
    graph = create_graph()
    ```

=== "requirements.txt"

    ```
    langgraph>=0.1.0
    ag2>=0.2.0
    langchain-core>=0.1.0
    langchain-openai>=0.0.5
    ```

=== "langgraph.json"

    ```json
    {
    "dependencies": ["."],
    "graphs": {
        "autogen_agent": "./agent.py:graph"
    },
    "env": ".env"
    }
    ```


## 5. LangGraph Platform에 배포

LangGraph Platform CLI를 사용하여 그래프를 배포합니다:

```
pip install -U langgraph-cli
```

```
langgraph deploy --config langgraph.json
```
