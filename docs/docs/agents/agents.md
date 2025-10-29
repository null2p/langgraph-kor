---
search:
  boost: 2
tags:
  - agent
hide:
  - tags
---

# LangGraph quickstart

이 가이드는 에이전틱 시스템을 빠르고 안정적으로 구축할 수 있도록 설계된 LangGraph의 **사전 구축된**, **재사용 가능한** 컴포넌트를 설정하고 사용하는 방법을 보여줍니다.

## Prerequisites

이 튜토리얼을 시작하기 전에 다음이 준비되어 있는지 확인하세요:

- [Anthropic](https://console.anthropic.com/settings/keys) API 키

## 1. Install dependencies

아직 설치하지 않았다면 LangGraph와 LangChain을 설치하세요:

:::python

```
pip install -U langgraph "langchain[anthropic]"
```

!!! info

    `langchain[anthropic]`은 에이전트가 [모델](https://python.langchain.com/docs/integrations/chat/)을 호출할 수 있도록 설치됩니다.

:::

:::js

```bash
npm install @langchain/langgraph @langchain/core @langchain/anthropic
```

!!! info

    `@langchain/core` `@langchain/anthropic`은 에이전트가 [모델](https://js.langchain.com/docs/integrations/chat/)을 호출할 수 있도록 설치됩니다.

:::

## 2. Create an agent

:::python
에이전트를 생성하려면 @[`create_react_agent`][create_react_agent]를 사용합니다:

```python
from langgraph.prebuilt import create_react_agent

def get_weather(city: str) -> str:  # (1)!
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",  # (2)!
    tools=[get_weather],  # (3)!
    prompt="You are a helpful assistant"  # (4)!
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

1. 에이전트가 사용할 도구를 정의합니다. 도구는 일반 Python 함수로 정의할 수 있습니다. 더 고급 도구 사용 및 커스터마이징은 [tools](../how-tos/tool-calling.md) 페이지를 확인하세요.
2. 에이전트가 사용할 언어 모델을 제공합니다. 에이전트용 언어 모델 구성에 대한 자세한 내용은 [models](./models.md) 페이지를 확인하세요.
3. 모델이 사용할 도구 목록을 제공합니다.
4. 에이전트가 사용하는 언어 모델에 시스템 프롬프트(지침)를 제공합니다.
   :::

:::js
에이전트를 생성하려면 [`createReactAgent`](https://langchain-ai.github.io/langgraphjs/reference/functions/langgraph_prebuilt.createReactAgent.html)를 사용합니다:

```typescript
import { ChatAnthropic } from "@langchain/anthropic";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const getWeather = tool(
  // (1)!
  async ({ city }) => {
    return `It's always sunny in ${city}!`;
  },
  {
    name: "get_weather",
    description: "Get weather for a given city.",
    schema: z.object({
      city: z.string().describe("The city to get weather for"),
    }),
  }
);

const agent = createReactAgent({
  llm: new ChatAnthropic({ model: "anthropic:claude-3-5-sonnet-latest" }), // (2)!
  tools: [getWeather], // (3)!
  stateModifier: "You are a helpful assistant", // (4)!
});

// Run the agent
await agent.invoke({
  messages: [{ role: "user", content: "what is the weather in sf" }],
});
```

1. 에이전트가 사용할 도구를 정의합니다. 도구는 `tool` 함수를 사용하여 정의할 수 있습니다. 더 고급 도구 사용 및 커스터마이징은 [tools](./tools.md) 페이지를 확인하세요.
2. 에이전트가 사용할 언어 모델을 제공합니다. 에이전트용 언어 모델 구성에 대한 자세한 내용은 [models](./models.md) 페이지를 확인하세요.
3. 모델이 사용할 도구 목록을 제공합니다.
4. 에이전트가 사용하는 언어 모델에 시스템 프롬프트(지침)를 제공합니다.
   :::

## 3. Configure an LLM

:::python
temperature와 같은 특정 매개변수로 LLM을 구성하려면 [init_chat_model](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html)을 사용합니다:

```python
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

# highlight-next-line
model = init_chat_model(
    "anthropic:claude-3-7-sonnet-latest",
    # highlight-next-line
    temperature=0
)

agent = create_react_agent(
    # highlight-next-line
    model=model,
    tools=[get_weather],
)
```

:::

:::js
temperature와 같은 특정 매개변수로 LLM을 구성하려면 모델 인스턴스를 사용합니다:

```typescript
import { ChatAnthropic } from "@langchain/anthropic";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

// highlight-next-line
const model = new ChatAnthropic({
  model: "claude-3-5-sonnet-latest",
  // highlight-next-line
  temperature: 0,
});

const agent = createReactAgent({
  // highlight-next-line
  llm: model,
  tools: [getWeather],
});
```

:::

LLM 구성 방법에 대한 자세한 내용은 [Models](./models.md)를 참조하세요.

## 4. Add a custom prompt

프롬프트는 LLM에게 어떻게 동작해야 하는지 지시합니다. 다음 유형의 프롬프트 중 하나를 추가합니다:

- **Static**: 문자열은 **시스템 메시지**로 해석됩니다.
- **Dynamic**: 입력 또는 구성을 기반으로 **런타임**에 생성되는 메시지 리스트입니다.

=== "Static prompt"

    고정된 프롬프트 문자열 또는 메시지 리스트를 정의합니다:

    :::python
    ```python
    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_weather],
        # A static prompt that never changes
        # highlight-next-line
        prompt="Never answer questions about the weather."
    )

    agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
    ```
    :::

    :::js
    ```typescript
    import { createReactAgent } from "@langchain/langgraph/prebuilt";
    import { ChatAnthropic } from "@langchain/anthropic";

    const agent = createReactAgent({
      llm: new ChatAnthropic({ model: "anthropic:claude-3-5-sonnet-latest" }),
      tools: [getWeather],
      // A static prompt that never changes
      // highlight-next-line
      stateModifier: "Never answer questions about the weather."
    });

    await agent.invoke({
      messages: [{ role: "user", content: "what is the weather in sf" }]
    });
    ```
    :::

=== "Dynamic prompt"

    :::python
    에이전트의 state와 구성을 기반으로 메시지 리스트를 반환하는 함수를 정의합니다:

    ```python
    from langchain_core.messages import AnyMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.prebuilt.chat_agent_executor import AgentState
    from langgraph.prebuilt import create_react_agent

    # highlight-next-line
    def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:  # (1)!
        user_name = config["configurable"].get("user_name")
        system_msg = f"You are a helpful assistant. Address the user as {user_name}."
        return [{"role": "system", "content": system_msg}] + state["messages"]

    agent = create_react_agent(
        model="anthropic:claude-3-7-sonnet-latest",
        tools=[get_weather],
        # highlight-next-line
        prompt=prompt
    )

    agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
        # highlight-next-line
        config={"configurable": {"user_name": "John Smith"}}
    )
    ```

    1. 동적 프롬프트를 사용하면 LLM에 대한 입력을 구성할 때 메시지가 아닌 [context](./context.md)를 포함할 수 있습니다. 예를 들면:

        - `user_id`나 API 자격 증명과 같은 런타임에 전달되는 정보 (`config` 사용).
        - 다단계 추론 프로세스 중에 업데이트되는 내부 에이전트 state (`state` 사용).

        동적 프롬프트는 `state`와 `config`를 받아 LLM에 전송할 메시지 리스트를 반환하는 함수로 정의할 수 있습니다.
    :::

    :::js
    에이전트의 state와 구성을 기반으로 메시지를 반환하는 함수를 정의합니다:

    ```typescript
    import { type BaseMessageLike } from "@langchain/core/messages";
    import { type RunnableConfig } from "@langchain/core/runnables";
    import { createReactAgent } from "@langchain/langgraph/prebuilt";

    // highlight-next-line
    const dynamicPrompt = (state: { messages: BaseMessageLike[] }, config: RunnableConfig): BaseMessageLike[] => {  // (1)!
      const userName = config.configurable?.user_name;
      const systemMsg = `You are a helpful assistant. Address the user as ${userName}.`;
      return [{ role: "system", content: systemMsg }, ...state.messages];
    };

    const agent = createReactAgent({
      llm: "anthropic:claude-3-5-sonnet-latest",
      tools: [getWeather],
      // highlight-next-line
      stateModifier: dynamicPrompt
    });

    await agent.invoke(
      { messages: [{ role: "user", content: "what is the weather in sf" }] },
      // highlight-next-line
      { configurable: { user_name: "John Smith" } }
    );
    ```

    1. 동적 프롬프트를 사용하면 LLM에 대한 입력을 구성할 때 메시지가 아닌 [context](./context.md)를 포함할 수 있습니다. 예를 들면:

        - `user_id`나 API 자격 증명과 같은 런타임에 전달되는 정보 (`config` 사용).
        - 다단계 추론 프로세스 중에 업데이트되는 내부 에이전트 state (`state` 사용).

        동적 프롬프트는 `state`와 `config`를 받아 LLM에 전송할 메시지 리스트를 반환하는 함수로 정의할 수 있습니다.
    :::

자세한 내용은 [Context](./context.md)를 참조하세요.

## 5. Add memory

에이전트와의 다회전 대화를 허용하려면 에이전트를 생성할 때 checkpointer를 제공하여 [persistence](../concepts/persistence.md)를 활성화해야 합니다. 런타임에는 대화(세션)의 고유 식별자인 `thread_id`가 포함된 config를 제공해야 합니다:

:::python

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

# highlight-next-line
checkpointer = InMemorySaver()

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    # highlight-next-line
    checkpointer=checkpointer  # (1)!
)

# Run the agent
# highlight-next-line
config = {"configurable": {"thread_id": "1"}}
sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    # highlight-next-line
    config  # (2)!
)
ny_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what about new york?"}]},
    # highlight-next-line
    config
)
```

1. `checkpointer`를 사용하면 에이전트가 도구 호출 루프의 모든 단계에서 상태를 저장할 수 있습니다. 이를 통해 [단기 메모리](../how-tos/memory/add-memory.md#add-short-term-memory) 및 [human-in-the-loop](../concepts/human_in_the_loop.md) 기능이 가능합니다.
2. `thread_id`가 포함된 구성을 전달하여 향후 에이전트 호출 시 동일한 대화를 재개할 수 있습니다.
   :::

:::js

```typescript
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { MemorySaver } from "@langchain/langgraph";

// highlight-next-line
const checkpointer = new MemorySaver();

const agent = createReactAgent({
  llm: "anthropic:claude-3-5-sonnet-latest",
  tools: [getWeather],
  // highlight-next-line
  checkpointSaver: checkpointer, // (1)!
});

// Run the agent
// highlight-next-line
const config = { configurable: { thread_id: "1" } };
const sfResponse = await agent.invoke(
  { messages: [{ role: "user", content: "what is the weather in sf" }] },
  // highlight-next-line
  config // (2)!
);
const nyResponse = await agent.invoke(
  { messages: [{ role: "user", content: "what about new york?" }] },
  // highlight-next-line
  config
);
```

1. `checkpointSaver`를 사용하면 에이전트가 도구 호출 루프의 모든 단계에서 상태를 저장할 수 있습니다. 이를 통해 [단기 메모리](../how-tos/memory/add-memory.md#add-short-term-memory) 및 [human-in-the-loop](../concepts/human_in_the_loop.md) 기능이 가능합니다.
2. `thread_id`가 포함된 구성을 전달하여 향후 에이전트 호출 시 동일한 대화를 재개할 수 있습니다.
   :::

:::python
checkpointer를 활성화하면 제공된 checkpointer 데이터베이스(또는 `InMemorySaver`를 사용하는 경우 메모리)의 모든 단계에서 에이전트 state를 저장합니다.
:::

:::js
checkpointer를 활성화하면 제공된 checkpointer 데이터베이스(또는 `MemorySaver`를 사용하는 경우 메모리)의 모든 단계에서 에이전트 state를 저장합니다.
:::

위의 예제에서 에이전트가 동일한 `thread_id`로 두 번째로 호출될 때 첫 번째 대화의 원래 메시지 히스토리가 새 사용자 입력과 함께 자동으로 포함됩니다.

자세한 내용은 [Memory](../how-tos/memory/add-memory.md)를 참조하세요.

## 6. Configure structured output

:::python
스키마에 맞는 구조화된 응답을 생성하려면 `response_format` 매개변수를 사용합니다. 스키마는 `Pydantic` 모델 또는 `TypedDict`로 정의할 수 있습니다. 결과는 `structured_response` 필드를 통해 액세스할 수 있습니다.

```python
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent

class WeatherResponse(BaseModel):
    conditions: str

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    # highlight-next-line
    response_format=WeatherResponse  # (1)!
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

# highlight-next-line
response["structured_response"]
```

1.  `response_format`이 제공되면 에이전트 루프 끝에 별도의 단계가 추가됩니다: 에이전트 메시지 히스토리가 구조화된 출력을 생성하기 위해 구조화된 출력이 있는 LLM에 전달됩니다.

        이 LLM에 시스템 프롬프트를 제공하려면 튜플 `(prompt, schema)`를 사용합니다. 예: `response_format=(prompt, WeatherResponse)`.

    :::

:::js
스키마에 맞는 구조화된 응답을 생성하려면 `responseFormat` 매개변수를 사용합니다. 스키마는 `Zod` 스키마로 정의할 수 있습니다. 결과는 `structuredResponse` 필드를 통해 액세스할 수 있습니다.

```typescript
import { z } from "zod";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

const WeatherResponse = z.object({
  conditions: z.string(),
});

const agent = createReactAgent({
  llm: "anthropic:claude-3-5-sonnet-latest",
  tools: [getWeather],
  // highlight-next-line
  responseFormat: WeatherResponse, // (1)!
});

const response = await agent.invoke({
  messages: [{ role: "user", content: "what is the weather in sf" }],
});

// highlight-next-line
response.structuredResponse;
```

1.  `responseFormat`이 제공되면 에이전트 루프 끝에 별도의 단계가 추가됩니다: 에이전트 메시지 히스토리가 구조화된 출력을 생성하기 위해 구조화된 출력이 있는 LLM에 전달됩니다.

        이 LLM에 시스템 프롬프트를 제공하려면 객체 `{ prompt, schema }`를 사용합니다. 예: `responseFormat: { prompt, schema: WeatherResponse }`.

    :::

!!! Note "LLM post-processing"

    구조화된 출력은 스키마에 따라 응답을 형식화하기 위해 LLM에 대한 추가 호출이 필요합니다.

## Next steps

- [에이전트를 로컬에 배포](../tutorials/langgraph-platform/local-server.md)
- [사전 구축된 에이전트에 대해 자세히 알아보기](../agents/overview.md)
- [LangGraph Platform quickstart](../cloud/quick_start.md)
