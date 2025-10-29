# Models

LangGraph는 LangChain 라이브러리를 통해 [LLM(언어 모델)](https://python.langchain.com/docs/concepts/chat_models/)에 대한 기본 지원을 제공합니다. 이를 통해 다양한 LLM을 에이전트와 워크플로우에 쉽게 통합할 수 있습니다.

## Initialize a model

:::python
[`init_chat_model`](https://python.langchain.com/docs/how_to/chat_models_universal_init/)을 사용하여 모델을 초기화합니다:

{% include-markdown "../../snippets/chat_model_tabs.md" %}
:::

:::js
모델 제공자 클래스를 사용하여 모델을 초기화합니다:

=== "OpenAI"

    ```typescript
    import { ChatOpenAI } from "@langchain/openai";

    const model = new ChatOpenAI({
      model: "gpt-4o",
      temperature: 0,
    });
    ```

=== "Anthropic"

    ```typescript
    import { ChatAnthropic } from "@langchain/anthropic";

    const model = new ChatAnthropic({
      model: "claude-3-5-sonnet-20240620",
      temperature: 0,
      maxTokens: 2048,
    });
    ```

=== "Google"

    ```typescript
    import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

    const model = new ChatGoogleGenerativeAI({
      model: "gemini-1.5-pro",
      temperature: 0,
    });
    ```

=== "Groq"

    ```typescript
    import { ChatGroq } from "@langchain/groq";

    const model = new ChatGroq({
      model: "llama-3.1-70b-versatile",
      temperature: 0,
    });
    ```

:::

:::python

### Instantiate a model directly

모델 제공자가 `init_chat_model`을 통해 사용할 수 없는 경우, 제공자의 모델 클래스를 직접 인스턴스화할 수 있습니다. 모델은 [BaseChatModel 인터페이스](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html)를 구현하고 도구 호출을 지원해야 합니다:

```python
# Anthropic is already supported by `init_chat_model`,
# but you can also instantiate it directly.
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(
  model="claude-3-7-sonnet-latest",
  temperature=0,
  max_tokens=2048
)
```

:::

!!! important "Tool calling support"

    모델이 외부 도구를 호출해야 하는 에이전트나 워크플로우를 구축하는 경우, 기본 언어 모델이 [도구 호출](../concepts/tools.md)을 지원하는지 확인하세요. 호환 가능한 모델은 [LangChain 통합 디렉토리](https://python.langchain.com/docs/integrations/chat/)에서 찾을 수 있습니다.

## Use in an agent

:::python
`create_react_agent`를 사용할 때 모델 이름 문자열로 모델을 지정할 수 있으며, 이는 `init_chat_model`을 사용하여 모델을 초기화하는 단축 표기법입니다. 이를 통해 모델을 직접 임포트하거나 인스턴스화할 필요 없이 모델을 사용할 수 있습니다.

=== "model name"

      ```python
      from langgraph.prebuilt import create_react_agent

      create_react_agent(
         # highlight-next-line
         model="anthropic:claude-3-7-sonnet-latest",
         # other parameters
      )
      ```

=== "model instance"

      ```python
      from langchain_anthropic import ChatAnthropic
      from langgraph.prebuilt import create_react_agent

      model = ChatAnthropic(
          model="claude-3-7-sonnet-latest",
          temperature=0,
          max_tokens=2048
      )
      # Alternatively
      # model = init_chat_model("anthropic:claude-3-7-sonnet-latest")

      agent = create_react_agent(
        # highlight-next-line
        model=model,
        # other parameters
      )
      ```

:::

:::js
`createReactAgent`를 사용할 때 모델 인스턴스를 직접 전달할 수 있습니다:

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { createReactAgent } from "@langchain/langgraph/prebuilt";

const model = new ChatOpenAI({
  model: "gpt-4o",
  temperature: 0,
});

const agent = createReactAgent({
  llm: model,
  tools: tools,
});
```

:::

:::python

### Dynamic model selection

런타임에 동적으로 모델을 선택하려면 `create_react_agent`에 호출 가능한 함수를 전달합니다. 이는 사용자 입력, 구성 설정 또는 기타 런타임 조건을 기반으로 모델을 선택하려는 시나리오에 유용합니다.

선택자 함수는 채팅 모델을 반환해야 합니다. 도구를 사용하는 경우, 선택자 함수 내에서 도구를 모델에 바인딩해야 합니다.

  ```python
from dataclasses import dataclass
from typing import Literal
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.runtime import Runtime

@tool
def weather() -> str:
    """Returns the current weather conditions."""
    return "It's nice and sunny."


# Define the runtime context
@dataclass
class CustomContext:
    provider: Literal["anthropic", "openai"]

# Initialize models
openai_model = init_chat_model("openai:gpt-4o")
anthropic_model = init_chat_model("anthropic:claude-sonnet-4-20250514")


# Selector function for model choice
def select_model(state: AgentState, runtime: Runtime[CustomContext]) -> BaseChatModel:
    if runtime.context.provider == "anthropic":
        model = anthropic_model
    elif runtime.context.provider == "openai":
        model = openai_model
    else:
        raise ValueError(f"Unsupported provider: {runtime.context.provider}")

    # With dynamic model selection, you must bind tools explicitly
    return model.bind_tools([weather])


# Create agent with dynamic model selection
agent = create_react_agent(select_model, tools=[weather])

# Invoke with context to select model
output = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Which model is handling this?",
            }
        ]
    },
    context=CustomContext(provider="openai"),
)

print(output["messages"][-1].text())
```

!!! version-added "Added in version 0.6.0"

:::

## Advanced model configuration

### Disable streaming

:::python
개별 LLM 토큰의 스트리밍을 비활성화하려면 모델을 초기화할 때 `disable_streaming=True`를 설정합니다:

=== "`init_chat_model`"

    ```python
    from langchain.chat_models import init_chat_model

    model = init_chat_model(
        "anthropic:claude-3-7-sonnet-latest",
        # highlight-next-line
        disable_streaming=True
    )
    ```

=== "`ChatModel`"

    ```python
    from langchain_anthropic import ChatAnthropic

    model = ChatAnthropic(
        model="claude-3-7-sonnet-latest",
        # highlight-next-line
        disable_streaming=True
    )
    ```

`disable_streaming`에 대한 자세한 내용은 [API reference](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html#langchain_core.language_models.chat_models.BaseChatModel.disable_streaming)를 참조하세요.
:::

:::js
개별 LLM 토큰의 스트리밍을 비활성화하려면 모델을 초기화할 때 `streaming: false`를 설정합니다:

```typescript
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({
  model: "gpt-4o",
  streaming: false,
});
```

:::

### Add model fallbacks

:::python
`model.with_fallbacks([...])`를 사용하여 다른 모델 또는 다른 LLM 제공자로의 폴백을 추가할 수 있습니다:

=== "`init_chat_model`"

    ```python
    from langchain.chat_models import init_chat_model

    model_with_fallbacks = (
        init_chat_model("anthropic:claude-3-5-haiku-latest")
        # highlight-next-line
        .with_fallbacks([
            init_chat_model("openai:gpt-4.1-mini"),
        ])
    )
    ```

=== "`ChatModel`"

    ```python
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI

    model_with_fallbacks = (
        ChatAnthropic(model="claude-3-5-haiku-latest")
        # highlight-next-line
        .with_fallbacks([
            ChatOpenAI(model="gpt-4.1-mini"),
        ])
    )
    ```

모델 폴백에 대한 자세한 내용은 이 [가이드](https://python.langchain.com/docs/how_to/fallbacks/#fallback-to-better-model)를 참조하세요.
:::

:::js
`model.withFallbacks([...])`를 사용하여 다른 모델 또는 다른 LLM 제공자로의 폴백을 추가할 수 있습니다:

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";

const modelWithFallbacks = new ChatOpenAI({
  model: "gpt-4o",
}).withFallbacks([
  new ChatAnthropic({
    model: "claude-3-5-sonnet-20240620",
  }),
]);
```

모델 폴백에 대한 자세한 내용은 이 [가이드](https://js.langchain.com/docs/how_to/fallbacks/#fallback-to-better-model)를 참조하세요.
:::

:::python

### Use the built-in rate limiter

Langchain에는 내장된 인메모리 속도 제한기가 포함되어 있습니다. 이 속도 제한기는 스레드 안전하며 동일한 프로세스의 여러 스레드에서 공유될 수 있습니다.

```python
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_anthropic import ChatAnthropic

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.
)

model = ChatAnthropic(
   model_name="claude-3-opus-20240229",
   rate_limiter=rate_limiter
)
```

[속도 제한을 처리하는 방법](https://python.langchain.com/docs/how_to/chat_model_rate_limiting/)에 대한 자세한 내용은 LangChain 문서를 참조하세요.
:::

## Bring your own model

원하는 LLM이 LangChain에서 공식적으로 지원되지 않는 경우 다음 옵션을 고려하세요:

:::python

1. **커스텀 LangChain 채팅 모델 구현**: [LangChain 채팅 모델 인터페이스](https://python.langchain.com/docs/how_to/custom_chat_model/)를 준수하는 모델을 생성합니다. 이를 통해 LangGraph의 에이전트 및 워크플로우와 완전히 호환되지만 LangChain 프레임워크에 대한 이해가 필요합니다.

   :::

:::js

1. **커스텀 LangChain 채팅 모델 구현**: [LangChain 채팅 모델 인터페이스](https://js.langchain.com/docs/how_to/custom_chat/)를 준수하는 모델을 생성합니다. 이를 통해 LangGraph의 에이전트 및 워크플로우와 완전히 호환되지만 LangChain 프레임워크에 대한 이해가 필요합니다.

   :::

2. **커스텀 스트리밍을 사용한 직접 호출**: `StreamWriter`를 사용하여 [커스텀 스트리밍 로직을 추가](../how-tos/streaming.md#use-with-any-llm)하여 모델을 직접 사용합니다.
   자세한 지침은 [커스텀 스트리밍 문서](../how-tos/streaming.md#use-with-any-llm)를 참조하세요. 이 접근 방식은 사전 구축된 에이전트 통합이 필요하지 않은 커스텀 워크플로우에 적합합니다.

## Additional resources

:::python

- [Multimodal inputs](https://python.langchain.com/docs/how_to/multimodal_inputs/)
- [Structured outputs](https://python.langchain.com/docs/how_to/structured_output/)
- [Model integration directory](https://python.langchain.com/docs/integrations/chat/)
- [Force model to call a specific tool](https://python.langchain.com/docs/how_to/tool_choice/)
- [All chat model how-to guides](https://python.langchain.com/docs/how_to/#chat-models)
- [Chat model integrations](https://python.langchain.com/docs/integrations/chat/)

  :::

:::js

- [Multimodal inputs](https://js.langchain.com/docs/how_to/multimodal_inputs/)
- [Structured outputs](https://js.langchain.com/docs/how_to/structured_output/)
- [Model integration directory](https://js.langchain.com/docs/integrations/chat/)
- [Force model to call a specific tool](https://js.langchain.com/docs/how_to/tool_choice/)
- [All chat model how-to guides](https://js.langchain.com/docs/how_to/#chat-models)
- [Chat model integrations](https://js.langchain.com/docs/integrations/chat/)

  :::
