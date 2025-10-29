# 도구

많은 AI 애플리케이션은 자연어를 통해 사용자와 상호작용합니다. 그러나 일부 사용 사례에서는 모델이 구조화된 입력을 사용하여 API, 데이터베이스 또는 파일 시스템과 같은 외부 시스템과 직접 인터페이스해야 합니다. 이러한 시나리오에서는 [도구 호출](../how-tos/tool-calling.md)을 통해 모델이 지정된 입력 스키마를 준수하는 요청을 생성할 수 있습니다.

:::python
**Tools** encapsulate a callable function and its input schema. These can be passed to compatible [chat models](https://python.langchain.com/docs/concepts/chat_models), allowing the model to decide whether to invoke a tool and with what arguments.
:::

:::js
**Tools** encapsulate a callable function and its input schema. These can be passed to compatible [chat models](https://js.langchain.com/docs/concepts/chat_models), allowing the model to decide whether to invoke a tool and with what arguments.
:::

## 도구 호출

![Diagram of a tool call by a model](./img/tool_call.png)

도구 호출은 일반적으로 **조건부**입니다. 사용자 입력과 사용 가능한 도구를 기반으로 모델은 도구 호출 요청을 발행할 수 있습니다. 이 요청은 도구 이름과 입력 인수를 지정하는 `tool_calls` 필드를 포함하는 `AIMessage` 객체로 반환됩니다:

:::python

```python
llm_with_tools.invoke("What is 2 multiplied by 3?")
# -> AIMessage(tool_calls=[{'name': 'multiply', 'args': {'a': 2, 'b': 3}, ...}])
```

```
AIMessage(
  tool_calls=[
    ToolCall(name="multiply", args={"a": 2, "b": 3}),
    ...
  ]
)
```

:::

:::js

```typescript
await llmWithTools.invoke("What is 2 multiplied by 3?");
```

```
AIMessage {
  tool_calls: [
    ToolCall {
      name: "multiply",
      args: { a: 2, b: 3 },
      ...
    },
    ...
  ]
}
```

:::

입력이 어떤 도구와도 관련이 없는 경우, 모델은 자연어 메시지만 반환합니다:

:::python

```python
llm_with_tools.invoke("Hello world!")  # -> AIMessage(content="Hello!")
```

:::

:::js

```typescript
await llmWithTools.invoke("Hello world!"); // { content: "Hello!" }
```

:::

중요한 점은 모델이 도구를 실행하지 않는다는 것입니다. 모델은 요청만 생성합니다. 도구 호출을 처리하고 결과를 반환하는 것은 별도의 실행자(예: 런타임 또는 에이전트)의 책임입니다.

자세한 내용은 [도구 호출 가이드](../how-tos/tool-calling.md)를 참조하세요.

## 사전 구축된 도구

LangChain은 API, 데이터베이스, 파일 시스템 및 웹 데이터를 포함한 일반적인 외부 시스템에 대한 사전 구축된 도구 통합을 제공합니다.

:::python
사용 가능한 도구는 [통합 디렉토리](https://python.langchain.com/docs/integrations/tools/)에서 찾아보세요.
:::

:::js
사용 가능한 도구는 [통합 디렉토리](https://js.langchain.com/docs/integrations/tools/)에서 찾아보세요.
:::

일반적인 카테고리:

- **Search**: Bing, SerpAPI, Tavily
- **Code execution**: Python REPL, Node.js REPL
- **Databases**: SQL, MongoDB, Redis
- **Web data**: Scraping and browsing
- **APIs**: OpenWeatherMap, NewsAPI, etc.

## 커스텀 도구

:::python
`@tool` 데코레이터 또는 일반 Python 함수를 사용하여 커스텀 도구를 정의할 수 있습니다. 예를 들어:

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
```

:::

:::js
`tool` 함수를 사용하여 커스텀 도구를 정의할 수 있습니다. 예를 들어:

```typescript
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const multiply = tool(
  (input) => {
    return input.a * input.b;
  },
  {
    name: "multiply",
    description: "Multiply two numbers.",
    schema: z.object({
      a: z.number(),
      b: z.number(),
    }),
  }
);
```

:::

자세한 내용은 [도구 호출 가이드](../how-tos/tool-calling.md)를 참조하세요.

## 도구 실행

모델은 도구를 호출할 시기를 결정하지만, 도구 호출의 실행은 런타임 컴포넌트에서 처리해야 합니다.

LangGraph는 이를 위한 사전 구축된 컴포넌트를 제공합니다:

:::python

- @[`ToolNode`][ToolNode]: A prebuilt node that executes tools.
- @[`create_react_agent`][create_react_agent]: Constructs a full agent that manages tool calling automatically.
:::

:::js

- @[ToolNode]: A prebuilt node that executes tools.
- @[`createReactAgent`][create_react_agent]: Constructs a full agent that manages tool calling automatically.
:::
