# 기본 챗봇 만들기

이 튜토리얼에서는 기본 챗봇을 구축합니다. 이 챗봇은 다음 일련의 튜토리얼의 기반이 되며, 점진적으로 더 정교한 기능을 추가하면서 LangGraph의 핵심 개념을 소개받게 됩니다. 시작해 봅시다! 🌟

## 사전 요구 사항

이 튜토리얼을 시작하기 전에 [OpenAI](https://platform.openai.com/api-keys),
[Anthropic](https://console.anthropic.com/settings/keys), 또는
[Google Gemini](https://ai.google.dev/gemini-api/docs/api-key)와 같은 도구 호출 기능을 지원하는 LLM에 대한 액세스 권한이 있는지 확인하세요.

## 1. 패키지 설치

필요한 패키지를 설치합니다:

:::python

```bash
pip install -U langgraph langsmith
```

:::

:::js
=== "npm"

    ```bash
    npm install @langchain/langgraph @langchain/core zod
    ```

=== "yarn"

    ```bash
    yarn add @langchain/langgraph @langchain/core zod
    ```

=== "pnpm"

    ```bash
    pnpm add @langchain/langgraph @langchain/core zod
    ```

=== "bun"

    ```bash
    bun add @langchain/langgraph @langchain/core zod
    ```

:::

!!! tip

    LangSmith에 가입하여 LangGraph 프로젝트의 문제를 빠르게 찾고 성능을 향상시키세요. LangSmith를 사용하면 추적 데이터를 사용하여 LangGraph로 구축한 LLM 앱을 디버그, 테스트 및 모니터링할 수 있습니다. 시작 방법에 대한 자세한 내용은 [LangSmith 문서](https://docs.smith.langchain.com)를 참조하세요.

## 2. `StateGraph` 생성

이제 LangGraph를 사용하여 기본 챗봇을 만들 수 있습니다. 이 챗봇은 사용자 메시지에 직접 응답합니다.

먼저 `StateGraph`를 생성합니다. `StateGraph` 객체는 챗봇의 구조를 "상태 머신"으로 정의합니다. LLM과 챗봇이 호출할 수 있는 함수를 나타내는 `노드`를 추가하고, 봇이 이러한 함수 간에 전환하는 방법을 지정하는 `엣지`를 추가합니다.

:::python

```python
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    # Messages는 "list" 타입을 가집니다. 주석의 `add_messages` 함수는
    # 이 상태 키를 업데이트하는 방법을 정의합니다
    # (이 경우 메시지를 덮어쓰지 않고 목록에 추가합니다)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
```

:::

:::js

```typescript
import { StateGraph, MessagesZodState, START } from "@langchain/langgraph";
import { z } from "zod";

const State = z.object({ messages: MessagesZodState.shape.messages });

const graph = new StateGraph(State).compile();
```

:::

이제 그래프는 두 가지 주요 작업을 처리할 수 있습니다:

1. 각 `노드`는 현재 `State`를 입력으로 받고 상태에 대한 업데이트를 출력할 수 있습니다.
2. `messages`에 대한 업데이트는 미리 빌드된 reducer 함수 덕분에 덮어쓰지 않고 기존 목록에 추가됩니다.

!!! tip "개념"

    그래프를 정의할 때 첫 번째 단계는 `State`를 정의하는 것입니다. `State`에는 그래프의 스키마와 상태 업데이트를 처리하는 [reducer 함수](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers)가 포함됩니다. 예제에서 `State`는 `messages`라는 하나의 키를 가진 스키마입니다. reducer 함수는 메시지를 덮어쓰지 않고 목록에 추가하는 데 사용됩니다. reducer 주석이 없는 키는 이전 값을 덮어씁니다.

    state, reducer 및 관련 개념에 대한 자세한 내용은 [LangGraph 레퍼런스 문서](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.message.add_messages)를 참조하세요.

## 3. 노드 추가

다음으로 "`chatbot`" 노드를 추가합니다. **노드**는 작업 단위를 나타내며 일반적으로 일반 함수입니다.

먼저 채팅 모델을 선택합니다:

:::python

{% include-markdown "../../../snippets/chat_model_tabs.md" %}

<!---
```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
```
-->

:::

:::js

```typescript
import { ChatOpenAI } from "@langchain/openai";
// or import { ChatAnthropic } from "@langchain/anthropic";

const llm = new ChatOpenAI({
  model: "gpt-4o",
  temperature: 0,
});
```

:::

이제 채팅 모델을 간단한 노드에 통합할 수 있습니다:

:::python

```python

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# 첫 번째 인수는 고유한 노드 이름입니다
# 두 번째 인수는 노드가 사용될 때마다 호출될 함수 또는 객체입니다.
graph_builder.add_node("chatbot", chatbot)
```

:::

:::js

```typescript hl_lines="7-9"
import { StateGraph, MessagesZodState, START } from "@langchain/langgraph";
import { z } from "zod";

const State = z.object({ messages: MessagesZodState.shape.messages });

const graph = new StateGraph(State)
  .addNode("chatbot", async (state: z.infer<typeof State>) => {
    return { messages: [await llm.invoke(state.messages)] };
  })
  .compile();
```

:::

`chatbot` 노드 함수가 현재 `State`를 입력으로 받고 "messages" 키 아래에 업데이트된 `messages` 목록을 포함하는 딕셔너리를 반환하는 방식에 **주목하세요**. 이것이 모든 LangGraph 노드 함수의 기본 패턴입니다.

:::python
`State`의 `add_messages` 함수는 LLM의 응답 메시지를 상태에 이미 있는 메시지에 추가합니다.
:::

:::js
`MessagesZodState` 내에서 사용되는 `addMessages` 함수는 LLM의 응답 메시지를 상태에 이미 있는 메시지에 추가합니다.
:::

## 4. `entry` 포인트 추가

그래프가 실행될 때마다 **작업을 시작할 위치**를 알려주기 위해 `entry` 포인트를 추가합니다:

:::python

```python
graph_builder.add_edge(START, "chatbot")
```

:::

:::js

```typescript hl_lines="10"
import { StateGraph, MessagesZodState, START } from "@langchain/langgraph";
import { z } from "zod";

const State = z.object({ messages: MessagesZodState.shape.messages });

const graph = new StateGraph(State)
  .addNode("chatbot", async (state: z.infer<typeof State>) => {
    return { messages: [await llm.invoke(state.messages)] };
  })
  .addEdge(START, "chatbot")
  .compile();
```

:::

## 5. `exit` 포인트 추가

**그래프가 실행을 종료해야 하는 위치**를 나타내기 위해 `exit` 포인트를 추가합니다. 이것은 더 복잡한 흐름에 유용하지만, 이와 같은 간단한 그래프에서도 종료 노드를 추가하면 명확성이 향상됩니다.

:::python

```python
graph_builder.add_edge("chatbot", END)
```

:::

:::js

```typescript hl_lines="11"
import { StateGraph, MessagesZodState, START, END } from "@langchain/langgraph";
import { z } from "zod";

const State = z.object({ messages: MessagesZodState.shape.messages });

const graph = new StateGraph(State)
  .addNode("chatbot", async (state: z.infer<typeof State>) => {
    return { messages: [await llm.invoke(state.messages)] };
  })
  .addEdge(START, "chatbot")
  .addEdge("chatbot", END)
  .compile();
```

:::

이것은 챗봇 노드를 실행한 후 그래프를 종료하도록 지시합니다.

## 6. 그래프 컴파일

그래프를 실행하기 전에 컴파일해야 합니다. 그래프 빌더에서 `compile()`을 호출하여 이를 수행할 수 있습니다. 이렇게 하면 상태에서 호출할 수 있는 `CompiledGraph`가 생성됩니다.

:::python

```python
graph = graph_builder.compile()
```

:::

:::js

```typescript hl_lines="12"
import { StateGraph, MessagesZodState, START, END } from "@langchain/langgraph";
import { z } from "zod";

const State = z.object({ messages: MessagesZodState.shape.messages });

const graph = new StateGraph(State)
  .addNode("chatbot", async (state: z.infer<typeof State>) => {
    return { messages: [await llm.invoke(state.messages)] };
  })
  .addEdge(START, "chatbot")
  .addEdge("chatbot", END)
  .compile();
```

:::

## 7. 그래프 시각화 (선택 사항)

:::python
`get_graph` 메서드와 `draw_ascii` 또는 `draw_png`와 같은 "draw" 메서드 중 하나를 사용하여 그래프를 시각화할 수 있습니다. 각 `draw` 메서드에는 추가 종속성이 필요합니다.

```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # 이것은 추가 종속성이 필요하며 선택 사항입니다
    pass
```

:::

:::js
`getGraph` 메서드를 사용하여 그래프를 시각화하고 `drawMermaidPng` 메서드로 그래프를 렌더링할 수 있습니다.

```typescript
import * as fs from "node:fs/promises";

const drawableGraph = await graph.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("basic-chatbot.png", imageBuffer);
```

:::

![basic chatbot diagram](basic-chatbot.png)

## 8. 챗봇 실행

이제 챗봇을 실행하세요!

!!! tip

    `quit`, `exit` 또는 `q`를 입력하여 언제든지 채팅 루프를 종료할 수 있습니다.

:::python

```python
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # input()을 사용할 수 없는 경우 대체
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
```

:::

:::js

```typescript
import { HumanMessage } from "@langchain/core/messages";

async function streamGraphUpdates(userInput: string) {
  const stream = await graph.stream({
    messages: [new HumanMessage(userInput)],
  });

import * as readline from "node:readline/promises";
import { StateGraph, MessagesZodState, START, END } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

const llm = new ChatOpenAI({ model: "gpt-4o-mini" });

const State = z.object({ messages: MessagesZodState.shape.messages });

const graph = new StateGraph(State)
  .addNode("chatbot", async (state: z.infer<typeof State>) => {
    return { messages: [await llm.invoke(state.messages)] };
  })
  .addEdge(START, "chatbot")
  .addEdge("chatbot", END)
  .compile();

async function generateText(content: string) {
  const stream = await graph.stream(
    { messages: [{ type: "human", content }] },
    { streamMode: "values" }
  );

  for await (const event of stream) {
    for (const value of Object.values(event)) {
      console.log(
        "Assistant:",
        value.messages[value.messages.length - 1].content
      );
    const lastMessage = event.messages.at(-1);
    if (lastMessage?.getType() === "ai") {
      console.log(`Assistant: ${lastMessage.text}`);
    }
  }
}

const prompt = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

while (true) {
  const human = await prompt.question("User: ");
  if (["quit", "exit", "q"].includes(human.trim())) break;
  await generateText(human || "What do you know about LangGraph?");
}

prompt.close();
```

:::

```
Assistant: LangGraph is a library designed to help build stateful multi-agent applications using language models. It provides tools for creating workflows and state machines to coordinate multiple AI agents or language model interactions. LangGraph is built on top of LangChain, leveraging its components while adding graph-based coordination capabilities. It's particularly useful for developing more complex, stateful AI applications that go beyond simple query-response interactions.
```

:::python

```
Goodbye!
```

:::

**축하합니다!** LangGraph를 사용하여 첫 번째 챗봇을 구축했습니다. 이 봇은 사용자 입력을 받아 LLM을 사용하여 응답을 생성함으로써 기본적인 대화를 진행할 수 있습니다. 위 호출에 대한 [LangSmith Trace](https://smith.langchain.com/public/7527e308-9502-4894-b347-f34385740d5a/r)를 확인할 수 있습니다.

:::python

다음은 이 튜토리얼의 전체 코드입니다:

```python
from typing import Annotated

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# 첫 번째 인수는 고유한 노드 이름입니다
# 두 번째 인수는 노드가 사용될 때마다 호출될 함수 또는 객체입니다.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()
```

:::

:::js

```typescript
import { StateGraph, START, END, MessagesZodState } from "@langchain/langgraph";
import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";

const llm = new ChatOpenAI({
  model: "gpt-4o",
  temperature: 0,
});

const State = z.object({ messages: MessagesZodState.shape.messages });

const graph = new StateGraph(State);
  // 첫 번째 인수는 고유한 노드 이름입니다
  // 두 번째 인수는 노드가 사용될 때마다 호출될 함수 또는 객체입니다.
  .addNode("chatbot", async (state) => {
    return { messages: [await llm.invoke(state.messages)] };
  });
  .addEdge(START, "chatbot");
  .addEdge("chatbot", END)
  .compile();
```

:::

## 다음 단계

봇의 지식이 훈련 데이터에 있는 것으로 제한되어 있다는 것을 알아챘을 것입니다. 다음 부분에서는 봇의 지식을 확장하고 더 강력하게 만들기 위해 [웹 검색 도구를 추가](./2-add-tools.md)할 것입니다.
