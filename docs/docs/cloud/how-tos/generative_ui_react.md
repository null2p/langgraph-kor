# LangGraph를 사용하여 생성형 사용자 인터페이스 구현하는 방법

!!! info "사전 요구사항"

    - [LangGraph Platform](../../concepts/langgraph_platform.md)
    - [LangGraph Server](../../concepts/langgraph_server.md)
    - [`useStream()` React Hook](./use_stream_react.md)

생성형 사용자 인터페이스(Generative UI)를 사용하면 에이전트가 텍스트를 넘어 풍부한 사용자 인터페이스를 생성할 수 있습니다. 이를 통해 대화 흐름과 AI 응답에 따라 UI가 적응하는 더 인터랙티브하고 컨텍스트 인식 애플리케이션을 만들 수 있습니다.

![Generative UI Sample](./img/generative_ui_sample.jpg)

LangGraph Platform은 React 컴포넌트를 그래프 코드와 함께 배치하는 것을 지원합니다. 이를 통해 그래프에 대한 특정 UI 컴포넌트를 구축하는 데 집중하면서 [Agent Chat](https://agentchat.vercel.app)와 같은 기존 채팅 인터페이스에 쉽게 연결하고 실제로 필요할 때만 코드를 로드할 수 있습니다.

## 튜토리얼

### 1. UI 컴포넌트 정의 및 구성

먼저 첫 번째 UI 컴포넌트를 만듭니다. 각 컴포넌트에 대해 그래프 코드에서 컴포넌트를 참조하는 데 사용할 고유 식별자를 제공해야 합니다.

```tsx title="src/agent/ui.tsx"
const WeatherComponent = (props: { city: string }) => {
  return <div>Weather for {props.city}</div>;
};

export default {
  weather: WeatherComponent,
};
```

다음으로, `langgraph.json` 구성에서 UI 컴포넌트를 정의합니다:

=== "Python agent"

    ```json title="langgraph.json"
    {
      "node_version": "20",
      "graphs": {
        "agent": "./src/agent.py:graph"
      },
      "ui": {
        "agent": "./src/agent/ui.tsx"
      }
    }
    ```

=== "JS agent"

    ```json title="langgraph.json"
    {
      "node_version": "20",
      "graphs": {
        "agent": "./src/agent/index.ts:graph"
      },
      "ui": {
        "agent": "./src/agent/ui.tsx"
      }
    }
    ```

`ui` 섹션은 그래프에서 사용할 UI 컴포넌트를 가리킵니다. 기본적으로 그래프 이름과 동일한 키를 사용하는 것을 권장하지만, 원하는 대로 컴포넌트를 분할할 수 있습니다. 자세한 내용은 [UI 컴포넌트의 네임스페이스 커스터마이징](#customise-the-namespace-of-ui-components)을 참조하세요.

LangGraph Platform은 UI 컴포넌트 코드와 스타일을 자동으로 번들링하고 `LoadExternalComponent` 컴포넌트로 로드할 수 있는 외부 자산으로 제공합니다. `react` 및 `react-dom`과 같은 일부 종속성은 번들에서 자동으로 제외됩니다.

CSS 및 Tailwind 4.x도 기본적으로 지원되므로 UI 컴포넌트에서 Tailwind 클래스와 `shadcn/ui`를 자유롭게 사용할 수 있습니다.

=== "`src/agent/ui.tsx`"

    ```tsx
    import "./styles.css";

    const WeatherComponent = (props: { city: string }) => {
      return <div className="bg-red-500">Weather for {props.city}</div>;
    };

    export default {
      weather: WeatherComponent,
    };
    ```

=== "`src/agent/styles.css`"

    ```css
    @import "tailwindcss";
    ```

### 2. 그래프에서 UI 컴포넌트 전송

=== "Python"

    ```python title="src/agent.py"
    import uuid
    from typing import Annotated, Sequence, TypedDict

    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.graph.ui import AnyUIMessage, ui_message_reducer, push_ui_message


    class AgentState(TypedDict):  # noqa: D101
        messages: Annotated[Sequence[BaseMessage], add_messages]
        ui: Annotated[Sequence[AnyUIMessage], ui_message_reducer]


    async def weather(state: AgentState):
        class WeatherOutput(TypedDict):
            city: str

        weather: WeatherOutput = (
            await ChatOpenAI(model="gpt-4o-mini")
            .with_structured_output(WeatherOutput)
            .with_config({"tags": ["nostream"]})
            .ainvoke(state["messages"])
        )

        message = AIMessage(
            id=str(uuid.uuid4()),
            content=f"Here's the weather for {weather['city']}",
        )

        # Emit UI elements associated with the message
        push_ui_message("weather", weather, message=message)
        return {"messages": [message]}


    workflow = StateGraph(AgentState)
    workflow.add_node(weather)
    workflow.add_edge("__start__", "weather")
    graph = workflow.compile()
    ```

=== "JS"

    에이전트 노드에서 UI 요소를 내보내려면 `typedUi` 유틸리티를 사용하세요:

    ```typescript title="src/agent/index.ts"
    import {
      typedUi,
      uiMessageReducer,
    } from "@langchain/langgraph-sdk/react-ui/server";

    import { ChatOpenAI } from "@langchain/openai";
    import { v4 as uuidv4 } from "uuid";
    import { z } from "zod";

    import type ComponentMap from "./ui.js";

    import {
      Annotation,
      MessagesAnnotation,
      StateGraph,
      type LangGraphRunnableConfig,
    } from "@langchain/langgraph";

    const AgentState = Annotation.Root({
      ...MessagesAnnotation.spec,
      ui: Annotation({ reducer: uiMessageReducer, default: () => [] }),
    });

    export const graph = new StateGraph(AgentState)
      .addNode("weather", async (state, config) => {
        // Provide the type of the component map to ensure
        // type safety of `ui.push()` calls as well as
        // pushing the messages to the `ui` and sending a custom event as well.
        const ui = typedUi<typeof ComponentMap>(config);

        const weather = await new ChatOpenAI({ model: "gpt-4o-mini" })
          .withStructuredOutput(z.object({ city: z.string() }))
          .withConfig({ tags: ["nostream"] })
          .invoke(state.messages);

        const response = {
          id: uuidv4(),
          type: "ai",
          content: `Here's the weather for ${weather.city}`,
        };

        // Emit UI elements associated with the AI message
        ui.push({ name: "weather", props: weather }, { message: response });

        return { messages: [response] };
      })
      .addEdge("__start__", "weather")
      .compile();
    ```

### 3. React 애플리케이션에서 UI 요소 처리

클라이언트 측에서는 `useStream()` 및 `LoadExternalComponent`를 사용하여 UI 요소를 표시할 수 있습니다.

```tsx title="src/app/page.tsx"
"use client";

import { useStream } from "@langchain/langgraph-sdk/react";
import { LoadExternalComponent } from "@langchain/langgraph-sdk/react-ui";

export default function Page() {
  const { thread, values } = useStream({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",
  });

  return (
    <div>
      {thread.messages.map((message) => (
        <div key={message.id}>
          {message.content}
          {values.ui
            ?.filter((ui) => ui.metadata?.message_id === message.id)
            .map((ui) => (
              <LoadExternalComponent key={ui.id} stream={thread} message={ui} />
            ))}
        </div>
      ))}
    </div>
  );
}
```

내부적으로 `LoadExternalComponent`는 LangGraph Platform에서 UI 컴포넌트의 JS 및 CSS를 가져와 shadow DOM에서 렌더링하여 애플리케이션의 나머지 부분과 스타일 격리를 보장합니다.

## 사용 가이드

### 클라이언트 측에서 커스텀 컴포넌트 제공

클라이언트 애플리케이션에 이미 로드된 컴포넌트가 있는 경우, LangGraph Platform에서 UI 코드를 가져오지 않고 직접 렌더링할 수 있는 이러한 컴포넌트의 맵을 제공할 수 있습니다.

```tsx
const clientComponents = {
  weather: WeatherComponent,
};

<LoadExternalComponent
  stream={thread}
  message={ui}
  components={clientComponents}
/>;
```

### 컴포넌트 로딩 중 로딩 UI 표시

컴포넌트가 로딩 중일 때 렌더링할 fallback UI를 제공할 수 있습니다.

```tsx
<LoadExternalComponent
  stream={thread}
  message={ui}
  fallback={<div>Loading...</div>}
/>
```

### UI 컴포넌트의 네임스페이스 커스터마이징 {#customise-the-namespace-of-ui-components}

기본적으로 `LoadExternalComponent`는 `useStream()` 훅의 `assistantId`를 사용하여 UI 컴포넌트의 코드를 가져옵니다. `LoadExternalComponent` 컴포넌트에 `namespace` prop을 제공하여 이를 커스터마이징할 수 있습니다.

=== "`src/app/page.tsx`"

    ```tsx
    <LoadExternalComponent
      stream={thread}
      message={ui}
      namespace="custom-namespace"
    />
    ```

=== "`langgraph.json`"

    ```json
    {
      "ui": {
        "custom-namespace": "./src/agent/ui.tsx"
      }
    }
    ```

### UI 컴포넌트에서 thread 상태 액세스 및 상호작용

`useStreamContext` 훅을 사용하여 UI 컴포넌트 내에서 thread 상태에 액세스할 수 있습니다.

```tsx
import { useStreamContext } from "@langchain/langgraph-sdk/react-ui";

const WeatherComponent = (props: { city: string }) => {
  const { thread, submit } = useStreamContext();
  return (
    <>
      <div>Weather for {props.city}</div>

      <button
        onClick={() => {
          const newMessage = {
            type: "human",
            content: `What's the weather in ${props.city}?`,
          };

          submit({ messages: [newMessage] });
        }}
      >
        Retry
      </button>
    </>
  );
};
```

### 클라이언트 컴포넌트에 추가 컨텍스트 전달

`LoadExternalComponent` 컴포넌트에 `meta` prop을 제공하여 클라이언트 컴포넌트에 추가 컨텍스트를 전달할 수 있습니다.

```tsx
<LoadExternalComponent stream={thread} message={ui} meta={{ userId: "123" }} />
```

그런 다음 `useStreamContext` 훅을 사용하여 UI 컴포넌트에서 `meta` prop에 액세스할 수 있습니다.

```tsx
import { useStreamContext } from "@langchain/langgraph-sdk/react-ui";

const WeatherComponent = (props: { city: string }) => {
  const { meta } = useStreamContext<
    { city: string },
    { MetaType: { userId?: string } }
  >();

  return (
    <div>
      Weather for {props.city} (user: {meta?.userId})
    </div>
  );
};
```

### 서버에서 UI 메시지 스트리밍

`useStream()` 훅의 `onCustomEvent` 콜백을 사용하여 노드 실행이 완료되기 전에 UI 메시지를 스트리밍할 수 있습니다. 이는 LLM이 응답을 생성하는 동안 UI 컴포넌트를 업데이트할 때 특히 유용합니다.

```tsx
import { uiMessageReducer } from "@langchain/langgraph-sdk/react-ui";

const { thread, submit } = useStream({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",
  onCustomEvent: (event, options) => {
    options.mutate((prev) => {
      const ui = uiMessageReducer(prev.ui ?? [], event);
      return { ...prev, ui };
    });
  },
});
```

그런 다음 업데이트하려는 UI 메시지와 동일한 ID로 `ui.push()` / `push_ui_message()`를 호출하여 UI 컴포넌트에 업데이트를 푸시할 수 있습니다.

=== "Python"

    ```python
    from typing import Annotated, Sequence, TypedDict

    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
    from langgraph.graph import StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.graph.ui import AnyUIMessage, push_ui_message, ui_message_reducer


    class AgentState(TypedDict):  # noqa: D101
        messages: Annotated[Sequence[BaseMessage], add_messages]
        ui: Annotated[Sequence[AnyUIMessage], ui_message_reducer]


    class CreateTextDocument(TypedDict):
        """Prepare a document heading for the user."""

        title: str


    async def writer_node(state: AgentState):
        model = ChatAnthropic(model="claude-3-5-sonnet-latest")
        message: AIMessage = await model.bind_tools(
            tools=[CreateTextDocument],
            tool_choice={"type": "tool", "name": "CreateTextDocument"},
        ).ainvoke(state["messages"])

        tool_call = next(
            (x["args"] for x in message.tool_calls if x["name"] == "CreateTextDocument"),
            None,
        )

        if tool_call:
            ui_message = push_ui_message("writer", tool_call, message=message)
            ui_message_id = ui_message["id"]

            # We're already streaming the LLM response to the client through UI messages
            # so we don't need to stream it again to the `messages` stream mode.
            content_stream = model.with_config({"tags": ["nostream"]}).astream(
                f"Create a document with the title: {tool_call['title']}"
            )

            content: AIMessageChunk | None = None
            async for chunk in content_stream:
                content = content + chunk if content else chunk

                push_ui_message(
                    "writer",
                    {"content": content.text()},
                    id=ui_message_id,
                    message=message,
                    # Use `merge=rue` to merge props with the existing UI message
                    merge=True,
                )

        return {"messages": [message]}
    ```

=== "JS"

    ```tsx
    import {
      Annotation,
      MessagesAnnotation,
      type LangGraphRunnableConfig,
    } from "@langchain/langgraph";
    import { z } from "zod";
    import { ChatAnthropic } from "@langchain/anthropic";
    import {
      typedUi,
      uiMessageReducer,
    } from "@langchain/langgraph-sdk/react-ui/server";
    import type { AIMessageChunk } from "@langchain/core/messages";

    import type ComponentMap from "./ui";

    const AgentState = Annotation.Root({
      ...MessagesAnnotation.spec,
      ui: Annotation({ reducer: uiMessageReducer, default: () => [] }),
    });

    async function writerNode(
      state: typeof AgentState.State,
      config: LangGraphRunnableConfig
    ): Promise<typeof AgentState.Update> {
      const ui = typedUi<typeof ComponentMap>(config);

      const model = new ChatAnthropic({ model: "claude-3-5-sonnet-latest" });
      const message = await model
        .bindTools(
          [
            {
              name: "create_text_document",
              description: "Prepare a document heading for the user.",
              schema: z.object({ title: z.string() }),
            },
          ],
          { tool_choice: { type: "tool", name: "create_text_document" } }
        )
        .invoke(state.messages);

      type ToolCall = { name: "create_text_document"; args: { title: string } };
      const toolCall = message.tool_calls?.find(
        (tool): tool is ToolCall => tool.name === "create_text_document"
      );

      if (toolCall) {
        const { id, name } = ui.push(
          { name: "writer", props: { title: toolCall.args.title } },
          { message }
        );

        const contentStream = await model
          // We're already streaming the LLM response to the client through UI messages
          // so we don't need to stream it again to the `messages` stream mode.
          .withConfig({ tags: ["nostream"] })
          .stream(`Create a short poem with the topic: ${message.text}`);

        let content: AIMessageChunk | undefined;
        for await (const chunk of contentStream) {
          content = content?.concat(chunk) ?? chunk;

          ui.push(
            { id, name, props: { content: content?.text } },
            // Use `merge: true` to merge props with the existing UI message
            { message, merge: true }
          );
        }
      }

      return { messages: [message] };
    }
    ```

=== "`ui.tsx`"

    ```tsx
    function WriterComponent(props: { title: string; content?: string }) {
      return (
        <article>
          <h2>{props.title}</h2>
          <p style={{ whiteSpace: "pre-wrap" }}>{props.content}</p>
        </article>
      );
    }

    export default {
      weather: WriterComponent,
    };
    ```

### 상태에서 UI 메시지 제거

RemoveMessage를 추가하여 상태에서 메시지를 제거할 수 있는 것과 유사하게, UI 메시지의 ID로 `remove_ui_message` / `ui.delete`를 호출하여 상태에서 UI 메시지를 제거할 수 있습니다.

=== "Python"

    ```python
    from langgraph.graph.ui import push_ui_message, delete_ui_message

    # push message
    message = push_ui_message("weather", {"city": "London"})

    # remove said message
    delete_ui_message(message["id"])
    ```

=== "JS"

    ```tsx
    // push message
    const message = ui.push({ name: "weather", props: { city: "London" } });

    // remove said message
    ui.delete(message.id);
    ```

## 더 알아보기

- [JS/TS SDK 레퍼런스](../reference/sdk/js_ts_sdk_ref.md)
