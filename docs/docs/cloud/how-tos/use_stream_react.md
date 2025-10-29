# React 애플리케이션에 LangGraph를 통합하는 방법

!!! info "사전 요구사항"

    - [LangGraph Platform](../../concepts/langgraph_platform.md)
    - [LangGraph Server](../../concepts/langgraph_server.md)

`useStream()` React 훅은 LangGraph를 React 애플리케이션에 통합하는 원활한 방법을 제공합니다. 스트리밍, 상태 관리 및 분기 논리의 모든 복잡성을 처리하여 훌륭한 채팅 경험을 구축하는 데 집중할 수 있게 합니다.

주요 기능:

- 메시지 스트리밍: 메시지 청크 스트림을 처리하여 완전한 메시지 형성
- 메시지, 인터럽트, 로딩 상태 및 오류에 대한 자동 상태 관리
- 대화 분기: 채팅 히스토리의 모든 지점에서 대체 대화 경로 생성
- UI 독립적 설계: 자신만의 컴포넌트 및 스타일링 사용

React 애플리케이션에서 `useStream()`을 사용하는 방법을 살펴보겠습니다.

`useStream()`은 맞춤형 채팅 경험을 만들기 위한 견고한 기반을 제공합니다. 미리 만들어진 채팅 컴포넌트 및 인터페이스의 경우 [CopilotKit](https://docs.copilotkit.ai/coagents/quickstart/langgraph) 및 [assistant-ui](https://www.assistant-ui.com/docs/runtimes/langgraph)도 확인하는 것이 좋습니다.

## 설치

```bash
npm install @langchain/langgraph-sdk @langchain/core
```

## 예제

```tsx
"use client";

import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";

export default function App() {
  const thread = useStream<{ messages: Message[] }>({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",
    messagesKey: "messages",
  });

  return (
    <div>
      <div>
        {thread.messages.map((message) => (
          <div key={message.id}>{message.content as string}</div>
        ))}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();

          const form = e.target as HTMLFormElement;
          const message = new FormData(form).get("message") as string;

          form.reset();
          thread.submit({ messages: [{ type: "human", content: message }] });
        }}
      >
        <input type="text" name="message" />

        {thread.isLoading ? (
          <button key="stop" type="button" onClick={() => thread.stop()}>
            Stop
          </button>
        ) : (
          <button keytype="submit">Send</button>
        )}
      </form>
    </div>
  );
}
```

## UI 커스터마이징

`useStream()` 훅은 모든 복잡한 상태 관리를 백그라운드에서 처리하며, UI를 구축할 수 있는 간단한 인터페이스를 제공합니다. 기본적으로 제공되는 기능:

- 스레드 상태 관리
- 로딩 및 오류 상태
- 인터럽트
- 메시지 처리 및 업데이트
- 분기 지원

다음은 이러한 기능을 효과적으로 사용하는 몇 가지 예제입니다:

### 로딩 상태

`isLoading` 속성은 스트림이 활성화되어 있는지 알려주며, 다음을 수행할 수 있습니다:

- 로딩 표시기 표시
- 처리 중 입력 필드 비활성화
- 취소 버튼 표시

```tsx
export default function App() {
  const { isLoading, stop } = useStream<{ messages: Message[] }>({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",
    messagesKey: "messages",
  });

  return (
    <form>
      {isLoading && (
        <button key="stop" type="button" onClick={() => stop()}>
          Stop
        </button>
      )}
    </form>
  );
}
```

### 페이지 새로고침 후 스트림 재개

`useStream()` 훅은 `reconnectOnMount: true`를 설정하여 마운트 시 진행 중인 실행을 자동으로 재개할 수 있습니다. 이는 페이지 새로고침 후 스트림을 계속하는 데 유용하며, 다운타임 동안 생성된 메시지 및 이벤트가 손실되지 않도록 보장합니다.

```tsx
const thread = useStream<{ messages: Message[] }>({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",
  reconnectOnMount: true,
});
```

기본적으로 생성된 실행의 ID는 `window.sessionStorage`에 저장되며, `reconnectOnMount`에 커스텀 스토리지를 전달하여 바꿀 수 있습니다. 스토리지는 스레드의 진행 중인 실행 ID를 지속하는 데 사용됩니다(`lg:stream:${threadId}` 키 아래).

```tsx
const thread = useStream<{ messages: Message[] }>({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",
  reconnectOnMount: () => window.localStorage,
});
```

또한 실행 콜백을 사용하여 실행 메타데이터를 지속하고 `joinStream` 함수를 사용하여 스트림을 재개하는 방식으로 재개 프로세스를 수동으로 관리할 수도 있습니다. 실행을 생성할 때 `streamResumable: true`를 전달해야 합니다. 그렇지 않으면 일부 이벤트가 손실될 수 있습니다.

```tsx
import type { Message } from "@langchain/langgraph-sdk";
import { useStream } from "@langchain/langgraph-sdk/react";
import { useCallback, useState, useEffect, useRef } from "react";

export default function App() {
  const [threadId, onThreadId] = useSearchParam("threadId");

  const thread = useStream<{ messages: Message[] }>({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",

    threadId,
    onThreadId,

    onCreated: (run) => {
      window.sessionStorage.setItem(`resume:${run.thread_id}`, run.run_id);
    },
    onFinish: (_, run) => {
      window.sessionStorage.removeItem(`resume:${run?.thread_id}`);
    },
  });

  // Ensure that we only join the stream once per thread.
  const joinedThreadId = useRef<string | null>(null);
  useEffect(() => {
    if (!threadId) return;

    const resume = window.sessionStorage.getItem(`resume:${threadId}`);
    if (resume && joinedThreadId.current !== threadId) {
      thread.joinStream(resume);
      joinedThreadId.current = threadId;
    }
  }, [threadId]);

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        const form = e.target as HTMLFormElement;
        const message = new FormData(form).get("message") as string;
        thread.submit(
          { messages: [{ type: "human", content: message }] },
          { streamResumable: true }
        );
      }}
    >
      <div>
        {thread.messages.map((message) => (
          <div key={message.id}>{message.content as string}</div>
        ))}
      </div>
      <input type="text" name="message" />
      <button type="submit">Send</button>
    </form>
  );
}

// Utility method to retrieve and persist data in URL as search param
function useSearchParam(key: string) {
  const [value, setValue] = useState<string | null>(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get(key) ?? null;
  });

  const update = useCallback(
    (value: string | null) => {
      setValue(value);

      const url = new URL(window.location.href);
      if (value == null) {
        url.searchParams.delete(key);
      } else {
        url.searchParams.set(key, value);
      }

      window.history.pushState({}, "", url.toString());
    },
    [key]
  );

  return [value, update] as const;
}
```

### 스레드 관리

내장된 스레드 관리로 대화를 추적하세요. 현재 스레드 ID에 액세스하고 새 스레드가 생성될 때 알림을 받을 수 있습니다:

```tsx
const [threadId, setThreadId] = useState<string | null>(null);

const thread = useStream<{ messages: Message[] }>({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",

  threadId: threadId,
  onThreadId: setThreadId,
});
```

페이지 새로고침 후 사용자가 대화를 재개할 수 있도록 URL의 쿼리 매개변수에 `threadId`를 저장하는 것이 좋습니다.

### 메시지 처리

`useStream()` 훅은 서버에서 수신한 메시지 청크를 추적하고 이를 연결하여 완전한 메시지를 형성합니다. 완성된 메시지 청크는 `messages` 속성을 통해 검색할 수 있습니다.

기본적으로 `messagesKey`는 `messages`로 설정되어 `values["messages"]`에 새 메시지 청크를 추가합니다. 다른 키에 메시지를 저장하는 경우 `messagesKey`의 값을 변경할 수 있습니다.

```tsx
import type { Message } from "@langchain/langgraph-sdk";
import { useStream } from "@langchain/langgraph-sdk/react";

export default function HomePage() {
  const thread = useStream<{ messages: Message[] }>({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",
    messagesKey: "messages",
  });

  return (
    <div>
      {thread.messages.map((message) => (
        <div key={message.id}>{message.content as string}</div>
      ))}
    </div>
  );
}
```

내부적으로 `useStream()` 훅은 `streamMode: "messages-tuple"`을 사용하여 그래프 노드 내부의 모든 LangChain 채팅 모델 호출에서 메시지 스트림(즉, 개별 LLM 토큰)을 수신합니다. 메시지 스트리밍에 대한 자세한 내용은 [스트리밍](../how-tos/streaming.md#messages) 가이드를 참조하세요.

### 인터럽트

`useStream()` 훅은 스레드의 마지막 인터럽트로 채워지는 `interrupt` 속성을 노출합니다. 인터럽트를 사용하여 다음을 수행할 수 있습니다:

- 노드 실행 전 확인 UI 렌더링
- 에이전트가 사용자에게 명확한 질문을 할 수 있도록 사람의 입력 대기

인터럽트에 대한 자세한 내용은 [인터럽트 처리 방법](../../how-tos/human_in_the_loop/wait-user-input.ipynb) 가이드를 참조하세요.

```tsx
const thread = useStream<{ messages: Message[] }, { InterruptType: string }>({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",
  messagesKey: "messages",
});

if (thread.interrupt) {
  return (
    <div>
      Interrupted! {thread.interrupt.value}
      <button
        type="button"
        onClick={() => {
          // `resume` can be any value that the agent accepts
          thread.submit(undefined, { command: { resume: true } });
        }}
      >
        Resume
      </button>
    </div>
  );
}
```

### 분기

각 메시지에 대해 `getMessagesMetadata()`를 사용하여 메시지가 처음 표시된 체크포인트를 가져올 수 있습니다. 그런 다음 처음 표시된 체크포인트 이전의 체크포인트에서 새 실행을 생성하여 스레드에 새 분기를 만들 수 있습니다.

분기는 다음과 같은 방법으로 생성할 수 있습니다:

1. 이전 사용자 메시지 편집
2. 이전 어시스턴트 메시지 재생성 요청

```tsx
"use client";

import type { Message } from "@langchain/langgraph-sdk";
import { useStream } from "@langchain/langgraph-sdk/react";
import { useState } from "react";

function BranchSwitcher({
  branch,
  branchOptions,
  onSelect,
}: {
  branch: string | undefined;
  branchOptions: string[] | undefined;
  onSelect: (branch: string) => void;
}) {
  if (!branchOptions || !branch) return null;
  const index = branchOptions.indexOf(branch);

  return (
    <div className="flex items-center gap-2">
      <button
        type="button"
        onClick={() => {
          const prevBranch = branchOptions[index - 1];
          if (!prevBranch) return;
          onSelect(prevBranch);
        }}
      >
        Prev
      </button>
      <span>
        {index + 1} / {branchOptions.length}
      </span>
      <button
        type="button"
        onClick={() => {
          const nextBranch = branchOptions[index + 1];
          if (!nextBranch) return;
          onSelect(nextBranch);
        }}
      >
        Next
      </button>
    </div>
  );
}

function EditMessage({
  message,
  onEdit,
}: {
  message: Message;
  onEdit: (message: Message) => void;
}) {
  const [editing, setEditing] = useState(false);

  if (!editing) {
    return (
      <button type="button" onClick={() => setEditing(true)}>
        Edit
      </button>
    );
  }

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        const form = e.target as HTMLFormElement;
        const content = new FormData(form).get("content") as string;

        form.reset();
        onEdit({ type: "human", content });
        setEditing(false);
      }}
    >
      <input name="content" defaultValue={message.content as string} />
      <button type="submit">Save</button>
    </form>
  );
}

export default function App() {
  const thread = useStream({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",
    messagesKey: "messages",
  });

  return (
    <div>
      <div>
        {thread.messages.map((message) => {
          const meta = thread.getMessagesMetadata(message);
          const parentCheckpoint = meta?.firstSeenState?.parent_checkpoint;

          return (
            <div key={message.id}>
              <div>{message.content as string}</div>

              {message.type === "human" && (
                <EditMessage
                  message={message}
                  onEdit={(message) =>
                    thread.submit(
                      { messages: [message] },
                      { checkpoint: parentCheckpoint }
                    )
                  }
                />
              )}

              {message.type === "ai" && (
                <button
                  type="button"
                  onClick={() =>
                    thread.submit(undefined, { checkpoint: parentCheckpoint })
                  }
                >
                  <span>Regenerate</span>
                </button>
              )}

              <BranchSwitcher
                branch={meta?.branch}
                branchOptions={meta?.branchOptions}
                onSelect={(branch) => thread.setBranch(branch)}
              />
            </div>
          );
        })}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();

          const form = e.target as HTMLFormElement;
          const message = new FormData(form).get("message") as string;

          form.reset();
          thread.submit({ messages: [message] });
        }}
      >
        <input type="text" name="message" />

        {thread.isLoading ? (
          <button key="stop" type="button" onClick={() => thread.stop()}>
            Stop
          </button>
        ) : (
          <button key="submit" type="submit">
            Send
          </button>
        )}
      </form>
    </div>
  );
}
```

고급 사용 사례의 경우 `experimental_branchTree` 속성을 사용하여 스레드의 트리 표현을 가져올 수 있으며, 이는 메시지 기반이 아닌 그래프에 대한 분기 제어를 렌더링하는 데 사용할 수 있습니다.

### 낙관적 업데이트

에이전트에 네트워크 요청을 수행하기 전에 클라이언트 상태를 낙관적으로 업데이트하여 에이전트가 요청을 보기 전에 사용자 메시지를 즉시 표시하는 등 사용자에게 즉각적인 피드백을 제공할 수 있습니다.

```tsx
const stream = useStream({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",
  messagesKey: "messages",
});

const handleSubmit = (text: string) => {
  const newMessage = { type: "human" as const, content: text };

  stream.submit(
    { messages: [newMessage] },
    {
      optimisticValues(prev) {
        const prevMessages = prev.messages ?? [];
        const newMessages = [...prevMessages, newMessage];
        return { ...prev, messages: newMessages };
      },
    }
  );
};
```

### 캐시된 스레드 표시

`initialValues` 옵션을 사용하여 히스토리가 서버에서 로드되는 동안 캐시된 스레드 데이터를 즉시 표시합니다. 이렇게 하면 기존 스레드로 이동할 때 캐시된 데이터를 즉시 표시하여 사용자 경험이 향상됩니다.

```tsx
import { useStream } from "@langchain/langgraph-sdk/react";

const CachedThreadExample = ({ threadId, cachedThreadData }) => {
  const stream = useStream({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",
    threadId,
    // Show cached data immediately while history loads
    initialValues: cachedThreadData?.values,
    messagesKey: "messages",
  });

  return (
    <div>
      {stream.messages.map((message) => (
        <div key={message.id}>{message.content as string}</div>
      ))}
    </div>
  );
};
```

### 낙관적 스레드 생성

`submit` 함수의 `threadId` 옵션을 사용하여 스레드가 실제로 생성되기 전에 스레드 ID를 알아야 하는 낙관적 UI 패턴을 활성화합니다.

```tsx
import { useState } from "react";
import { useStream } from "@langchain/langgraph-sdk/react";

const OptimisticThreadExample = () => {
  const [threadId, setThreadId] = useState<string | null>(null);
  const [optimisticThreadId] = useState(() => crypto.randomUUID());

  const stream = useStream({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",
    threadId,
    onThreadId: setThreadId, // (3) Updated after thread has been created.
    messagesKey: "messages",
  });

  const handleSubmit = (text: string) => {
    // (1) Perform a soft navigation to /threads/${optimisticThreadId}
    // without waiting for thread creation.
    window.history.pushState({}, "", `/threads/${optimisticThreadId}`);

    // (2) Submit message to create thread with the predetermined ID.
    stream.submit(
      { messages: [{ type: "human", content: text }] },
      { threadId: optimisticThreadId }
    );
  };

  return (
    <div>
      <p>Thread ID: {threadId ?? optimisticThreadId}</p>
      {/* Rest of component */}
    </div>
  );
};
```

### TypeScript

`useStream()` 훅은 TypeScript로 작성된 앱에 친화적이며, 더 나은 타입 안전성과 IDE 지원을 위해 상태에 대한 타입을 지정할 수 있습니다.

```tsx
// Define your types
type State = {
  messages: Message[];
  context?: Record<string, unknown>;
};

// Use them with the hook
const thread = useStream<State>({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",
  messagesKey: "messages",
});
```

다음과 같은 다양한 시나리오에 대한 타입을 선택적으로 지정할 수도 있습니다:

- `ConfigurableType`: `config.configurable` 속성의 타입 (기본값: `Record<string, unknown>`)
- `InterruptType`: 인터럽트 값의 타입 - 즉, `interrupt(...)` 함수의 내용 (기본값: `unknown`)
- `CustomEventType`: 커스텀 이벤트의 타입 (기본값: `unknown`)
- `UpdateType`: submit 함수의 타입 (기본값: `Partial<State>`)

```tsx
const thread = useStream<
  State,
  {
    UpdateType: {
      messages: Message[] | Message;
      context?: Record<string, unknown>;
    };
    InterruptType: string;
    CustomEventType: {
      type: "progress" | "debug";
      payload: unknown;
    };
    ConfigurableType: {
      model: string;
    };
  }
>({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",
  messagesKey: "messages",
});
```

LangGraph.js를 사용하는 경우 그래프의 주석 타입을 재사용할 수도 있습니다. 그러나 전체 LangGraph.js 런타임을 가져오지 않도록 주석 스키마의 타입만 가져와야 합니다(즉, `import type { ... }` 지시문을 통해).

```tsx
import {
  Annotation,
  MessagesAnnotation,
  type StateType,
  type UpdateType,
} from "@langchain/langgraph/web";

const AgentState = Annotation.Root({
  ...MessagesAnnotation.spec,
  context: Annotation<string>(),
});

const thread = useStream<
  StateType<typeof AgentState.spec>,
  { UpdateType: UpdateType<typeof AgentState.spec> }
>({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",
  messagesKey: "messages",
});
```

## 이벤트 처리

`useStream()` 훅은 다양한 이벤트에 응답할 수 있도록 여러 콜백 옵션을 제공합니다:

- `onError`: 오류가 발생할 때 호출됩니다.
- `onFinish`: 스트림이 완료될 때 호출됩니다.
- `onUpdateEvent`: 업데이트 이벤트를 수신할 때 호출됩니다.
- `onCustomEvent`: 커스텀 이벤트를 수신할 때 호출됩니다. 커스텀 이벤트를 스트리밍하는 방법은 [스트리밍](../../how-tos/streaming.md#stream-custom-data) 가이드를 참조하세요.
- `onMetadataEvent`: Run ID와 Thread ID를 포함하는 메타데이터 이벤트를 수신할 때 호출됩니다.

## 더 알아보기

- [JS/TS SDK 레퍼런스](../reference/sdk/js_ts_sdk_ref.md)
