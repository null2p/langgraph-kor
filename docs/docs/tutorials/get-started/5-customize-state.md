# 상태 사용자 정의

이 튜토리얼에서는 메시지 목록에 의존하지 않고 복잡한 동작을 정의하기 위해 상태에 추가 필드를 추가합니다. 챗봇은 검색 도구를 사용하여 특정 정보를 찾고 사람에게 검토를 위해 전달합니다.

!!! note

    이 튜토리얼은 [사람이 개입하는 제어 추가](./4-human-in-the-loop.md)를 기반으로 합니다.

## 1. 상태에 키 추가

상태에 `name` 및 `birthday` 키를 추가하여 엔티티의 생일을 조사하도록 챗봇을 업데이트합니다:

:::python

```python
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]
    # highlight-next-line
    name: str
    # highlight-next-line
    birthday: str
```

:::

:::js

```typescript
import { MessagesZodState } from "@langchain/langgraph";
import { z } from "zod";

const State = z.object({
  messages: MessagesZodState.shape.messages,
  // highlight-next-line
  name: z.string(),
  // highlight-next-line
  birthday: z.string(),
});
```

:::

이 정보를 상태에 추가하면 다른 그래프 노드(정보를 저장하거나 처리하는 다운스트림 노드 등)와 그래프의 지속성 레이어에서 쉽게 액세스할 수 있습니다.

## 2. 도구 내부에서 상태 업데이트

:::python

이제 `human_assistance` 도구 내부에서 상태 키를 채웁니다. 이를 통해 사람이 상태에 저장되기 전에 정보를 검토할 수 있습니다. 도구 내부에서 상태 업데이트를 발행하려면 [`Command`](../../concepts/low_level.md#use-inside-tools)를 사용합니다.

```python
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool

from langgraph.types import Command, interrupt

@tool
# 상태 업데이트를 위해 ToolMessage를 생성하므로,
# 일반적으로 해당 도구 호출의 ID가 필요합니다. LangChain의
# InjectedToolCallId를 사용하여 이 인수가 도구의 스키마에서
# 모델에 공개되지 않아야 함을 나타낼 수 있습니다.
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """사람에게 도움을 요청합니다."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    # 정보가 올바른 경우 상태를 그대로 업데이트합니다.
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    # 그렇지 않으면 사람 검토자로부터 정보를 받습니다.
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # 이번에는 도구 내부에서 ToolMessage로 명시적으로 상태를 업데이트합니다.
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # 도구에서 Command 객체를 반환하여 상태를 업데이트합니다.
    return Command(update=state_update)
```

:::

:::js

Now, populate the state keys inside of the `humanAssistance` tool. This allows a human to review the information before it is stored in the state. Use [`Command`](../../concepts/low_level.md#use-inside-tools) to issue a state update from inside the tool.

```typescript
import { tool } from "@langchain/core/tools";
import { ToolMessage } from "@langchain/core/messages";
import { Command, interrupt } from "@langchain/langgraph";

const humanAssistance = tool(
  async (input, config) => {
    // 상태 업데이트를 위해 ToolMessage를 생성하므로,
    // 일반적으로 해당 도구 호출의 ID가 필요합니다.
    // 이는 도구의 config에서 사용할 수 있습니다.
    const toolCallId = config?.toolCall?.id as string | undefined;
    if (!toolCallId) throw new Error("Tool call ID is required");

    const humanResponse = await interrupt({
      question: "Is this correct?",
      name: input.name,
      birthday: input.birthday,
    });

    // 도구 내부에서 ToolMessage로 명시적으로 상태를 업데이트합니다.
    const stateUpdate = (() => {
      // 정보가 올바른 경우 상태를 그대로 업데이트합니다.
      if (humanResponse.correct?.toLowerCase().startsWith("y")) {
        return {
          name: input.name,
          birthday: input.birthday,
          messages: [
            new ToolMessage({ content: "Correct", tool_call_id: toolCallId }),
          ],
        };
      }

      // 그렇지 않으면 사람 검토자로부터 정보를 받습니다.
      return {
        name: humanResponse.name || input.name,
        birthday: humanResponse.birthday || input.birthday,
        messages: [
          new ToolMessage({
            content: `Made a correction: ${JSON.stringify(humanResponse)}`,
            tool_call_id: toolCallId,
          }),
        ],
      };
    })();

    // 도구에서 Command 객체를 반환하여 상태를 업데이트합니다.
    return new Command({ update: stateUpdate });
  },
  {
    name: "humanAssistance",
    description: "Request assistance from a human.",
    schema: z.object({
      name: z.string().describe("엔티티의 이름"),
      birthday: z.string().describe("엔티티의 생일/출시일"),
    }),
  }
);
```

:::

그래프의 나머지 부분은 동일하게 유지됩니다.

## 3. 챗봇에 프롬프트 전달

:::python
LangGraph 라이브러리의 "생일"을 조회하도록 챗봇에 프롬프트를 전달하고 필요한 정보를 얻으면 `human_assistance` 도구에 연락하도록 챗봇에 지시합니다. 도구의 인수에 `name`과 `birthday`를 설정하면 챗봇이 이러한 필드에 대한 제안을 생성하도록 강제합니다.

```python
user_input = (
    "Can you look up when LangGraph was released? "
    "When you have the answer, use the human_assistance tool for review."
)
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

:::

:::js
LangGraph 라이브러리의 "생일"을 조회하도록 챗봇에 프롬프트를 전달하고 필요한 정보를 얻으면 `humanAssistance` 도구에 연락하도록 챗봇에 지시합니다. 도구의 인수에 `name`과 `birthday`를 설정하면 챗봇이 이러한 필드에 대한 제안을 생성하도록 강제합니다.

```typescript
import { isAIMessage } from "@langchain/core/messages";

const userInput =
  "Can you look up when LangGraph was released? " +
  "When you have the answer, use the humanAssistance tool for review.";

const events = await graph.stream(
  { messages: [{ role: "user", content: userInput }] },
  { configurable: { thread_id: "1" }, streamMode: "values" }
);

for await (const event of events) {
  if ("messages" in event) {
    const lastMessage = event.messages.at(-1);

    console.log(
      "=".repeat(32),
      `${lastMessage?.getType()} Message`,
      "=".repeat(32)
    );
    console.log(lastMessage?.text);

    if (
      lastMessage &&
      isAIMessage(lastMessage) &&
      lastMessage.tool_calls?.length
    ) {
      console.log("Tool Calls:");
      for (const call of lastMessage.tool_calls) {
        console.log(`  ${call.name} (${call.id})`);
        console.log(`  Args: ${JSON.stringify(call.args)}`);
      }
    }
  }
}
```

:::

```
================================ Human Message =================================

Can you look up when LangGraph was released? When you have the answer, use the human_assistance tool for review.
================================== Ai Message ==================================

[{'text': "Certainly! I'll start by searching for information about LangGraph's release date using the Tavily search function. Then, I'll use the human_assistance tool for review.", 'type': 'text'}, {'id': 'toolu_01JoXQPgTVJXiuma8xMVwqAi', 'input': {'query': 'LangGraph release date'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
Tool Calls:
  tavily_search_results_json (toolu_01JoXQPgTVJXiuma8xMVwqAi)
 Call ID: toolu_01JoXQPgTVJXiuma8xMVwqAi
  Args:
    query: LangGraph release date
================================= Tool Message =================================
Name: tavily_search_results_json

[{"url": "https://blog.langchain.dev/langgraph-cloud/", "content": "We also have a new stable release of LangGraph. By LangChain 6 min read Jun 27, 2024 (Oct '24) Edit: Since the launch of LangGraph Platform, we now have multiple deployment options alongside LangGraph Studio - which now fall under LangGraph Platform. LangGraph Platform is synonymous with our Cloud SaaS deployment option."}, {"url": "https://changelog.langchain.com/announcements/langgraph-cloud-deploy-at-scale-monitor-carefully-iterate-boldly", "content": "LangChain - Changelog | ☁ 🚀 LangGraph Platform: Deploy at scale, monitor LangChain LangSmith LangGraph LangChain LangSmith LangGraph LangChain LangSmith LangGraph LangChain Changelog Sign up for our newsletter to stay up to date DATE: The LangChain Team LangGraph LangGraph Platform ☁ 🚀 LangGraph Platform: Deploy at scale, monitor carefully, iterate boldly DATE: June 27, 2024 AUTHOR: The LangChain Team LangGraph Platform is now in closed beta, offering scalable, fault-tolerant deployment for LangGraph agents. LangGraph Platform also includes a new playground-like studio for debugging agent failure modes and quick iteration: Join the waitlist today for LangGraph Platform. And to learn more, read our blog post announcement or check out our docs. Subscribe By clicking subscribe, you accept our privacy policy and terms and conditions."}]
================================== Ai Message ==================================

[{'text': "Based on the search results, it appears that LangGraph was already in existence before June 27, 2024, when LangGraph Platform was announced. However, the search results don't provide a specific release date for the original LangGraph. \n\nGiven this information, I'll use the human_assistance tool to review and potentially provide more accurate information about LangGraph's initial release date.", 'type': 'text'}, {'id': 'toolu_01JDQAV7nPqMkHHhNs3j3XoN', 'input': {'name': 'Assistant', 'birthday': '2023-01-01'}, 'name': 'human_assistance', 'type': 'tool_use'}]
Tool Calls:
  human_assistance (toolu_01JDQAV7nPqMkHHhNs3j3XoN)
 Call ID: toolu_01JDQAV7nPqMkHHhNs3j3XoN
  Args:
    name: Assistant
    birthday: 2023-01-01
```

:::python
`human_assistance` 도구에서 다시 `interrupt`에 도달했습니다.
:::

:::js
`humanAssistance` 도구에서 다시 `interrupt`에 도달했습니다.
:::

## 4. 사람의 도움 추가

챗봇이 올바른 날짜를 식별하지 못했으므로 정보를 제공합니다:

:::python

```python
human_command = Command(
    resume={
        "name": "LangGraph",
        "birthday": "Jan 17, 2024",
    },
)

events = graph.stream(human_command, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

:::

:::js

```typescript
import { Command } from "@langchain/langgraph";

const humanCommand = new Command({
  resume: {
    name: "LangGraph",
    birthday: "Jan 17, 2024",
  },
});

const resumeEvents = await graph.stream(humanCommand, {
  configurable: { thread_id: "1" },
  streamMode: "values",
});

for await (const event of resumeEvents) {
  if ("messages" in event) {
    const lastMessage = event.messages.at(-1);

    console.log(
      "=".repeat(32),
      `${lastMessage?.getType()} Message`,
      "=".repeat(32)
    );
    console.log(lastMessage?.text);

    if (
      lastMessage &&
      isAIMessage(lastMessage) &&
      lastMessage.tool_calls?.length
    ) {
      console.log("Tool Calls:");
      for (const call of lastMessage.tool_calls) {
        console.log(`  ${call.name} (${call.id})`);
        console.log(`  Args: ${JSON.stringify(call.args)}`);
      }
    }
  }
}
```

:::

```
================================== Ai Message ==================================

[{'text': "Based on the search results, it appears that LangGraph was already in existence before June 27, 2024, when LangGraph Platform was announced. However, the search results don't provide a specific release date for the original LangGraph. \n\nGiven this information, I'll use the human_assistance tool to review and potentially provide more accurate information about LangGraph's initial release date.", 'type': 'text'}, {'id': 'toolu_01JDQAV7nPqMkHHhNs3j3XoN', 'input': {'name': 'Assistant', 'birthday': '2023-01-01'}, 'name': 'human_assistance', 'type': 'tool_use'}]
Tool Calls:
  human_assistance (toolu_01JDQAV7nPqMkHHhNs3j3XoN)
 Call ID: toolu_01JDQAV7nPqMkHHhNs3j3XoN
  Args:
    name: Assistant
    birthday: 2023-01-01
================================= Tool Message =================================
Name: human_assistance

Made a correction: {'name': 'LangGraph', 'birthday': 'Jan 17, 2024'}
================================== Ai Message ==================================

Thank you for the human assistance. I can now provide you with the correct information about LangGraph's release date.

LangGraph was initially released on January 17, 2024. This information comes from the human assistance correction, which is more accurate than the search results I initially found.

To summarize:
1. LangGraph's original release date: January 17, 2024
2. LangGraph Platform announcement: June 27, 2024

It's worth noting that LangGraph had been in development and use for some time before the LangGraph Platform announcement, but the official initial release of LangGraph itself was on January 17, 2024.
```

이제 이러한 필드가 상태에 반영된 것을 확인할 수 있습니다:

:::python

```python
snapshot = graph.get_state(config)

{k: v for k, v in snapshot.values.items() if k in ("name", "birthday")}
```

```
{'name': 'LangGraph', 'birthday': 'Jan 17, 2024'}
```

:::

:::js

```typescript
const snapshot = await graph.getState(config);

const relevantState = Object.fromEntries(
  Object.entries(snapshot.values).filter(([k]) =>
    ["name", "birthday"].includes(k)
  )
);
```

```
{ name: 'LangGraph', birthday: 'Jan 17, 2024' }
```

:::

이를 통해 다운스트림 노드(예: 정보를 추가로 처리하거나 저장하는 노드)에서 쉽게 액세스할 수 있습니다.

## 5. 수동으로 상태 업데이트

:::python
LangGraph는 애플리케이션 상태에 대한 높은 수준의 제어를 제공합니다. 예를 들어, 언제든지(중단된 경우 포함) `graph.update_state`를 사용하여 키를 수동으로 재정의할 수 있습니다:

```python
graph.update_state(config, {"name": "LangGraph (library)"})
```

```
{'configurable': {'thread_id': '1',
  'checkpoint_ns': '',
  'checkpoint_id': '1efd4ec5-cf69-6352-8006-9278f1730162'}}
```

:::

:::js
LangGraph는 애플리케이션 상태에 대한 높은 수준의 제어를 제공합니다. 예를 들어, 언제든지(중단된 경우 포함) `graph.updateState`를 사용하여 키를 수동으로 재정의할 수 있습니다:

```typescript
await graph.updateState(
  { configurable: { thread_id: "1" } },
  { name: "LangGraph (library)" }
);
```

```typescript
{
  configurable: {
    thread_id: '1',
    checkpoint_ns: '',
    checkpoint_id: '1efd4ec5-cf69-6352-8006-9278f1730162'
  }
}
```

:::

## 6. 새 값 확인

:::python
`graph.get_state`를 호출하면 새 값이 반영된 것을 볼 수 있습니다:

```python
snapshot = graph.get_state(config)

{k: v for k, v in snapshot.values.items() if k in ("name", "birthday")}
```

```
{'name': 'LangGraph (library)', 'birthday': 'Jan 17, 2024'}
```

:::

:::js
`graph.getState`를 호출하면 새 값이 반영된 것을 볼 수 있습니다:

```typescript
const updatedSnapshot = await graph.getState(config);

const updatedRelevantState = Object.fromEntries(
  Object.entries(updatedSnapshot.values).filter(([k]) =>
    ["name", "birthday"].includes(k)
  )
);
```

```typescript
{ name: 'LangGraph (library)', birthday: 'Jan 17, 2024' }
```

:::

수동 상태 업데이트는 LangSmith에서 [trace를 생성](https://smith.langchain.com/public/7ebb7827-378d-49fe-9f6c-5df0e90086c8/r)합니다. 원하는 경우 [사람이 개입하는 워크플로를 제어](../../how-tos/human_in_the_loop/add-human-in-the-loop.md)하는 데에도 사용할 수 있습니다. 일반적으로 `interrupt` 함수 사용이 권장되는데, 이는 상태 업데이트와 독립적으로 사람이 개입하는 상호작용에서 데이터를 전송할 수 있기 때문입니다.

**축하합니다!** 더 복잡한 워크플로를 용이하게 하기 위해 상태에 사용자 정의 키를 추가했으며 도구 내부에서 상태 업데이트를 생성하는 방법을 배웠습니다.

이 튜토리얼의 그래프를 검토하려면 아래 코드 스니펫을 확인하세요:

:::python

{% include-markdown "../../../snippets/chat_model_tabs.md" %}

<!---
```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
```
-->

```python
from typing import Annotated

from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str

@tool
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    return Command(update=state_update)


tool = TavilySearch(max_results=2)
tools = [tool, human_assistance]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert(len(message.tool_calls) <= 1)
    return {"messages": [message]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```

:::

:::js

```typescript
import {
  Command,
  interrupt,
  MessagesZodState,
  MemorySaver,
  StateGraph,
  END,
  START,
} from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { ChatAnthropic } from "@langchain/anthropic";
import { TavilySearch } from "@langchain/tavily";
import { ToolMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const State = z.object({
  messages: MessagesZodState.shape.messages,
  name: z.string(),
  birthday: z.string(),
});

const humanAssistance = tool(
  async (input, config) => {
    // Note that because we are generating a ToolMessage for a state update, we
    // generally require the ID of the corresponding tool call. This is available
    // in the tool's config.
    const toolCallId = config?.toolCall?.id as string | undefined;
    if (!toolCallId) throw new Error("Tool call ID is required");

    const humanResponse = await interrupt({
      question: "Is this correct?",
      name: input.name,
      birthday: input.birthday,
    });

    // 도구 내부에서 ToolMessage로 명시적으로 상태를 업데이트합니다.
    const stateUpdate = (() => {
      // 정보가 올바른 경우 상태를 그대로 업데이트합니다.
      if (humanResponse.correct?.toLowerCase().startsWith("y")) {
        return {
          name: input.name,
          birthday: input.birthday,
          messages: [
            new ToolMessage({ content: "Correct", tool_call_id: toolCallId }),
          ],
        };
      }

      // 그렇지 않으면 사람 검토자로부터 정보를 받습니다.
      return {
        name: humanResponse.name || input.name,
        birthday: humanResponse.birthday || input.birthday,
        messages: [
          new ToolMessage({
            content: `Made a correction: ${JSON.stringify(humanResponse)}`,
            tool_call_id: toolCallId,
          }),
        ],
      };
    })();

    // 도구에서 Command 객체를 반환하여 상태를 업데이트합니다.
    return new Command({ update: stateUpdate });
  },
  {
    name: "humanAssistance",
    description: "Request assistance from a human.",
    schema: z.object({
      name: z.string().describe("엔티티의 이름"),
      birthday: z.string().describe("엔티티의 생일/출시일"),
    }),
  }
);

const searchTool = new TavilySearch({ maxResults: 2 });

const tools = [searchTool, humanAssistance];
const llmWithTools = new ChatAnthropic({
  model: "claude-3-5-sonnet-latest",
}).bindTools(tools);

const memory = new MemorySaver();

const chatbot = async (state: z.infer<typeof State>) => {
  const message = await llmWithTools.invoke(state.messages);
  return { messages: message };
};

const graph = new StateGraph(State)
  .addNode("chatbot", chatbot)
  .addNode("tools", new ToolNode(tools))
  .addConditionalEdges("chatbot", toolsCondition, ["tools", END])
  .addEdge("tools", "chatbot")
  .addEdge(START, "chatbot")
  .compile({ checkpointer: memory });
```

:::

## 다음 단계

LangGraph 기본 튜토리얼을 마치기 전에 검토할 개념이 하나 더 있습니다: `checkpointing`과 `state updates`를 [시간 여행](./6-time-travel.md)에 연결하는 것입니다.
