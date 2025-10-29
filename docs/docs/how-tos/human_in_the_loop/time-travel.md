# 타임 트래블 사용

LangGraph에서 [타임 트래블](../../concepts/time-travel.md)을 사용하려면:

:::python

1. [그래프 실행](#1-run-the-graph): @[`invoke`][CompiledStateGraph.invoke] 또는 @[`stream`][CompiledStateGraph.stream] 메서드를 사용하여 초기 입력으로 그래프를 실행합니다.
2. [기존 스레드에서 체크포인트 식별](#2-identify-a-checkpoint): @[`get_state_history()`][get_state_history] 메서드를 사용하여 특정 `thread_id`에 대한 실행 히스토리를 가져오고 원하는 `checkpoint_id`를 찾습니다.
   또는 실행을 일시 중지하려는 노드 앞에 [interrupt](../../how-tos/human_in_the_loop/add-human-in-the-loop.md)를 설정합니다. 그러면 해당 인터럽트까지 기록된 가장 최근 체크포인트를 찾을 수 있습니다.
3. [그래프 상태 업데이트 (선택사항)](#3-update-the-state-optional): @[`update_state`][update_state] 메서드를 사용하여 체크포인트에서 그래프의 상태를 수정하고 대안 상태에서 실행을 재개합니다.
4. [체크포인트에서 실행 재개](#4-resume-execution-from-the-checkpoint): 입력이 `None`이고 적절한 `thread_id` 및 `checkpoint_id`를 포함하는 구성과 함께 `invoke` 또는 `stream` 메서드를 사용합니다.
   :::

:::js

1. [그래프 실행](#1-run-the-graph): @[`invoke`][CompiledStateGraph.invoke] 또는 @[`stream`][CompiledStateGraph.stream] 메서드를 사용하여 초기 입력으로 그래프를 실행합니다.
2. [기존 스레드에서 체크포인트 식별](#2-identify-a-checkpoint): @[`getStateHistory()`][get_state_history] 메서드를 사용하여 특정 `thread_id`에 대한 실행 히스토리를 가져오고 원하는 `checkpoint_id`를 찾습니다.
   또는 실행을 일시 중지하려는 노드 앞에 [breakpoint](../../concepts/breakpoints.md)를 설정합니다. 그러면 해당 breakpoint까지 기록된 가장 최근 체크포인트를 찾을 수 있습니다.
3. [그래프 상태 업데이트 (선택사항)](#3-update-the-state-optional): @[`updateState`][update_state] 메서드를 사용하여 체크포인트에서 그래프의 상태를 수정하고 대안 상태에서 실행을 재개합니다.
4. [체크포인트에서 실행 재개](#4-resume-execution-from-the-checkpoint): 입력이 `null`이고 적절한 `thread_id` 및 `checkpoint_id`를 포함하는 구성과 함께 `invoke` 또는 `stream` 메서드를 사용합니다.
   :::

!!! tip

    타임 트래블에 대한 개념적 개요는 [Time travel](../../concepts/time-travel.md)을 참조하세요.

## 워크플로우에서

이 예제는 농담 주제를 생성하고 LLM을 사용하여 농담을 작성하는 간단한 LangGraph 워크플로우를 구축합니다. 그래프를 실행하고, 과거 실행 체크포인트를 검색하고, 선택적으로 상태를 수정하고, 선택한 체크포인트에서 실행을 재개하여 대안 결과를 탐색하는 방법을 보여줍니다.

### 설정

먼저 필요한 패키지를 설치해야 합니다

:::python

```python
%%capture --no-stderr
%pip install --quiet -U langgraph langchain_anthropic
```

:::

:::js

```bash
npm install @langchain/langgraph @langchain/anthropic
```

:::

Next, we need to set API keys for Anthropic (the LLM we will use)

:::python

```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")
```

:::

:::js

```typescript
process.env.ANTHROPIC_API_KEY = "YOUR_API_KEY";
```

:::

<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

:::python

```python
import uuid

from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    topic: NotRequired[str]
    joke: NotRequired[str]


llm = init_chat_model(
    "anthropic:claude-3-7-sonnet-latest",
    temperature=0,
)


def generate_topic(state: State):
    """LLM call to generate a topic for the joke"""
    msg = llm.invoke("Give me a funny topic for a joke")
    return {"topic": msg.content}


def write_joke(state: State):
    """LLM call to write a joke based on the topic"""
    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}


# Build workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("generate_topic", generate_topic)
workflow.add_node("write_joke", write_joke)

# Add edges to connect nodes
workflow.add_edge(START, "generate_topic")
workflow.add_edge("generate_topic", "write_joke")
workflow.add_edge("write_joke", END)

# Compile
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
graph
```

:::

:::js

```typescript
import { v4 as uuidv4 } from "uuid";
import { z } from "zod";
import { StateGraph, START, END } from "@langchain/langgraph";
import { ChatAnthropic } from "@langchain/anthropic";
import { MemorySaver } from "@langchain/langgraph";

const State = z.object({
  topic: z.string().optional(),
  joke: z.string().optional(),
});

const llm = new ChatAnthropic({
  model: "claude-3-5-sonnet-latest",
  temperature: 0,
});

// Build workflow
const workflow = new StateGraph(State)
  // Add nodes
  .addNode("generateTopic", async (state) => {
    // LLM call to generate a topic for the joke
    const msg = await llm.invoke("Give me a funny topic for a joke");
    return { topic: msg.content };
  })
  .addNode("writeJoke", async (state) => {
    // LLM call to write a joke based on the topic
    const msg = await llm.invoke(`Write a short joke about ${state.topic}`);
    return { joke: msg.content };
  })
  // Add edges to connect nodes
  .addEdge(START, "generateTopic")
  .addEdge("generateTopic", "writeJoke")
  .addEdge("writeJoke", END);

// Compile
const checkpointer = new MemorySaver();
const graph = workflow.compile({ checkpointer });
```

:::

### 1. Run the graph

:::python

```python
config = {
    "configurable": {
        "thread_id": uuid.uuid4(),
    }
}
state = graph.invoke({}, config)

print(state["topic"])
print()
print(state["joke"])
```

:::

:::js

```typescript
const config = {
  configurable: {
    thread_id: uuidv4(),
  },
};

const state = await graph.invoke({}, config);

console.log(state.topic);
console.log();
console.log(state.joke);
```

:::

**Output:**

```
How about "The Secret Life of Socks in the Dryer"? You know, exploring the mysterious phenomenon of how socks go into the laundry as pairs but come out as singles. Where do they go? Are they starting new lives elsewhere? Is there a sock paradise we don't know about? There's a lot of comedic potential in the everyday mystery that unites us all!

# The Secret Life of Socks in the Dryer

I finally discovered where all my missing socks go after the dryer. Turns out they're not missing at all—they've just eloped with someone else's socks from the laundromat to start new lives together.

My blue argyle is now living in Bermuda with a red polka dot, posting vacation photos on Sockstagram and sending me lint as alimony.
```

### 2. Identify a checkpoint

:::python

```python
# The states are returned in reverse chronological order.
states = list(graph.get_state_history(config))

for state in states:
    print(state.next)
    print(state.config["configurable"]["checkpoint_id"])
    print()
```

**Output:**

```
()
1f02ac4a-ec9f-6524-8002-8f7b0bbeed0e

('write_joke',)
1f02ac4a-ce2a-6494-8001-cb2e2d651227

('generate_topic',)
1f02ac4a-a4e0-630d-8000-b73c254ba748

('__start__',)
1f02ac4a-a4dd-665e-bfff-e6c8c44315d9
```

:::

:::js

```typescript
// The states are returned in reverse chronological order.
const states = [];
for await (const state of graph.getStateHistory(config)) {
  states.push(state);
}

for (const state of states) {
  console.log(state.next);
  console.log(state.config.configurable?.checkpoint_id);
  console.log();
}
```

**Output:**

```
[]
1f02ac4a-ec9f-6524-8002-8f7b0bbeed0e

['writeJoke']
1f02ac4a-ce2a-6494-8001-cb2e2d651227

['generateTopic']
1f02ac4a-a4e0-630d-8000-b73c254ba748

['__start__']
1f02ac4a-a4dd-665e-bfff-e6c8c44315d9
```

:::

:::python

```python
# This is the state before last (states are listed in chronological order)
selected_state = states[1]
print(selected_state.next)
print(selected_state.values)
```

**Output:**

```
('write_joke',)
{'topic': 'How about "The Secret Life of Socks in the Dryer"? You know, exploring the mysterious phenomenon of how socks go into the laundry as pairs but come out as singles. Where do they go? Are they starting new lives elsewhere? Is there a sock paradise we don\\'t know about? There\\'s a lot of comedic potential in the everyday mystery that unites us all!'}
```

:::

:::js

```typescript
// This is the state before last (states are listed in chronological order)
const selectedState = states[1];
console.log(selectedState.next);
console.log(selectedState.values);
```

**Output:**

```
['writeJoke']
{'topic': 'How about "The Secret Life of Socks in the Dryer"? You know, exploring the mysterious phenomenon of how socks go into the laundry as pairs but come out as singles. Where do they go? Are they starting new lives elsewhere? Is there a sock paradise we don\\'t know about? There\\'s a lot of comedic potential in the everyday mystery that unites us all!'}
```

:::

### 3. Update the state (optional)

:::python
`update_state` will create a new checkpoint. The new checkpoint will be associated with the same thread, but a new checkpoint ID.

```python
new_config = graph.update_state(selected_state.config, values={"topic": "chickens"})
print(new_config)
```

**Output:**

```
{'configurable': {'thread_id': 'c62e2e03-c27b-4cb6-8cea-ea9bfedae006', 'checkpoint_ns': '', 'checkpoint_id': '1f02ac4a-ecee-600b-8002-a1d21df32e4c'}}
```

:::

:::js
`updateState` will create a new checkpoint. The new checkpoint will be associated with the same thread, but a new checkpoint ID.

```typescript
const newConfig = await graph.updateState(selectedState.config, {
  topic: "chickens",
});
console.log(newConfig);
```

**Output:**

```
{'configurable': {'thread_id': 'c62e2e03-c27b-4cb6-8cea-ea9bfedae006', 'checkpoint_ns': '', 'checkpoint_id': '1f02ac4a-ecee-600b-8002-a1d21df32e4c'}}
```

:::

### 4. Resume execution from the checkpoint

:::python

```python
graph.invoke(None, new_config)
```

**Output:**

```python
{'topic': 'chickens',
 'joke': 'Why did the chicken join a band?\n\nBecause it had excellent drumsticks!'}
```

:::

:::js

```typescript
await graph.invoke(null, newConfig);
```

**Output:**

```typescript
{
  'topic': 'chickens',
  'joke': 'Why did the chicken join a band?\n\nBecause it had excellent drumsticks!'
}
```

:::
